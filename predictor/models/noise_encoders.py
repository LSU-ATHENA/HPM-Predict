import torch
import torch.nn as nn


class ResidualConv(nn.Module):

    def __init__(self, in_channels: int = 4, spatial_size: int = 128):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.SiLU()
        self.do1 = nn.Dropout2d(0.3)
        self.skip_1 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0)

        self.ds1 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.ds_bn1 = nn.BatchNorm2d(64)
        self.ds_act1 = nn.SiLU()
        self.ds_do1 = nn.Dropout2d(0.3)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.SiLU()
        self.do2 = nn.Dropout2d(0.3)

        self.ds2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.ds_bn2 = nn.BatchNorm2d(128)
        self.ds_act2 = nn.SiLU()
        self.ds_do2 = nn.Dropout2d(0.3)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.SiLU()
        self.do3 = nn.Dropout2d(0.3)

        self.ds3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.ds_bn3 = nn.BatchNorm2d(256)
        self.ds_act3 = nn.SiLU()
        self.ds_do3 = nn.Dropout2d(0.3)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.act4 = nn.SiLU()
        self.do4 = nn.Dropout2d(0.3)

        self.ds4 = nn.Conv2d(256, 1024, kernel_size=5, stride=2, padding=2)
        self.ds_bn4 = nn.BatchNorm2d(1024)
        self.ds_act4 = nn.SiLU()
        self.ds_do4 = nn.Dropout2d(0.3)

        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.final_do = nn.Dropout(0.3)

        self._output_dim = 1024

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip_1(x)
        x = self.do1(self.act1(self.bn1(self.conv1(x)))) + identity
        x = self.ds_do1(self.ds_act1(self.ds_bn1(self.ds1(x))))
        x = self.do2(self.act2(self.bn2(self.conv2(x)))) + x
        x = self.ds_do2(self.ds_act2(self.ds_bn2(self.ds2(x))))
        x = self.do3(self.act3(self.bn3(self.conv3(x)))) + x
        x = self.ds_do3(self.ds_act3(self.ds_bn3(self.ds3(x))))
        x = self.do4(self.act4(self.bn4(self.conv4(x)))) + x
        x = self.ds_do4(self.ds_act4(self.ds_bn4(self.ds4(x))))

        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.final_do(x)
        return x


class SpatialShrinkNoiseEncoder(nn.Module):
    """Legacy lightweight noise encoder used by older PAINE checkpoints.

    nn.Sequential of N stride-2 Conv2d(in_ch, in_ch, k=3) -> BN -> SiLU stages,
    where N is chosen so the output spatial size lands at 16x16. Channels stay
    at in_channels throughout, so output dim = in_channels * 16 * 16 = 1024
    for the standard 4-channel DiT/SDXL latents.

    Hunyuan/PixArt/SDXL/DS at spatial_size=128 -> 3 stages (128 -> 64 -> 32 -> 16).
    PixArt-Alpha at spatial_size=64 -> 2 stages (64 -> 32 -> 16). Both produce
    the same output_dim=1024 so downstream fusion is unchanged.
    """

    TARGET_SPATIAL = 16  # final spatial dim before flatten

    def __init__(self, in_channels: int = 4, spatial_size: int = 128):
        super().__init__()
        n_stages = 0
        s = spatial_size
        while s > self.TARGET_SPATIAL:
            n_stages += 1
            s //= 2
        if s != self.TARGET_SPATIAL:
            raise ValueError(
                f"spatial_size={spatial_size} not reachable from {self.TARGET_SPATIAL} "
                f"by repeated halving (need a power-of-2 multiple)."
            )
        layers = []
        for _ in range(n_stages):
            layers += [
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(),
            ]
        self.conv = nn.Sequential(*layers)
        self._output_dim = in_channels * self.TARGET_SPATIAL * self.TARGET_SPATIAL

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x.flatten(start_dim=1)


def get_noise_encoder(name: str = 'residualconv', spatial_size: int = 128, **kwargs) -> nn.Module:
    name = (name or 'residualconv').lower()
    if name in ('residualconv', 'custom'):
        return ResidualConv(spatial_size=spatial_size, **kwargs)
    if name == 'spatial_shrink':
        return SpatialShrinkNoiseEncoder(spatial_size=spatial_size, **kwargs)
    raise ValueError(
        f"Unknown noise encoder name: '{name}'. "
        f"Available: residualconv, custom, spatial_shrink"
    )
