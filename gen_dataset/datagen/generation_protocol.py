"""Fixed generation protocols used by dataset and predictor inference code."""


GENERATION_PROTOCOLS = {
    "sdxl": {
        "num_inference_steps": 50,
        "guidance_scale": 5.5,
        "scheduler": "DPMSolverMultistepScheduler",
        "protocol_note": "fixed SDXL experimental protocol",
    },
    "dreamshaper": {
        "num_inference_steps": 50,
        "guidance_scale": 2.0,
        "scheduler": "DPMSolverMultistepScheduler",
        "protocol_note": "DreamShaper controlled SDXL-family 50-step protocol",
    },
    "hunyuan_dit": {
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "scheduler": None,
        "protocol_note": "Diffusers Hunyuan-DiT call defaults",
    },
    "pixart_sigma": {
        "num_inference_steps": 20,
        "guidance_scale": 4.5,
        "scheduler": None,
        "protocol_note": "Diffusers PixArt-Sigma call defaults",
    },
    "sana_sprint": {
        "num_inference_steps": 4,
        "guidance_scale": 4.5,
        "scheduler": None,
        "protocol_note": "SANA-Sprint existing project protocol",
    },
}


def get_generation_protocol(model_type: str) -> dict:
    if model_type not in GENERATION_PROTOCOLS:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Available: {sorted(GENERATION_PROTOCOLS)}"
        )
    return GENERATION_PROTOCOLS[model_type].copy()


def apply_generation_scheduler(pipe, scheduler_name: str | None):
    if not scheduler_name:
        return pipe

    import diffusers

    scheduler_cls = getattr(diffusers, scheduler_name)
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
    return pipe
