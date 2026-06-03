import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Set


def load_train_prompts(path='data/pickscore_train_prompts.json', n=5000, seed=42, data_dir='data/'):
    captions = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            caption = json.loads(line).get('caption', '').strip()
            if caption:
                captions.append(caption)

    seen = set()
    unique = [c for c in captions if not (c in seen or seen.add(c))]

    eval_prompts = get_eval_prompt_set(data_dir)
    filtered = [c for c in unique if c not in eval_prompts]

    print(f"[prompts] total={len(captions)} unique={len(unique)} filtered={len(filtered)}")

    if len(filtered) < n:
        print(f"[prompts] WARNING: only {len(filtered)} available, requested {n}")
        n = len(filtered)

    rng = random.Random(seed)
    rng.shuffle(filtered)
    return filtered[:n]


def load_eval_prompts(benchmark: str, data_dir='data/') -> List[Dict]:
    csv_map = {
        'pickscore': 'pickscore.csv',
        'hpd': 'HPD_prompt.csv',
        'drawbench': 'drawbench.csv',
    }
    csv_path = Path(data_dir) / csv_map[benchmark]

    results = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            prompt = (row.get('prompt') or row.get('Prompt') or
                      row.get('caption') or row.get('Caption') or '').strip()
            if not prompt:
                continue
            entry = {'prompt': prompt}
            cat = row.get('Category') or row.get('category')
            if cat:
                entry['category'] = cat.strip()
            results.append(entry)
    return results


def get_eval_prompt_set(data_dir='data/') -> Set[str]:
    eval_prompts = set()
    for bm in ['pickscore', 'hpd', 'drawbench']:
        try:
            for e in load_eval_prompts(bm, data_dir):
                eval_prompts.add(e['prompt'])
        except (FileNotFoundError, KeyError):
            pass

    # Also exclude GenEval prompts (553 structured evaluation tasks)
    geneval_path = Path(data_dir) / 'evaluation_metadata.jsonl'
    if geneval_path.exists():
        with open(geneval_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    eval_prompts.add(json.loads(line)['prompt'].strip())

    # Exclude style evaluation prompts
    for style_file in ['anime.json', 'concept-art.json', 'paintings.json', 'photo.json']:
        style_path = Path(data_dir) / style_file
        if style_path.exists():
            with open(style_path, 'r') as f:
                for prompt in json.load(f):
                    if isinstance(prompt, str) and prompt.strip():
                        eval_prompts.add(prompt.strip())

    return eval_prompts
