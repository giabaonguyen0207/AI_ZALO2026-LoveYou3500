"""Save the uniform training-question sample used by the third inference run."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'src'))
    from datamate.data.split import DatasetSplit, ensure_outside_dataset
    from datamate.domain import json_text
    from tools.diagnose_train import SAMPLE_SIZE, SAMPLING_SEED, diagnostic_plan

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-root', default=str(root / 'train/train'))
    parser.add_argument('--output-dir', default=str(root / 'outputs/inference-round3-random100'))
    parser.add_argument('--size', type=int, default=SAMPLE_SIZE)
    parser.add_argument('--seed', type=int, default=SAMPLING_SEED)
    args = parser.parse_args()
    split = DatasetSplit(args.split_root)
    questions = split.questions()
    selected, population, _, _ = diagnostic_plan(questions, {}, seed=args.seed,
                                                sample_size=args.size, sampling_mode='random')
    ids = [q.id for q in selected]
    content = ''.join(json_text({'id': q.id, 'business_id': q.business_id, **q.prompt_dict()}) + '\n' for q in selected)
    manifest = {'sampling': 'random', 'sampling_scope': 'all_training_questions',
                'algorithm': 'random.Random(seed).sample(questions sorted by id, sample_size)',
                'seed': args.seed, 'sample_size': args.size, 'population_size': len(questions),
                'population_businesses': population, 'sampling_without_replacement': True,
                'stratification': None, 'forced_questions': [], 'excluded_question_ids': [],
                'holdout_excluded': False, 'references_used_for_selection': False,
                'source_questions_sha256': hashlib.sha256((split.root / 'questions.jsonl').read_bytes()).hexdigest(),
                'selected_questions_sha256': hashlib.sha256(content.encode()).hexdigest(),
                'question_ids': ids, 'status': 'selected_not_inferred'}
    output = ensure_outside_dataset(args.output_dir, split.root)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'selection_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Thư mục đã có mẫu khác; chọn output-dir mới')
    (output / 'questions.jsonl').write_text(content, encoding='utf-8', newline='\n')
    (output / 'question_ids.json').write_text(json_text(ids, indent=2) + '\n', encoding='utf-8')
    manifest_path.write_text(json_text(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Đã chọn {len(ids)}/{len(questions)} câu, seed={args.seed}: {output}')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
