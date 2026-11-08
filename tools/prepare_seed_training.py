"""Prepare the reviewed 72+100 TRAIN seed; no model inference or GPU training."""
import argparse
from datamate.domain import json_text
from datamate.training.seed import DEFAULT_RUNS,prepare_seed,verify_seed_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-root',default='train/train')
    parser.add_argument('--output',default='outputs/training/seed72_100_gemma_v1')
    parser.add_argument('--reviews',default='configs/seed72_100_reviews.json')
    parser.add_argument('--sources',default='configs/training_sources.json')
    parser.add_argument('--cache-dir',default='.cache/datamate-v3-probes')
    parser.add_argument('--runs',nargs='+',default=list(DEFAULT_RUNS))
    parser.add_argument('--max-length',type=int,default=16384)
    parser.add_argument('--max-target-tokens',type=int,default=768)
    args=parser.parse_args()
    summary=prepare_seed(args.split_root,args.output,reviews=args.reviews,sources=args.sources,
                         cache_dir=args.cache_dir,max_length=args.max_length,max_target_tokens=args.max_target_tokens,runs=args.runs,
                         tokenizer_dirs={'gemma4-12b':'.cache/tokenizers/gemma4-12b'})
    verify_seed_dataset(args.output)
    print(json_text(summary,indent=2),flush=True)


if __name__=='__main__': main()
