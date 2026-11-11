"""Count a saved conversation with a local tokenizer, without loading weights/data.

Input: JSON array, or an object with input_messages/initial_messages.
Run this in the GPU environment (or a tokenizer-only CPU environment).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from datamate.config import configure_offline
from datamate.domain import DataMateError
from datamate.models.chat_format import context_budget, encode_messages, input_ids


def check(tokenizer, messages, output_tokens, context_tokens):
    if not isinstance(messages, list) or not messages or any(
        not isinstance(m, dict) or m.get('role') not in {'system', 'user', 'assistant', 'tool'} or 'content' not in m
        for m in messages
    ):
        raise DataMateError('Cần JSON array conversation hợp lệ, có role/content')
    count = len(input_ids(encode_messages(tokenizer, messages)))
    return context_budget(count, output_tokens, context_tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--messages', required=True, type=Path)
    parser.add_argument('--output-tokens', required=True, type=int)
    parser.add_argument('--context-tokens', required=True, type=int)
    args = parser.parse_args()
    try:
        payload = json.loads(args.messages.read_text(encoding='utf-8-sig'))
        messages = payload if isinstance(payload, list) else payload.get('input_messages', payload.get('initial_messages', payload.get('prompt_messages')))
        configure_offline('.cache')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True, trust_remote_code=False)
        print(json.dumps(check(tokenizer, messages, args.output_tokens, args.context_tokens)))
    except (ImportError, DataMateError, OSError, ValueError, AttributeError) as exc:
        parser.exit(2, f'Context preflight: {exc}\n')


if __name__ == '__main__':
    main()
