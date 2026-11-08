"""Seal the prepared dataset ZIP and a release lock for the Colab notebook."""
import argparse
import json
from pathlib import Path
import zipfile

from datamate.domain import json_text
from datamate.models.artifacts import sha256
from datamate.training.seed import verify_seed_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default='outputs/training/seed72_100_gemma_v1')
    parser.add_argument('--output',default='outputs/seed72_100_gemma_v1.zip')
    parser.add_argument('--lock',default='configs/gemma_seed72_100_release.json')
    args=parser.parse_args()
    dataset=Path(args.dataset);manifest=verify_seed_dataset(dataset)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.resolve().is_relative_to(dataset.resolve()): raise ValueError('ZIP phải nằm ngoài dataset')
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as handle:
        for name in sorted(set(manifest['files'])|{'manifest.json'}):
            handle.write(dataset/name,name)
    with zipfile.ZipFile(output) as handle:
        assert handle.testzip() is None
        assert all(handle.read(name)==(dataset/name).read_bytes() for name in handle.namelist())
    lock={'dataset_name':dataset.name,'target_model_key':manifest['target_model_key'],'archive_filename':output.name,'archive_sha256':sha256(output),
          'dataset_manifest_sha256':sha256(dataset/'manifest.json'),
          'source_fingerprint':manifest['source_fingerprint'],'train_examples':manifest['train_examples'],
          'target_pipeline_version':manifest['target_pipeline_version']}
    Path(args.lock).write_text(json_text(lock,indent=2)+'\n',encoding='utf-8')
    print(json_text(lock,indent=2),flush=True)


if __name__=='__main__': main()
