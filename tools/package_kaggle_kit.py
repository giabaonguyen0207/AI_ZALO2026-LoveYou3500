"""Combine three kits as separate directories; safe for Kaggle auto-extraction."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def package(current,baseline,benchmark,output):
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    files={}
    inputs={'current':Path(current),'baseline':Path(baseline),'benchmark':Path(benchmark)}
    with zipfile.ZipFile(baseline) as frozen:
        commit=json.loads(frozen.comment)['git_commit']
    manifest={'format':'datamate-kaggle-kit-v1','baseline_commit':commit,
              'source_archives':{kind:{'filename':path.name,'sha256':digest(path)} for kind,path in inputs.items()},
              'files':files}
    temporary=output.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as destination:
        for kind,path in inputs.items():
            with zipfile.ZipFile(path) as source:
                for member in source.infolist():
                    if member.is_dir():continue
                    parts=Path(member.filename).parts
                    if Path(member.filename).is_absolute() or '..' in parts:
                        raise ValueError('Unsafe source archive path')
                    if member.filename.lower().endswith('.zip'):
                        raise ValueError('Upload kit must contain directories, not nested ZIPs')
                    name=kind+'/'+member.filename
                    sha=hashlib.sha256()
                    with source.open(member) as incoming,destination.open(name,'w') as outgoing:
                        while chunk:=incoming.read(1024*1024):
                            outgoing.write(chunk);sha.update(chunk)
                    files[name]=sha.hexdigest()
        destination.writestr('kaggle_kit.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(output)
    Path(str(output)+'.manifest.json').write_text(json.dumps({'sha256':digest(output),'files':len(files),
                  'baseline_commit':commit},indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--benchmark',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=package(args.current,args.baseline,args.benchmark,args.output)
    print(json.dumps({'file':str(args.output),'files':len(result['files']),'sha256':digest(args.output)}))


if __name__=='__main__':main()
