"""Generate the Kaggle notebook from the copyable cells in huongdankaggle.md."""
import hashlib
import json
from pathlib import Path
import re


def build(project, kind='benchmark'):
    project=Path(project)
    guide=(project/'huongdankaggle.md').read_text(encoding='utf-8')
    ranges={'benchmark':('## 3. Chạy thử','## 6. So sánh'),
            'compare':('## 6. So sánh','## 8. Cấu hình')}
    first,last=ranges[kind]
    start,end=guide.index(first),guide.index(last)
    cells=[]
    def add(kind,source):
        if not source.strip(): return
        cell={'cell_type':kind,'id':hashlib.sha256((kind+source).encode()).hexdigest()[:12],
              'metadata':{},'source':source.splitlines(keepends=True)}
        if kind=='code':
            compile(source,'kaggle-benchmark-cell','exec')
            cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    add('markdown',guide[:guide.index('## 3. Chạy thử')])
    position=start
    for match in re.finditer(r'^```python\n(.*?)^```\s*$',guide[start:end],re.S|re.M):
        absolute_start=start+match.start()
        add('markdown',guide[position:absolute_start])
        add('code',match.group(1))
        position=start+match.end()
    add('markdown',guide[position:end])
    add('markdown','Xem huongdankaggle.md để đi tiếp theo thứ tự; tất cả bước chuẩn bị, chạy và so sánh đều trên Kaggle.')
    return {'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                                    'language_info':{'name':'python','version':'3.11'}},
            'nbformat':4,'nbformat_minor':5}


def main():
    root=Path(__file__).resolve().parents[1]
    names={'benchmark':'DataMate_Kaggle_Runtime_Benchmark.ipynb','compare':'DataMate_Kaggle_Compare.ipynb'}
    for kind,name in names.items():
        output=root/'notebooks'/name
        output.write_text(json.dumps(build(root,kind),ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
        print(output)


if __name__=='__main__': main()
