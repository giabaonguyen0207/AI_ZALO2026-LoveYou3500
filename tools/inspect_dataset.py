"""Read-only dataset inventory; writes only audit metadata under docs/."""
from pathlib import Path
from collections import Counter
import csv
import difflib
import json
import sqlite3
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
report = {'audit_date': '2026-10-31', 'splits': {}}
for name in ('train', 'public_test'):
    base = ROOT / name / name
    questions = [json.loads(s) for s in (base / 'questions.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    businesses = sorted(p for p in (base / 'data').iterdir() if p.is_dir())
    files = list((base / 'data').rglob('*'))
    files = [p for p in files if p.is_file()]
    item = {'questions': len(questions), 'businesses': len(businesses),
            'question_keys': sorted({k for q in questions for k in q}),
            'questions_per_business': dict(sorted(Counter(q['business_id'] for q in questions).items())),
            'file_formats': dict(Counter(p.suffix for p in files)),
            'bytes': sum(p.stat().st_size for p in files), 'tables': [],
            'readme_summaries': {p.name: (p/'README.md').read_text(encoding='utf-8-sig').splitlines()[:6] for p in businesses}}
    if name == 'train':
        answers = [json.loads(s) for s in (base/'answers.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        item['answers'] = len(answers)
        item['answer_types'] = dict(Counter(a['type'] for a in answers))
        item['answer_keys'] = sorted({k for a in answers for k in a})
        item['duplicate_question_ids'] = len(questions) - len({q['id'] for q in questions})
        item['duplicate_answer_ids'] = len(answers) - len({a['id'] for a in answers})
        item['id_sets_equal'] = {q['id'] for q in questions} == {a['id'] for a in answers}
    for p in sorted(files):
        rel = str(p.relative_to(ROOT))
        if p.suffix == '.csv':
            with p.open(encoding='utf-8-sig', newline='') as f:
                sample = f.read(8192)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=',;\t|')
                except csv.Error:
                    dialect = csv.excel
                rows = csv.reader(f, dialect)
                header = next(rows, [])
                n = sum(1 for _ in rows)
            item['tables'].append({'file': rel, 'rows': n, 'header': header})
        elif p.suffix == '.sqlite':
            with sqlite3.connect(p.as_uri() + '?mode=ro', uri=True) as con:
                for (table,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
                    quoted = '"' + table.replace('"', '""') + '"'
                    cols = [r[1] for r in con.execute('PRAGMA table_info(' + quoted + ')')]
                    n = con.execute('SELECT count(*) FROM ' + quoted).fetchone()[0]
                    item['tables'].append({'file': rel, 'table': table, 'rows': n, 'header': cols})
        elif p.suffix == '.xlsx':
            with zipfile.ZipFile(p) as z:
                wb = ET.fromstring(z.read('xl/workbook.xml'))
                sheets = [s.attrib['name'] for s in wb.findall('s:sheets/s:sheet', NS)]
                sheet_paths = sorted(n for n in z.namelist() if n.startswith('xl/worksheets/sheet') and n.endswith('.xml'))
                for sp in sheet_paths:
                    # Dimension is a worksheet extent, not a count of business records.
                    with z.open(sp) as f:
                        dimension = None
                        for event, element in ET.iterparse(f, events=('start',)):
                            if element.tag.endswith('}dimension'):
                                dimension = element.attrib.get('ref')
                                break
                            if element.tag.endswith('}sheetData'):
                                break
                    item['tables'].append({'file': rel, 'sheet_xml': sp,
                                           'workbook_sheets': sheets, 'dimension': dimension})
    report['splits'][name] = item

out = ROOT / 'docs' / 'dataset-audit.json'
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
train_rules = (ROOT/'train/train/rulebook.md').read_text(encoding='utf-8-sig')
public_rules = (ROOT/'public_test/public_test/rulebook.md').read_text(encoding='utf-8-sig')
diff = '\n'.join(difflib.unified_diff(train_rules.splitlines(), public_rules.splitlines(), fromfile='train/rulebook.md', tofile='public_test/rulebook.md', lineterm=''))
(ROOT/'docs/rulebook-diff.txt').write_text(diff, encoding='utf-8')
for name, item in report['splits'].items():
    print(json.dumps({'split': name, **{k: v for k, v in item.items() if k not in ('tables', 'readme_summaries', 'questions_per_business')},
                      'csv_sqlite_rows': sum(t.get('rows', 0) for t in item['tables']),
                      'xlsx_sheets': sum('sheet_xml' in t for t in item['tables'])}, ensure_ascii=False))
print(diff)
for name, item in report['splits'].items():
    print(name, 'README summaries:', json.dumps(item['readme_summaries'], ensure_ascii=False))
