"""Integration audit: read each business, build scratch catalog, record failures."""
import argparse
import time
from pathlib import Path
from datamate.data.ingest import Ingestor
from datamate.data.split import DatasetSplit
from datamate.domain import json_text

args=argparse.ArgumentParser()
args.add_argument("--split-root",default="train/train")
args.add_argument("--output",default="outputs/ingestion-check.json")
options=args.parse_args()
split=DatasetSplit(options.split_root)
ingestor=Ingestor(split,".cache/datamate")
report=[]
for business in split.businesses():
    before=time.monotonic()
    try:
        catalog=ingestor.catalog(business)
        row={"business_id":business,"ok":True,"tables_and_views":len(catalog.tables),
             "raw_rows":sum(t.get("rows",0) for t in catalog.tables if t["name"]!="calendar"),
             "warnings":catalog.warnings,"header_after_first_row":[{"table":t["name"],"row":t["header_row"]}
                for t in catalog.tables if t.get("header_row",0)>1]}
    except Exception as exc:
        row={"business_id":business,"ok":False,"error":str(exc)}
    row["ingestion_seconds"]=round(time.monotonic()-before,3)
    report.append(row)
    print(json_text({k:v for k,v in row.items() if k not in {"warnings","header_after_first_row"}}),flush=True)
out=Path(options.output)
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json_text(report,indent=2),encoding="utf-8")
if any(not row["ok"] for row in report):
    raise SystemExit(1)
