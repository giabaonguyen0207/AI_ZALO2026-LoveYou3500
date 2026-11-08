"""Resolve serial-date epochs from source metadata and cross-column evidence."""
from datetime import datetime
import re

from .transforms import iso_date,typed_date


def date_epoch(table, rows, binding, cutoff):
    if table.get('excel_epoch'):
        return table['excel_epoch'], {'source':'workbook_metadata'}
    dates=[column for role,column in binding['roles'].items() if role in
           {'order_date','payment_date','purchase_date','stock_timestamp'}]
    serials=[];paired=[]
    timestamp=binding['roles'].get('timestamp')
    for row in rows:
        for column in dates:
            value=row.get(column)
            if value is None or not re.fullmatch(r'\d{5}(?:\.\d+)?',str(value).strip()):continue
            serials.append(value)
            if timestamp and column==binding['roles'].get('order_date'):
                try:paired.append((value,iso_date(str(row[timestamp])[:10])))
                except (ValueError,TypeError):pass
    if not serials:return None,None
    candidates=[]
    for epoch in ('1899-12-30T00:00:00','1904-01-01T00:00:00'):
        try:
            resolved=[typed_date(value,None,epoch) for value in set(serials)]
            agrees=all(typed_date(value,None,epoch)==day for value,day in paired)
            # A declared last source date is independent of question as_of_date.
            if agrees and (not cutoff or max(resolved)<=cutoff) and (paired or cutoff):candidates.append(epoch)
        except (ValueError,TypeError,OverflowError):pass
    if len(candidates)==1:
        return candidates[0],{'source':'timestamp_or_declared_coverage','paired_rows':len(paired),
                              'serial_rows':len(serials),'declared_end':cutoff,'candidates':candidates}
    return None,{'source':'unresolved_serial_epoch','serial_rows':len(serials),'candidates':candidates}
