"""Merge dimension attributes by observed identity; ambiguous matches stay explicit."""
from collections import defaultdict
import json

from .transforms import search_key

PRODUCT_FIELDS=('product_group','prescription','base_unit','price','cost')
CUSTOMER_FIELDS=('customer_group','gender')


def dimensions(bindings, clean, entity):
    fields=PRODUCT_FIELDS if entity=='product' else CUSTOMER_FIELDS
    identifier,name=entity+'_id',entity+'_name'
    groups=defaultdict(list)
    relevant=[b for b in bindings if b['kind']==('products' if entity=='product' else 'customers')]
    relevant.sort(key=lambda b:-len(b['roles']))
    for binding in relevant:
        for row in clean[binding['table']]:
            pid,pname=(row.get(binding['roles'].get(role,'')) for role in (identifier,name))
            if not pid and not pname:continue
            key=('id',str(pid)) if pid is not None and str(pid).strip() else ('name',search_key(pname))
            record={role:row.get(binding['roles'].get(role,'')) for role in fields}
            if entity=='product':record['base_unit']=record.get('base_unit') or row.get(binding['roles'].get('unit',''))
            issues=list(row.get('_dm_issues',[]))
            issues += ['invalid_dimension:'+role for role in fields
                       if 'parse:'+binding['roles'].get(role,'') in issues]
            record.update({identifier:pid,name:pname,'_dm_source_table':binding['table'],
                           '_dm_issues':issues,
                           **{k:row.get(k) for k in ('_dm_file','_dm_sheet','_dm_row')}})
            groups[key].append(record)
    result=[];by_id={};by_name=defaultdict(list)
    for key,records in groups.items():
        record={**records[0]}
        issues=list(dict.fromkeys(issue for r in records for issue in r.get('_dm_issues',[])))
        names=[r[name] for r in records if r.get(name)]
        record[name]=names[0] if names else None
        for field in fields:
            values=[r[field] for r in records if r.get(field) is not None and str(r[field]).strip()]
            if field=='prescription':
                parsed=[]
                for value in values:
                    if search_key(str(value)) not in {'0','1','true','false','co','khong'}:
                        issues.append('invalid_dimension:'+field)
                    else:parsed.append(int(search_key(str(value)) in {'1','true','co'}))
                values=parsed
            keys={search_key(v) if isinstance(v,str) else v for v in values}
            record[field]=values[0] if len(keys)==1 else None
            if len(keys)>1:issues.append('conflicting_dimension:'+field)
        record['_dm_issues']=json.dumps(issues,ensure_ascii=False)
        result.append(record)
        if key[0]=='id':by_id[key[1]]=record
        for alias in {search_key(v) for v in names}:by_name[alias].append(record)
    return result,by_id,by_name


def resolve_dimension(identifier,name,index):
    _,by_id,by_name=index
    if identifier is not None and str(identifier).strip():
        return by_id.get(str(identifier)),False
    matches=by_name.get(search_key(name),[])
    return (matches[0],False) if len(matches)==1 else (None,len(matches)>1)
