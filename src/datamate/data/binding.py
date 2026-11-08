"""Schema roles from the supplied README, with explicit header fallbacks.

No business/question identifiers or reference answers enter schema binding.
"""
from __future__ import annotations

import re
from .transforms import search_key


ALIASES = {
    "order_id": {"ma don hang", "ma hoa don", "order id", "ma bill", "don",'ma giao dich','so ct','so phieu','ma phieu','ma gd','so hoa don','orderid','invoice id'},
    "order_date": {"ngay ban", "ngay", "ngay thang", "ngay dat hang", "ngay giao dich", "ngay mua", "ngay lap"},
    "status": {"trang thai", "trang thai don", "trang thai don hang", "trang thai xu ly", "tinh trang don", "tinh trang", "status", "ket qua", "ghi chu tt",'state','order status'},
    "product_id": {"ma hang", "ma hang hoa", "ma san pham", "ma sku", "ma hh",'ma sp','ma mon','id sp'},
    "product_name": {"ten hang", "ten hang hoa", "ten san pham", "mat hang", "san pham", "hang", "hang hoa", "product", "mon",'chi tiet hang','ten sp'},
    "customer_id": {"ma kh", "ma khach hang", "customer id", "ma khach mua",'ma kh mua','ma khach le si'},
    "customer_name": {"khach hang", "ten khach hang", "ten khach", "ten kh", "khach", "kh", "tenkh",'khach hang ten','ben mua'},
    "quantity": {"so luong", "sl ban", "so luong san pham", "so luong ban", "so luong hang", "sl", "s luong", "s l",'slg','sluong','sl nhap','so luong nhap'},
    "base_quantity": {"so luong quy doi", "sl quy doi", "so luong le", "sl le", "sl co ban", "quy ra dv le", "sl quy ra le"},
    "line_amount": {"thanh tien", "thanh tien d", "tien ban", "tri gia", "thanh tien", "tien", "amount", "tong"},
    "header_revenue": {"tong tien hang", "tien hang", "tien hang truoc giam"},
    "order_total": {"tong sau giam gia", "phai thu", "tong tien", "tongtien", "tong thanh toan"},
    "balance": {"con no", "du no", "chua tra", "con phai thu"},
    "paid": {"khach da tra", "da tra", "da thu", "thanh toan"},
    "unit": {"don vi tinh", "don vi", "dvt", "dv tinh", "unit", "donvi", "d vi"},
    "base_unit": {"don vi co ban", "don vi co so", "don vi goc", "dv nho", "dv goc"},
    "large_unit": {"don vi quy doi", "dv lon", "dv nhap"},
    "factor": {"ti le quy doi", "he so", "he so qd", "he so quy doi", "quy doi", "bang"},
    "stock": {"ton cuoi ky", "ton cuoi", "ton kho", "ton thuc"},
    "price": {"don gia", "gia san pham", "gia ban", "gia ban le", "gia 1 dv", "gia le", "price",'gia','gia niem yet','gia nhap'},
    "cost": {"gia von", "gia von hang", "gia von luc ban", "gia von hien tai"},
    "line_no": {"stt", "tt dong"},
    "timestamp": {"thoi gian tao", "thoi gian lap", "gio tao don"}, "hour": {"gio"},
    "channel": {"kenh ban", "kenh ban hang", "nguon don", "loai ban"},
    "employee": {"nguoi ban", "ma nv ban"},
    "payment_method": {"phuong thuc thanh toan", "phuong thuc tt", "hinh thuc thanh toan", "phuong thuc", "pttt"},
    "payment_date": {"thoi diem", "thoi gian thu", "ngay gio thu", "ngay thu"}, "payment_amount": {"so tien thu", "gia tri"},
    "return_quantity": {"so luong tra", "so luong tra dvt co ban", "sl tra dvt co ban"},
    "cancel_reason": {"ly do huy", "nguyen nhan huy",'cancelreason','cancel reason','cancellation reason'},
    'purchase_date':{'ngay nhap','ngay nhap hang','ngay phieu nhap'},
    'purchase_id':{'ma nhap hang','ma phieu nhap','ma don nhap'},
    'receipt_id':{'phieu thu','ma phieu thu'},
    'product_group':{'nhom hang','nhom hang hoa','nhom hang 3 cap','danh muc','phan loai'},
    'customer_group':{'nhom khach','nhom khach hang','loai khach hang'},
    'gender':{'gioi tinh','nam nu','gender'},
    'prescription':{'ke don','thuoc ke don','ban theo don'},
    'return_reason':{'ly do tra','ly do tra hang','nguyen nhan tra hang'},
    'expiry_date':{'han dung','han su dung','ngay het han','expiry date'},
    'lot_id':{'ma lo','so lo'},
    'stock_timestamp':{'ngay gio phat sinh','thoi diem phat sinh'},
    'location_id':{'ma chi nhanh','ma kho','warehouse id'},
}


def header_key(column):
    separated = ''.join((' ' if i and char.isupper() and
        (column[i-1].islower() or column[i-1].isdigit() or
         column[i-1].isupper() and i+1<len(column) and column[i+1].islower()) else '')+char
        for i,char in enumerate(column))
    return search_key(separated)


def readme_descriptions(readme, table):
    result, active, active_level = {}, False, 0
    for line in readme.splitlines():
        if re.match(r'^#{1,4}\s', line):
            level=len(line)-len(line.lstrip('#'))
            quoted = re.findall(r"`([^`]+)`", line)
            # Subheadings without a new file keep the parent file's scope.
            file_heading=bool(quoted and quoted[0].casefold().endswith(('.csv','.xlsx','.sqlite','.db')))
            if not active or level<=active_level or file_heading:
                active = bool(quoted and quoted[0].casefold() == table["file"].casefold())
                if len(quoted) > 1:
                    active = active and quoted[1].casefold() == table.get("sheet", "").casefold()
                active_level=level
        if active:
            match = re.match(r"\s*[-*]\s*(?:\*\*|`)(.+?)(?:\*\*|`)\s*:\s*(.+)", line)
            if match:
                result[match[1]] = match[2]
            if line.strip().startswith('|'):
                cells=[cell.strip().strip('`* ') for cell in line.strip().strip('|').split('|')]
                if len(cells)>=2 and cells[0] in table['columns']:
                    result[cells[0]]=' '.join(cells[1:])
    return result


def description_role(description):
    text = search_key(description) or ""
    text = re.sub(r'^la\s+', '', text)
    # Specific roles precede the broad 'mã hàng', 'số lượng', 'tổng tiền'.
    patterns = [
        ("return_quantity", "so luong tra"), ("base_quantity", "so luong da doi"),
        ("base_quantity", "so luong da quy"), ("factor", "he so"),
        ("payment_date", "thoi diem thu"), ("payment_date", "ngay thu"),
        ("payment_amount", "so tien"), ("purchase_date", "ngay nhap"),
        ('receipt_id','ma phieu thu'),
        ("purchase_id", "ma phieu nhap"), ("purchase_id", "ma don nhap"),
        ("line_no", "so thu tu dong"), ("timestamp", "thoi diem dat"),
        ("timestamp", "thoi diem tao don"), ("order_date", "ngay ban"),
        ("order_date", "ngay dat"), ("order_id", "ma don"),
        ("status", "trang thai don"), ("customer_id", "ma khach"),
        ("customer_name", "ten khach"), ("product_id", "ma hang"),
        ("product_name", "ten hang"), ("product_name", "ten mat hang"),
        ("line_amount", "thanh tien sau giam"), ("line_amount", "thanh tien cua dong"),
        ("header_revenue", "tong tien hang"),
        ("order_total", "tong tien don"), ("order_total", "tong tien ca don"),
        ("order_total", "so khach phai tra cho ca don"),
        ("paid", "da thanh toan"), ("balance", "con no"), ("stock", "ton kho"),
        ("payment_method", "hinh thuc thanh toan"), ("payment_method", "phuong thuc thanh toan"),
        ("unit", "don vi tinh"), ("quantity", "so luong"),
        ("price", "don gia"), ("cost", "gia von"),
        ("cancel_reason", "ly do huy"), ("cancel_reason", "nguyen nhan huy"),
    ]
    if "cot trong" in text or "khong co du lieu" in text:
        return "empty"
    return next((role for role, pattern in patterns if text.startswith(pattern)), None)


def bind_table(table, readme):
    descriptions = readme_descriptions(readme, table)
    candidates, origins = {}, {}
    for column in table["columns"]:
        role = description_role(descriptions.get(column)) if column in descriptions else None
        origin = "README" if role else "header"
        if not role:
            # Split both MaKhachHang and acronym prefixes such as SLQuyDoi.
            # Unicode ranges such as à-ỹ also contain uppercase Vietnamese
            # letters. Use character categories so ĐVT stays one acronym.
            key = header_key(column)
            role = next((name for name, aliases in ALIASES.items() if key in aliases), None)
        if role and role != "empty":
            candidates.setdefault(role, []).append(column)
            origins[column] = origin
    # An explicit README definition outranks a generic header alias. In a
    # receipt ledger 'Mã phiếu' is a receipt key, while README may explicitly
    # identify 'Mã hóa đơn' as the order key. Do not discard both as ambiguous.
    selected={}
    for role,columns in candidates.items():
        documented=[c for c in columns if origins[c]=='README']
        selected[role]=documented if documented else columns
    ambiguous = {role: cols for role, cols in selected.items() if len(cols)>1}
    roles = {role: cols[0] for role, cols in selected.items() if len(cols)==1}
    generic_names=[c for c in table['columns'] if header_key(c) in {'ten', 'ten goi', 'name'}]
    if len(generic_names)==1 and 'order_id' not in roles:
        name_role = 'customer_name' if 'customer_id' in roles and 'product_id' not in roles else 'product_name' if 'product_id' in roles and 'customer_id' not in roles else None
        if name_role and name_role not in roles and name_role not in ambiguous:
            roles[name_role]=generic_names[0]
            origins[generic_names[0]]='header_and_identity_context'
    # A line amount called 'Tổng tiền' in a line-item export is not an order
    # total. The README generally resolves this; retain this structural fallback.
    if "order_total" in roles and "line_amount" not in roles and ("product_name" in roles or "product_id" in roles) and "quantity" in roles:
        roles["line_amount"] = roles.pop("order_total")
    # 'Thời gian' is shared by invoices and purchases. Use the declared file
    # relationship, not arbitrary filename/business identifiers.
    if "order_id" in roles and "order_date" not in roles and "Thời gian" in table["columns"]:
        roles["order_date"] = "Thời gian"
    if "order_id" in roles and "payment_amount" in roles and "payment_date" not in roles and "Thời gian" in table["columns"] and "status" not in roles:
        roles["payment_date"] = "Thời gian"
        roles.pop("order_date", None)
    if "order_id" in roles and "payment_date" in roles and "line_amount" in roles:
        roles["payment_amount"] = roles.pop("line_amount")
    if 'purchase_date' in roles and 'order_id' in roles and 'purchase_id' not in roles:
        roles['purchase_id']=roles.pop('order_id')
    if 'customer_id' in roles and 'order_id' not in roles and 'product_id' not in roles:
        for column in table['columns']:
            if header_key(column) in {'nhom','hang'}:
                roles.setdefault('customer_group',column)
    if 'order_id' in roles and 'product_id' not in roles and 'return_quantity' not in roles:
        for column in table['columns']:
            if header_key(column)=='nguyen nhan':roles.setdefault('return_reason',column)
    kind = "other"
    if "stock" in roles and ("product_id" in roles or "product_name" in roles):
        kind = "inventory"
    elif "factor" in roles and ("product_id" in roles or "product_name" in roles):
        kind = "conversion"
    elif "return_quantity" in roles and "order_id" in roles:
        kind = "returns"
    elif "order_id" in roles and "payment_date" in roles and "payment_amount" in roles:
        kind = "payments"
    elif "line_amount" in roles and ("product_name" in roles or "product_id" in roles) and "purchase_date" not in roles and 'purchase_id' not in roles:
        kind = "sales"
    elif "order_date" in roles and "order_id" in roles and "status" in roles:
        kind = "orders"
    elif "product_id" in roles and ("product_name" in roles or 'prescription' in roles) and "quantity" not in roles and 'lot_id' not in roles:
        kind = "products"
    elif "customer_id" in roles and "customer_name" in roles and "order_id" not in roles:
        kind = "customers"
    return {"table": table["name"], "kind": kind, "roles": roles, "ambiguous": ambiguous, "origins": origins,
            'unmapped':[c for c in table['columns'] if c not in roles.values()]}


def resolve_order_relationships(connection,bindings,tables):
    """Resolve competing foreign keys from full-column overlap, not IDs/names.

    Only a complete match to known order keys with no competing match is
    accepted. Uncertain relationships remain explicit rather than being guessed.
    """
    def quote(value):
        return '"'+value.replace('"','""')+'"'
    anchors=set()
    for binding in bindings:
        column=binding['roles'].get('order_id')
        if column and binding['kind'] in {'orders','sales'}:
            anchors.update(str(row[0]) for row in connection.execute(f'SELECT DISTINCT {quote(column)} FROM {quote(binding["table"])}') if row[0] is not None and str(row[0]).strip())
    if not anchors:
        return
    for binding in bindings:
        candidates=binding['ambiguous'].get('order_id',[])
        if not candidates or 'payment_amount' not in binding['roles']:
            continue
        scores=[]
        for column in candidates:
            values={str(row[0]) for row in connection.execute(f'SELECT DISTINCT {quote(column)} FROM {quote(binding["table"])}') if row[0] is not None and str(row[0]).strip()}
            scores.append({'column':column,'distinct_values':len(values),'matched':len(values&anchors),
                           'coverage':len(values&anchors)/len(values) if values else 0})
        ranked=sorted(scores,key=lambda x:x['coverage'],reverse=True)
        binding['relationship_evidence']={'scope':'all_distinct_values','candidates':scores,'anchor':'known_order_keys'}
        if ranked[0]['coverage']==1 and (len(ranked)==1 or ranked[1]['coverage']==0):
            column=ranked[0]['column'];binding['roles']['order_id']=column
            binding['origins'][column]='full_column_relationship'
            del binding['ambiguous']['order_id']
            table=next(t for t in tables if t['name']==binding['table'])
            if 'payment_date' not in binding['roles'] and 'Thời gian' in table['columns'] and 'status' not in binding['roles']:
                binding['roles']['payment_date']='Thời gian'
                binding['origins']['Thời gian']='payment_ledger_context'
            if all(role in binding['roles'] for role in ('order_id','payment_date','payment_amount')):
                binding['kind']='payments'
            binding['unmapped']=[c for c in table['columns'] if c not in binding['roles'].values()]
