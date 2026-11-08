"""Small lexical hints. The original request and entity strings stay untouched."""
import re
import unicodedata

ALIASES = {'dthu':'doanh thu','revenue':'doanh thu','sp':'sản phẩm','sl':'số lượng','dc':'được',
           'nhiu':'nhiều','bnhiu':'bao nhiêu','bn':'bao nhiêu','trc':'trước','hnay':'hôm nay',
           'hqua':'hôm qua','order':'đơn','orders':'đơn','vs':'với','ko':'không','khoong':'không','cos':'có'}
AMBIGUOUS = {'dt':['doanh thu','điện thoại'],'ck':['chuyển khoản','chiết khấu'],'tb':['trung bình','thông báo']}
PHRASES = {'bua qua':'hôm qua, một ngày trước as_of_date'}


def folded(text):
    value=unicodedata.normalize('NFKD',text.replace('đ','d').replace('Đ','D')).lower()
    return ''.join(ch for ch in value if not unicodedata.combining(ch))


def abbreviation_hints(text):
    tokens=set(re.findall(r'\b\w+\b',folded(text)))
    result={key:value for key,value in ALIASES.items() if key in tokens}
    candidates={key:value for key,value in AMBIGUOUS.items() if key in tokens}
    text_key=' '+re.sub(r'\s+',' ',folded(text))+' '
    phrases={key:value for key,value in PHRASES.items() if ' '+key+' ' in text_key}
    return {'aliases':result,'ambiguous_candidates':candidates,'phrase_hints':phrases,
            'advisory_only':True,'keep_original_entities_numbers_dates':True} if result or candidates or phrases else None
