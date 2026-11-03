"""Strict parser for Gemma's single native call envelope, not JSON repair.

Recognizes unquoted argument keys and the checkpoint's <|"|> string delimiter.
The whole input must be one closed call; prose, extra calls and bare values fail.
"""
from __future__ import annotations

import json
import re

from ..domain import DataMateError


class NativeParser:
    def __init__(self,text):
        self.text,self.pos=text,0

    def whitespace(self):
        while self.pos<len(self.text) and self.text[self.pos].isspace():
            self.pos+=1

    def take(self,token):
        self.whitespace()
        if not self.text.startswith(token,self.pos):
            raise DataMateError('Native call thiếu token '+repr(token))
        self.pos+=len(token)

    def string(self):
        self.whitespace()
        marker='<|"|>'
        if self.text.startswith(marker,self.pos):
            self.pos+=len(marker)
            end=self.text.find(marker,self.pos)
            if end<0:
                raise DataMateError('Native string chưa đóng')
            value=self.text[self.pos:end]
            self.pos=end+len(marker)
            return value
        value,end=json.JSONDecoder().raw_decode(self.text,self.pos)
        if not isinstance(value,str):
            raise DataMateError('Native key/string không hợp lệ')
        self.pos=end
        return value

    def value(self,depth=0):
        if depth>32:
            raise DataMateError('Native call lồng quá sâu')
        self.whitespace()
        if self.text.startswith('"',self.pos) or self.text.startswith('<|"|>',self.pos):
            return self.string()
        if self.text.startswith('{',self.pos):
            self.pos+=1;result={};self.whitespace()
            if self.text.startswith('}',self.pos):
                self.pos+=1;return result
            while True:
                self.whitespace()
                if self.text.startswith(('"','<|"|>'),self.pos):
                    key=self.string()
                else:
                    match=re.match(r'[A-Za-z_][A-Za-z_0-9]*',self.text[self.pos:])
                    if not match:
                        raise DataMateError('Native argument key không hợp lệ')
                    key=match[0];self.pos+=len(key)
                if key in result:
                    raise DataMateError('Native call trùng khóa')
                self.take(':');result[key]=self.value(depth+1);self.whitespace()
                if self.text.startswith('}',self.pos):
                    self.pos+=1;return result
                self.take(',')
        if self.text.startswith('[',self.pos):
            self.pos+=1;result=[];self.whitespace()
            if self.text.startswith(']',self.pos):
                self.pos+=1;return result
            while True:
                result.append(self.value(depth+1));self.whitespace()
                if self.text.startswith(']',self.pos):
                    self.pos+=1;return result
                self.take(',')
        for literal,value in [('true',True),('false',False),('null',None)]:
            if self.text.startswith(literal,self.pos):
                self.pos+=len(literal);return value
        number=re.match(r'-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?',self.text[self.pos:])
        if number:
            self.pos+=len(number[0]);return json.loads(number[0])
        raise DataMateError('Native value không hợp lệ; không đoán bare text hoặc JSON dở')


def parse_native_call(text,actions):
    marked=text.startswith('<|tool_call>')
    if marked:
        if not text.endswith('<tool_call|>'):
            raise DataMateError('Native tool call chưa đóng')
        text=text[len('<|tool_call>'):-len('<tool_call|>')].strip()
    match=re.match(r'^call:([A-Za-z_][A-Za-z_0-9]*)\s*:?',text)
    if not match or match[1] not in actions:
        raise DataMateError('Native call không thuộc bộ công cụ đã khai báo')
    name=match[1];parser=NativeParser(text[match.end():])
    try:
        value=parser.value()
        parser.whitespace()
    except (ValueError,TypeError,RecursionError) as exc:
        raise DataMateError('Native call sai cấu trúc') from exc
    if parser.pos!=len(parser.text) or not isinstance(value,dict):
        raise DataMateError('Cần đúng một native call, không có phần dư')
    if 'action' in value and value['action']!=name:
        raise DataMateError('Native function name khác action JSON')
    return {'action':name,**value}
