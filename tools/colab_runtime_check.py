"""Reject stale source/imports before copying or loading any weights."""
import ast
import hashlib
from pathlib import Path
import sys


def _version(path, class_name, field):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==class_name)
    return next(ast.literal_eval(node.value) for node in cls.body if isinstance(node,ast.Assign)
                and any(isinstance(target,ast.Name) and target.id==field for target in node.targets))


def verify_runtime(work):
    root=Path(work);source=root/'src/datamate'
    pipeline=_version(source/'agent/runner.py','Agent','pipeline_version')
    backend=_version(source/'models/backends.py','TransformersBackend','protocol_version')
    if pipeline<6 or backend<3:
        raise RuntimeError('Source ZIP cũ: cần pipeline >=6/backend >=3. Thay datamate-colab-source.zip trên Drive rồi chạy lại từ ô đầu.')
    for module_name,class_name,field,expected in (
        ('datamate.agent.runner','Agent','pipeline_version',pipeline),
        ('datamate.models.backends','TransformersBackend','protocol_version',backend)):
        module=sys.modules.get(module_name)
        if module and getattr(getattr(module,class_name),field)!=expected:
            raise RuntimeError('Module cũ đang ở RAM; restart kernel trong cùng runtime rồi chạy từ ô đầu. Giữ cache /content/models.')
    normalization=sys.modules.get('datamate.data.normalization')
    if normalization and normalization.semantic_version()!=normalization._source_version():
        raise RuntimeError('Code chuẩn hóa trên đĩa khác module RAM; restart kernel trước khi nạp trọng số.')
    return {'pipeline_version':pipeline,'backend_protocol':backend,
            'runner_sha256':hashlib.sha256((source/'agent/runner.py').read_bytes()).hexdigest()}
