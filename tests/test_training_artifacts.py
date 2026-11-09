import json
import struct

import pytest
from datamate.domain import DataMateError, json_text
from datamate.models.artifacts import parameter_count, seal_artifact, verify_artifact
from datamate.training.dataset import prepare_training, source_manifest


def test_training_holdout_and_fingerprint_protection(split_root, tmp_path):
    manifest = source_manifest(split_root)
    sources = tmp_path / "sources.json"
    sources.write_text(json_text({"sources":[{**manifest,"role":"competition_training"}]}),encoding="utf-8")
    output=tmp_path/"prepared"
    report=prepare_training(split_root,output,sources=sources)
    assert set(report["train_businesses"]).isdisjoint(report["validation_businesses"])
    assert report["train_examples"] + report["validation_examples"] == 3
    (split_root/"answers.jsonl").write_text("{}\n",encoding="utf-8")
    with pytest.raises(DataMateError,match="TRAIN"):
        prepare_training(split_root,tmp_path/"rejected",sources=sources)


def test_unapproved_public_data_is_rejected(split_root, tmp_path):
    sources=tmp_path/"sources.json"
    sources.write_text('{"sources":[]}',encoding="utf-8")
    with pytest.raises(DataMateError):
        prepare_training(split_root,tmp_path/"out",sources=sources)


def test_parameter_count_uses_all_tensor_shapes(tmp_path):
    header=json.dumps({"a":{"shape":[3,4],"dtype":"BF16","data_offsets":[0,24]},"b":{"shape":[2],"dtype":"F32","data_offsets":[24,32]}}).encode()
    (tmp_path/"model.safetensors").write_bytes(struct.pack("<Q",len(header))+header+b"\0"*32)
    assert parameter_count(tmp_path)==14


def test_artifact_hash_detects_modified_weights(tmp_path, monkeypatch):
    from datamate.models.registry import MODELS, ModelSpec
    spec = ModelSpec("fixture", "fixture/model", "a"*40, 3, "2026-01-01T00:00:00Z")
    monkeypatch.setitem(MODELS, "fixture", spec)
    header=json.dumps({"a":{"shape":[3],"dtype":"BF16","data_offsets":[0,6]}}).encode()
    weight=tmp_path/"model.safetensors"
    weight.write_bytes(struct.pack("<Q",len(header))+header+b"\0"*6)
    seal_artifact(tmp_path,spec)
    assert verify_artifact(tmp_path,"fixture")["tensor_parameters"]==3
    weight.write_bytes(weight.read_bytes()[:-1]+b"x")
    with pytest.raises(DataMateError,match="Hash"):
        verify_artifact(tmp_path,"fixture")


