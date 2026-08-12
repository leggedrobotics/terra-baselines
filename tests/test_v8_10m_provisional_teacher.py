import hashlib
import json
from types import SimpleNamespace

from scripts import v8_10m_provisional_teacher as teacher
from utils import accepted_bank


def test_legacy_teacher_sampler_is_separate_from_bounded_replay(monkeypatch, tmp_path):
    observed = {}
    fake_bank = SimpleNamespace(
        levels=(
            SimpleNamespace(condition_id="foundation"),
            SimpleNamespace(condition_id="trench"),
        ),
        sampling_probabilities=(1.0, 3.0),
        map_count_per_condition=96,
    )

    def fake_load(root, arm, revision, *, curriculum_stage, sampler_profile):
        observed.update(
            root=root,
            arm=arm,
            revision=revision,
            curriculum_stage=curriculum_stage,
            sampler_profile=sampler_profile,
        )
        return fake_bank

    monkeypatch.setattr(accepted_bank, "load_accepted_bank", fake_load)
    payload = {
        "stage": "full",
        "conditions": ["foundation", "trench"],
        "declared_weights": [1.0, 3.0],
        "probabilities": [0.25, 0.75],
        "maps_per_condition": 96,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    monkeypatch.setattr(teacher, "LEGACY_FULL_SAMPLING_SHA256", digest)

    result = teacher.legacy_full_sampling_contract({"root": tmp_path})

    assert observed["curriculum_stage"] == "full"
    assert observed["sampler_profile"] == "bank_v4"
    assert result["sampler_profile"] == "bank_v4"
    assert result["sha256"] == digest
    assert result["probabilities"] == [0.25, 0.75]
