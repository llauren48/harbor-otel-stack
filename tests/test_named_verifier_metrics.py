import json
from types import SimpleNamespace

from harborstack.convert.scan import load_reward
from harborstack.convert.scores import score_items


def test_flat_verifier_metrics_preserve_names_without_inventing_reward(tmp_path):
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "reward.json").write_text(json.dumps({"accuracy": 1.0, "valid": True}))
    reward, details = load_reward(tmp_path)
    assert reward is None
    assert details == {"accuracy": 1.0}
    assert score_items(SimpleNamespace(reward=reward, reward_details=details, phenotypes=[])) == [
        {"name": "reward.accuracy", "value": 1.0, "dataType": "NUMERIC"}
    ]


def test_explicit_reward_and_details_remain_authoritative(tmp_path):
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "reward.json").write_text(json.dumps({
        "reward": 0.5, "accuracy": 0.0, "details": {"accuracy": 1.0}
    }))
    assert load_reward(tmp_path) == (0.5, {"accuracy": 1.0})
