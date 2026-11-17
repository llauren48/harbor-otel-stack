"""读取层：发现 Trial 目录，解析 ATIF 轨迹与 verifier 判分。

只依赖标准库。目录名按 Harbor 约定 ``task__agent__model__attempt`` 回退解析，
``result.json`` 存在时以其中的元数据为准。

发现阶段区分两种情况，避免 case 静默消失：

* **skip**：存在 ``agent/trajectory.json`` 但读不出来（截断、非法 JSON 等）；
* **missing**：目录在、``result.json`` 也在，却没有 ``agent/trajectory.json``
  （容器没起来、超时被杀、agent 未落盘），连同 result.json 里的异常信息一并上报。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

TRAJECTORY_REL = Path("agent") / "trajectory.json"


@dataclass
class StepRecord:
    step_id: int
    source: str
    timestamp: str | None
    model_name: str | None
    message: str
    reasoning_content: str | None
    tool_calls: list[dict[str, Any]]
    observation_text: str
    metrics: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)
    observation_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TrialRecord:
    trial_dir: Path
    task_id: str
    agent_name: str
    model_name: str
    attempt: str
    run_id: str
    session_id: str
    task_prompt: str
    final_message: str
    steps: list[StepRecord]
    final_metrics: dict[str, Any]
    subagent_trajectories: list[dict[str, Any]]
    reward: float | None
    reward_details: dict[str, float]
    phenotypes: list[str]
    exception: dict[str, Any] | None = None


class ScanError(Exception):
    """Trial 目录不可解析。"""


# ---------- 目录发现 ----------

def looks_like_trial(path: Path) -> bool:
    return (path / TRAJECTORY_REL).is_file()


def looks_like_case_dir(path: Path) -> bool:
    """Harbor 的 case 目录名为 ``task__agent__model__attempt``。

    只认带 ``__`` 且至少落了一个产物文件的目录；job 根目录（如
    ``hs-path-trace-llm2``）虽然也有 result.json，但名字不含 ``__``，不会被误判。
    """
    return "__" in path.name and (
        (path / TRAJECTORY_REL).is_file() or (path / "result.json").is_file()
    )


def iter_case_dirs(root: str | Path) -> Iterator[Path]:
    """发现全部 case 目录：既有正常 Trial，也有只剩 result.json 的残缺目录。"""
    root = Path(root)
    for path in (root, *sorted(root.rglob("*"))):
        if not path.is_dir() or not looks_like_case_dir(path):
            continue
        if path != root and looks_like_trial(path.parent):
            continue  # Trial 内部嵌套的同名目录不算独立 case
        yield path


def _read_result(trial_dir: Path) -> dict[str, Any]:
    path = trial_dir / "result.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def missing_entry(trial_dir: Path) -> dict[str, Any]:
    """缺 ``agent/trajectory.json`` 的 case。

    只记定位信息（case 名、绝对路径、原因），细节留在原始目录里由人回查；
    这条记录不参与任何失败判定。
    """
    entry: dict[str, Any] = {
        "trial": trial_dir.name,
        "path": str(trial_dir.resolve()),
        "kind": "missing_atif",
        "reason": f"缺少 {TRAJECTORY_REL}",
    }
    exc_info = _read_result(trial_dir).get("exception_info")
    if isinstance(exc_info, dict) and exc_info:
        kind = exc_info.get("exception_type") or "Exception"
        entry["reason"] = f"{entry['reason']}（agent 异常：{kind}）"
    return entry


# ---------- 元数据 ----------

def _parse_dir_name(name: str) -> dict[str, str]:
    parts = name.split("__")
    if len(parts) >= 4:
        return {"task_id": parts[0], "agent_name": parts[1],
                "model_name": "__".join(parts[2:-1]), "attempt": parts[-1]}
    if len(parts) == 3:
        return {"task_id": parts[0], "agent_name": parts[1],
                "model_name": parts[2], "attempt": "1"}
    return {"task_id": name, "agent_name": "unknown",
            "model_name": "unknown", "attempt": "1"}


def _meta_from_result(result: dict[str, Any]) -> dict[str, Any]:
    agent = result.get("agent_name") or (result.get("agent") or {}).get("name")
    model = result.get("model_name") or (result.get("agent") or {}).get("model_name")
    task = (result.get("task_name") or result.get("task_id")
            or (result.get("task") or {}).get("name"))
    run_id = result.get("run_id") or result.get("job_id")
    res = (result.get("result") or result.get("trial_result")
           or result.get("verifier_result") or {})
    reward = res.get("reward") if isinstance(res, dict) else None
    if reward is None:
        reward = result.get("reward")
    return {"agent_name": agent, "model_name": model, "task_id": task,
            "run_id": run_id, "reward": reward}


# ---------- 内容兼容 ----------

def text_of(content: Any) -> str:
    """ATIF v1.6+ 起 message/content 可为字符串或 ContentPart 列表。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


# ---------- 判分 ----------

def load_reward(trial_dir: Path) -> tuple[float | None, dict[str, float]]:
    vdir = trial_dir / "verifier"
    txt = vdir / "reward.txt"
    if txt.is_file():
        try:
            return float(txt.read_text(encoding="utf-8").strip()), {}
        except ValueError:
            pass

    rj = vdir / "reward.json"
    if not rj.is_file():
        return None, {}
    try:
        data = json.loads(rj.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, {}
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        return float(data), {}
    if not isinstance(data, dict):
        return None, {}

    reward = data.get("reward", data.get("score"))
    details = {
        str(k): float(v) for k, v in (data.get("details") or {}).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    # Harbor verifiers may emit named metrics directly, e.g. {"accuracy": 1.0}.
    # Preserve their names without inventing an aggregate reward.
    for name, value in data.items():
        if name not in {"reward", "score", "details"} and \
                isinstance(value, (int, float)) and not isinstance(value, bool):
            details.setdefault(str(name), float(value))
    try:
        reward_f = float(reward) if reward is not None else None
    except (TypeError, ValueError):
        reward_f = None
    return reward_f, details


def load_reward_details(trial_dir: Path) -> dict[str, float]:
    """Reward Kit 的 reward-details.json：逐判据得分，结构宽容解析。"""
    path = trial_dir / "verifier" / "reward-details.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    out: dict[str, float] = {}

    def eat(obj: Any) -> None:
        if isinstance(obj, dict):
            if isinstance(obj.get("criteria"), list):
                for item in obj["criteria"]:
                    eat(item)
                return
            name = obj.get("name") or obj.get("criterion") or obj.get("id")
            score = obj.get("score", obj.get("value"))
            if name is not None and isinstance(score, (int, float)) \
                    and not isinstance(score, bool):
                out[str(name)] = float(score)
                return
            for key, value in obj.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    out[str(key)] = float(value)
        elif isinstance(obj, list):
            for item in obj:
                eat(item)

    eat(data)
    return out


# ---------- Trial 解析 ----------

def load_trial(trial_dir: str | Path, run_id_hint: str = "") -> TrialRecord:
    trial_dir = Path(trial_dir)
    traj_path = trial_dir / TRAJECTORY_REL
    if not traj_path.is_file():
        raise ScanError(f"缺少 {TRAJECTORY_REL}：{trial_dir}")
    try:
        traj = json.loads(traj_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScanError(f"trajectory.json 解析失败：{exc}") from exc

    result = _read_result(trial_dir)

    base = _parse_dir_name(trial_dir.name)
    from_result = _meta_from_result(result)
    agent_cfg = traj.get("agent") or {}
    task_id = from_result.get("task_id") or base["task_id"]
    agent_name = from_result.get("agent_name") or base["agent_name"]
    model_name = from_result.get("model_name") or base["model_name"]
    if agent_name == "unknown":
        agent_name = agent_cfg.get("name", agent_name)
    if model_name == "unknown":
        model_name = agent_cfg.get("model_name", model_name)
    run_id = (from_result.get("run_id") or run_id_hint
              or trial_dir.parent.name or "unknown-run")

    steps: list[StepRecord] = []
    for raw in traj.get("steps") or []:
        obs = raw.get("observation") or {}
        obs_text = "\n".join(
            text_of(r.get("content")) for r in (obs.get("results") or [])
        )
        steps.append(StepRecord(
            step_id=int(raw.get("step_id") or 0),
            source=str(raw.get("source") or ""),
            timestamp=raw.get("timestamp"),
            model_name=raw.get("model_name") or agent_cfg.get("model_name"),
            message=text_of(raw.get("message")),
            reasoning_content=raw.get("reasoning_content"),
            tool_calls=list(raw.get("tool_calls") or []),
            observation_text=obs_text,
            observation_results=[r for r in (obs.get("results") or [])
                                 if isinstance(r, dict)],
            metrics=dict(raw.get("metrics") or {}),
            extra=dict(raw.get("extra") or {}),
        ))

    task_prompt = next((s.message for s in steps
                        if s.source == "user" and s.message), "") \
        or next((s.message for s in steps
                 if s.source == "system" and s.message), "")
    final_message = next((s.message for s in reversed(steps)
                          if s.source == "agent" and s.message), "")

    reward, details = load_reward(trial_dir)
    if reward is None and from_result.get("reward") is not None:
        try:
            reward = float(from_result["reward"])
        except (TypeError, ValueError):
            reward = None
    if not details:
        details = load_reward_details(trial_dir)

    extra = traj.get("extra") or {}
    phenotypes = [str(t) for t in
                  (extra.get("phenotypes") or result.get("phenotypes") or [])]
    exc_info = result.get("exception_info")
    exception = dict(exc_info) if isinstance(exc_info, dict) else None

    return TrialRecord(
        trial_dir=trial_dir,
        task_id=str(task_id),
        agent_name=str(agent_name),
        model_name=str(model_name),
        attempt=str(base["attempt"]),
        run_id=str(run_id),
        session_id=str(traj.get("session_id") or f"{run_id}:{trial_dir.name}"),
        task_prompt=task_prompt,
        final_message=final_message,
        steps=steps,
        final_metrics=dict(traj.get("final_metrics") or {}),
        subagent_trajectories=list(traj.get("subagent_trajectories") or []),
        reward=reward,
        reward_details=details,
        phenotypes=phenotypes,
        exception=exception,
    )


def load_trials(root: str | Path, trial_filter: str = "all",
                run_id_hint: str = "", on_skip=None, on_missing=None
                ) -> tuple[list[TrialRecord], int, list[dict[str, Any]]]:
    """发现并解析全部 Trial，返回 ``(记录, skip 数, 缺文件明细)``。

    skip 只统计「有 trajectory.json 但读不动」；缺 trajectory.json 的 case
    单独记入 missing 明细，不再静默消失。
    """
    records: list[TrialRecord] = []
    skipped = 0
    missing: list[dict[str, Any]] = []
    for path in iter_case_dirs(root):
        if not looks_like_trial(path):
            entry = missing_entry(path)
            missing.append(entry)
            if on_missing is not None:
                on_missing(path, entry["reason"])
            continue
        try:
            rec = load_trial(path, run_id_hint=run_id_hint)
        except Exception as exc:  # noqa: BLE001 - 单个 Trial 坏掉不影响整批
            skipped += 1
            if on_skip is not None:
                on_skip(path, exc)
            continue
        if trial_filter == "success" and not (rec.reward is not None and rec.reward > 0):
            continue
        if trial_filter == "failure" and not (rec.reward is None or rec.reward <= 0):
            continue
        records.append(rec)
    return records, skipped, missing
