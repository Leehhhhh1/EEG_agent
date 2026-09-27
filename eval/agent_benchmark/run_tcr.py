"""Run EEGAgent benchmark cases and compute TCR, R-ACC and TCE.

Examples:
    python eval/agent_benchmark/run_tcr.py --dry-run
    python eval/agent_benchmark/run_tcr.py --level L1 --repeat 1
    python eval/agent_benchmark/run_tcr.py --case-id L1-001 --repeat 1
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.agent_benchmark.tcr import TraceCollector, load_cases, score_case, summarize_metrics


BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_CASE_FILES = (
    BENCHMARK_DIR / "l1_atomic.jsonl",
    BENCHMARK_DIR / "l2_sequential.jsonl",
    BENCHMARK_DIR / "l3_multiturn.jsonl",
)
PAIRED_EEG_FILES = {
    "GPED": "data/gped_049_a_6.edf",
    "ABDO": "data/aaaaabdo_s003_t000.edf",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or rescore EEGAgent benchmark metrics.")
    parser.add_argument("--level", choices=("L1", "L2", "L3"))
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument(
        "--cases-file",
        action="append",
        type=Path,
        default=[],
        help="Use one or more JSONL files instead of the default 30-case suite.",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--paired-two-eeg",
        action="store_true",
        help="Run every selected case once per GPED and ABDO record; adapt GPED L2-001 to 120 s.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rescore-runs",
        type=Path,
        help="Score a saved runs.jsonl without running the agent or calling the model API.",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _select_cases(cases: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = cases
    if args.level:
        selected = [case for case in selected if case.get("level") == args.level]
    if args.case_id:
        requested = set(args.case_id)
        selected = [case for case in selected if case.get("id") in requested]
        missing = requested - {str(case.get("id")) for case in selected}
        if missing:
            raise ValueError(f"Unknown case ids: {', '.join(sorted(missing))}")
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")
    return selected


def _pair_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paired = []
    for dataset, file_path in PAIRED_EEG_FILES.items():
        if not (PROJECT_ROOT / file_path).is_file():
            raise FileNotFoundError(f"EEG dataset not found: {file_path}")
        for original in cases:
            case = deepcopy(original)
            case["id"] = f"{original['id']}@{dataset}"
            case["dataset"] = dataset
            if case.get("session", {}).get("mode") == "loaded":
                case["session"]["file"] = file_path
            else:
                case["dataset_note"] = "No EDF session; dataset-independent control case."
            if dataset == "GPED" and original["id"] == "L2-001":
                turn = case["turns"][0]
                turn["query"] = "请先调用基础信息工具重新确认采样率，再检查前2分钟是否存在发作样事件。"
                for expectation in turn["expected_arguments"]:
                    if expectation["tool"] == "detect_eeg_events":
                        expectation["arguments"]["end"] = 120
                case["adaptation"] = "GPED lasts 241 s; the 600 s detection window was shortened to 120 s."
            paired.append(case)
    return paired


def _open_session(bridge, case: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    session = case.get("session") or {}
    if session.get("mode") == "none":
        return None, None
    file_path = (PROJECT_ROOT / str(session["file"])).resolve()
    opened = bridge.call_tool("open_eeg_session", {"file_path": str(file_path)})
    if opened.get("is_error") or not opened.get("structured_content"):
        raise RuntimeError(f"Unable to open EEG session for {file_path}: {opened}")
    session_id = opened["structured_content"]["session_id"]
    basic = bridge.call_tool("get_eeg_basic_information", {"session_id": session_id})
    if basic.get("is_error"):
        raise RuntimeError(f"Unable to load basic EEG information: {basic}")
    return session_id, basic.get("structured_content") or {}


def _run_setup(bridge, case: dict[str, Any], session_id: str | None) -> None:
    for call in case.get("setup_tool_calls") or []:
        if session_id is None:
            raise RuntimeError("A setup tool call requires an EEG session.")
        arguments = dict(call.get("arguments") or {})
        arguments["session_id"] = session_id
        result = bridge.call_tool(str(call["tool"]), arguments)
        if result.get("is_error"):
            raise RuntimeError(f"Setup tool {call['tool']} failed: {result}")


def _run_case(bridge, shared_retriever, case: dict[str, Any]) -> dict[str, Any]:
    from agent_runtime.mcp_chat_agent import MCPChatAgent

    session_id = None
    turn_runs: list[dict[str, Any]] = []
    setup_error = None
    try:
        session_id, basic_info = _open_session(bridge, case)
        _run_setup(bridge, case, session_id)
        agent = MCPChatAgent(
            bridge,
            session_id=session_id,
            initial_basic_info=basic_info,
        )
        agent.set_rag_retriever(shared_retriever)
        for turn in case.get("turns") or []:
            collector = TraceCollector()
            try:
                result = agent.run_stream(
                    str(turn["query"]),
                    on_trace=collector,
                )
                turn_runs.append({
                    "response": result.get("response", ""),
                    "result": result,
                    "events": collector.events,
                    "error": None,
                })
            except Exception as exc:
                turn_runs.append({
                    "response": "",
                    "events": collector.events,
                    "error": f"{exc.__class__.__name__}: {exc}",
                })
                break
    except Exception as exc:
        setup_error = f"{exc.__class__.__name__}: {exc}"
    finally:
        if session_id is not None:
            try:
                bridge.call_tool("close_eeg_session", {"session_id": session_id})
            except Exception:
                pass

    score = score_case(case, turn_runs)
    if setup_error:
        score["passed"] = False
        score["route_pass"] = False
        score["tce"] = None
        score["setup_error"] = setup_error
    return {
        "case": {"id": case.get("id"), "level": case.get("level")},
        "score": score,
        "turn_runs": turn_runs,
    }


def _write_markdown_report(
    path: Path,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    overall = summary["overall"]
    def percentage(value: float | None) -> str:
        return f"{value:.2%}" if value is not None else "—"

    lines = [
        "# EEGAgent 分层评测报告",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 任务数：{overall['total']}",
        f"- 总体 TCR：{percentage(overall['tcr'])}（{overall['completed']}/{overall['total']}）",
        f"- 总体 R-ACC：{percentage(overall['r_acc'])}（{overall['correct_routes']}/{overall['total']}）",
        f"- 总体 TCE：{percentage(overall['tce'])}（{overall['tce_cases']} 个完成任务参与计算）",
        "",
        "## 分等级结果",
        "",
        "| 等级 | 任务数 | TCR | R-ACC | TCE | TCE 样本数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for level, result in summary["by_level"].items():
        lines.append(
            f"| {level} | {result['total']} | {percentage(result['tcr'])} "
            f"({result['completed']}/{result['total']}) | {percentage(result['r_acc'])} "
            f"({result['correct_routes']}/{result['total']}) | "
            f"{percentage(result['tce'])} | {result['tce_cases']} |"
        )
    if summary.get("by_dataset_level"):
        lines.extend([
            "",
            "## 数据集 × 等级",
            "",
            "| 数据集 | 等级 | 任务数 | TCR | R-ACC | TCE | TCE 样本数 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for dataset, by_level in summary["by_dataset_level"].items():
            for level, result in by_level.items():
                lines.append(
                    f"| {dataset} | {level} | {result['total']} | "
                    f"{percentage(result['tcr'])} ({result['completed']}/{result['total']}) | "
                    f"{percentage(result['r_acc'])} ({result['correct_routes']}/{result['total']}) | "
                    f"{percentage(result['tce'])} | {result['tce_cases']} |"
                )
    lines.extend([
        "",
        "## 逐任务结果",
        "",
        "| 重复 | 用例 | 数据集 | 等级 | TCR | R-ACC | TCE | 工具调用 | 失败原因 |",
        "|---:|---|---|---|---|---|---:|---:|---|",
    ])
    for record in records:
        score = record["score"]
        reasons = []
        if score.get("setup_error"):
            reasons.append(str(score["setup_error"]))
        for turn_score in score.get("turn_scores") or []:
            reasons.extend(str(reason) for reason in turn_score.get("reasons") or [])
        reason_text = "；".join(reasons).replace("|", "\\|") or "—"
        lines.append(
            f"| {record.get('repetition', 1)} | {record['case']['id']} | "
            f"{score.get('dataset') or '—'} | {record['case']['level']} | "
            f"{'PASS' if score['passed'] else 'FAIL'} | "
            f"{'PASS' if score['route_pass'] else 'FAIL'} | {percentage(score['tce'])} | "
            f"{score['required_tool_calls']}/{score['actual_tool_calls']} 必需/全部 | {reason_text} |"
        )
    lines.extend([
        "",
        "## 判定口径",
        "",
        "TCR 为结构性任务完成率：必需工具成功、禁用工具未被调用、顺序和关键参数正确、无非预期工具错误，且最终回答非空。R-ACC 要求一个用例中每轮都路由到预期 Skill。TCE 只在完成且发生工具调用的任务上计算：必需工具调用数 ÷ Agent 实际工具调用数，先逐任务计算再取平均；评测器预置调用不计入。没有可计算样本时显示为“—”。自然语言回答检查项仍需另行复核。",
        "",
    ])
    adaptations = sorted({record.get("case", {}).get("adaptation") for record in records if record.get("case", {}).get("adaptation")})
    if adaptations:
        lines.extend(["## 用例适配", ""])
        lines.extend(f"- {adaptation}" for adaptation in adaptations)
        lines.extend(["- L1-010 是未加载 EDF 的对照任务；在两组中重复运行，但不读取对应脑电记录。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_results(output_dir: Path, records: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_path = output_dir / "runs.jsonl"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    traces_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, default=str) + "\n" for record in records),
        encoding="utf-8",
    )
    scores = [record["score"] for record in records]
    summary = summarize_metrics(scores)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown_report(report_path, summary, records)
    return traces_path, summary_path, report_path


def main() -> int:
    args = _arguments()
    case_files = tuple(
        path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        for path in args.cases_file
    ) or DEFAULT_CASE_FILES
    cases = _select_cases(load_cases(case_files), args)
    if args.paired_two_eeg:
        cases = _pair_cases(cases)
    if args.rescore_runs:
        runs_path = args.rescore_runs.resolve()
        case_by_id = {str(case["id"]): case for case in cases}
        records: list[dict[str, Any]] = []
        with runs_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                case_id = str(record.get("case", {}).get("id"))
                if case_id not in case_by_id:
                    raise ValueError(f"Unknown case {case_id} in {runs_path}:{line_number}")
                score = score_case(case_by_id[case_id], record.get("turn_runs") or [])
                setup_error = record.get("score", {}).get("setup_error")
                if setup_error:
                    score.update(passed=False, route_pass=False, tce=None, setup_error=setup_error)
                record["score"] = score
                records.append(record)
        summary = summarize_metrics([record["score"] for record in records])
        output_dir = args.output_dir or runs_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "metrics_summary.json"
        report_path = output_dir / "metrics_report.md"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_markdown_report(report_path, summary, records)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Summary: {summary_path}")
        print(f"Report: {report_path}")
        return 0
    planned_runs = len(cases) * args.repeat
    print(f"Selected {len(cases)} cases, repeat={args.repeat}, planned runs={planned_runs}")
    if args.dry_run:
        print("Dry run passed: JSONL loaded and filters are valid. No model API was called.")
        return 0

    from agent_runtime.mcp_client import MCPClientBridge
    from RAG.retriever import EEGRetriever

    output_dir = args.output_dir or (
        BENCHMARK_DIR / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "runs.jsonl"
    if checkpoint_path.exists():
        raise FileExistsError(f"Results already exist; choose another --output-dir: {checkpoint_path}")

    bridge = MCPClientBridge(PROJECT_ROOT)
    records: list[dict[str, Any]] = []
    try:
        bridge.start()
        shared_retriever = EEGRetriever()
        with checkpoint_path.open("w", encoding="utf-8") as checkpoint:
            for repetition in range(1, args.repeat + 1):
                for index, case in enumerate(cases, start=1):
                    print(
                        f"[{repetition}/{args.repeat}] [{index}/{len(cases)}] "
                        f"running {case['id']}",
                        flush=True,
                    )
                    record = _run_case(bridge, shared_retriever, case)
                    record["repetition"] = repetition
                    record["case"]["adaptation"] = case.get("adaptation")
                    records.append(record)
                    checkpoint.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    checkpoint.flush()
                    print(
                        f"{case['id']}: {'PASS' if record['score']['passed'] else 'FAIL'}",
                        flush=True,
                    )
    finally:
        bridge.close()

    traces_path, summary_path, report_path = _write_results(output_dir, records)
    summary = summarize_metrics([record["score"] for record in records])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Runs: {traces_path}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
