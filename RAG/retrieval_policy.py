"""Deterministic routing and first-pass history expansion for RAG queries."""

from dataclasses import dataclass
import re
from typing import Literal


RetrievalMode = Literal["skip", "retrieve", "probe"]


@dataclass(frozen=True)
class RetrievalDecision:
    mode: RetrievalMode
    reason: str
    retrieval_query: str


KNOWLEDGE_MARKERS = (
    "什么是", "是什么", "定义", "解释", "标准", "指南", "区别", "分类",
    "诊断标准", "acns", "术语", "规范", "报告", "summary", "report",
    "guideline", "definition", "difference", "criteria", "terminology",
    "what is", "explain",
)
RECORDING_MARKERS = (
    "当前记录", "这份脑电", "这个患者", "本次脑电", "导联", "脑区",
    "秒", "分钟", "前60", "有没有", "是否存在", "观察到", "检测",
    "分析这", "current recording", "this eeg", "this patient", "channel",
    "first minute", "seconds", "detect", "observed",
)
NO_RAG_MARKERS = (
    "你好", "谢谢", "再见", "清空对话", "移除数据", "加载文件",
    "调用工具", "工具失败", "执行失败", "为什么报错", "你是谁",
    "hello", "thanks", "thank you", "clear conversation", "tool failed",
)
FOLLOW_UP_MARKERS = (
    "它", "这个", "这种", "那个", "这些", "那些", "上述", "刚才",
    "前者", "后者", "还有呢", "那呢", "其",
)
ENGLISH_FOLLOW_UP = re.compile(
    r"\b(it|this|that|they|these|those|former|latter)\b|"
    r"\b(what|how) about\b",
    re.IGNORECASE,
)


def is_follow_up(query: str) -> bool:
    text = query.strip()
    lowered = text.lower()
    return any(marker in lowered for marker in FOLLOW_UP_MARKERS) or bool(
        ENGLISH_FOLLOW_UP.search(text)
    )


def build_retrieval_query(current_query: str, previous_user_query: str | None) -> str:
    """Expand only clear follow-ups with the immediately preceding user query."""
    current = current_query.strip()
    previous = (previous_user_query or "").strip()
    if previous and is_follow_up(current):
        return f"{previous}\n{current}"
    return current


def decide_retrieval(
    query: str,
    *,
    has_eeg_session: bool,
    previous_user_query: str | None = None,
) -> RetrievalDecision:
    """Choose rule-based retrieval, skipping, or a FAISS relevance probe."""
    lowered = query.strip().lower()
    retrieval_query = build_retrieval_query(query, previous_user_query)

    # Knowledge/standard requests take precedence over recording markers so a
    # query such as "按 ACNS 标准解释当前记录" can use both RAG and EEG tools.
    if any(marker in lowered for marker in KNOWLEDGE_MARKERS):
        return RetrievalDecision("retrieve", "knowledge_or_guideline", retrieval_query)
    if has_eeg_session and any(marker in lowered for marker in RECORDING_MARKERS):
        return RetrievalDecision("skip", "recording_specific_tool_query", retrieval_query)
    if any(marker in lowered for marker in NO_RAG_MARKERS):
        return RetrievalDecision("skip", "conversation_or_ui_intent", retrieval_query)
    return RetrievalDecision("probe", "semantic_probe", retrieval_query)
