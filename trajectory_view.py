"""PySide6 analysis trajectory widgets for the desktop client."""

from __future__ import annotations

import json
import time
from typing import Any

from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


LANE_LABELS = ("用户 / 上下文", "助手", "工具")
LANE_INDEX = {"input": 0, "model": 1, "tool": 2}
KIND_LABELS = {
    "user": "用户",
    "context": "上下文",
    "assistant": "助手",
    # Backward-compatible display for traces created before the four-type model.
    "system": "上下文",
    "model": "助手",
    "tool": "工具",
}
STATUS_LABELS = {
    "running": "进行中",
    "complete": "完成",
    "error": "失败",
}
KIND_COLORS = {
    "user": QColor("#5b84e8"),
    "context": QColor("#58b67a"),
    "assistant": QColor("#8467ad"),
    "system": QColor("#58b67a"),
    "model": QColor("#8467ad"),
    "tool": QColor("#e58a28"),
}


def _format_seconds(seconds: float) -> str:
    """Format a compact duration for the aggregate strip."""
    seconds = max(0.0, seconds)
    if seconds >= 60:
        minutes = int(seconds // 60)
        remainder = seconds - minutes * 60
        return f"{minutes} 分 {remainder:.1f} 秒"
    return f"{seconds:.1f} 秒"


def _format_tokens(tokens: int) -> str:
    """Format token counts using compact K/M suffixes."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.1f}K"
    return str(tokens)


def summarize_trajectory(events: list[dict[str, Any]], now: float | None = None) -> str:
    """Build the DeepSeek-Harness-style aggregate status text."""
    now = time.time() if now is None else now
    turns = {
        event.get("turn") for event in events
        if isinstance(event.get("turn"), int)
    }
    steps = sum(1 for event in events if event.get("type") != "turn/end")
    model_events = [
        event for event in events
        if event.get("kind") in {"assistant", "model"}
    ]
    tool_events = [event for event in events if event.get("kind") == "tool"]
    model_time = sum(_event_duration(event, now) for event in model_events)
    tool_time = sum(_event_duration(event, now) for event in tool_events)

    ttfts = [
        float(event["ttft"]) for event in model_events
        if isinstance(event.get("ttft"), (int, float))
    ]
    average_ttft = sum(ttfts) / len(ttfts) if ttfts else None

    prompt_values = [
        int(event["prompt_tokens"]) for event in model_events
        if isinstance(event.get("prompt_tokens"), (int, float))
    ]
    completion_values = [
        int(event["completion_tokens"]) for event in model_events
        if isinstance(event.get("completion_tokens"), (int, float))
    ]
    prompt_tokens = sum(prompt_values)
    completion_tokens = sum(completion_values)
    latest_turn = max(turns) if turns else None
    current_turn_model_events = (
        [event for event in model_events if event.get("turn") == latest_turn]
        if latest_turn is not None
        else model_events
    )
    cache_hit_tokens = sum(
        int(event["cache_hit_tokens"]) for event in current_turn_model_events
        if isinstance(event.get("cache_hit_tokens"), (int, float))
    )
    cache_miss_tokens = sum(
        int(event["cache_miss_tokens"]) for event in current_turn_model_events
        if isinstance(event.get("cache_miss_tokens"), (int, float))
    )
    has_cache_usage = any(
        isinstance(event.get("cache_hit_tokens"), (int, float))
        or isinstance(event.get("cache_miss_tokens"), (int, float))
        for event in current_turn_model_events
    )
    cache_total = cache_hit_tokens + cache_miss_tokens
    cache_text = (
        f"{cache_hit_tokens / cache_total:.0%}"
        if has_cache_usage and cache_total > 0
        else "—"
    )

    measured_output_tokens = 0
    generation_seconds = 0.0
    for event in model_events:
        tokens = event.get("completion_tokens")
        ttft = event.get("ttft")
        if not isinstance(tokens, (int, float)) or not isinstance(ttft, (int, float)):
            continue
        after_first_token = _event_duration(event, now) - float(ttft)
        if after_first_token <= 0:
            continue
        measured_output_tokens += int(tokens)
        generation_seconds += after_first_token
    speed_text = (
        f"{measured_output_tokens / generation_seconds:.0f} tok/s"
        if measured_output_tokens > 0 and generation_seconds > 0
        else "— tok/s"
    )
    ttft_text = _format_seconds(average_ttft) if average_ttft is not None else "—"
    prompt_text = _format_tokens(prompt_tokens) if prompt_values else "—"
    completion_text = _format_tokens(completion_tokens) if completion_values else "—"
    latest_context_event = next((
        event for event in reversed(model_events)
        if isinstance(event.get("prompt_tokens"), (int, float))
        and isinstance(event.get("context_limit_tokens"), (int, float))
    ), None)
    context_group = ""
    if latest_context_event is not None:
        context_tokens = int(latest_context_event["prompt_tokens"])
        context_limit = int(latest_context_event["context_limit_tokens"])
        context_ratio = context_tokens / context_limit if context_limit > 0 else 0.0
        context_group = (
            f"  |  上下文 {_format_tokens(context_tokens)} / "
            f"{_format_tokens(context_limit)} tok（{context_ratio:.0%}）"
        )

    return (
        f"{len(turns)} 轮 · {steps} 步"
        f"  |  LLM {_format_seconds(model_time)} · 工具调用 {_format_seconds(tool_time)}"
        f"  |  首 token 平均 {ttft_text} · {speed_text}"
        f"  |  本轮缓存命中 {cache_text}"
        f"  |  输入 {prompt_text} tok · 输出 {completion_text} tok"
        f"{context_group}"
    )


def _event_end(event: dict[str, Any], now: float | None = None) -> float:
    """Return a usable end timestamp for a completed or running event."""
    ended_at = event.get("ended_at")
    if isinstance(ended_at, (int, float)):
        return float(ended_at)
    started_at = event.get("started_at")
    if event.get("status") == "running" and isinstance(started_at, (int, float)):
        return now if now is not None else time.time()
    return float(started_at) if isinstance(started_at, (int, float)) else 0.0


def _event_duration(event: dict[str, Any], now: float | None = None) -> float:
    """Return a non-negative event duration in seconds."""
    duration = event.get("duration")
    if isinstance(duration, (int, float)):
        return max(0.0, float(duration))
    started_at = event.get("started_at")
    if not isinstance(started_at, (int, float)):
        return 0.0
    return max(0.0, _event_end(event, now) - float(started_at))


def _sequence_bar_layout(durations: list[float], plot_width: int) -> list[tuple[int, int]]:
    """Fill the order axis with adjacent bars weighted by event duration."""
    count = len(durations)
    if count == 0:
        return []
    gap = 2 if plot_width >= count * 3 else 0
    available = max(count, plot_width - gap * (count - 1))
    base_width = min(5.0, available / count)
    weights = [max(0.0, duration) ** 0.5 for duration in durations]
    if sum(weights) <= 0:
        weights = [1.0] * count
    remaining = max(0.0, available - base_width * count)
    weight_total = sum(weights)
    raw_widths = [base_width + remaining * weight / weight_total for weight in weights]
    widths = [max(1, int(width)) for width in raw_widths]
    unallocated = available - sum(widths)
    fractions = sorted(
        range(count),
        key=lambda index: raw_widths[index] - int(raw_widths[index]),
        reverse=True,
    )
    for index in fractions[:max(0, unallocated)]:
        widths[index] += 1

    layout: list[tuple[int, int]] = []
    cursor = 0
    for width in widths:
        layout.append((cursor, width))
        cursor += width + gap
    return layout


def _hit_test_bar(
    targets: list[tuple[QRectF, str]],
    x: float,
    y: float,
) -> str | None:
    """Return the closest timeline event whose padded bar contains the point."""
    matches = [
        (abs(rect.center().x() - x), event_id)
        for rect, event_id in targets
        if rect.adjusted(-3, -5, 3, 5).contains(x, y)
    ]
    return min(matches, default=(0.0, None))[1]


class TrajectorySummaryBar(QLabel):
    """Live aggregate strip shared with the conversation page."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._events: list[dict[str, Any]] = []
        self.setMinimumHeight(20)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setStyleSheet(
            "background: transparent; color: #53657a; border: 0; "
            "padding: 0 8px; font-size: 12px;"
        )
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()
        self._refresh()

    def set_events(self, events: list[dict[str, Any]]) -> None:
        """Replace the event snapshot and immediately refresh metrics."""
        self._events = events
        self._refresh()

    def _refresh(self) -> None:
        self.setText(summarize_trajectory(self._events))


class TrajectoryTimeline(QWidget):
    """Compact three-lane overview arranged by event call order."""

    event_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._events: list[dict[str, Any]] = []
        self._bar_hit_targets: list[tuple[QRectF, str]] = []
        self._selected_id: str | None = None
        self.setMinimumHeight(150)
        self.setMaximumHeight(210)
        self.setCursor(Qt.PointingHandCursor)
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self.update)

    def set_events(self, events: list[dict[str, Any]]) -> None:
        """Replace the displayed event snapshot."""
        self._events = events
        if any(event.get("status") == "running" for event in events):
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def set_selected_event(self, event_id: str | None) -> None:
        """Highlight the timeline bar corresponding to the selected table row."""
        if self._selected_id == event_id:
            return
        self._selected_id = event_id
        self.update()

    def mousePressEvent(self, event) -> None:
        """Select the ledger step represented by the clicked timeline bar."""
        if event.button() == Qt.LeftButton:
            position = event.position()
            event_id = _hit_test_bar(
                self._bar_hit_targets,
                position.x(),
                position.y(),
            )
            if event_id is not None:
                self.event_selected.emit(event_id)
                event.accept()
                return
        super().mousePressEvent(event)

    def paintEvent(self, _event) -> None:
        """Paint lane labels, call-order ticks, and duration-scaled spans."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))

        left = 104
        right = 18
        top = 22
        bottom = 24
        plot_width = max(1, self.width() - left - right)
        lane_height = max(28, (self.height() - top - bottom) / len(LANE_LABELS))

        painter.setPen(QColor("#526273"))
        for lane, label in enumerate(LANE_LABELS):
            center_y = top + lane_height * lane + lane_height / 2
            painter.drawText(10, int(center_y + 4), label)
            painter.setPen(QPen(QColor("#e3e8ee"), 1))
            painter.drawLine(left, int(center_y), left + plot_width, int(center_y))
            painter.setPen(QColor("#526273"))

        ordered = [
            event for event in self._events
            if event.get("type") != "turn/end"
        ]
        ordered.sort(key=lambda event: int(event.get("sequence", 0)))
        self._bar_hit_targets = []
        if not ordered:
            painter.setPen(QColor("#8b98a7"))
            painter.drawText(left + 12, top + 20, "暂无轨迹，发送问题后将在这里实时显示。")
            return

        event_count = len(ordered)
        now = time.time()
        durations = [_event_duration(event, now) for event in ordered]
        bar_layout = _sequence_bar_layout(durations, plot_width)
        painter.setPen(QColor("#7a8795"))
        if event_count <= 5:
            tick_indexes = list(range(event_count))
        else:
            tick_indexes = sorted({
                round(index * (event_count - 1) / 4)
                for index in range(5)
            })
        for index in tick_indexes:
            offset, bar_width = bar_layout[index]
            x = left + offset + bar_width // 2
            painter.drawLine(x, top - 3, x, self.height() - bottom + 2)
            sequence = ordered[index].get("sequence", index + 1)
            painter.drawText(x - 12, self.height() - 6, f"#{sequence}")

        for index, event in enumerate(ordered):
            offset, bar_width = bar_layout[index]
            x = left + offset
            lane = LANE_INDEX.get(str(event.get("lane")), 0)
            y = int(top + lane * lane_height + lane_height / 2 - 7)
            color = QColor("#d9534f") if event.get("status") == "error" else KIND_COLORS.get(
                str(event.get("kind")), QColor("#8994a3")
            )
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            bar_rect = QRectF(x, y, bar_width, 14)
            painter.drawRoundedRect(bar_rect, 3, 3)
            event_id = str(event.get("id", event.get("sequence", index + 1)))
            self._bar_hit_targets.append((bar_rect, event_id))
            if event_id == self._selected_id:
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor("#263f59"), 2))
                painter.drawRoundedRect(bar_rect.adjusted(-1, -1, 1, 1), 4, 4)


class AnalysisTrajectoryPage(QWidget):
    """Searchable trajectory ledger with timeline and event inspector."""

    clear_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._events: list[dict[str, Any]] = []
        self._selected_id: str | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 16)
        root.setSpacing(10)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("轨迹")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        subtitle = QLabel("按轮次记录知识检索、Skill 路由、模型生成与本地工具调用。")
        subtitle.setStyleSheet("color: #657587;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box, 1)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索轨迹")
        self.search.setMaximumWidth(260)
        self.search.textChanged.connect(self._rebuild_table)
        header.addWidget(self.search)
        clear = QPushButton("清空轨迹")
        clear.clicked.connect(self.clear_requested.emit)
        header.addWidget(clear)
        root.addLayout(header)

        timeline_frame = QFrame()
        timeline_frame.setStyleSheet(
            "QFrame { background: white; border: 1px solid #d6dde5; border-radius: 6px; }"
        )
        timeline_layout = QVBoxLayout(timeline_frame)
        timeline_layout.setContentsMargins(8, 6, 8, 6)
        timeline_layout.addWidget(QLabel("调用顺序概览"))
        self.timeline = TrajectoryTimeline()
        self.timeline.event_selected.connect(self._select_timeline_event)
        timeline_layout.addWidget(self.timeline)
        root.addWidget(timeline_frame)

        content = QSplitter(Qt.Horizontal)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(("#", "类型", "轮次 / 步骤", "状态", "摘要", "耗时"))
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._show_selected_event)
        content.addWidget(self.table)

        detail_panel = QWidget()
        detail_layout = QVBoxLayout(detail_panel)
        detail_layout.setContentsMargins(10, 0, 0, 0)
        detail_title = QLabel("事件详情")
        detail_title.setStyleSheet("font-size: 15px; font-weight: 600;")
        detail_layout.addWidget(detail_title)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText("选择一条轨迹记录以查看输入、输出和计时。")
        detail_layout.addWidget(self.details)
        content.addWidget(detail_panel)
        content.setSizes([900, 420])
        root.addWidget(content, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._refresh_running_durations)
        self._timer.start()

    def set_events(self, events: list[dict[str, Any]]) -> None:
        """Replace the page's event snapshot and preserve selection by id."""
        self._events = events
        self.timeline.set_events(events)
        self._rebuild_table()

    def _matching_events(self) -> list[dict[str, Any]]:
        query = self.search.text().strip().casefold()
        if not query:
            return self._events
        return [
            event for event in self._events
            if query in " ".join(
                str(event.get(key, ""))
                for key in ("kind", "type", "title", "summary", "status", "turn", "step")
            ).casefold()
        ]

    def _rebuild_table(self) -> None:
        selected_id = self._selected_id
        events = self._matching_events()
        self.table.setRowCount(len(events))
        if not events:
            self._selected_id = None
            self.details.clear()
            return
        selected_row = -1
        for row, event in enumerate(events):
            event_id = str(event.get("id", row))
            sequence = int(event.get("sequence", row + 1))
            kind = str(event.get("kind", "system"))
            turn = event.get("turn")
            step = event.get("step")
            position = f"第 {turn} 轮" if turn is not None else "—"
            if step is not None:
                position += f" / 步骤 {step}"
            values = (
                str(sequence),
                KIND_LABELS.get(kind, kind),
                position,
                STATUS_LABELS.get(str(event.get("status")), str(event.get("status", ""))),
                str(event.get("summary") or event.get("title") or event.get("type") or ""),
                self._format_duration(event),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, event_id)
                if event.get("status") == "error":
                    item.setForeground(QColor("#b73832"))
                self.table.setItem(row, column, item)
            if event_id == selected_id:
                selected_row = row
        if selected_row >= 0:
            self.table.selectRow(selected_row)
        elif self._selected_id is None:
            self.table.selectRow(len(events) - 1)
            self.table.scrollToBottom()

    def _show_selected_event(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.table.item(rows[0].row(), 0)
        if item is None:
            return
        event_id = str(item.data(Qt.UserRole))
        event = next((entry for entry in self._events if str(entry.get("id")) == event_id), None)
        if event is None:
            return
        self._selected_id = event_id
        self.timeline.set_selected_event(event_id)
        metadata = {
            key: value for key, value in event.items()
            if key not in {"input", "output"}
        }
        sections = [json.dumps(metadata, ensure_ascii=False, indent=2, default=str)]
        detail_fields = (("输入", "input"),)
        if event.get("type") != "turn/end":
            detail_fields += (("输出", "output"),)
        for label, key in detail_fields:
            value = event.get(key)
            if value in (None, ""):
                continue
            rendered = value if isinstance(value, str) else json.dumps(
                value, ensure_ascii=False, indent=2, default=str
            )
            sections.append(f"{label}\n{'─' * 24}\n{rendered}")
        self.details.setPlainText("\n\n".join(sections))

    def _select_timeline_event(self, event_id: str) -> None:
        """Select, reveal, and inspect the table row clicked in the timeline."""
        self._selected_id = event_id
        matching_row = self._table_row_for_event(event_id)
        if matching_row < 0 and self.search.text():
            self.search.clear()
            matching_row = self._table_row_for_event(event_id)
        if matching_row < 0:
            return
        self.table.selectRow(matching_row)
        item = self.table.item(matching_row, 0)
        if item is not None:
            self.table.scrollToItem(item, QAbstractItemView.PositionAtCenter)
        self.timeline.set_selected_event(event_id)

    def _table_row_for_event(self, event_id: str) -> int:
        """Find an event row in the currently displayed ledger."""
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and str(item.data(Qt.UserRole)) == event_id:
                return row
        return -1

    def _refresh_running_durations(self) -> None:
        if not any(event.get("status") == "running" for event in self._events):
            return
        for row, event in enumerate(self._matching_events()):
            item = self.table.item(row, 5)
            if item is not None and event.get("status") == "running":
                item.setText(self._format_duration(event))

    @staticmethod
    def _format_duration(event: dict[str, Any]) -> str:
        duration = _event_duration(event, time.time())
        if event.get("status") == "running":
            return f"{duration:.2f} 秒…"
        return f"{duration:.2f} 秒"
