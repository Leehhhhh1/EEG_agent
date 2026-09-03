import html
import json
import sys
import time
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Qt, Signal, QTimer
from PySide6.QtGui import QAction, QColor, QKeyEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from agent_runtime.mcp_chat_agent import MCPChatAgent
from agent_runtime.mcp_client import MCPClientBridge


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"
PROMPT_PLACEHOLDER = "输入有关当前脑电记录的问题，按 Enter 发送，Shift+Enter 换行。"


class AgentLoader(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, bridge: MCPClientBridge, file_path: str):
        """初始化对象状态。"""
        super().__init__()
        self.bridge = bridge
        self.file_path = file_path

    def run(self):
        """执行当前任务流程。"""
        try:
            self.bridge.start()
            opened = self.bridge.call_tool("open_eeg_session", {"file_path": self.file_path})
            if opened["is_error"]:
                raise RuntimeError("打开 EDF 失败：" + ("\n".join(opened["content"]) or "MCP Server 未返回具体错误。"))
            session_info = opened["structured_content"]
            if not session_info:
                raise RuntimeError("打开 EDF 失败：MCP Server 未返回会话信息。")
            session_id = session_info["session_id"]
            basic = self.bridge.call_tool("get_eeg_basic_information", {"session_id": session_id})
            if basic["is_error"]:
                raise RuntimeError("读取脑电基本信息失败：" + ("\n".join(basic["content"]) or "MCP Server 未返回具体错误。"))
            basic_info = basic["structured_content"] or {}
            self.finished.emit({
                "session_id": session_id,
                "summary": session_info,
                "basic_info": basic_info,
            })
        except Exception as exc:
            self.failed.emit(f"加载 EEG 会话失败：{exc}")


class Spinner(QWidget):
    def __init__(self):
        """初始化对象状态。"""
        super().__init__()
        self._step = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self.setFixedSize(22, 22)
        self.hide()

    def start(self):
        """启动当前流程。"""
        self.show()
        self._timer.start(80)

    def stop(self):
        """停止当前流程。"""
        self._timer.stop()
        self.hide()

    def _advance(self):
        """处理 advance 相关逻辑。"""
        self._step = (self._step + 1) % 12
        self.update()

    def paintEvent(self, _event):
        """绘制当前控件。"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(self.width() / 2, self.height() / 2)
        pen = QPen()
        pen.setWidth(3)
        pen.setCapStyle(Qt.RoundCap)
        for index in range(12):
            alpha = 45 + ((index + self._step) % 12) * 17
            pen.setColor(QColor(23, 105, 170, alpha))
            painter.setPen(pen)
            painter.drawLine(0, -8, 0, -4)
            painter.rotate(30)


class ChatWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str)
    delta = Signal(str)
    tool_started = Signal(str)
    tool_finished = Signal(str)
    tool_call_detected = Signal()

    def __init__(self, agent, prompt: str):
        """初始化对象状态。"""
        super().__init__()
        self.agent = agent
        self.prompt = prompt

    def run(self):
        """执行当前任务流程。"""
        try:
            self.finished.emit(self.agent.run_stream(
                self.prompt,
                on_delta=self.delta.emit,
                on_tool_start=self.tool_started.emit,
                on_tool_end=self.tool_finished.emit,
                on_tool_call_detected=self.tool_call_detected.emit,
            ))
        except Exception as exc:
            self.failed.emit(str(exc))


class RAGPreloader(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def run(self):
        """Load both local retrieval models before the first user request."""
        try:
            from RAG.retriever import EEGRetriever

            self.finished.emit(EEGRetriever())
        except Exception as exc:
            self.failed.emit(f"本地检索模型初始化失败：{exc}")


class MessageComposer(QPlainTextEdit):
    send_requested = Signal()

    def focusInEvent(self, event):
        # 输入法预编辑文本不会触发 textChanged，因此获得焦点时隐藏占位提示。
        """处理输入框获得焦点事件。"""
        self.setPlaceholderText("")
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        """处理输入框失去焦点事件。"""
        super().focusOutEvent(event)
        if not self.toPlainText():
            self.setPlaceholderText(PROMPT_PLACEHOLDER)

    def keyPressEvent(self, event: QKeyEvent):
        """处理键盘输入事件。"""
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
            else:
                # Enter sends all text currently in the composer; Shift+Enter
                # remains available for composing a multi-line message.
                self.send_requested.emit()
            return
        super().keyPressEvent(event)


class EEGAgentWindow(QMainWindow):
    def __init__(self):
        """初始化对象状态。"""
        super().__init__()
        self.agent = None
        self.preloaded_rag_retriever = None
        self.rag_ready = False
        self.eeg_session_id = None
        self.mcp_bridge = MCPClientBridge(PROJECT_ROOT)
        self.active_thread = None
        self.active_worker = None
        self.transcript = []
        self.streaming_label = None
        self.streaming_response = ""

        self.setWindowTitle("EEGAgent 脑电分析客户端")
        self.resize(1280, 820)
        self._build_menu()
        self._build_ui()
        self._start_rag_preload()

    def _build_menu(self):
        """构建 build menu 所需内容。"""
        file_menu = self.menuBar().addMenu("文件")
        open_action = QAction("打开 EDF 文件...", self)
        open_action.triggered.connect(self.choose_edf)
        file_menu.addAction(open_action)

        export_action = QAction("导出对话...", self)
        export_action.triggered.connect(self.export_conversation)
        file_menu.addAction(export_action)
        file_menu.addSeparator()

        quit_action = QAction("退出", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        session_menu = self.menuBar().addMenu("会话")
        clear_action = QAction("清空对话", self)
        clear_action.triggered.connect(self.clear_conversation)
        session_menu.addAction(clear_action)

    def _build_ui(self):
        """构建 build ui 所需内容。"""
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_recording_panel())
        splitter.addWidget(self._build_chat_panel())
        splitter.addWidget(self._build_details_panel())
        splitter.setSizes([310, 650, 320])
        self.setCentralWidget(splitter)
        self.setStyleSheet(
            "QMainWindow { background: #f5f7fa; }"
            "QFrame#messageUser { background: #dceeff; border: 1px solid #a8c9e8; border-radius: 7px; }"
            "QFrame#messageAssistant { background: #ffffff; border: 1px solid #d6dde5; border-radius: 7px; }"
            "QPlainTextEdit, QLineEdit { background: #ffffff; border: 1px solid #c9d3df; border-radius: 4px; }"
            "QPushButton { background: #1769aa; color: white; border: 0; border-radius: 4px; padding: 7px 12px; }"
            "QPushButton:disabled { background: #9caaba; }"
            "QPushButton:hover:!disabled { background: #12588f; }"
        )

    def _build_recording_panel(self):
        """构建 build recording panel 所需内容。"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("脑电记录")
        title.setStyleSheet("font-size: 17px; font-weight: 600;")
        layout.addWidget(title)

        layout.addWidget(QLabel("当前脑电数据"))
        self.file_input = QLineEdit()
        self.file_input.setReadOnly(True)
        self.file_input.setPlaceholderText("未加载脑电数据，可直接开始对话")
        layout.addWidget(self.file_input)

        buttons = QHBoxLayout()
        self.open_button = QPushButton("选择 EDF")
        self.open_button.clicked.connect(self.choose_edf)
        self.load_button = QPushButton("加载")
        self.load_button.clicked.connect(self.load_selected_edf)
        self.load_button.setEnabled(False)
        self.detach_button = QPushButton("移除数据")
        self.detach_button.clicked.connect(self.detach_recording)
        self.detach_button.setEnabled(False)
        buttons.addWidget(self.open_button)
        buttons.addWidget(self.load_button)
        buttons.addWidget(self.detach_button)
        layout.addLayout(buttons)

        layout.addWidget(QLabel("记录信息"))
        self.recording_info = QPlainTextEdit()
        self.recording_info.setReadOnly(True)
        self.recording_info.setPlaceholderText("加载脑电记录后将在此显示基本信息。")
        layout.addWidget(self.recording_info, 1)
        return panel

    def _build_chat_panel(self):
        """构建 build chat panel 所需内容。"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("脑电分析对话")
        title.setStyleSheet("font-size: 17px; font-weight: 600;")
        layout.addWidget(title)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(4, 4, 4, 4)
        self.chat_layout.setSpacing(10)
        self.chat_layout.addStretch()
        self.chat_scroll.setWidget(self.chat_container)
        layout.addWidget(self.chat_scroll, 1)

        self.prompt_input = MessageComposer()
        self.prompt_input.setPlaceholderText(PROMPT_PLACEHOLDER)
        self.prompt_input.setFixedHeight(84)
        self.prompt_input.send_requested.connect(self.send_message)
        self.prompt_input.textChanged.connect(self._update_prompt_placeholder)
        layout.addWidget(self.prompt_input)

        controls = QHBoxLayout()
        self.clear_button = QPushButton("清空")
        self.clear_button.clicked.connect(self.clear_conversation)
        self.send_button = QPushButton("发送")
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setEnabled(True)
        controls.addWidget(self.clear_button)
        controls.addStretch()
        controls.addWidget(self.send_button)
        layout.addLayout(controls)
        return panel

    def _build_details_panel(self):
        """构建 build details panel 所需内容。"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("分析状态")
        title.setStyleSheet("font-size: 17px; font-weight: 600;")
        layout.addWidget(title)

        self.spinner = Spinner()
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #405266;")
        status_row = QHBoxLayout()
        status_row.addWidget(self.spinner)
        status_row.addWidget(self.status_label, 1)
        layout.addLayout(status_row)

        layout.addWidget(QLabel("最近一次运行"))
        self.metrics = QPlainTextEdit()
        self.metrics.setReadOnly(True)
        self.metrics.setMaximumHeight(130)
        self.metrics.setPlaceholderText("运行指标将在此显示。")
        layout.addWidget(self.metrics)

        layout.addWidget(QLabel("临床提示"))
        note = QLabel(
            "本客户端用于辅助脑电分析与报告整理。"
            "分析结果须由具备资质的临床医生审核。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #6b4b00; background: #fff6d8; padding: 8px; border-radius: 4px;")
        layout.addWidget(note)
        layout.addStretch()
        return panel

    def choose_edf(self):
        """打开文件选择器以选择 EDF 脑电文件。"""
        if self.active_thread is not None:
            self._set_status("后台初始化尚未完成，请稍候。")
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择脑电记录",
            str(PROJECT_ROOT / "data"),
            "EDF 脑电文件 (*.edf);;所有文件 (*.*)",
        )
        if not file_path:
            return
        self.file_input.setText(file_path)
        self.load_button.setEnabled(True)
        self._set_status("已选择脑电记录，请加载后将其附加到当前会话。")

    def load_selected_edf(self):
        """加载当前选择的 EDF 文件并创建分析会话。"""
        if self.active_thread is not None:
            return
        file_path = self.file_input.text().strip()
        if not file_path:
            return
        self._set_busy(True, "正在加载脑电数据和本地模型...")
        worker = AgentLoader(self.mcp_bridge, file_path)
        self._run_worker(worker, worker.finished, self._agent_loaded, worker.failed, self._job_failed)

    def _start_rag_preload(self):
        """Preload the embedding and reranking models on a worker thread."""
        self.spinner.start()
        self._set_busy(True, "正在后台加载 BGE-M3 和重排模型...")
        worker = RAGPreloader()
        self._run_worker(
            worker,
            worker.finished,
            self._rag_preloaded,
            worker.failed,
            self._rag_preload_failed,
        )

    def _rag_preloaded(self, retriever):
        """Keep the loaded models alive for every subsequent chat turn."""
        self.preloaded_rag_retriever = retriever
        self.rag_ready = True
        if self.agent is not None:
            self.agent.set_rag_retriever(retriever)
        self.spinner.stop()
        self._set_status("本地检索模型加载完成，可以开始对话或加载脑电记录。")

    def _rag_preload_failed(self, message):
        """Report startup model failures without retrying on the first prompt."""
        self.rag_ready = False
        self._job_failed(message)

    def send_message(self):
        """发送用户输入并启动后台回答任务。"""
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt or self.active_thread is not None:
            return
        if not self.rag_ready:
            self._set_status("本地检索模型尚未加载完成，暂时不能发送消息。")
            return
        try:
            agent = self._current_agent()
        except ValueError as exc:
            self._add_message("系统", f"发生错误：{exc}", False)
            self._set_status("无法开始会话。")
            return
        self.prompt_input.clear()
        self._add_message("你", prompt, True)
        self.transcript.append(("你", prompt))
        has_recording = self.eeg_session_id is not None
        self._set_busy(True, "正在分析脑电并生成回答..." if has_recording else "正在生成回答...")
        # Keep composing available while the current answer streams. Sending
        # remains blocked by active_thread until this worker has finished.
        self.prompt_input.setEnabled(True)
        self.prompt_input.setFocus()
        self.streaming_response = ""
        self.streaming_label = self._add_message("EEGAgent", "正在生成回答...", False)
        worker = ChatWorker(agent, prompt)
        self._run_worker(worker, worker.finished, self._chat_finished, worker.failed, self._job_failed)

    def _run_worker(self, worker, success_signal, success_handler, failure_signal, failure_handler):
        """处理 run worker 相关逻辑。"""
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        success_signal.connect(success_handler)
        failure_signal.connect(failure_handler)
        if isinstance(worker, ChatWorker):
            worker.delta.connect(self._append_stream_delta)
            worker.tool_call_detected.connect(self._reset_stream_response)
            worker.tool_started.connect(self._tool_started)
            worker.tool_finished.connect(self._tool_finished)
        success_signal.connect(thread.quit)
        failure_signal.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._worker_finished)
        self.active_thread = thread
        # 保持 Python 包装对象存活，直到工作线程完全退出。
        self.active_worker = worker
        thread.start()

    def _agent_loaded(self, loaded):
        """处理 agent loaded 相关逻辑。"""
        new_session_id = loaded["session_id"]
        try:
            self._current_agent()
        except Exception as exc:
            try:
                self.mcp_bridge.call_tool("close_eeg_session", {"session_id": new_session_id})
            except Exception:
                pass
            self._job_failed(str(exc))
            return
        if self.eeg_session_id is not None:
            try:
                self.mcp_bridge.call_tool("close_eeg_session", {"session_id": self.eeg_session_id})
            except Exception:
                pass
        self.eeg_session_id = new_session_id
        self.agent.attach_session(new_session_id, loaded["basic_info"])
        self.recording_info.setPlainText(json.dumps(loaded["basic_info"], ensure_ascii=False, indent=2))
        self.detach_button.setEnabled(True)
        file_name = loaded["summary"]["recording_name"]
        self._add_message("系统", f"已加载脑电数据：{file_name}。后续问题可结合该数据进行分析。", False)
        self.transcript.append(("系统", f"已加载脑电数据：{file_name}。"))
        self._set_status("脑电记录加载完成，已附加到当前会话。")

    def _chat_finished(self, result):
        """处理 chat finished 相关逻辑。"""
        response = result.get("response", "未返回分析结果。")
        if self.streaming_label is not None:
            self._set_message_text(self.streaming_label, "EEGAgent", response)
        else:
            self._add_message("EEGAgent", response, False)
        self.transcript.append(("EEGAgent", response))
        self.streaming_label = None
        self.streaming_response = ""
        self.spinner.stop()
        routing = result.get("routing") or {}
        if not routing.get("enabled"):
            routing_lines = [
                "处理模式：普通 RAG 问答",
                "路由过程：未加载 EDF → 跳过 Skill 路由 → 不开放 EEG 工具",
                "Skill 路由：未启用（未加载 EDF）",
                "开放工具：0 个",
            ]
        else:
            source_labels = {
                "keyword": "关键词匹配",
                "embedding": "BGE-M3 语义匹配",
                "fallback": "语义低置信度兜底",
            }
            route_processes = {
                "keyword": "EDF 会话就绪 → 关键词命中 → 选择 Skill",
                "embedding": "EDF 会话就绪 → 关键词未命中 → BGE-M3 语义匹配 → 选择 Skill",
                "fallback": "EDF 会话就绪 → 关键词未命中 → 语义置信度不足 → general_eeg",
            }
            source = routing.get("source")
            routing_lines = [
                f"路由过程：{route_processes.get(source, 'EDF 会话就绪 → 选择 Skill')}",
                f"Skill：{routing.get('skill', 'unknown')}",
                f"路由方式：{source_labels.get(source, source or 'unknown')}",
            ]
            keyword_matches = routing.get("keyword_matches") or []
            if keyword_matches:
                routing_lines.append("命中关键词：" + "、".join(keyword_matches))
            candidates = routing.get("candidates") or []
            if candidates:
                routing_lines.append(
                    "语义候选：" + "；".join(
                        f"{candidate.get('name')} {candidate.get('score', 0):.3f}"
                        for candidate in candidates
                    )
                )
                routing_lines.append(f"候选分差：{routing.get('margin', 0):.3f}")
            allowed_tools = routing.get("allowed_tools") or []
            routing_lines.append(
                "开放工具：" + ("、".join(allowed_tools) if allowed_tools else "无")
            )
        self.metrics.setPlainText(
            "\n".join(routing_lines) + "\n\n"
            f"交互轮数：{result.get('rounds', 0)}\n"
            f"模型耗时：{result.get('model_time', 0):.2f} 秒\n"
            f"本地工具耗时：{result.get('local_tool_time', 0):.2f} 秒\n"
            f"总耗时：{result.get('total_time', 0):.2f} 秒"
        )
        self._set_status("分析完成。")

    def _job_failed(self, message):
        """处理 job failed 相关逻辑。"""
        if self.streaming_label is not None:
            self._set_message_text(self.streaming_label, "系统", f"发生错误：{message}")
        else:
            self._add_message("系统", f"发生错误：{message}", False)
        self.streaming_label = None
        self.streaming_response = ""
        self.spinner.stop()
        self._set_status("操作未能完成。")

    def _worker_finished(self):
        """处理 worker finished 相关逻辑。"""
        self.active_thread = None
        self.active_worker = None
        self.open_button.setEnabled(True)
        self.load_button.setEnabled(bool(self.file_input.text().strip()))
        self.detach_button.setEnabled(self.eeg_session_id is not None)
        self.send_button.setEnabled(self.rag_ready)
        self.prompt_input.setEnabled(self.rag_ready)
        self.spinner.stop()

    def _set_busy(self, busy: bool, status: str):
        """处理 set busy 相关逻辑。"""
        self._set_status(status)
        self.open_button.setEnabled(not busy)
        self.load_button.setEnabled(not busy and bool(self.file_input.text().strip()))
        self.detach_button.setEnabled(not busy and self.eeg_session_id is not None)
        self.send_button.setEnabled(not busy and self.rag_ready)
        self.prompt_input.setEnabled(not busy and self.rag_ready)

    def _set_status(self, status: str):
        """处理 set status 相关逻辑。"""
        self.status_label.setText(status)
        self.statusBar().showMessage(status)

    def _update_prompt_placeholder(self):
        """处理 update prompt placeholder 相关逻辑。"""
        if self.prompt_input.toPlainText() or self.prompt_input.hasFocus():
            self.prompt_input.setPlaceholderText("")
        else:
            self.prompt_input.setPlaceholderText(PROMPT_PLACEHOLDER)

    def _append_stream_delta(self, delta: str):
        """处理 append stream delta 相关逻辑。"""
        if self.streaming_label is None:
            return
        self.streaming_response += delta
        self._set_message_text(self.streaming_label, "EEGAgent", self.streaming_response)

    def _reset_stream_response(self):
        """处理 reset stream response 相关逻辑。"""
        self.streaming_response = ""
        if self.streaming_label is not None:
            self._set_message_text(self.streaming_label, "EEGAgent", "正在准备调用本地分析工具...")

    def _tool_started(self, tool_name: str):
        """处理 tool started 相关逻辑。"""
        self.spinner.start()
        self._set_status(f"正在调用本地工具：{tool_name}")
        if self.streaming_label is not None:
            self._set_message_text(self.streaming_label, "EEGAgent", f"正在调用工具：{tool_name}")

    def _tool_finished(self, tool_name: str):
        """处理 tool finished 相关逻辑。"""
        self.spinner.stop()
        self._set_status(f"工具调用完成：{tool_name}，正在继续生成回答...")

    def _set_message_text(self, label: QLabel, speaker: str, text: str):
        """处理 set message text 相关逻辑。"""
        label.setText(f"<b>{html.escape(speaker)}</b><br>{html.escape(text).replace(chr(10), '<br>')}")

    def _add_message(self, speaker: str, text: str, is_user: bool):
        """处理 add message 相关逻辑。"""
        frame = QFrame()
        frame.setObjectName("messageUser" if is_user else "messageAssistant")
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        label = QLabel()
        self._set_message_text(label, speaker, text)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(label)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, frame)
        self.chat_scroll.verticalScrollBar().rangeChanged.connect(
            lambda _minimum, maximum: self.chat_scroll.verticalScrollBar().setValue(maximum)
        )
        return label

    def clear_conversation(self):
        """清空当前聊天记录并重置对话上下文。"""
        if self.active_thread is not None:
            return
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.transcript = []
        self.metrics.clear()
        if self.agent is not None:
            self.agent.reset()
        if self.eeg_session_id is not None:
            self._add_message("系统", "对话已清空，当前脑电数据仍保持加载。", False)
        else:
            self._add_message("系统", "对话已清空，可以直接开始会话或选择脑电数据。", False)

    def _current_agent(self):
        """处理 current agent 相关逻辑。"""
        if self.agent is None:
            self.agent = MCPChatAgent(self.mcp_bridge)
        if self.preloaded_rag_retriever is not None:
            self.agent.set_rag_retriever(self.preloaded_rag_retriever)
        return self.agent

    def detach_recording(self):
        """移除当前 EEG 数据会话并回到普通对话。"""
        if self.active_thread is not None:
            return
        if self.eeg_session_id is None:
            return
        file_name = Path(self.file_input.text()).name
        if self.eeg_session_id:
            try:
                self.mcp_bridge.call_tool("close_eeg_session", {"session_id": self.eeg_session_id})
            except Exception:
                pass
        self.agent.detach_session()
        self.eeg_session_id = None
        self.file_input.clear()
        self.recording_info.clear()
        self.load_button.setEnabled(False)
        self.detach_button.setEnabled(False)
        self._add_message("系统", f"已移除脑电数据：{file_name}。后续消息将作为直接对话处理。", False)
        self.transcript.append(("系统", f"已移除脑电数据：{file_name}。"))
        self._set_status("脑电数据已移除，当前可直接对话。")

    def closeEvent(self, event):
        """处理 close Event 相关逻辑。"""
        self.mcp_bridge.close()
        super().closeEvent(event)

    def export_conversation(self):
        """导出 export conversation 相关结果。"""
        if not self.transcript:
            self._set_status("当前没有可导出的对话内容。")
            return
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出对话记录",
            str(PROJECT_ROOT / "eegagent-conversation.txt"),
            "文本文件 (*.txt)",
        )
        if not output_path:
            return
        text = "\n\n".join(f"{speaker}:\n{content}" for speaker, content in self.transcript)
        Path(output_path).write_text(text, encoding="utf-8")
        self._set_status("对话记录已导出。")


def main():
    """启动程序入口。"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = EEGAgentWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
