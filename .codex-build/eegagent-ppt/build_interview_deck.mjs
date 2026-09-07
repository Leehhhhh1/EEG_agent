import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "F:\\codex\\EEGAgent-main";
const SKILL_DIR = "C:\\Users\\Administrator\\.codex\\plugins\\cache\\openai-primary-runtime\\presentations\\26.904.11930\\skills\\presentations";
const TMP_DIR = path.join(workspaceDir, ".codex-build", "eegagent-ppt", "interview");
const FINAL_PPTX = path.join(workspaceDir, "output", "presentations", "EEGAgent智能体技术架构_面试版_v3.pptx");
const RUNTIME_PYTHON = "C:\\Users\\Administrator\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe";

const { makeNativeBulletParagraphs, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools", "artifact_tool_utils.mjs")).href,
);

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

const W = 1280;
const H = 720;
const FONT = "Noto Sans SC";
const C = {
  bg: "#06111E",
  bg2: "#081827",
  surface: "#10263B",
  surface2: "#122E48",
  surface3: "#0B2033",
  white: "#F7FAFC",
  text: "#E6EEF7",
  muted: "#A4B5C7",
  dim: "#70849A",
  line: "#29465F",
  cyan: "#22D3EE",
  cyan2: "#67E8F9",
  violet: "#8B5CF6",
  green: "#34D399",
  orange: "#F59E0B",
  red: "#FB7185",
};

const presentation = Presentation.create({ slideSize: { width: W, height: H } });

function textBox(slide, text, x, y, w, h, opt = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: opt.font ?? FONT,
    fontSize: opt.size ?? 22,
    bold: opt.bold ?? false,
    color: opt.color ?? C.text,
    alignment: opt.align ?? "left",
    verticalAlignment: opt.valign ?? "top",
    autoFit: opt.autoFit ?? "none",
    wrap: "square",
    lineSpacing: opt.lineSpacing ?? 1.12,
    insets: opt.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
  };
  return shape;
}

function box(slide, x, y, w, h, opt = {}) {
  const geometry = opt.geometry ?? "rect";
  const config = {
    geometry,
    position: { left: x, top: y, width: w, height: h },
    fill: opt.fill ?? C.surface,
    line: opt.line ?? { style: "solid", fill: C.line, width: 1 },
    shadow: opt.shadow ?? "shadow-none",
  };
  if (["rect", "textbox", "roundRect"].includes(geometry)) {
    config.borderRadius = opt.radius ?? 16;
  }
  return slide.shapes.add(config);
}

function title(slide, section, heading, subtitle = "") {
  textBox(slide, section, 68, 30, 400, 24, { size: 14, bold: true, color: C.cyan, valign: "middle" });
  textBox(slide, heading, 68, 62, 1120, 60, { size: 42, bold: true, color: C.white, valign: "middle" });
  box(slide, 68, 134, 1144, 2, { fill: C.line, line: { fill: "none", width: 0 }, radius: 0 });
  if (subtitle) textBox(slide, subtitle, 68, 151, 1110, 44, { size: 20, color: C.muted, valign: "middle" });
}

function footer(slide, page, source) {
  if (source) textBox(slide, source, 68, 681, 1050, 20, { size: 11, color: C.dim, valign: "middle" });
  textBox(slide, String(page).padStart(2, "0"), 1160, 681, 52, 20, { size: 13, color: C.muted, align: "right", valign: "middle" });
}

function notes(slide, lines) {
  slide.speakerNotes.textFrame.setText(lines);
}

function tag(slide, text, x, y, w, color = C.cyan) {
  const s = box(slide, x, y, w, 30, { fill: `${color}/14`, line: { style: "solid", fill: `${color}/55`, width: 1 }, radius: 15 });
  s.text = text;
  s.text.style = { typeface: FONT, fontSize: 15, bold: true, color, alignment: "center", verticalAlignment: "middle", autoFit: "none", insets: { left: 8, right: 8, top: 2, bottom: 2 } };
  return s;
}

function labeledBox(slide, heading, body, x, y, w, h, accent = C.cyan, opt = {}) {
  const b = box(slide, x, y, w, h, { fill: opt.fill ?? C.surface, line: { style: "solid", fill: accent, width: opt.lineWidth ?? 1.5 }, radius: opt.radius ?? 18, shadow: opt.shadow ?? "shadow-none" });
  textBox(slide, heading, x + 20, y + 17, w - 40, 32, { size: opt.headingSize ?? 22, bold: true, color: accent, align: opt.align ?? "left", valign: "middle" });
  if (body) textBox(slide, body, x + 20, y + 60, w - 40, h - 75, { size: opt.bodySize ?? 18, color: opt.bodyColor ?? C.text, align: opt.align ?? "left", lineSpacing: 1.18 });
  return b;
}

function connect(slide, from, to, opt = {}) {
  return slide.shapes.connect(from, to, {
    kind: opt.kind ?? "straight",
    fromSide: opt.fromSide ?? "right",
    toSide: opt.toSide ?? "left",
    line: { style: opt.style ?? "solid", fill: opt.color ?? "#55758F", width: opt.width ?? 2 },
    tail: { type: opt.arrow ?? "triangle", width: "sm", length: "sm" },
  });
}

function bulletBox(slide, items, x, y, w, h, opt = {}) {
  const s = textBox(slide, "", x, y, w, h, { size: opt.size ?? 20, color: opt.color ?? C.text });
  s.text = makeNativeBulletParagraphs(items, {
    marginLeftPoints: opt.margin ?? 18,
    hangingPoints: opt.hanging ?? 9,
    spaceAfterPoints: opt.spaceAfter ?? 9,
  });
  s.text.style = { typeface: FONT, fontSize: opt.size ?? 20, color: opt.color ?? C.text, autoFit: "none", lineSpacing: opt.lineSpacing ?? 1.15, insets: { left: 0, right: 0, top: 0, bottom: 0 } };
  return s;
}

// 1 Cover
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  const coverPath = path.join(workspaceDir, "assets", "presentation", "eegagent-cover.png");
  const cover = await fs.readFile(coverPath);
  slide.images.add({ blob: cover, contentType: "image/png", alt: "脑电信号与智能体网络的概念视觉", fit: "cover", position: { left: 0, top: 0, width: W, height: H } });
  box(slide, 0, 0, 720, H, { fill: "linear(0deg, #06111E 0%, #06111E/97 72%, #06111E/25 100%)", line: { fill: "none", width: 0 }, radius: 0 });
  tag(slide, "INTERVIEW TECHNICAL DECK", 72, 96, 260, C.cyan);
  textBox(slide, "EEGAgent", 72, 164, 560, 82, { size: 62, bold: true, color: C.white, valign: "middle" });
  textBox(slide, "智能体技术架构与工程实现", 76, 264, 570, 58, { size: 31, bold: true, color: C.cyan2 });
  textBox(slide, "规划机制、记忆模块\nMCP 工具调用、RAG 建库与检索", 76, 344, 570, 84, { size: 21, color: C.muted, lineSpacing: 1.2 });
  textBox(slide, "面试讲解版  10 页", 76, 632, 320, 25, { size: 15, color: C.dim });
  notes(slide, [
    "建议开场：EEGAgent 不是把 LLM 直接接到 EEG 数据，而是用运行时把规划、工具、记忆和知识检索拆成可控模块。",
    "来源：README.md；agent_runtime；RAG；mcp_server。",
    "封面视觉为项目已有生成资源，仅用于技术主题表达。",
  ]);
}

// 2 Architecture overview
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "01  系统总览", "五层技术架构", "核心目标是让 LLM 负责决策，让确定性代码负责边界、执行和状态管理。");

  const layers = [
    ["交互层", "PySide6 桌面端", "EDF 选择、对话流、执行轨迹、取消与导出", C.cyan],
    ["Agent 运行时", "Planner + Memory + ReAct Loop", "路由 Skill、管理上下文、驱动多轮工具调用", C.violet],
    ["协议层", "MCP over stdio", "持久连接、工具发现、结构化调用与错误归一化", C.orange],
    ["能力层", "EEG Tools + Session", "基础信息、片段探索、事件检测、报告生成", C.green],
    ["知识层", "Local RAG", "Docling 建库、BGE-M3 混合召回、重排与临时注入", C.cyan2],
  ];
  for (let i = 0; i < layers.length; i++) {
    const y = 220 + i * 82;
    box(slide, 68, y, 1144, 62, { fill: i % 2 === 0 ? "#0C2135" : "#0A1C2E", line: { style: "solid", fill: `${layers[i][3]}/70`, width: 1 }, radius: 10 });
    textBox(slide, layers[i][0], 88, y + 13, 130, 34, { size: 19, bold: true, color: layers[i][3], valign: "middle" });
    textBox(slide, layers[i][1], 238, y + 13, 330, 34, { size: 21, bold: true, color: C.white, valign: "middle" });
    textBox(slide, layers[i][2], 594, y + 13, 580, 34, { size: 18, color: C.muted, valign: "middle" });
  }
  footer(slide, 2, "来源：desktop_app.py、agent_runtime、mcp_server、eeg_core、RAG");
  notes(slide, [
    "面试讲法：用分层先建立全局认知。规划、记忆、调用协议、领域能力和知识库各自可替换。",
    "当前桌面端通过 stdio 启动本地 MCP 服务，Agent 通过 OpenAI 兼容接口调用 DeepSeek。",
  ]);
}

// 3 Planning and routing
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg2;
  title(slide, "02  规划机制", "两级 Skill 路由与安全降级", "规划首先确定本轮任务边界，再决定是否生成工具调用。");

  const q = labeledBox(slide, "用户问题", "例如：检查前十分钟有没有癫痫样放电", 68, 242, 250, 112, C.cyan, { align: "center", bodySize: 17 });
  const kw = labeledBox(slide, "关键词路由", "统计命中数，再按 priority 与名称排序", 380, 218, 256, 118, C.violet, { align: "center", bodySize: 17 });
  const sem = labeledBox(slide, "语义路由", "BGE-M3 对描述和 routing_examples 编码", 380, 376, 256, 118, C.violet, { align: "center", bodySize: 17 });
  const specialized = labeledBox(slide, "专用 Skill", "top ≥ 0.65 且 margin ≥ 0.05", 704, 205, 242, 108, C.green, { align: "center", bodySize: 17 });
  const general = labeledBox(slide, "general_eeg", "专用未通过，但 top ≥ 0.45", 704, 342, 242, 108, C.orange, { align: "center", bodySize: 17 });
  const none = labeledBox(slide, "no_skill", "低置信度时禁用 EEG 工具", 704, 479, 242, 108, C.red, { align: "center", bodySize: 17 });
  const out = labeledBox(slide, "本轮计划", "Skill 指令 + allowed_tools + 输出规则", 1008, 315, 204, 140, C.cyan2, { align: "center", bodySize: 17 });

  connect(slide, q, kw, { color: C.cyan });
  connect(slide, kw, specialized, { color: C.violet });
  connect(slide, kw, sem, { kind: "straight", fromSide: "bottom", toSide: "top", color: C.dim });
  textBox(slide, "无命中", 555, 344, 70, 22, { size: 13, bold: true, color: C.dim, align: "center" });
  connect(slide, sem, specialized, { kind: "elbow", color: C.violet });
  connect(slide, sem, general, { kind: "elbow", color: C.orange });
  connect(slide, sem, none, { kind: "elbow", color: C.red });
  connect(slide, specialized, out, { kind: "elbow", color: C.green });
  connect(slide, general, out, { color: C.orange });
  connect(slide, none, out, { kind: "elbow", color: C.red });

  textBox(slide, "语义分数取每个 Skill 最相似的 2 条样例均值，减少单条样例偶然命中。", 68, 618, 950, 32, { size: 18, bold: true, color: C.cyan2 });
  footer(slide, 3, "来源：agent_runtime/skills/selector.py、semantic_selector.py、registry.py");
  notes(slide, [
    "关键词路由适合高精度意图，命中后不再走语义选择。",
    "语义路由使用本地 BGE-M3，默认专用阈值 0.65，候选分差 0.05，general_eeg 下限 0.45。",
    "每个 Skill 的描述与多条 routing_examples 都参与编码，每个 Skill 取最相似两条的均值。",
  ]);
}

// 4 ReAct loop
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "03  规划执行", "受约束的流式 ReAct 循环", "模型可以连续调用工具，但运行时掌握会话、授权、终止条件和错误处理。");

  const nodes = [];
  const xs = [68, 264, 460, 656, 852, 1048];
  const heads = ["组织上下文", "模型决策", "解析 tool_calls", "运行时校验", "MCP 执行", "写回结果"];
  const bodies = ["系统指令、会话摘要、本轮 Skill、RAG", "流式输出文本、推理内容与函数参数", "按 index 拼接增量参数", "检查 session 与 allowed_tools", "自动注入 session_id", "tool 消息进入下一轮模型请求"];
  const accents = [C.cyan, C.violet, C.violet, C.red, C.orange, C.green];
  for (let i = 0; i < xs.length; i++) {
    nodes.push(labeledBox(slide, heads[i], bodies[i], xs[i], 250, 156, 164, accents[i], { align: "center", headingSize: 19, bodySize: 16 }));
  }
  for (let i = 0; i < nodes.length - 1; i++) connect(slide, nodes[i], nodes[i + 1], { color: "#587A95" });

  box(slide, 168, 478, 944, 114, { fill: "#0A1F32", line: { style: "solid", fill: C.line, width: 1 }, radius: 20 });
  textBox(slide, "继续循环", 204, 506, 145, 30, { size: 20, bold: true, color: C.cyan });
  textBox(slide, "存在 tool_calls 时追加 assistant 与 tool 消息，再请求模型。没有工具调用时输出最终答案。", 350, 500, 705, 52, { size: 19, color: C.text, valign: "middle" });
  textBox(slide, "硬上限：8 轮工具调用。达到上限后移除工具 schema，要求模型基于已有证据收敛。", 204, 558, 850, 28, { size: 17, bold: true, color: C.orange });
  footer(slide, 4, "来源：agent_runtime/mcp_chat_agent.py");
  notes(slide, [
    "这是一个工程化 ReAct 循环：模型提出 action，MCP 返回 observation，模型继续决策。",
    "运行时收集流式 reasoning_content 和 tool_call 增量，确保 DeepSeek 思考模式在后续工具轮次保持上下文连续。",
    "MAX_TOOL_ROUNDS 为 8。错误会转为模型可见的结构化结果，不直接让整轮崩溃。",
  ]);
}

// 5 Memory
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg2;
  title(slide, "04  记忆模块", "三层记忆与低频压缩", "系统分别保存对话过程、稳定事实和 EEG 领域状态，避免把所有信息都塞进聊天历史。");

  const layers = [
    ["短期对话记忆", "messages", "完整 user、assistant、tool 交换，保留 reasoning_content 供工具轮次续接", C.cyan],
    ["结构化会话摘要", "session_summary", "recording、patient、analyses、findings、reports、conversation，每类最多 12 条", C.violet],
    ["领域状态记忆", "EEGSession", "raw、processed_data、bipolar_channels、findings、reports，由 session_id 隔离", C.green],
  ];
  for (let i = 0; i < layers.length; i++) {
    const y = 222 + i * 112;
    box(slide, 68, y, 1144, 92, { fill: i === 1 ? "#132444" : "#0D2236", line: { style: "solid", fill: layers[i][3], width: 1.2 }, radius: 14 });
    textBox(slide, layers[i][0], 90, y + 18, 210, 30, { size: 21, bold: true, color: layers[i][3] });
    textBox(slide, layers[i][1], 90, y + 53, 210, 22, { size: 15, color: C.dim });
    textBox(slide, layers[i][2], 322, y + 19, 842, 58, { size: 18, color: C.text, lineSpacing: 1.22, valign: "middle" });
  }

  box(slide, 68, 574, 1144, 74, { fill: "#0B1B2C", line: { style: "solid", fill: C.orange, width: 1 }, radius: 16 });
  textBox(slide, "压缩策略", 90, 594, 130, 30, { size: 20, bold: true, color: C.orange });
  textBox(slide, "32K token 上限。达到 80% 后压缩最旧完整轮次，目标降到 55%。系统提示、会话摘要和最新轮次继续保留。", 240, 588, 920, 42, { size: 18, color: C.text, valign: "middle" });
  footer(slide, 5, "来源：agent_runtime/mcp_chat_agent.py、token_budget.py、eeg_core/session.py");
  notes(slide, [
    "会话摘要不是让模型自由总结工具原始输出。代码只抽取白名单字段，避免把大 payload 长期保存。",
    "历史压缩以完整 user turn 为单位，tool_calls 与对应 tool 消息保持原子性。",
    "RAG 文本只临时注入当前 user message，模型返回后会替换回原问题，避免知识片段污染长期记忆。",
  ]);
}

// 6 Tool calling
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "05  工具调用", "MCP 工具链与双重授权", "模型看见稳定 schema，运行时仍会在真正执行前做二次检查。");

  const model = labeledBox(slide, "LLM", "固定展示所有运行时 Skill 所引用工具的 schema", 68, 240, 245, 138, C.violet, { align: "center", bodySize: 17 });
  const runtime = labeledBox(slide, "Agent Runtime", "解析 JSON 参数\n校验 Skill allowlist\n注入 session_id", 394, 220, 290, 178, C.cyan, { align: "center", bodySize: 18 });
  const bridge = labeledBox(slide, "MCP Client Bridge", "持久 stdio 连接\n同步接口桥接异步 event loop", 766, 240, 245, 138, C.orange, { align: "center", bodySize: 17 });
  const server = labeledBox(slide, "FastMCP Server", "会话查找\n领域函数执行\n结构化结果", 1067, 240, 145, 138, C.green, { align: "center", bodySize: 16 });
  connect(slide, model, runtime, { color: C.violet });
  connect(slide, runtime, bridge, { color: C.cyan });
  connect(slide, bridge, server, { color: C.orange });

  box(slide, 68, 464, 1144, 150, { fill: "#0B1F32", line: { style: "solid", fill: C.line, width: 1 }, radius: 20 });
  textBox(slide, "关键工程处理", 96, 490, 210, 32, { size: 22, bold: true, color: C.white });
  bulletBox(slide, [
    "模型 schema 中删除 session_id，运行时绑定当前会话后自动注入",
    "调用异常转成 ok=false、error_type、message、retryable，作为 tool 结果回传",
    "事件检测结果先做代表性压缩，降低大规模事件列表对上下文的占用",
  ], 330, 484, 830, 112, { size: 18, color: C.text, spaceAfter: 7 });
  footer(slide, 6, "来源：agent_runtime/mcp_client.py、mcp_chat_agent.py、mcp_server/server.py");
  notes(slide, [
    "双重授权包括 schema 暴露和执行前 allowlist 校验。当前实现为了保持提示词前缀稳定，模型在会话内看见固定工具 schema 联集。",
    "真正调用时仍按当前 Skill 检查权限，所以 schema 稳定不会扩大执行能力。",
    "MCPClientBridge 在后台线程维护异步 stdio 会话，对桌面端提供同步 list_tools 与 call_tool。",
  ]);
}

// 7 RAG build
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg2;
  title(slide, "06  RAG 建库", "结构感知的父子分块与双索引", "建库优先保留标题层级、表格和页码，再为召回与上下文扩展分别设计粒度。");

  const xs = [68, 254, 440, 626, 812, 998];
  const heads = ["知识文件", "Docling 解析", "结构块", "父子分块", "BGE-M3 编码", "索引落盘"];
  const bodies = ["PDF、DOCX、MD\nTXT、HTML、XHTML", "OCR、表格结构\n标题层级、页码", "heading\nparagraph\nlist、table", "Parent 保持章节\nChild 面向召回", "Dense 向量\nSparse token 权重", "FAISS Dense\nSparse 倒排\nParent / Child 元数据"];
  const accents = [C.cyan, C.violet, C.violet, C.orange, C.cyan2, C.green];
  const nodes = [];
  for (let i = 0; i < xs.length; i++) nodes.push(labeledBox(slide, heads[i], bodies[i], xs[i], 230, 150, 170, accents[i], { align: "center", headingSize: 18, bodySize: 16 }));
  for (let i = 0; i < nodes.length - 1; i++) connect(slide, nodes[i], nodes[i + 1], { color: "#587A95" });

  box(slide, 68, 452, 1144, 140, { fill: "#0B1E31", line: { style: "solid", fill: C.line, width: 1 }, radius: 20 });
  textBox(slide, "分块参数", 96, 480, 150, 32, { size: 22, bold: true, color: C.orange });
  textBox(slide, "Parent", 268, 480, 90, 28, { size: 18, bold: true, color: C.white });
  textBox(slide, "目标 900，最大 1400 tokens", 350, 480, 300, 28, { size: 18, color: C.muted });
  textBox(slide, "Child", 268, 529, 90, 28, { size: 18, bold: true, color: C.white });
  textBox(slide, "目标 280，最大 400，重叠 50 tokens", 350, 529, 360, 28, { size: 18, color: C.muted });
  textBox(slide, "增量重建", 760, 480, 120, 28, { size: 18, bold: true, color: C.green });
  textBox(slide, "registry 比较文件 mtime、size、解析与分块配置。版本变化自动全量重建。", 760, 516, 385, 52, { size: 17, color: C.text, lineSpacing: 1.18 });
  footer(slide, 7, "来源：RAG/docling_parser.py、chunker.py、indexer.py");
  notes(slide, [
    "Parent 用于保存章节级上下文和来源元数据，Child 用于精细召回。",
    "表格和列表按行拆分，普通段落按句子边界拆分；稳定 ID 使用 SHA-1 摘要。",
    "FAISS 使用 IndexFlatIP，Sparse 使用 token_id 到文档权重的倒排表。",
  ]);
}

// 8 RAG retrieval
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "07  RAG 检索", "策略路由、三路召回与重排", "检索结果最多返回 3 个父块，并在当前用户消息中临时注入。");

  const steps = [
    ["检索策略", "知识问题直接检索。记录特异性工具请求跳过。模糊问题先做 FAISS 探测", 1110, C.cyan],
    ["双路召回", "Dense Top 20 + Sparse Top 20，RRF 权重 0.6 / 0.4，合并 Top 40", 990, C.cyan2],
    ["三路融合", "Dense 0.35 + Sparse 0.20 + ColBERT 0.45，保留 Top 15", 870, C.violet],
    ["Cross Encoder", "bge-reranker-v2-m3 计算 query-passage 相关度", 750, C.orange],
    ["最终评分", "0.2 × coarse_normalized + 0.8 × rerank_score，默认阈值 0.5", 630, C.green],
    ["上下文扩展", "按 parent 去重，补充命中 child 的前后相邻块，最终 Top 0 至 3", 510, C.cyan],
  ];
  for (let i = 0; i < steps.length; i++) {
    const x = (W - steps[i][2]) / 2;
    const y = 214 + i * 69;
    box(slide, x, y, steps[i][2], 54, { fill: i % 2 === 0 ? "#10283F" : "#0D2337", line: { style: "solid", fill: `${steps[i][3]}/65`, width: 1 }, radius: 10 });
    textBox(slide, steps[i][0], x + 18, y + 10, 150, 34, { size: 18, bold: true, color: steps[i][3], valign: "middle" });
    textBox(slide, steps[i][1], x + 175, y + 10, steps[i][2] - 195, 34, { size: 17, color: C.text, align: "center", valign: "middle" });
  }
  textBox(slide, "隔离原则：检索片段只服务本轮知识回答，工具结果才形成患者特异性证据。", 68, 642, 1020, 28, { size: 18, bold: true, color: C.red });
  footer(slide, 8, "来源：RAG/retrieval_policy.py、retriever.py、ranking.py、searcher.py");
  notes(slide, [
    "模糊问题先用 Dense Top-1 做 FAISS probe，默认阈值 0.35。",
    "三路融合使用 rank 而不是直接混合原始分数，降低不同打分尺度造成的偏差。",
    "reranker 输出 sigmoid 相关度，低于 0.5 的候选会被过滤，因此最终可能返回 0 条。",
  ]);
}

// 9 End-to-end trace
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg2;
  title(slide, "08  端到端示例", "一次事件筛查请求的执行轨迹", "示例问题：前一分钟是否存在癫痫样放电？如果有，出现在哪些导联？");

  box(slide, 118, 222, 3, 360, { fill: C.line, line: { fill: "none", width: 0 }, radius: 0 });
  const events = [
    ["01", "规划", "关键词命中 detection，加载 allowed_tools", C.violet],
    ["02", "检索", "识别为记录特异性请求，本轮跳过 RAG", C.cyan],
    ["03", "模型", "记录时长未知时，先请求 get_eeg_basic_information", C.cyan2],
    ["04", "执行", "运行时注入 session_id，经 MCP 获取记录范围", C.orange],
    ["05", "模型", "构造 detect_eeg_events，窗口 0 至 60 秒", C.violet],
    ["06", "记忆", "把事件数量、时间、导联、脑区和置信度写入摘要", C.green],
    ["07", "回答", "只引用工具证据，并说明需要专业人员复核", C.red],
  ];
  for (let i = 0; i < events.length; i++) {
    const y = 214 + i * 58;
    box(slide, 104, y + 9, 30, 30, { geometry: "ellipse", fill: events[i][3], line: { fill: "none", width: 0 }, radius: 0 });
    textBox(slide, events[i][0], 151, y, 52, 38, { size: 18, bold: true, color: events[i][3], valign: "middle" });
    textBox(slide, events[i][1], 218, y, 116, 38, { size: 20, bold: true, color: C.white, valign: "middle" });
    textBox(slide, events[i][2], 348, y, 815, 38, { size: 18, color: C.text, valign: "middle" });
  }
  box(slide, 750, 608, 462, 48, { fill: "#14263A", line: { style: "solid", fill: C.cyan, width: 1 }, radius: 12 });
  textBox(slide, "可观测性：路由分数、工具耗时、上下文占用、缓存命中率", 772, 618, 418, 28, { size: 16, bold: true, color: C.cyan2, align: "center", valign: "middle" });
  footer(slide, 9, "来源：agent_runtime/mcp_chat_agent.py、detection Skill、mcp_server/server.py");
  notes(slide, [
    "这个例子适合面试时按 Trace 顺序讲，能够把规划、RAG 策略、MCP、记忆和安全输出串起来。",
    "detection Skill 规定窗口不超过 600 秒；本例使用 0 至 60 秒。",
    "路由、RAG、模型请求、工具调用和 turn end 都会发出可视化 trace 事件。",
  ]);
}

// 10 Tradeoffs
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "09  工程总结", "当前实现与演进方向", "面试中需要同时说明已经落地的能力，以及下一阶段可以怎样扩展。");

  textBox(slide, "当前实现", 68, 214, 480, 38, { size: 26, bold: true, color: C.green });
  bulletBox(slide, [
    "确定性关键词与本地语义路由结合，低置信度安全降级",
    "Skill 指令按轮次隔离，执行前强制校验工具权限",
    "三层记忆配合 token 预算，RAG 只在当前轮临时注入",
    "MCP 解耦 Agent 与 EEG 算法，并提供完整执行轨迹",
  ], 68, 274, 510, 250, { size: 19, color: C.text, spaceAfter: 12 });

  textBox(slide, "演进方向", 662, 214, 480, 38, { size: 26, bold: true, color: C.orange });
  bulletBox(slide, [
    "增加显式任务图，让复杂问题拆成可检查的多 Skill 子计划",
    "把关键事件与报告写入可持久化记忆，并设计版本与权限策略",
    "建立路由准确率、工具成功率、引用正确性与临床复核指标",
    "对独立工具调用引入并行执行，同时保持会话状态一致性",
  ], 662, 274, 520, 250, { size: 19, color: C.text, spaceAfter: 12 });

  box(slide, 68, 570, 1114, 72, { fill: "linear(90deg, #0E2A42 0%, #14274A 100%)", line: { style: "solid", fill: C.cyan, width: 1.5 }, radius: 18 });
  textBox(slide, "面试总结", 92, 590, 140, 30, { size: 20, bold: true, color: C.cyan });
  textBox(slide, "EEGAgent 的技术重点是把 LLM 的灵活决策放进一个可控、可追踪、可扩展的领域运行时。", 246, 586, 900, 38, { size: 21, bold: true, color: C.white, valign: "middle" });
  footer(slide, 10, "来源：当前仓库实现。演进方向为基于现状的设计建议");
  notes(slide, [
    "最后一页把技术亮点与不足同时讲清楚。右侧均为建议，不代表当前已经实现。",
    "可根据应聘岗位调整重点：Agent 岗强调规划和记忆，平台岗强调 MCP 与可观测性，算法岗强调 RAG 与 EEG 工具。",
  ]);
}

const stagingDir = path.join(workspaceDir, ".codex-finalizer");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "EEGAgent智能体技术架构_面试版_candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

for (let i = 0; i < presentation.slides.items.length; i++) {
  const slide = presentation.slides.items[i];
  const preview = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(TMP_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const result = await finalizePresentation({
  explicitTotalSlideCount: 10,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
  ],
  fontPolicy: { basis: "design", families: [FONT] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "EEGAgent智能体技术架构_面试版_v3.validation.json"),
});

console.log(JSON.stringify({ finalPath: FINAL_PPTX, slideCount: presentation.slides.items.length, validation: result }, null, 2));
