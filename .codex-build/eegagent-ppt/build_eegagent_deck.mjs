import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "F:\\codex\\EEGAgent-main";
const SKILL_DIR = "C:\\Users\\Administrator\\.codex\\plugins\\cache\\openai-primary-runtime\\presentations\\26.904.11930\\skills\\presentations";
const TMP_DIR = path.join(workspaceDir, ".codex-build", "eegagent-ppt");
const FINAL_PPTX = path.join(workspaceDir, "output", "presentations", "EEGAgent项目介绍_中文_v2.pptx");
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
  bg: "#07111F",
  bg2: "#0A182A",
  surface: "#10243A",
  surface2: "#132B44",
  white: "#F7FAFC",
  text: "#E7EEF7",
  muted: "#A8B7C8",
  cyan: "#22D3EE",
  cyan2: "#67E8F9",
  violet: "#8B5CF6",
  green: "#34D399",
  orange: "#F59E0B",
  red: "#FB7185",
  line: "#28435E",
};

const presentation = Presentation.create({ slideSize: { width: W, height: H } });

function addText(slide, text, x, y, w, h, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: options.font ?? FONT,
    fontSize: options.size ?? 24,
    bold: options.bold ?? false,
    color: options.color ?? C.text,
    alignment: options.align ?? "left",
    verticalAlignment: options.valign ?? "top",
    autoFit: options.autoFit ?? "none",
    wrap: "square",
    lineSpacing: options.lineSpacing ?? 1.12,
    insets: options.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
  };
  return shape;
}

function addBox(slide, x, y, w, h, options = {}) {
  return slide.shapes.add({
    geometry: options.geometry ?? "rect",
    position: { left: x, top: y, width: w, height: h },
    fill: options.fill ?? C.surface,
    line: options.line ?? { style: "solid", fill: C.line, width: 1 },
    borderRadius: options.radius ?? 18,
    shadow: options.shadow ?? "shadow-none",
  });
}

function addTitle(slide, title, section) {
  addText(slide, section.toUpperCase(), 68, 34, 210, 24, {
    size: 14, bold: true, color: C.cyan, valign: "middle",
  });
  addText(slide, title, 68, 66, 1090, 60, {
    size: 42, bold: true, color: C.white, valign: "middle",
  });
  addBox(slide, 68, 137, 1144, 2, { fill: C.line, line: { fill: "none", width: 0 }, radius: 0 });
}

function addFooter(slide, page, source = "") {
  addText(slide, String(page).padStart(2, "0"), 1158, 680, 54, 20, {
    size: 13, color: C.muted, align: "right", valign: "middle",
  });
  if (source) addText(slide, source, 68, 680, 1000, 20, { size: 11, color: "#71839A", valign: "middle" });
}

function addNotes(slide, lines) {
  slide.speakerNotes.textFrame.setText(lines);
}

function addLabel(slide, text, x, y, w, color = C.cyan) {
  const shape = addBox(slide, x, y, w, 30, {
    fill: `${color}/14`, line: { style: "solid", fill: `${color}/55`, width: 1 }, radius: 15,
  });
  shape.text = text;
  shape.text.style = {
    typeface: FONT, fontSize: 15, bold: true, color,
    alignment: "center", verticalAlignment: "middle", autoFit: "none",
    insets: { left: 8, right: 8, top: 2, bottom: 2 },
  };
  return shape;
}

function addStep(slide, n, title, body, x, y, w, accent = C.cyan) {
  addText(slide, String(n).padStart(2, "0"), x, y, 55, 44, { size: 30, bold: true, color: accent });
  addText(slide, title, x + 62, y + 2, w - 62, 34, { size: 23, bold: true, color: C.white });
  addText(slide, body, x + 62, y + 39, w - 62, 60, { size: 17, color: C.muted, lineSpacing: 1.18 });
}

function connect(slide, a, b, fromSide = "right", toSide = "left", color = C.cyan) {
  return slide.shapes.connect(a, b, {
    kind: "straight",
    fromSide,
    toSide,
    line: { style: "solid", fill: color, width: 2 },
    tail: { type: "triangle", width: "sm", length: "sm" },
  });
}

// 1. Cover
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  const cover = await fs.readFile(path.join(workspaceDir, "assets", "presentation", "eegagent-cover.png"));
  slide.images.add({
    blob: cover,
    contentType: "image/png",
    alt: "抽象脑电波形、神经网络与智能体连接的科学视觉",
    fit: "cover",
    position: { left: 0, top: 0, width: W, height: H },
  });
  addBox(slide, 0, 0, 660, H, {
    fill: "linear(0deg, #07111F 0%, #07111F/96 65%, #07111F/20 100%)",
    line: { fill: "none", width: 0 }, radius: 0,
  });
  addLabel(slide, "GENERAL-PURPOSE EEG ANALYSIS", 72, 116, 300, C.cyan);
  addText(slide, "EEGAgent", 72, 178, 540, 92, { size: 68, bold: true, color: C.white, valign: "middle" });
  addText(slide, "面向脑电自动分析的大模型智能体框架", 76, 282, 500, 72, {
    size: 28, bold: true, color: C.cyan2, lineSpacing: 1.08,
  });
  addText(slide, "从自然语言问题出发，调度脑电工具，保留分析轨迹，并生成结构化筛查结果。", 76, 382, 460, 100, {
    size: 21, color: C.muted, lineSpacing: 1.28,
  });
  addText(slide, "项目介绍  ·  2026", 76, 636, 300, 24, { size: 15, color: "#7E91A8" });
  addNotes(slide, [
    "本页概述 EEGAgent 的定位。",
    "来源：项目 README.md；arXiv:2511.09947v2。",
    "封面视觉由 OpenAI 图像生成工具创建，仅用于概念表达。",
  ]);
}

// 2. Positioning
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  addTitle(slide, "为什么需要 EEGAgent", "01  项目定位");
  addText(slide, "传统 EEG 模型通常围绕单一任务训练。真实分析却需要连续追问、跨任务判断与多轮工具调用。", 68, 174, 1120, 82, {
    size: 28, bold: true, color: C.white, lineSpacing: 1.18,
  });

  addBox(slide, 68, 296, 468, 250, { fill: "#0C1B2D", line: { style: "solid", fill: "#29445F", width: 1 }, radius: 22 });
  addText(slide, "单任务模型", 100, 326, 370, 38, { size: 27, bold: true, color: C.orange });
  const leftBullets = addText(slide, "", 100, 386, 382, 124, { size: 20, color: C.muted });
  leftBullets.text = makeNativeBulletParagraphs([
    "输入与输出形式固定",
    "任务切换依赖独立模型",
    "难以承接连续分析上下文",
  ], { marginLeftPoints: 18, hangingPoints: 9, spaceAfterPoints: 10 });
  leftBullets.text.style = { typeface: FONT, fontSize: 20, color: C.muted, autoFit: "none", lineSpacing: 1.12, insets: { left: 0, right: 0, top: 0, bottom: 0 } };

  addBox(slide, 570, 270, 642, 302, { fill: "linear(135deg, #102942 0%, #152651 100%)", line: { style: "solid", fill: C.cyan, width: 2 }, radius: 24, shadow: "shadow-md" });
  addText(slide, "EEGAgent", 610, 309, 280, 42, { size: 30, bold: true, color: C.cyan2 });
  addText(slide, "大模型理解问题并规划下一步，工具完成信号处理与检测，运行时负责权限边界和会话状态。", 610, 374, 530, 94, {
    size: 23, color: C.text, lineSpacing: 1.22,
  });
  addText(slide, "结果以可解释的分析链路返回，而不是只给出一个分类标签。", 610, 493, 520, 52, {
    size: 19, bold: true, color: C.green,
  });
  addFooter(slide, 2, "来源：README.md；arXiv:2511.09947v2");
  addNotes(slide, [
    "核心观点来自论文摘要：现有模型多面向单一任务，EEGAgent 使用 LLM 调度多个工具完成多任务、连续推理。",
    "来源：https://arxiv.org/abs/2511.09947；项目 README.md。",
  ]);
}

// 3. Capabilities
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg2;
  addTitle(slide, "四类核心能力", "02  功能概览");
  const xs = [68, 355, 642, 929];
  const nums = ["01", "02", "03", "04"];
  const titles = ["基础感知", "片段探索", "事件检测", "报告生成"];
  const bodies = [
    "读取记录时长、采样率、原始通道与可用蒙太奇。",
    "按时间窗查看背景节律、振幅、频谱、对称性与异常筛查特征。",
    "筛查发作样活动与癫痫样放电，返回时间、导联和脑区证据。",
    "汇总当前会话中已保存的分析结果，生成结构化筛查草稿。",
  ];
  const accents = [C.cyan, C.violet, C.orange, C.green];
  for (let i = 0; i < 4; i++) {
    addText(slide, nums[i], xs[i], 190, 110, 70, { size: 48, bold: true, color: accents[i] });
    addText(slide, titles[i], xs[i], 282, 215, 44, { size: 26, bold: true, color: C.white });
    addBox(slide, xs[i], 345, 215, 3, { fill: accents[i], line: { fill: "none", width: 0 }, radius: 0 });
    addText(slide, bodies[i], xs[i], 380, 220, 156, { size: 19, color: C.muted, lineSpacing: 1.28 });
    if (i < 3) addBox(slide, xs[i] + 245, 186, 1, 370, { fill: C.line, line: { fill: "none", width: 0 }, radius: 0 });
  }
  addText(slide, "所有患者特异性结论都必须来自工具返回值，并由专业人员复核原始脑电图。", 68, 615, 1000, 34, {
    size: 18, bold: true, color: C.red,
  });
  addFooter(slide, 3, "来源：mcp_server/server.py；agent_runtime/skills/definitions");
  addNotes(slide, [
    "能力对应 MCP 工具：get_eeg_basic_information、explore_eeg_segment、detect_eeg_events、generate_eeg_report。",
    "来源：mcp_server/server.py；agent_runtime/skills/definitions/*/SKILL.md。",
  ]);
}

// 4. Architecture
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  addTitle(slide, "总体架构", "03  系统设计");
  addText(slide, "问题驱动", 68, 196, 174, 36, { size: 24, bold: true, color: C.cyan });
  addText(slide, "用户用自然语言描述任务，智能体结合上下文进行规划。", 68, 242, 170, 108, { size: 18, color: C.muted, lineSpacing: 1.25 });
  addText(slide, "工具执行", 68, 388, 174, 36, { size: 24, bold: true, color: C.violet });
  addText(slide, "预处理、特征提取与检测模型提供可调用的分析能力。", 68, 434, 170, 108, { size: 18, color: C.muted, lineSpacing: 1.25 });

  const framework = await fs.readFile(path.join(workspaceDir, "framework.png"));
  addBox(slide, 264, 174, 948, 481, { fill: "#FFFFFF", line: { style: "solid", fill: "#FFFFFF", width: 1 }, radius: 12, shadow: "shadow-lg" });
  slide.images.add({
    blob: framework,
    contentType: "image/png",
    alt: "EEGAgent 原始总体框架图",
    fit: "contain",
    position: { left: 278, top: 188, width: 920, height: 454 },
  });
  addFooter(slide, 4, "图源：项目 framework.png");
  addNotes(slide, [
    "框架图直接取自项目仓库 framework.png，未改动内容。",
    "图中展示环境信息、脑电知识库、智能体规划、工具箱、上下文和四类任务。",
  ]);
}

// 5. Runtime
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg2;
  addTitle(slide, "一次请求如何完成", "04  运行时链路");
  addText(slide, "桌面端把自然语言请求转换为受约束的脑电工具调用，并把结果写回同一会话。", 68, 158, 1080, 42, { size: 22, color: C.muted });

  const x = [68, 266, 464, 662, 860, 1058];
  const titles = ["用户问题", "Skill 路由", "权限收敛", "MCP 工具", "会话状态", "可见回答"];
  const subs = ["任务与时间窗", "关键词或语义", "allowed_tools", "分析与检测", "保存结果", "证据与限制"];
  const accents = [C.cyan, C.violet, C.orange, C.cyan, C.green, C.violet];
  const boxes = [];
  for (let i = 0; i < x.length; i++) {
    const b = addBox(slide, x[i], 270, 154, 148, { fill: "#10243A", line: { style: "solid", fill: accents[i], width: 1.5 }, radius: 18 });
    boxes.push(b);
    addText(slide, String(i + 1), x[i] + 16, 286, 28, 28, { size: 18, bold: true, color: accents[i], align: "center", valign: "middle" });
    addText(slide, titles[i], x[i] + 16, 330, 122, 35, { size: 21, bold: true, color: C.white, align: "center" });
    addText(slide, subs[i], x[i] + 16, 376, 122, 28, { size: 15, color: C.muted, align: "center" });
  }
  for (let i = 0; i < boxes.length - 1; i++) connect(slide, boxes[i], boxes[i + 1], "right", "left", "#4B718F");

  addBox(slide, 150, 492, 980, 112, { fill: "#0B1D31", line: { style: "solid", fill: C.line, width: 1 }, radius: 20 });
  addText(slide, "会话级约束", 184, 518, 170, 30, { size: 20, bold: true, color: C.cyan });
  addText(slide, "只有加载 EDF 并建立会话后，专用 EEG Skill 与工具才会启用。运行时拒绝调用当前 Skill 未授权的工具。", 350, 512, 730, 62, {
    size: 19, color: C.text, lineSpacing: 1.22, valign: "middle",
  });
  addFooter(slide, 5, "来源：README.md；agent_runtime/mcp_chat_agent.py；mcp_client.py");
  addNotes(slide, [
    "运行时先进行高精度关键词路由，再在无命中时使用本地 BGE-M3 做语义匹配。",
    "选中的 Skill 声明 allowed_tools，运行时在执行前强制检查。",
    "来源：README.md；agent_runtime/mcp_chat_agent.py；agent_runtime/mcp_client.py。",
  ]);
}

// 6. Skill/tool boundaries
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  addTitle(slide, "Skill 把分析目标变成工具边界", "05  可控调用");
  addText(slide, "每个 Skill 同时定义触发条件、处理流程、允许工具和输出规则。", 68, 160, 1060, 40, { size: 22, color: C.muted });

  const rows = [
    ["基础信息", "读取元数据", "get_eeg_basic_information", C.cyan],
    ["片段探索", "最长 60 秒", "basic_information + explore_eeg_segment", C.violet],
    ["事件检测", "最长 600 秒", "basic_information + detect_eeg_events", C.orange],
    ["报告生成", "只汇总已有结果", "basic_information + generate_eeg_report", C.green],
  ];
  addText(slide, "SKILL", 82, 232, 190, 25, { size: 14, bold: true, color: "#71839A" });
  addText(slide, "执行约束", 354, 232, 190, 25, { size: 14, bold: true, color: "#71839A" });
  addText(slide, "允许调用", 642, 232, 420, 25, { size: 14, bold: true, color: "#71839A" });
  for (let i = 0; i < rows.length; i++) {
    const y = 274 + i * 82;
    addBox(slide, 68, y, 1144, 64, { fill: i % 2 === 0 ? "#0C1D30" : "#0A192A", line: { fill: "none", width: 0 }, radius: 10 });
    addBox(slide, 68, y, 6, 64, { fill: rows[i][3], line: { fill: "none", width: 0 }, radius: 3 });
    addText(slide, rows[i][0], 90, y + 14, 220, 34, { size: 22, bold: true, color: C.white, valign: "middle" });
    addText(slide, rows[i][1], 354, y + 14, 220, 34, { size: 19, color: C.muted, valign: "middle" });
    addText(slide, rows[i][2], 642, y + 14, 520, 34, { size: 17, color: rows[i][3], valign: "middle" });
  }
  addText(slide, "一般脑电问题进入受限的 general_eeg。没有会话证据时，系统不得声称患者特异性发现。", 68, 626, 1100, 32, { size: 18, bold: true, color: C.red });
  addFooter(slide, 6, "来源：agent_runtime/skills/definitions/*/SKILL.md");
  addNotes(slide, [
    "表中约束均来自运行时 Skill 定义。",
    "片段探索窗口不超过 60 秒，事件检测窗口不超过 600 秒。",
    "报告 Skill 不会静默执行检测或探索，只整理会话已有结果。",
  ]);
}

// 7. RAG
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg2;
  addTitle(slide, "本地 RAG 如何补充脑电知识", "06  知识检索");
  addText(slide, "知识问题进入两阶段检索。记录特异性工具请求会跳过 RAG，避免把通用资料混入患者证据。", 68, 158, 1100, 60, { size: 21, color: C.muted, lineSpacing: 1.2 });

  const steps = [
    ["策略判断", "跳过、直接检索或 FAISS 探测"],
    ["双路召回", "Dense 与 Sparse 各召回 20 个子块"],
    ["三路融合", "加入 ColBERT，保留前 15 个候选"],
    ["重排过滤", "bge-reranker-v2-m3 按阈值筛选"],
    ["邻接扩展", "按父块去重并补充相邻子块"],
    ["临时注入", "仅给当前问题附加 0 至 3 段资料"],
  ];
  const widths = [1040, 920, 800, 680, 560, 440];
  const accents = [C.cyan, C.cyan, C.violet, C.orange, C.green, C.cyan2];
  for (let i = 0; i < steps.length; i++) {
    const w = widths[i];
    const x = (W - w) / 2;
    const y = 240 + i * 62;
    addBox(slide, x, y, w, 48, { fill: i % 2 === 0 ? "#102A42" : "#0D2338", line: { style: "solid", fill: `${accents[i]}/55`, width: 1 }, radius: 10 });
    addText(slide, steps[i][0], x + 18, y + 9, 150, 30, { size: 18, bold: true, color: accents[i], valign: "middle" });
    addText(slide, steps[i][1], x + 175, y + 9, w - 195, 30, { size: 17, color: C.text, align: "center", valign: "middle" });
  }
  addText(slide, "最终评分  0.2 × 归一化融合分数 + 0.8 × 重排分数", 335, 635, 610, 30, { size: 18, bold: true, color: C.white, align: "center" });
  addFooter(slide, 7, "来源：README.md；RAG/retriever.py；retrieval_policy.py");
  addNotes(slide, [
    "检索流程、召回数量、融合数量、公式与注入段数来自 README.md 的 RAG retrieval pipeline。",
    "默认阈值：模糊问题 FAISS 探测 0.35，重排过滤 0.5；可通过环境变量调整。",
    "检索文本只附加到当前用户消息，不进入系统提示，也不保留到后续轮次。",
  ]);
}

// 8. Desktop experience
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  addTitle(slide, "桌面端使用体验", "07  人机协作");
  addText(slide, "用户围绕当前 EDF 记录持续提问，界面同时呈现回答、工具状态和分析轨迹。", 68, 158, 1080, 42, { size: 22, color: C.muted });
  addStep(slide, 1, "选择 EDF 文件", "桌面端启动本地 MCP 服务并建立记录会话。", 68, 236, 500, C.cyan);
  addStep(slide, 2, "提出分析问题", "支持直接对话，也支持指定时间窗、导联和分析目标。", 68, 360, 500, C.violet);
  addStep(slide, 3, "观察执行轨迹", "界面显示 Skill 路由、候选分数、允许工具与工具调用状态。", 68, 484, 500, C.orange);

  addBox(slide, 650, 232, 562, 344, { fill: "linear(145deg, #0E2A43 0%, #151C38 100%)", line: { style: "solid", fill: C.line, width: 1 }, radius: 26, shadow: "shadow-md" });
  addText(slide, "“先看前 30 秒的背景节律，\n再检查是否存在左右不对称。”", 700, 278, 460, 102, {
    size: 28, bold: true, color: C.white, lineSpacing: 1.22,
  });
  addBox(slide, 700, 406, 424, 2, { fill: C.cyan, line: { fill: "none", width: 0 }, radius: 0 });
  addText(slide, "系统可流式返回文字，并在工具运行时更新状态。对话记录可以导出，移除数据后回到普通对话。", 700, 438, 445, 94, {
    size: 19, color: C.muted, lineSpacing: 1.26,
  });
  addFooter(slide, 8, "来源：desktop_app.py；trajectory_view.py；README.md");
  addNotes(slide, [
    "桌面端由 PySide6 构建。",
    "源代码提供 EDF 文件选择、流式消息、工具状态、轨迹页、对话清空与导出等功能。",
    "来源：desktop_app.py；trajectory_view.py。",
  ]);
}

// 9. Evaluation and boundaries
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg2;
  addTitle(slide, "评估方向与使用边界", "08  证据与限制");

  addText(slide, "仓库中的评估入口", 68, 174, 470, 38, { size: 25, bold: true, color: C.white });
  const evals = [
    ["MDD", "抑郁相关 EEG 任务", C.violet],
    ["Sleep", "睡眠分期任务", C.cyan],
    ["TUSL", "事件检测相关任务", C.orange],
  ];
  for (let i = 0; i < evals.length; i++) {
    const y = 236 + i * 96;
    addText(slide, evals[i][0], 68, y, 108, 40, { size: 26, bold: true, color: evals[i][2] });
    addText(slide, evals[i][1], 188, y + 3, 350, 36, { size: 20, color: C.text });
    addBox(slide, 68, y + 56, 470, 1, { fill: C.line, line: { fill: "none", width: 0 }, radius: 0 });
  }
  addText(slide, "论文摘要说明系统在公开数据集上完成能力评估。仓库 README 未提供可直接引用的量化结果，本介绍因此不添加性能数字。", 68, 540, 480, 108, { size: 17, color: C.muted, lineSpacing: 1.22 });

  addBox(slide, 624, 174, 588, 390, { fill: "#0C1B2D", line: { style: "solid", fill: C.red, width: 1.5 }, radius: 24 });
  addText(slide, "临床使用边界", 670, 214, 450, 44, { size: 29, bold: true, color: C.red });
  const boundary = addText(slide, "", 670, 286, 460, 220, { size: 20, color: C.text });
  boundary.text = makeNativeBulletParagraphs([
    "结果定位为自动化筛查发现或报告草稿",
    "不得把统计探索直接解释为诊断结论",
    "事件、导联、脑区与置信度必须来自工具输出",
    "专业人员仍需复核原始 EEG 及临床背景",
  ], { marginLeftPoints: 18, hangingPoints: 9, spaceAfterPoints: 12 });
  boundary.text.style = { typeface: FONT, fontSize: 20, color: C.text, autoFit: "none", lineSpacing: 1.16, insets: { left: 0, right: 0, top: 0, bottom: 0 } };
  addFooter(slide, 9, "来源：README.md；MDD_eval.py；Sleep_eval.py；TUSL_eval.py；运行时 Skill");
  addNotes(slide, [
    "论文摘要只说明在公开数据集上进行了评估，没有在 README 中提供具体数值。",
    "MDD、Sleep、TUSL 三个评估方向来自仓库顶层评估脚本。",
    "临床边界来自 detection、exploration、reporting 与 general_eeg Skill 的输出规则。",
  ]);
}

// 10. Quick start and close
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  const cover = await fs.readFile(path.join(workspaceDir, "assets", "presentation", "eegagent-cover.png"));
  slide.images.add({
    blob: cover,
    contentType: "image/png",
    alt: "抽象脑电与智能体背景",
    fit: "cover",
    position: { left: 0, top: 0, width: W, height: H },
    crop: { left: 0.13, top: 0, right: 0, bottom: 0 },
  });
  addBox(slide, 0, 0, 780, H, { fill: "linear(0deg, #07111F 0%, #07111F/97 75%, #07111F/40 100%)", line: { fill: "none", width: 0 }, radius: 0 });
  addText(slide, "快速启动", 72, 76, 500, 62, { size: 44, bold: true, color: C.white });
  addText(slide, "1", 76, 188, 36, 34, { size: 24, bold: true, color: C.cyan });
  addText(slide, "安装 requirements.txt 中的依赖", 130, 188, 500, 34, { size: 21, color: C.text });
  addText(slide, "2", 76, 260, 36, 34, { size: 24, bold: true, color: C.violet });
  addText(slide, "配置 DEEPSEEK_API_KEY", 130, 260, 500, 34, { size: 21, color: C.text });
  addText(slide, "3", 76, 332, 36, 34, { size: 24, bold: true, color: C.orange });
  addText(slide, "运行 desktop_app.py", 130, 332, 500, 34, { size: 21, color: C.text });
  addText(slide, "4", 76, 404, 36, 34, { size: 24, bold: true, color: C.green });
  addText(slide, "选择 EDF 文件并开始提问", 130, 404, 500, 34, { size: 21, color: C.text });

  addText(slide, "EEGAgent 的价值在于把多种 EEG 能力组织成可追踪、可约束、可扩展的分析过程。", 76, 512, 610, 94, {
    size: 27, bold: true, color: C.cyan2, lineSpacing: 1.2,
  });
  addText(slide, "论文：arXiv:2511.09947v2", 76, 645, 420, 24, { size: 15, color: "#7E91A8" });
  addNotes(slide, [
    "快速启动命令与环境变量名称来自项目 README.md 和 main.py。",
    "项目论文：https://arxiv.org/abs/2511.09947。",
    "图像生成提示词：深色科学编辑风格，脑电波形与神经网络连接，左侧留出标题空间，无文字与标识。",
  ]);
}

const stagingDir = path.join(workspaceDir, ".codex-finalizer");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "EEGAgent项目介绍_中文_candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

// Save per-slide PNGs for visual review before final delivery.
for (let i = 0; i < presentation.slides.items.length; i++) {
  const slide = presentation.slides.items[i];
  const preview = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(TMP_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const requirements = {
  explicitTotalSlideCount: 10,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
};
const fontPolicy = { basis: "design", families: [FONT] };
const result = await finalizePresentation({
  ...requirements,
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
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "EEGAgent项目介绍_中文_v2.validation.json"),
});

console.log(JSON.stringify({ finalPath: FINAL_PPTX, slideCount: presentation.slides.items.length, validation: result }, null, 2));
