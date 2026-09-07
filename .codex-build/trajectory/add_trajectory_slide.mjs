import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "F:/codex/EEGAgent-main";
const sourcePath = `${workspaceDir}/output/presentations/EEGAgent智能体技术架构_面试版_v3.pptx`;
const skillDir = "C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations";
const buildDir = `${workspaceDir}/.codex-build/trajectory/build`;
const stagingDir = `${workspaceDir}/.codex-finalizer`;
const finalPath = `${workspaceDir}/output/presentations/EEGAgent智能体技术架构_面试版_v4_轨迹增强.pptx`;
const pythonExecutable = "C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";
process.env.RUNTIME_NODE_MODULES = "C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(stagingDir, { recursive: true });

const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePath));
const exampleSlide = presentation.slides.getItem(8);
const summarySlide = presentation.slides.getItem(9);
const trajectorySlide = exampleSlide.duplicate();
trajectorySlide.moveTo(8);

function shapeByName(slide, name) {
  const shape = slide.shapes.items.find((item) => item.name === name);
  if (!shape) throw new Error(`Missing shape ${name}`);
  return shape;
}

function replace(slide, name, oldText, newText) {
  const shape = shapeByName(slide, name);
  shape.text.replace(oldText, newText);
}

const replacements = [
  ["矩形 37", "端到端示例", "执行轨迹"],
  ["矩形 1", "一次事件筛查请求的执行轨迹", "事件模型、实时更新与指标聚合"],
  ["矩形 3", "示例问题：前一分钟是否存在癫痫样放电？如果有，出现在哪些导联？", "每个阶段发出带稳定 ID 的事件，桌面端实时更新状态并聚合性能指标。"],
  ["矩形 7", "规划", "输入"],
  ["矩形 8", "关键词命中 detection，加载 allowed_tools", "user/message：记录原始问题、turn 与起止时间"],
  ["矩形 11", "检索", "路由"],
  ["矩形 12", "识别为记录特异性请求，本轮跳过 RAG", "routing/decision：Skill、路由来源、关键词命中与置信度"],
  ["矩形 15", "模型", "检索"],
  ["矩形 16", "记录时长未知时，先请求 get_eeg_basic_information", "rag/retrieval：检索策略、参考片段、来源与耗时"],
  ["矩形 19", "执行", "模型"],
  ["矩形 20", "运行时注入 session_id，经 MCP 获取记录范围", "model/request：状态、TTFT、token 与缓存统计"],
  ["矩形 23", "模型", "工具"],
  ["矩形 24", "构造 detect_eeg_events，窗口 0 至 60 秒", "tool/call：工具名、参数、执行状态、输入输出与耗时"],
  ["矩形 27", "记忆", "更新"],
  ["矩形 28", "把事件数量、时间、导联、脑区和置信度写入摘要", "同一事件 ID 从 running 更新为 complete、error 或 cancelled"],
  ["矩形 31", "回答", "收束"],
  ["矩形 32", "只引用工具证据，并说明需要专业人员复核", "turn/end：模型轮数、工具轮数、上下文占用与总耗时"],
  ["矩形 34", "可观测性：路由分数、工具耗时、上下文占用、缓存命中率", "UI：时间轴、事件表、输入输出详情；摘要栏聚合耗时、TTFT、tokens 与缓存"],
];
for (const [name, oldText, newText] of replacements) {
  replace(trajectorySlide, name, oldText, newText);
}

shapeByName(trajectorySlide, "圆角矩形 33").position = {
  left: 540, top: 604, width: 672, height: 56,
};
shapeByName(trajectorySlide, "矩形 34").position = {
  left: 560, top: 610, width: 630, height: 40,
};

trajectorySlide.speakerNotes.textFrame.setText(
  "轨迹事件由 Agent Runtime 发出，通过稳定事件 ID 增量更新状态。\n"
  + "事件类型覆盖 user/message、routing/decision、rag/retrieval、model/request、tool/call 和 turn/end。\n"
  + "桌面端按轮次展示时间轴、可搜索事件表和输入输出详情，并聚合 LLM 耗时、工具耗时、平均 TTFT、生成速度、缓存命中率、token 与上下文占用。\n"
  + "轨迹详情通过 _trace_text 做长度限制，不写入模型消息历史。"
);

replace(exampleSlide, "矩形 36", "08", "09");
replace(summarySlide, "矩形 12", "9", "10");

const candidatePath = `${stagingDir}/EEGAgent_v4_trajectory_candidate.pptx`;
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const newPreview = await trajectorySlide.export({ format: "png", scale: 2 });
await fs.writeFile(`${buildDir}/trajectory-slide.png`, new Uint8Array(await newPreview.arrayBuffer()));
const montage = await presentation.export({ format: "webp", montage: true, scale: 0.8 });
await fs.writeFile(`${buildDir}/montage.webp`, new Uint8Array(await montage.arrayBuffer()));

const { finalizePresentation } = await import(pathToFileURL(
  path.join(skillDir, "container_tools/artifact_tool_utils.mjs"),
).href);

const requirements = {
  explicitTotalSlideCount: 11,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
};
const fontPolicy = {
  basis: "reference",
  families: ["Noto Sans SC", "Calibri"],
  referencePath: sourcePath,
  referenceSha256: "ec53f2d6cbe3fbffe218dd66e052355ccced0810d97d1b0411c75061683446af",
};

const result = await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable,
  integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
  ],
  requiredNativeTableOwnerSlides: [],
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: `${stagingDir}/EEGAgent智能体技术架构_面试版_v4_轨迹增强.validation.json`,
});

const check = await presentation.inspect({
  kind: "slide,textbox,notes",
  search: "执行轨迹|事件模型|turn/end|UI：",
  maxChars: 12000,
});
await fs.writeFile(`${buildDir}/verification.ndjson`, check.ndjson);
console.log(JSON.stringify({ finalPath, slides: presentation.slides.items.length, finalize: result }));
