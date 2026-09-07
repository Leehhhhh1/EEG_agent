import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "F:/codex/EEGAgent-main";
const sourcePath = `${workspaceDir}/.codex-finalizer/EEGAgent_v4_trajectory_candidate.pptx`;
const skillDir = "C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations";
const buildDir = `${workspaceDir}/.codex-build/cache/build`;
const stagingDir = `${workspaceDir}/.codex-finalizer`;
const finalPath = `${workspaceDir}/output/presentations/EEGAgent智能体技术架构_面试版_v5_轨迹与缓存增强.pptx`;
const pythonExecutable = "C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";
process.env.RUNTIME_NODE_MODULES = "C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(stagingDir, { recursive: true });

const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePath));
const cacheSlide = presentation.slides.getItem(5).duplicate();
cacheSlide.moveTo(7);

function shapeByName(slide, name) {
  const shape = slide.shapes.items.find((item) => item.name === name);
  if (!shape) throw new Error(`Missing shape ${name}`);
  return shape;
}

function replace(slide, name, oldText, newText) {
  shapeByName(slide, name).text.replace(oldText, newText);
}

const replacements = [
  ["矩形 33", "规划执行", "缓存复用"],
  ["矩形 1", "受约束的流式 ReAct 循环", "前缀缓存复用与失效边界"],
  ["矩形 3", "模型可以连续调用工具，但运行时掌握会话、授权、终止条件和错误处理。", "缓存复用的是连续相同的输入 token 前缀，模型新生成的输出 token 不参与复用。"],
  ["矩形 5", "组织上下文", "系统前缀"],
  ["矩形 6", "系统指令、会话摘要、本轮 Skill、用户提问、RAG", "身份指令\n固定工具 schema"],
  ["矩形 8", "模型决策", "会话摘要"],
  ["矩形 9", "流式输出文本、推理内容与函数参数", "保持稳定\n压缩时更新"],
  ["矩形 11", "解析 tool_calls", "历史消息"],
  ["矩形 12", "按 index 拼接增量参数", "user、assistant\n与 tool"],
  ["矩形 14", "运行时校验", "当前 user"],
  ["矩形 15", "检查 session 与 allowed_tools", "Skill、问题\n临时 RAG"],
  ["矩形 17", "MCP 执行", "assistant"],
  ["矩形 18", "自动注入 session_id", "tool_calls\n或最终回答"],
  ["矩形 20", "写回结果", "tool 结果"],
  ["矩形 21", "tool 消息进入下一轮模型请求", "追加结果\n再次请求模型"],
  ["矩形 28", "继续循环", "轮内复用"],
  ["矩形 29", "存在 tool_calls 时追加 assistant 与 tool 消息，再请求模型。没有工具调用时输出最终答案。", "同一 ReAct 循环只在尾部追加 assistant 与 tool，上一轮输入可成为下一次请求的完整缓存前缀。"],
  ["矩形 30", "硬上限：8 轮工具调用。达到上限后移除工具 schema，要求模型基于已有证据输出回答。", "跨轮失效：临时 RAG 清理、摘要压缩或工具 schema 变化会改变前缀。观测：cache hit / miss、common prefix tokens。"],
  ["矩形 32", "05", "07"],
];
for (const [name, oldText, newText] of replacements) {
  replace(cacheSlide, name, oldText, newText);
}

shapeByName(cacheSlide, "圆角矩形 27").position = { left: 168, top: 468, width: 944, height: 146 };
shapeByName(cacheSlide, "矩形 28").position = { left: 196, top: 488, width: 100, height: 30 };
shapeByName(cacheSlide, "矩形 29").position = { left: 310, top: 482, width: 770, height: 54 };
shapeByName(cacheSlide, "矩形 30").position = { left: 196, top: 548, width: 884, height: 46 };

cacheSlide.speakerNotes.textFrame.setText(
  "提示词缓存复用的是输入 token，不复用模型本轮新生成的输出 token。\n"
  + "命中条件是请求开头存在连续且完全一致的 token 前缀，因此 System Prompt、工具 schema、摘要和消息顺序应尽量稳定。\n"
  + "在同一 ReAct 循环中，运行时仅在尾部追加 assistant.tool_calls 与 tool 结果，下一次模型请求可复用上一轮的完整输入前缀。\n"
  + "本轮结束后当前实现会清理临时 RAG；下一轮请求会在旧 user 消息附近发生前缀分叉。摘要压缩、工具 schema 或顺序变化也会造成失效。\n"
  + "运行时记录服务商返回的 prompt_cache_hit_tokens 与 prompt_cache_miss_tokens，并计算 prompt fingerprint 和 common_prefix_tokens 用于诊断。"
);

const toolSlide = presentation.slides.getItem(8);
const trajectorySlide = presentation.slides.getItem(9);
const exampleSlide = presentation.slides.getItem(10);
const summarySlide = presentation.slides.getItem(11);
replace(toolSlide, "矩形 23", "07", "08");
replace(trajectorySlide, "矩形 36", "08", "09");
replace(exampleSlide, "矩形 36", "09", "10");
replace(summarySlide, "矩形 12", "10", "11");

const candidatePath = `${stagingDir}/EEGAgent_v5_cache_candidate.pptx`;
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const newPreview = await cacheSlide.export({ format: "png", scale: 2 });
await fs.writeFile(`${buildDir}/cache-reuse-slide.png`, new Uint8Array(await newPreview.arrayBuffer()));
const montage = await presentation.export({ format: "webp", montage: true, scale: 0.8 });
await fs.writeFile(`${buildDir}/montage.webp`, new Uint8Array(await montage.arrayBuffer()));

const { finalizePresentation } = await import(pathToFileURL(
  path.join(skillDir, "container_tools/artifact_tool_utils.mjs"),
).href);

const requirements = {
  explicitTotalSlideCount: 12,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
};
const fontPolicy = {
  basis: "reference",
  families: ["Noto Sans SC", "Calibri"],
  referencePath: sourcePath,
  referenceSha256: "a0930aefc0c283e8dbf196e2f26a0237a04aa7e273d6c15bcb0f89fbf5d97bb9",
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
  receiptPath: `${stagingDir}/EEGAgent智能体技术架构_面试版_v5_轨迹与缓存增强.validation.json`,
});

const check = await presentation.inspect({
  kind: "slide,textbox,notes",
  search: "缓存复用|完整缓存前缀|prompt_cache_hit_tokens|common_prefix_tokens",
  maxChars: 12000,
});
await fs.writeFile(`${buildDir}/verification.ndjson`, check.ndjson);
console.log(JSON.stringify({ finalPath, slides: presentation.slides.items.length, finalize: result }));
