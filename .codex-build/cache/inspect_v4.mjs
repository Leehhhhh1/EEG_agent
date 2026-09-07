import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const source = "F:/codex/EEGAgent-main/output/presentations/EEGAgent智能体技术架构_面试版_v4_轨迹增强.pptx";
const outDir = "F:/codex/EEGAgent-main/.codex-build/cache/inspect";
await fs.mkdir(outDir, { recursive: true });
const presentation = await PresentationFile.importPptx(await FileBlob.load(source));
const snapshot = await presentation.inspect({
  kind: "slide,textbox,shape,notes,layout",
  include: "id,slide,name,title,text,textPreview,bbox,bboxUnit",
  maxChars: 30000,
});
await fs.writeFile(`${outDir}/inspect.ndjson`, snapshot.ndjson);
const montage = await presentation.export({ format: "webp", montage: true, scale: 0.8 });
await fs.writeFile(`${outDir}/montage.webp`, new Uint8Array(await montage.arrayBuffer()));
for (const index of [5, 6, 7, 8]) {
  const preview = await presentation.slides.getItem(index).export({ format: "png", scale: 1.5 });
  await fs.writeFile(`${outDir}/slide-${index + 1}.png`, new Uint8Array(await preview.arrayBuffer()));
}
console.log(JSON.stringify({ slides: presentation.slides.items.length }));
