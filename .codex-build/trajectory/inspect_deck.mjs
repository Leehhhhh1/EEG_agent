import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const source = "F:/codex/EEGAgent-main/output/presentations/EEGAgent智能体技术架构_面试版_v3.pptx";
const outDir = "F:/codex/EEGAgent-main/.codex-build/trajectory/inspect";
await fs.mkdir(outDir, { recursive: true });

const presentation = await PresentationFile.importPptx(await FileBlob.load(source));
const snapshot = await presentation.inspect({
  kind: "deck,slide,textbox,shape,image,table,chart,notes,layout",
  include: "id,slide,name,title,text,textPreview,textChars,textLines,bbox,bboxUnit,isPlaceholder,placeholders",
  maxChars: 40000,
});
await fs.writeFile(`${outDir}/inspect.ndjson`, snapshot.ndjson);

const montage = await presentation.export({ format: "webp", montage: true, scale: 0.8 });
await fs.writeFile(`${outDir}/montage.webp`, new Uint8Array(await montage.arrayBuffer()));

for (let index = 0; index < presentation.slides.items.length; index += 1) {
  const slide = presentation.slides.getItem(index);
  const preview = await slide.export({ format: "png", scale: 1.5 });
  await fs.writeFile(`${outDir}/slide-${index + 1}.png`, new Uint8Array(await preview.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(`${outDir}/slide-${index + 1}.layout.json`, await layout.text());
}

console.log(JSON.stringify({
  slides: presentation.slides.items.length,
  slideSize: presentation.slideSize,
  masters: presentation.masters.items.length,
  layouts: presentation.layouts.items.length,
}));
