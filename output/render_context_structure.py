from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1920, 1080
OUT_DIR = Path(__file__).resolve().parent
REGULAR = r"C:\Windows\Fonts\msyh.ttc"
BOLD = r"C:\Windows\Fonts\msyhbd.ttc"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(BOLD if bold else REGULAR, size)


image = Image.new("RGB", (WIDTH, HEIGHT), "#F7F9FC")
draw = ImageDraw.Draw(image)


def rounded(box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def centered(text, box, text_font, fill="#10233F"):
    x1, y1, x2, y2 = box
    bounds = draw.textbbox((0, 0), text, font=text_font)
    tw, th = bounds[2] - bounds[0], bounds[3] - bounds[1]
    draw.text(((x1 + x2 - tw) / 2, (y1 + y2 - th) / 2 - bounds[1]), text, font=text_font, fill=fill)


def arrow_down(x, y1, y2, color="#94A3B8"):
    draw.line((x, y1, x, y2 - 10), fill=color, width=4)
    draw.polygon([(x - 9, y2 - 13), (x + 9, y2 - 13), (x, y2)], fill=color)


def arrow_right(x1, y, x2, color="#64748B"):
    draw.line((x1, y, x2 - 12, y), fill=color, width=4)
    draw.polygon([(x2 - 15, y - 9), (x2 - 15, y + 9), (x2, y)], fill=color)


def role_badge(x, y, label, fill):
    rounded((x, y, x + 142, y + 38), 19, fill)
    centered(label, (x, y, x + 142, y + 38), font(19, True), "#FFFFFF")


draw.text((92, 46), "EEGAgent Context 上下文结构", font=font(48, True), fill="#10233F")
draw.text((94, 108), "消息角色、动态知识注入与 ReAct 工具循环", font=font(24), fill="#64748B")

main_x1, main_x2 = 92, 1570

# System prompt
rounded((main_x1, 162, main_x2, 268), 22, "#EAF3FF", "#B8D5FF", 2)
role_badge(118, 184, "role: system", "#2563EB")
draw.text((292, 180), "System Prompt", font=font(30, True), fill="#153B70")
draw.text((292, 222), "身份设定 · 医疗安全约束 · 工具调用原则 · 历史 Skill 隔离规则", font=font(22), fill="#3F5F86")

arrow_down(831, 268, 292)

# Session summary
rounded((main_x1, 292, main_x2, 398), 22, "#F1ECFF", "#D5C7FF", 2)
role_badge(118, 314, "role: system", "#7C3AED")
draw.text((292, 310), "Session Summary", font=font(30, True), fill="#4C2588")
draw.text((292, 352), "EEG 记录信息 · 患者信息 · 已完成分析 · 关键发现 · 压缩后的对话摘要", font=font(22), fill="#6E4D9B")

arrow_down(831, 398, 422)

# History
rounded((main_x1, 422, main_x2, 554), 22, "#FFFFFF", "#D9E1EC", 2)
draw.text((118, 444), "历史消息", font=font(28, True), fill="#26384F")
draw.text((118, 488), "按时间顺序进入下一轮上下文", font=font(20), fill="#6B7B91")

history_nodes = [
    (410, "user", "#0F766E"),
    (650, "assistant", "#0369A1"),
    (920, "assistant.tool_calls", "#B45309"),
    (1270, "tool", "#9A3412"),
]
for x, label, color in history_nodes:
    rounded((x, 458, x + 205, 520), 15, "#F8FAFC", color, 2)
    centered(label, (x, 458, x + 205, 520), font(21, True), color)
for (x1, _, _), (x2, _, _) in zip(history_nodes, history_nodes[1:]):
    arrow_right(x1 + 205, 489, x2)

arrow_down(831, 554, 578)

# Current user
rounded((main_x1, 578, main_x2, 754), 22, "#EAFBF5", "#9BDCC6", 2)
role_badge(118, 600, "role: user", "#059669")
draw.text((292, 598), "当前 User Message", font=font(30, True), fill="#155B49")
draw.text((292, 638), "三个部分都位于同一条 user.content 中", font=font(20), fill="#457667")

current_nodes = [
    (118, 676, 486, "Skill 指令", "绑定 EEG 时存在", "#D8F3E8", "#047857"),
    (514, 676, 910, "用户原始问题", "<user_request>", "#DFF6EF", "#047857"),
    (938, 676, 1544, "临时 RAG 知识片段", "本轮有效 · 本轮结束后移除", "#FFF3D6", "#B45309"),
]
for x1, y1, x2, title, sub, fill, color in current_nodes:
    rounded((x1, y1, x2, 735), 14, fill, color, 2)
    draw.text((x1 + 18, y1 + 9), title, font=font(22, True), fill=color)
    draw.text((x1 + 18, y1 + 38), sub, font=font(17), fill=color)
arrow_right(486, 705, 514, "#5DAE93")
arrow_right(910, 705, 938, "#5DAE93")

arrow_down(831, 754, 778)

# ReAct loop
rounded((main_x1, 778, main_x2, 948), 22, "#FFF8EC", "#F0C981", 2)
draw.text((118, 800), "本轮 ReAct 输出与工具循环", font=font(28, True), fill="#75430C")
draw.text((118, 842), "RAG 在整个本轮循环中保留", font=font(20), fill="#91642E")

loop_nodes = [
    (410, 826, 622, "DeepSeek", "推理", "#FFFDF8", "#92400E"),
    (680, 826, 960, "assistant.tool_calls", "发起工具调用", "#FFFDF8", "#B45309"),
    (1018, 826, 1198, "tool", "返回结果", "#FFFDF8", "#C2410C"),
    (1256, 826, 1538, "assistant", "最终回答", "#FFFDF8", "#0369A1"),
]
for x1, y1, x2, title, sub, fill, color in loop_nodes:
    rounded((x1, y1, x2, 916), 14, fill, color, 2)
    centered(title, (x1, y1 + 8, x2, y1 + 49), font(20, True), color)
    centered(sub, (x1, y1 + 48, x2, y2 := 910), font(17), "#64748B")
for (x1, _, x2, *_), (nx1, *_rest) in zip(loop_nodes, loop_nodes[1:]):
    arrow_right(x2, 871, nx1, "#D09A45")

# Tool return loop
draw.arc((600, 894, 1215, 970), 0, 180, fill="#D09A45", width=3)
draw.polygon([(617, 926), (633, 916), (633, 936)], fill="#D09A45")
draw.text((750, 936), "工具结果返回模型，继续推理", font=font(17), fill="#91642E")

# Context ruler
rounded((1610, 162, 1830, 948), 22, "#FFFFFF", "#D9E1EC", 2)
centered("Context 预算", (1620, 182, 1820, 228), font(25, True), "#26384F")
centered("42k tokens", (1620, 230, 1820, 278), font(28, True), "#2563EB")

bar_x1, bar_x2, bar_top, bar_bottom = 1680, 1740, 318, 854
rounded((bar_x1, bar_top, bar_x2, bar_bottom), 22, "#EDF2F7")
draw.rounded_rectangle((bar_x1, 438, bar_x2, bar_bottom), radius=22, fill="#BFE3D5")
draw.line((1650, 420, 1770, 420), fill="#F59E0B", width=4)
draw.line((1650, 552, 1770, 552), fill="#7C3AED", width=4)
draw.text((1765, 400), "34.4k", font=font(18, True), fill="#B45309")
draw.text((1765, 424), "压缩触发", font=font(16), fill="#91642E")
draw.text((1765, 534), "23.7k", font=font(18, True), fill="#6D28D9")
draw.text((1765, 558), "压缩目标", font=font(16), fill="#7253A5")
draw.text((1646, 878), "0", font=font(18), fill="#64748B")

draw.text((92, 1000), "角色边界：Skill 与 RAG 属于当前 user；工具调用属于 assistant；工具结果属于 tool。", font=font(22, True), fill="#334155")
draw.text((92, 1037), "当前实现：RAG 仅在本轮请求及其工具循环内存在，结束后从历史 user 消息中移除。", font=font(19), fill="#64748B")

image.save(OUT_DIR / "eegagent-context-structure.jpg", quality=96, subsampling=0, optimize=True)
image.save(OUT_DIR / "eegagent-context-structure.png", optimize=True)
