---
name: exploration
description: 探索指定脑电片段的背景节律、振幅、频谱功率、对称性或筛查特征。
priority: 180
requires_session: true
trigger_keywords:
  - analyze background rhythm
  - check symmetry
  - analyze amplitude
  - frequency spectrum
  - 分析背景节律
  - 背景节律
  - 检查左右对称
  - 左右是否对称
  - 分析振幅
  - 查看频谱
routing_examples:
  - 分析前30秒的背景节律
  - 看一下左右两侧脑电是否对称
  - 检查指定时间段的振幅变化
  - 分析这段脑电的主要频率和频谱功率
  - 探索前一分钟的脑电背景特征
  - 查看这个片段是否存在明显的左右差异
allowed_tools:
  - get_eeg_basic_information
  - explore_eeg_segment
---
# 脑电片段探索

## 处理流程

1. 确定用户要求的开始时间、结束时间、导联和分析重点。
2. 如果有效记录范围未知，先调用 `get_eeg_basic_information`。
3. 分析片段不得超过60秒。
4. 将用户请求映射为 `background_rhythm`、`amplitude`、`symmetry`、`abnormality_screen` 或 `overview`。
5. 调用 `explore_eeg_segment`，并且只能解释工具返回的测量结果和筛查提示。

## 输出规则

- 明确区分统计探索与事件诊断。
- 不得编造波形形态或临床结论。
- 必须说明自动筛查发现需要结合原始脑电图复核。
