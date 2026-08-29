---
name: exploration
description: 探索指定脑电片段的背景节律、振幅、频谱功率、对称性或筛查特征。
priority: 180
requires_session: true
trigger_keywords:
  - rhythm
  - amplitude
  - symmetry
  - background
  - frequency
  - 频率
  - 振幅
  - 节律
  - 对称
  - 背景
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
