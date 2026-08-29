---
name: reporting
description: 根据当前会话中已经保存的分析结果生成结构化脑电筛查报告。
priority: 260
requires_session: true
trigger_keywords:
  - report
  - summary
  - 报告
  - 总结
allowed_tools:
  - get_eeg_basic_information
  - generate_eeg_report
---
# 脑电筛查报告

## 处理流程

1. 仅在需要记录元数据时调用 `get_eeg_basic_information`。
2. 调用 `generate_eeg_report` 汇总当前会话中已经保存的分析结果。
3. 不得静默执行事件检测或片段探索，因为这些工具不属于当前 Skill。
4. 如果用户要求的报告部分没有既有分析结果，应明确说明。

## 输出规则

- 将报告明确表述为自动化筛查草稿。
- 保留报告工具返回的局限性说明和诊断状态。
- 不得添加生成报告中不存在的发现。
