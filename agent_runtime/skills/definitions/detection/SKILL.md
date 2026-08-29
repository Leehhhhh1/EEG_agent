---
name: detection
description: 筛查当前脑电记录中的发作样活动、癫痫样放电及事件位置。
priority: 220
requires_session: true
trigger_keywords:
  - seizure
  - epilep
  - discharge
  - 发作
  - 癫痫
  - 放电
  - 定位
allowed_tools:
  - get_eeg_basic_information
  - detect_eeg_events
---
# 脑电事件检测

## 处理流程

1. 从用户请求中确定分析的开始时间和结束时间。
2. 如果记录时长未知，或者无法确定有效的结束时间，先调用 `get_eeg_basic_information`。
3. 除非用户明确要求 `sensitive` 或 `specific`，否则使用 `balanced` 灵敏度。
4. 仅对有效且不超过600秒的时间窗口调用 `detect_eeg_events`。
5. 只能根据工具返回的事件和证据回答。

## 输出规则

- 将结果表述为自动筛查发现，不得作为诊断结论。
- 不得编造事件时间、导联、脑区、置信度或事件类型。
- 必须说明需要由具备资质的专业人员复核原始脑电图。
