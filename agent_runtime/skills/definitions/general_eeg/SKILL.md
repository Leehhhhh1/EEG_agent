---
name: general_eeg
description: 处理未明确匹配某个专用运行时 Skill 的一般脑电知识问题或模糊请求。
priority: 0
requires_session: false
trigger_keywords: []
routing_examples:
  - 什么是癫痫样放电
  - 解释一下脑电背景活动
  - ACNS如何定义周期性放电
  - 脑电报告通常包括哪些内容
  - 请解释这个脑电术语
allowed_tools:
  - get_eeg_basic_information
  - explore_eeg_segment
  - detect_eeg_events
  - generate_eeg_report
---
# 一般脑电辅助

## 处理流程

1. 根据当前对话和检索到的参考资料回答一般脑电知识问题。
2. 如果请求需要分析当前记录，只使用完成任务所需的最少脑电工具。
3. 无法从请求中安全确定分析参数时，应要求用户提供时间范围。
4. 没有当前会话的 MCP Tool 结果时，不得声称存在患者特异性发现。

## 输出规则

- 明确区分一般知识与当前记录的分析证据。
- 自动化分析仅用于筛查，不能替代具备资质的临床专业人员复核。
