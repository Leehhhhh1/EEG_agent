---
name: basic_information
description: 读取当前脑电记录的基础元数据、时长、采样率、通道和蒙太奇信息。
priority: 120
requires_session: true
trigger_keywords:
  - 基本信息
  - 记录信息
  - 时长
  - 采样率
  - 通道数
  - 蒙太奇
  - duration
  - sampling rate
  - channel count
  - montage
routing_examples:
  - 这份脑电记录有多长
  - 当前记录的采样率是多少
  - 查看原始通道数量和通道名称
  - 当前脑电可以组成哪些双极导联
  - 显示这份记录的基础元数据
allowed_tools:
  - get_eeg_basic_information
---
# 脑电基础信息

## 处理流程

1. 对当前脑电会话调用 `get_eeg_basic_information`。
2. 只能根据工具返回的元数据回答。
3. 明确区分原始通道与可用双极导联。

## 输出规则

- 不得推断工具未提供的患者属性。
- 本 Skill 只读取记录元数据，不得描述脑电波形发现。
