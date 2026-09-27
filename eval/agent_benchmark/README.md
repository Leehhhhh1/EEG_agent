# EEGAgent 分层流程评测集

本目录包含 30 个 JSONL 评测用例：L1、L2、L3 各 10 个。每个 JSONL 文件一行一个完整用例。

## 文件

- `l1_atomic.jsonl`：显式、短流程的原子能力。
- `l2_sequential.jsonl`：工具依赖、参数更正和失败处理。
- `l3_multiturn.jsonl`：同一 EEG 会话中的跨轮路由、状态复用和报告整合。
- `tcr_two_eeg.jsonl`：两份 EEG 各执行一次 L1、L2、L3 的六用例成对实验。

## 执行约定

1. `session.mode=loaded` 时，在首轮前加载 `session.file`；`none` 时不建立 EEG 会话。
2. `setup_tool_calls` 由评测器直接调用 MCP 工具，用于确定性地准备已有分析结果，不计入 Agent 得分。
3. 同一用例的 `turns` 必须复用同一个 Agent 和 EEG 会话；不同用例必须使用全新 Agent 和会话。
4. 每个用例建议独立运行 5 次。工具异常、模型超时和无法解析的输出都计为失败，不得跳过。
5. `expected_arguments` 使用 `resolved_subset`：先应用 MCP 函数默认值，再比较列出的参数；自动注入的 `session_id` 不参与比较。
6. `expected_tool_order` 是评分顺序。`must_have_tools` 必须出现，`forbidden_tools` 不得出现，`optional_tools` 出现与否不影响完成判定，但会影响工具效率。
7. `acceptable_skills` 表示多个路由均可接受；JSON 的 `null` 表示未选择运行时 Skill。
8. `expected_outcome=reject_invalid_request_or_surface_tool_error` 表示模型可以在调用前拒绝非法参数，也可以调用工具并如实转述错误，但不得编造分析结果。

## 回答检查项

- `uses_only_tool_evidence`：患者特异性结论只能来自本轮或既有工具结果。
- `no_unsupported_patient_attributes`：不得推断工具未提供的患者属性。
- `distinguishes_raw_and_bipolar_channels`：明确区分原始通道与可用双极导联。
- `exploration_not_event_diagnosis`：统计探索不得表述成事件诊断。
- `screening_not_diagnosis`：检测结果必须表述为自动筛查，而不是临床确诊。
- `professional_review`：说明需由具备资质的专业人员复核原始脑电。
- `automated_screening_draft`：报告必须明确为自动化筛查草稿。
- `uses_prior_findings`：正确使用同一会话内已保存的探索或检测结果。
- `no_fabricated_findings`：不得添加工具结果中不存在的发现。
- `no_repeat_analysis`：用户要求只汇总时不得重新执行探索或检测。
- `general_knowledge_only`：仅回答一般知识，不声称当前记录存在相应发现。
- `clarifies_missing_session`：未加载 EDF 时要求用户先加载记录。
- `clarifies_missing_scope`：模糊的记录分析请求需要用户明确目标或范围。
- `preserves_user_correction`：后续轮必须采用用户修正后的时间、导联或灵敏度。
- `reports_invalid_request_or_tool_error`：如实说明非法请求或工具错误。
- `no_fabricated_result_on_error`：失败后不得生成具体事件、数值或诊断。
- `distinguishes_general_knowledge_from_record`：一般知识与当前记录证据必须分开。
- `mentions_missing_existing_results`：没有既有分析时明确指出缺少相应结果。
- `uses_referenced_prior_window`：正确解析“刚才的时间段”等跨轮指代。

## 三项指标的判定

L1、L2、L3 分别计算 TCR、R-ACC、TCE，再给总体值。三项指标使用同一批运行轨迹，但分别计分：

- **TCR（任务完成率）**：完成用例数 ÷ 全部用例数。当前自动评分是结构性完成：必须工具成功、禁用工具未被调用、顺序和关键参数正确、无非预期工具错误，且最终回答非空。L3 必须每轮都通过。`response_checks` 尚需人工或独立 judge 复核，因此当前 TCR 不能当作完整临床语义正确率。
- **R-ACC（路由准确率）**：路由正确的用例数 ÷ 全部用例数。每轮必须路由到 `expected_skill` 或 `acceptable_skills`；L3 的所有轮都正确才算该用例路由正确。这里评估的是当前架构的 Skill 路由，而非论文中的多 Agent 主管路由。
- **TCE（工具调用效率）**：仅对已完成且有 Agent 工具调用的用例，计算“必需调用数 ÷ Agent 实际全部工具调用数”，再在同一等级内取算术平均。评测器预置的工具调用不计入。没有符合条件的用例时显示“—”，不是 0%。

六用例实验每个等级只有两条记录、每条只跑一次；这是验证评测流程的小样本，不能替代论文的大规模统计。

把全部 30 个原始用例分别用于 GPED 和 ABDO，各跑一次（共 60 次完整用例运行）：

```powershell
python eval/agent_benchmark/run_tcr.py --paired-two-eeg --repeat 1
```

该模式将每个用例复制为 `@GPED` 和 `@ABDO` 两版。GPED 记录仅 241 秒，因此 `L2-001@GPED` 的有效检测窗口从 600 秒改为 120 秒，保留“先查基础信息、再检测”的目标；ABDO 仍为 600 秒。`L1-010` 没有 EDF 会话，两版只是重复的无记录对照，不读取对应数据。报告会分别给出两份数据每个等级的三项指标。

## 三指标执行器

`run_tcr.py` 对每个用例同时输出三项指标，并按等级汇总。自然语言回答检查项会保存在结果中，当前标记为待人工或独立 judge 复核。

仅验证用例和筛选条件，不调用模型：

```powershell
python eval/agent_benchmark/run_tcr.py --dry-run
```

先运行一个低成本用例：

```powershell
python eval/agent_benchmark/run_tcr.py --case-id L1-001 --repeat 1
```

运行一个等级，每个用例一次：

```powershell
python eval/agent_benchmark/run_tcr.py --level L1 --repeat 1
```

论文式重复五次：

```powershell
python eval/agent_benchmark/run_tcr.py --repeat 5
```

最后一条命令会产生 `30 × 5 = 150` 次完整用例运行；多轮用例会包含多次模型调用。原始轨迹和汇总默认写入 `eval/agent_benchmark/results/<时间戳>/`。

已保存的轨迹可直接重新评分，不会调用模型，也不会覆盖原报告：

```powershell
python eval/agent_benchmark/run_tcr.py --cases-file eval/agent_benchmark/tcr_two_eeg.jsonl --rescore-runs eval/agent_benchmark/results/<时间戳>/runs.jsonl
```

此命令在对应结果目录中生成 `metrics_summary.json` 和 `metrics_report.md`。修订前后的运行应分开汇总，不合并为同一批实验。
