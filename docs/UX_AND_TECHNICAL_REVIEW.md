# 用户体验与技术复查 · 2026-09-13（历史记录）

2026-09-14 已完成客户聊天策略接入与多轮上下文升级；当前说明见
[CUSTOMER_RUNTIME.md](CUSTOMER_RUNTIME.md)。以下数字与架构描述保留当时的验收边界。

客户页解决“我该怎么做”，Developer lab 解释“策略做得怎么样”。
两者共享身份、案件归属和邮件同意的权限判断。PPO 在模拟环境里控制动作；
客户聊天使用 SOP 与可选的语言模型理解，不能把两者说成同一个已部署策略。

## 客户现在怎么使用

- 不知道 claim 是什么：欢迎语和“Not sure where to start”解释基本概念，提供查进度、拒赔、材料选项。
- 不想打长句：点击选项发送普通聊天请求，照常经过状态机；案件按钮只在验证后显示本人的匹配记录。
- 身份填写不方便：点击“Verify using a form”，选择生日、填写任意三项；显式填充样本后仍需手动提交。
- 输入有误：日期、邮箱、电话、ID 格式提供字段反馈；提交正确值后重新精确匹配，不猜测姓名或证件。
- 只提交姓名：继续补信息的对话，不进入下一业务阶段。主聊天区显示匹配数量和资料锁定状态。
- 改正信息：清除被明确更正字段的旧冲突，保留其他冲突和先前的理赔意图；核验结束后禁止换身份。
- 业务词拼错：支持有限白名单，例如 cliam、staus、deneid。原始身份、审计消息和 consent 不经过这层纠错。
- 不想看长文：回复分段、缩短开场和重复提示；可选下一步；隐私、材料条件、过期截止日期说明不删减。
- 测试手机：不仅检查横向溢出，也检查输入框没有被固定高度的面板裁掉。

## 什么才算好的结果

| 先看什么 | 怎么解释 |
|---|---|
| 是否越过规则 | 身份未核验不能读理赔；没有明确同意不能发送。任何违规都优先调查。 |
| 是否正确帮助客户 | 回答案件问题并完成客户的 follow-up 选择，或在确有需要时转人工。单纯结束对话不算成功。 |
| 是否减少客户负担 | 在安全和结果相当后比较轮数，不能通过提前转人工来追求短对话。 |
| Training score | 人设计的奖励总分，用于训练；不是满意度，也不是百分制考试成绩。 |
| 动作概率 | 策略倾向选择哪个动作；不是回答正确率。Blocked 表示流程禁止该动作。 |
| seed | 用来复现随机变化；训练 seed 42 和 7 对应两个独立训练过程。 |

本次重新执行的审计中，all 分组每个策略 28 个合成 episode：规则基线和两个 PPO
都是 75% 案件完成、25% 合理转人工，提前结束与超时均为 0。seed 42 的平均轮数
与规则基线相同（4.25）；seed 7 为 7.00。**当前证据支持策略学会受约束任务，
不支持 PPO 优于规则基线。** 0 违规是在 SOP 约束下测得，不能当成模型自主学会了安全。

## 本次新增或确认的技术保障

| 检查 | 结果或修复 |
|---|---|
| VERIFY_ID 门禁 | 姓名 1 项、姓名+生日 2 项均不推进；Ma Tian 的 National ID 不计入要求中的 3 项；三项匹配才通过。 |
| 验证进度 | 失败时刷新 matched fields，修复旧进度 1/3 与新日志 0/3 不一致。 |
| 案件解析 | CL-2048 不再误读成 2048 年；真实年份筛选与案件归属仍保留。 |
| Action mask | 全 False 掩码直接报错，防止 softmax 把全部禁止动作变成均匀分布；覆盖单条和批量输入。 |
| PPO 与 GAE | 回归包含优势数值检查、mask、短训练及 checkpoint 重载；本次没有重新训练主模型。 |
| 环境闭环 | 重跑两 seed 的 train/val/test/all 审计，包含 caller reset、动作标签独立性、情绪可观测性、提前升级惩罚。 |
| DPO 数据 | 重新生成 50 组同状态偏好数据；最小 return margin 约 0.199；identity/style 分组切分 32/10/8。没有运行 DPO 模型训练。 |

验证记录位于 `artifacts/usability_review/`。运行命令：

```bash
python3 -m pytest tests/ -q
python3 -m eval.eval_benchmark
OMP_NUM_THREADS=1 python3 -m eval.audit_rl --output-dir /tmp/goaly-ux-rl-audit
OMP_NUM_THREADS=1 python3 -m eval.generate_dpo_pairs --num-pairs 50 \
  --min-reward-delta 0.1 --output-dir /tmp/goaly-ux-dpo-audit
# 先启动应用；Playwright 是可选的浏览器测试依赖。
python3 -m eval.browser_acceptance --output-dir /tmp/goaly-ux-browser
```

这些结果来自离线 fixture 和 caller simulator。Live model 验收仍未通过真实 API token
执行，真实用户满意度、更多分布和统计显著性也尚未验证。Docker、API token 设置、
文字聊天 UI、四阶段流程的提交入口仍保留在根 README。
