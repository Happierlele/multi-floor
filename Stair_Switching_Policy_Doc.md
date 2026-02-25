# 楼梯切换策略 (Stair Switching Policy) 方法文档

## 1. 概述 (Overview)

在多楼层探索任务中，机器人面临一个核心决策问题：**“什么时候应该离开当前楼层，前往下一层？”**

如果离开太早，可能会错过当前楼层的高价值区域；如果离开太晚，则会浪费宝贵的时间步数，导致总探索效率下降。为了解决这个问题，我们设计了一套**基于神经网络的楼梯切换策略 (Stair Switching Policy)**。

该策略采用了**“从模仿到自主” (Imitation-to-Autonomous)** 的渐进式学习框架。在初期，网络通过观察并模仿精心设计的“启发式规则”来快速冷启动；随着训练的进行，网络逐渐内化这些规则，并具备泛化能力，最终能够根据实时状态自主做出最优决策。

---

## 2. 网络架构 (Network Architecture)

我们设计了一个轻量级的多层感知机 (MLP) 作为决策大脑，名为 `StairSwitchNet`。

### 2.1 输入特征 (Input Features)
网络接收 3 个高度抽象的全局状态特征，这些特征经过归一化处理，确保数值分布稳定：

| 特征名称 | 符号 | 范围 | 物理含义 |
| :--- | :--- | :--- | :--- |
| **探索率** | $E_{rate}$ | $[0.0, 1.0]$ | 当前楼层的已知区域占比。$1.0$ 表示全图已开。 |
| **最大效用** | $U_{max}$ | $[0.0, +\infty)$ | 当前楼层所有候选前沿点中，预期收益最高的值（通常归一化后在 0-1 之间）。反映了“还有没有大鱼”。 |
| **时间进度** | $T_{ratio}$ | $[0.0, 1.0]$ | 当前步数占最大允许步数的比例。$1.0$ 表示时间耗尽。 |

### 2.2 网络结构
*   **Input Layer**: 3 神经元
*   **Hidden Layer 1**: 64 神经元 + ReLU 激活
*   **Hidden Layer 2**: 64 神经元 + ReLU 激活
*   **Output Layer**: 2 神经元 (Logits)

### 2.3 输出动作 (Output Actions)
网络输出两个动作的 Logits (未归一化的概率值)，经过 Softmax 后得到概率分布：

*   **Action 0 (Stay)**: 继续在当前楼层探索。
*   **Action 1 (Switch)**: 立刻前往最近的楼梯，准备切换到下一层。

---

## 3. 训练机制 (Training Mechanism)

我们采用 **模仿学习 (Imitation Learning)** 与 **Q-Learning** 相结合的训练范式。

### 3.1 启发式教师 (Heuristic Teacher)
为了解决强化学习初期的“盲目探索”问题，我们引入了一个基于规则的“教师”。教师根据以下逻辑判断是否应该切换：

1.  **收益递减原则**: 如果 $U_{max} < 15.0$ (当前层没什么油水了) **且** $T_{ratio} > 1/8$ (不是刚开局)，则建议切换。
2.  **完备性原则**: 如果 $E_{rate} > 0.8$ (当前层几乎看完了)，则建议切换。

### 3.2 数据流转 (Data Pipeline)

整个训练过程分为四个阶段：

#### 阶段 I: 观察 (Observation)
在每一个时间步 $t$，网络接收当前状态 $s_t$，前向传播计算出预测动作 $a_{net}$。
$$ a_{net} = \arg\max \text{Network}(s_t) $$

#### 阶段 II: 执行与矫正 (Execution & Correction)
虽然网络给出了建议，但在训练初期，机器人**强制执行**启发式教师的建议 $a_{teacher}$。这保证了机器人在早期也能表现出合理的行为，不会因为网络的随机初始化而乱跑。

#### 阶段 III: 记录 (Recording)
我们将这一步的经验元组存入专门的 `stair_switch_buffer` 经验池中：
$$ \text{Transition} = (s_t, a_{teacher}, r_t, s_{t+1}, done) $$
*注意：这里存储的动作是 $a_{teacher}$，这意味着我们在告诉网络：“在这种状态下，老师选了 $a_{teacher}$，你也应该这么选。”*

#### 阶段 IV: 训练 (Training)
后台训练进程定期从经验池中采样 Batch 数据，计算损失并更新网络参数。

*   **损失函数**: 均方误差损失 (MSE Loss)
    $$ L = \frac{1}{N} \sum (Q_{net}(s, a_{teacher}) - y_{target})^2 $$
    其中 $y_{target} = r + \gamma \max Q_{target}(s', a')$ (DQN 目标)

通过这种方式，网络最初是在**“背诵”**规则，但由于 Q-Learning 的引入，网络实际上是在学习状态-动作的价值函数 (Value Function)。这意味着在未来，即使面对规则未覆盖的边缘情况，网络也能根据预测的长期价值做出判断。

---

## 4. 核心代码实现 (Core Implementation)

### 4.1 网络定义 (`model.py`)
```python
class StairSwitchNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super(StairSwitchNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)  # Output: [logit_stay, logit_switch]
        )

    def forward(self, x):
        return self.net(x)
```

### 4.2 推理与数据收集 (`worker.py`)
```python
# 1. 准备输入
switch_input = torch.tensor([self.env.explored_rate, max_utility / 100.0, time_ratio])

# 2. 启发式规则判定 (Teacher)
cond_utility = enough_steps and (max_utility < 15.0)
cond_explored = self.env.explored_rate > 0.8
should_switch = cond_utility or cond_explored
actual_action = 1 if should_switch else 0

# 3. 存入经验池 (用于教导网络)
self.stair_switch_buffer.append({
    'state': switch_input,
    'action': torch.tensor([actual_action]),
    # ...后续填充 reward 和 next_state
})
```

---

## 5. 总结 (Summary)

这套策略巧妙地平衡了**规则的稳定性**和**学习的灵活性**。
*   **短期**: 依靠规则保证下限，确保机器人能顺利完成多层探索。
*   **长期**: 依靠网络提升上限，让机器人学会更细腻的权衡（比如在时间紧迫时适当降低切换门槛）。
