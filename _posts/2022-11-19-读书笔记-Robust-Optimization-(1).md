---
layout: post
title: "<读书笔记> Robust Optimization (1)"
date:   2022-11-19
tags: [读书笔记,RO]
comments: true
author: Zhang Yue
---



鲁棒优化（robust optimization）是最优化理论中的一类用来寻求在不确定环境中使优化问题具有一定程度的鲁棒性的方法。鲁棒优化的目的是求得这样一个解，对于可能出现的所有情况，约束条件均满足，并且使得最坏情况下的目标函数的函数值最优。 其核心思想是将原始问题以一定的近似程度转化为一个具有多项式计算复杂度的凸优化问题。鲁棒优化的关键是建立相应的鲁棒对等模型。然后利用相关的优化理论将其转化为可求解的“近似”鲁棒对等问题，并给出鲁棒最优解。

> To be uncertain is uncomfortable; but to be certain is ridiculous. (Goethe, 1749-1832)

--------------------------------------------

**Chapter 0 Preface**

**A. 优化数据中的不确定性**

在实际优化问题中，常常会出现数据不确定的情况，如对未来需求的预测等等。我们难以得到这类数据高精度的估计，而它们的小小改动却又会使得问题的最优解大相径庭。因此，在优化问题中，我们需要一种方法，能够检测数据不确定性将会影响解的质量的情况，并在这种情况下产生一个健壮的解，一个不受数据不确定性影响或影响较小的解。

**B. 鲁棒优化机制**

以线性规划为例，线性规划的标准形式为：


$$
\min_{x} \{c^Tx: Ax \leq b\}
$$


而在鲁棒优化中，一个不确定的LP问题可以描述为以下形式：


$$
\min_{x} \{c^Tx: Ax \leq b\},(c,A,B)\in \mathcal{U}
$$


其中，$\mathcal{U}$ 为不确定集。

而这个问题的答案，正如鲁棒优化以其最基本的形式所提供的，依赖于对潜在决策环境的三个隐含假设：

- 决策变量 $x$ 中的所有元素都表示为“此时此地”的决策：在问题解决后它们能够取得特定的实数值。
- 当且仅当实际数据在预估的不确定集 $\mathcal{U}$ 中时，决策者要对所作决策的后果全部负责
- 不确定LP问题中的约束均为硬约束，当实际数据处于 $\mathcal{U}$ 中时，决策者不能违反约束

当一个解能够保证无论真实数据为 $\mathcal{U}$ 中任何元素时均可行，那么我们称这样一个解是鲁棒可行的，鲁棒不可行的解并不具备研究意义。

而对于鲁棒可行解对应的目标函数值，我们采取“**以最坏情况作为导向**”的思想，鲁棒可行解 $x$ 的质量通过计算 $sup\{c^Tx:(c,A,b) \in \mathcal{U}\}$ 来得到。

因此，我们的问题可以转化为：


$$
\min_{x} \left \{ \sup_{(c,A,b)\in \mathcal{U}} c^Tx: Ax\leq b, \forall (c,A,b)\in \mathcal{U} \right \}
$$


也可写作：


$$
\min_{x,t}\left \{ t:c^Tx \leq t, Ax \leq b, \forall (c,A,b) \in \mathcal{U} \right \} \tag{RC}
$$


后一个问题也被称为原始不确定问题的 *Robust Counterpart (RC)*，RC的可行/最优解被称为不确定问题的鲁棒可行/最优解。

**Robust vs. Stochastic Optimization**

在随机规划 (SO) 中，不确定的数值数据往往被假设为随机的。在最简单的情况中，这些随机数据的先验概率分布是已知的。在此，每个不确定的LP问题同样与一个确定性问题相关联，最著名的是**机会约束规划**：


$$
\min_{x,t}\left \{ t:Prob_{(c,A,b)\sim P}\{c^Tx\leq t \space \& \space Ax\leq b \} \geq 1-\epsilon \right \}
$$


在此 $\epsilon \ll 1$ 是一个极小值，$P$ 为 $(c,A,b)$ 的分布函数。

当分布只是部分可知时，假设 $P$ 属于一个概率分布族 $\mathcal{P}$，则上述问题可以由**模糊机会约束规划**代替：


$$
\min_{x,t}\left \{ t:Prob_{(c,A,b)\sim P}\{c^Tx\leq t \space \& \space Ax\leq b \} \geq 1-\epsilon, \forall P \in \mathcal{P} \right \}
$$


可以看出，SO相较于RO得到的解更加不具有保守性。RO总是寻找最坏情况下仍能表现良好的解，而SO的目的在于寻找一个在所有情况中都表现良好的解。但是，SO对不确定数据集提出了严峻的要求，必须已知数据概率分布或概率分布族。而实际问题中，大部分数据都很难呈现严格的分布性质，因此SO往往被迫对实际分布进行过于简化的假设。

并且，机会约束规划并不那么容易被人接受。机会约束规划得到的解，往往在大量的试验次数下能做到表现最优，这是大数定律（Law of Large Numbers）所决定的。而对于单次乃至少数试验下，它们得到的解表现较为一般。

此外，机会约束规划并没有想象中的那么成功。当机会约束规划可解时，毫无疑问这是一种对抗不确定性的极佳方式。但问题恰在于此，机会约束规划的可解性实际上是十分稀少的，大多数情况下很难去判别给定的候选解是否可行。并且，机会约束往往是非凸可行集，使问题更加复杂化。

从上述几方面可以发现，RO的保守性其实在很多方面是它的优势。很多情况下，我们并不能取到人生的最优解，因为各种变数的存在。当我们需要进行决策时，不能总是去考虑各种变数平均下的最优，因为机会总是只有一次，同样重要的决策场景，很难再出现第二次，单次试验下的随机规划效果是很差的。而每次都从最坏情况出发考虑，可能它并不能得到很多情况下的最优解，但它一定能保证在最坏情况下为你保证一个下界。可能略显悲观，但十分有用。

总而言之，SO和RO都是处理优化问题中数据不确定性十分有效的方式，有着各自的优缺点。

**Robust Optimization and Sensitivity Analysis**

除随机优化外，另一个处理优化中数据不确定性的方法是灵敏度分析。这里最重要的问题是名义最优解作为关于潜在名义数据的函数下的连续性。灵敏度分析相较于RO和SO，其实处理的是一个完全不同的问题。



