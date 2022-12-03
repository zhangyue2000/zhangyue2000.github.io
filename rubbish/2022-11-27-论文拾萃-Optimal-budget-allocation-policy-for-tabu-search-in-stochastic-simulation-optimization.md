---
layout: post
title: "<论文拾萃> Optimal budget allocation policy for tabu search in stochastic simulation optimization"
date:   2022-11-27
tags: [论文拾萃,TS,Simulation]
comments: true
author: Zhang Yue
---





本文于2023年发表于期刊 *Computers & Operation Research* 上。禁忌搜索是解决组合优化问题的有力工具。但是在一些随机仿真优化问题中，由于仿真噪声的影响，禁忌搜索的搜索方向可能会受到影响从而无法收敛于高质量的解。这个问题可以通过提升仿真样本的数量来得到解决，但在真实应用中仿真样本数量的提升也带来了求解时间、成本的增加。在此研究中，我们定义"预算"为可供解的评估使用的仿真样本的数量，我们为TS提出了渐进式的最优预算分配策略，使最优策略能更容易地应用到实际问题中。

---------------

**Problem formulation**

仿真优化问题可以表达为：


$$
\min_{x\in \mathcal{\Theta}} y(x)
$$


此处 $\Theta$ 为搜索空间，$y(x)=E[Y(x)]=\frac{1}{n}\sum_{j=1}^nY_j(x)$ 为仿真输出 $Y(x)$ 的期望。预测精度随着 $n$ 的增大而增大，$\lim_{n\rightarrow \infin} \bar{Y}(x;n)\rightarrow y(x)$.

**Assumption 1.** 假定$Var[Y(x)] \lt \infin , \forall \space x \in \Theta$，且观察值$Y_1(x),...,Y_n(x)$独立同分布

**Assumption 2.** 假定$Y_j(x) \sim N(y(x),\sigma^2(x)), \forall \space j$

 

TS和普通局部搜索采用相同的方式，不断迭代从一个点到到另一个点，直到某个终止条件满足。

每个 $x\in \Theta$ 有一个相关的邻域 $N(x)$， $x$ 可以通过移动转变为另一个解 $x' \in N(x)$。局部搜索过程中采取下滑移动，即每一次移动后得到解都优于原解 $y(x') \lt y(x)$。局部搜索的最明显缺点也是它将在局部最优值处停止。

为了解决这一点，TS采取了以下的调整。首先，每次移动都将移动到邻域最优解上，无论这次移动能够改进原来的解。其次，$N(x)$ 中的某个子集根据禁忌列表 $T$ 被禁忌，防止解被困在局部最优值处。禁忌表被设定为短期内存，根据最近采用的移动更新，禁忌表中的元素由几种形式：解，解的属性，移动等等。此外，如果某个移动虽然被禁忌但它仍满足一定的赦免准则，那么该移动仍然可被采用。我们考虑最简单的赦免原则：当将要移动到的解是历史最优解时，这种移动无论是否被禁忌均为可行的。

 ![image-20221127212100244.png](https://s2.loli.net/2022/11/27/Oa7Dbe2CAdFQspk.png)

---------------------------------------------

**The budget allocation problem**

定义 $s^t = (x,x_0)$ 为 TS 在第 $i$ 次迭代时的状态，$x$ 为第 $i$ 次迭代时的当前解，$x_0$ 为历史最优解。TS 移动定义为 $m$，从 $s^t \rightarrow s^{t+1}$ 的转变。定义 $m(s^t)$ 为在状态 $s^t$ 下根据真实目标值 $y(x_0),...,y(x_k)$ 做出的正确移动，而 $\hat{m}(s^t,n_0,...,n_k,\xi)$ 为根据 $\bar{Y}(x_0;n_0),...,\bar{Y}(x_k;n_k)$ 做出的移动，$\bar{Y}(x_i;n_i)$ 为从 $n_i$ 个仿真样本中计算得到的样本均值，$\xi$ 为代表仿真随机性的参数。因此，正确移动的概率为 $PCM = P(\hat{m}(s^t,n_0,...,n_k,\xi)=m(s^t)) $。

固定预算下，优化分配策略是改善PCM的有效途径之一：


$$
\begin{align}
\max&_{\alpha_0,...,\alpha_k} PCM \\
s.t. \quad &\alpha_0+\alpha_1+...+\alpha_k = 1 \\
&\alpha_i \gt 0,i=0,1,...,k
\end{align}
$$


此处$\alpha_i,i=0,1,...,k$ 为取样比例，$n_i = \alpha_i n,i=1...,k$ 为对第 $i$ 个邻域解分配的预算。

在随机环境中，TS的最优性间隙往往由两个因素造成：

- 仿真噪声的影响
- TS元启发式算法的本质

在此本文将集中于解决第一个因素带来的影响。

-----------------

**Simulation budget allocation**













------------------------

参考文献：

*Yu C, Lahrichi N, Matta A. Optimal budget allocation policy for tabu search in stochastic simulation optimization[J]. Computers & Operations Research, 2023, 150: 106046.*