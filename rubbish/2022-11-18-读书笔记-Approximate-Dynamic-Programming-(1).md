---
layout: post
title: "<读书笔记> Approximate Dynamic Programming (1)"
date:   2022-11-18
tags: [读书笔记,ADP]
comments: true
author: Zhang Yue
---



本书由Warren B. Powell所作。

---------

**Chapter 1 The Challenges of Dynamic Programming**

随着时间推移的问题优化出现在很多的场景之下，从供暖系统的控制到金融积极的发展。其中的例子有，航班起落、设备更新、库存控制、汽车时刻表设定等等。

这些问题总是包含这三步：进行决策，观察信息，在信息的基础上继续进行更深入的决策。我们将之称为**序列决策问题** （*sequential decision problems*）。此类问题的建模总是很简单的，但解决它却从非易事。

----------------------

**1.1 A dynamic programming example: a shortest path problem**

最知名的动态规划应用莫过于最短路问题。

<img src="https://s2.loli.net/2022/11/18/W8yj9pz4BrtFQqo.png" alt="image-20221118184818028.png" style="zoom: 67%;" align='mid'/>

我们定义 $I$ 为节点的集合。如果司机处于节点 $i$ 处，他可以到达集合 $I^+_i$ 的子集元素中，成本为$c_{ij}$。现在他需要从起始节点 $q \in I$ 出发，找到一条总成本最小的路径到达目标节点 $r \in I$.

>解决方法：
>
>定义 $v_i$ 为从节点 $i \in I$ 到节点 $r \in I$ 的总成本，因此 $v_r = 0$.
>
>初始化：令 $v_i = M, i \in I -r$
>
>迭代：$v_i \leftarrow min\{v_i,min_{j\in I^+}\{c_{ij}+v_j\}\}, \forall i \in I$
>
>停止条件：所有 $v_i$ 的值都不再变动 

但这并不是一个高效的方式，因为在早期迭代中大部分 $v_j = M$ 并不会改变，使得它的收敛较为缓慢。

----------

**1.2 The three curses of dimensionality**

可以发现，所有动态规划都可以写作递归的形式，通过递归将某一阶段特定状态的值与下一时刻另一特定状态的值联系起来。对于确定性问题，这个等式可以写作：
$$
V_t(S_t) = \max_{a_t}(C_t(S_t,a_t) + V_{t+1}(S_{t+1}))
$$
$S_{t+1}$为我们当前处于状态$S_t$采取动作$a_t$后转移到的状态

上面这个等式通常被称为**Bellman's equation**

而本书中的大部分问题与一些形式的不确定性相关（价格、旅行时间、设备故障、天气等等）。

事实上，在大多数ADP的应用中存在着三类维度灾难（*curses of dimensionality*）：

- *State space* : 如果状态变量 $S_t = (S_{t1},...,S_{tI})$ 由 $I$ 个维度组成，而每一项可以取 $L$ 个可能值，那么状态的总数便高达 $L^I$ 。
- *Outcome space* : 随机变量 $W_t = (W_{t1},...,W_{tJ})$ 由 $J$ 个维度组成，而每一项可以取 $M$ 个可能值，那么状态的总数便高达 $M^J$ 。
- *Action space* : 决策变量 $x_t = (x_{t1},...,x_{tK})$ 由 $K$ 个维度组成， 而每一项可以取 $N$ 个可能值，那么状态的总数便高达 $N^K$ 。在数学规划中，我们称 *action space* 为可行域，而假定 $x_t$ 是离散或连续的

由于这三类维度灾难的存在，问题的求解将会变得异常困难，本书之后将会提供一些常用的高效算法。

------

**1.4 Problem classes**

在此，我们将常见的问题归为以下几类：

- Budgeting：分配固定数量的资源到一系列的活动上去，每个活动都需要花费一定的资源。
- Asset acquisition with concave costs
- Asset acquisition with lagged information processes
- Buying/selling an asset
- General resource allocation problems
- Demand management
- Storage problem
- Shortest paths
- dynamic assignment

----------

