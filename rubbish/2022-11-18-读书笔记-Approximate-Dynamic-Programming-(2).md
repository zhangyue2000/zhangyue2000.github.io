---
layout: post
title: "<读书笔记> Approximate Dynamic Programming (2)"
date:   2022-11-18
tags: [读书笔记,ADP]
comments: true
author: Zhang Yue

---



本书由Warren B. Powell所作。

---------

**Chapter 3 Introduction to Markov Decision Processes**

假设存在离散状态空间 $S = (1,2,...,|S|)$，$S$ 足够小到可以枚举，动作空间 $A$ 同样较小，则我们可以直接计算成本 $C(S,a)$。

采取动作$a_t$ 后，我们可以根据概率转移矩阵 $p_t(S_{t+1} | S_t,a_t)$确定我们在状态 $S_t$ 时采取动作 $a_t$ 后处于状态 $S_{t+1}$ 的概率

当状态连续或状态空间过大时，一步转移矩阵 $p_t(S_{t+1} | S_t,a_t)$ 将难以计算

----------------------------------------

**3.1 The optimality equations**

以确定性最短路问题为例。

当我们处于状态 $S_t = i$ 时，采取动作 $a_t = j$，转移函数将会告知我们接下来的状态 $S_{t+1} = S^M(S_t, a_t)$





