---
layout: post
title: "<论文拾萃> On-Demand Delivery from Stores: Dynamic Dispatching and Routing with Random Demand"
date:   2023-02-10
tags: [论文拾萃,ADP,Benders Decomposition,Column generation]
comments: true
author: Zhang Yue
---

本论文于2022年发表于知名期刊 *Manufacturing & Service Operations Management* 上。作者对时下流行的按需配送（On-demand Delivery）问题进行了建模，通过参数化分派和路线策略近似值函数，构建了新的结构化近似框架，解决了最后一公里配送系统中的随机动态骑手分派与路线问题，并利用Benders decomposition和column generation对近似问题进行了求解。

-----------------------

**1.1 问题描述**

系统通过batch process处理订单：在同一个时隙内加入的订单组成一批，共享同样的配送时间目标。当一批订单收集完成后，商店开始准备订单，系统将会把这批订单分派给空闲的骑手并指定他们的路线。在订单全部准备完成后，分派的骑手会在商店取走订单并开始配送。一个骑手可以同时配送多个订单。在完成分配的配送任务后，骑手会回到商店准备下一次分派。由于顾客高度重视配送速度和可靠性，该公司需要一个优秀的分派和路线策略来在控制配送队伍规模和人力成本的同时最小化配送时间。

-------------

**1.2 问题建模**

本节给出了有限时域内存在多次分派波的骑手分派和路线模型。

配送服务系统运行$N$个周期，$\Delta$为每个周期的长度（两次连续的决策时间点的间隔）。本文的建模并不需要$\Delta$为静态值，但决策时间点是提前预知的。

在周期$n \in \{1,...,N\}$中，空闲骑手数量$\bar{K}^n$是已知且固定的。每个骑手都有固定的旅行速度$v$，每次旅行中最多配送$Q$个物件。为了便于分析，不妨假设$\bar{K}^n=\bar{K},n=1,...,N$。

潜在的顾客地点集合为$\mathcal{I}=\{1,...,I\}$。在时间周期$n$的开始（决策时间点$t_n$），我们关注在$[t_{n-1},t_n)$内到达的订单数量，$t_0$为服务开始时间。在$t_n$时刻，我们对这些订单根据他们的地点$\mathcal{I}^n \in \mathcal{I}$和数量$q^n=(q_1^n,...,q_I^n)\in N^I(q_i^n=0 \space if \space i \notin \mathcal{I}^n)$进行分派。不失一般性，$q_i^n \leq Q$。我们假定准备和打包订单的固定时间为$t_p$，在周期$n-1$内到达的订单均会在$t_n+t_p$前准备完成。由于$t_p$可以从数据中估计，因此它对系统而言是已知的，接下来我们将它假设为0。

在每一个决策时间点$t_n$，系统了解所有骑手的时空位置。骑手的状态向量为$\zeta^n=(\zeta^n_1,...,\zeta_N^n)$。特别地，$\zeta_{n^\star}^n \in N$为在时期$n^\star$内在途骑手的数量。只有当$n^\star \geq n$时，$\zeta_{n^\star}^n$才是有意义的。

综上，整个系统在时间点$t_n$的状态为$(\mathcal{I}^n,q^n,\zeta^n)$。

在时期$n$开始，我们需要做出两类决策：（1）需要决定多少骑手来对已有的订单进行派送，通过$K^n$标识；（2）将已有订单分配给这$K^n$个骑手并为他们规划路线。被分派的骑手会在完成任务后回到原点。根据公司实际情况，我们假定在$\mathcal{I}^n$中的所有订单都在$t^n$时刻被分配给骑手。

令location 0 表示骑手的初始位置，$Q^n$为时期$n$内顾客地点和订单数量服从的联合分布。系统做出分派和路由的联合决策$Y^n=\{y_{ijk}^n \in \{0,1\}:i,j\in \mathcal{I}^n \cup \{0\},k=0,...,\bar{K}\}$，$y_{ijk}^n=1$表示在时期$n$内骑手$k$从地点$i$到地点$j$，否则为0，$y_{00k}^n=1$表示骑手$k$未被分派任务、一直待在原点。

因此，顾客端的准时度表现可以反映为$u_i(Y^n)$，表示按照决定$Y^n$从该订单准备派送到派送完成的持续时间。此外，对于每个订单还有一个硬时间约束$L_{max}$。可行决策集$\mathcal{D}(\mathcal{I}^n,q^n,\zeta^n)$必须满足：



$$
\begin{align}
& y_{ijk}^n=0,\space  \forall i,j \in \mathcal{I}^n,k=1,...,\zeta_n^n, \tag{1}
\\
\\
& \sum_{i \in \mathcal{I}^n \cup \{0\}}y_{0ik}^n = \sum_{i \in \mathcal{I}^n\cup\{0\}}y_{i0k}^n=1, \space \forall k=\zeta^n_n+1,...,\bar{K}, \tag{2}
\\
\\
& \sum_{j\in \mathcal{I}^n \cup \{0\}}y_{ijk}^n = \sum_{j \in \mathcal{I}^n \cup \{0\}}y_{jik}^n, \space \forall i \in \mathcal{I}^n,k=\zeta_n^n+1,...,\bar{K} \tag{3}
\\
\\
& \sum_{i \in S}\sum_{j \in S}y_{ijk}^n \leq \lvert S \rvert -1, \space \forall S \subset \mathcal{I}^n,k=\zeta_n^n+1,...,\bar{K} \tag{4}
\\
\\
& \sum_{i \in \mathcal{I}^n}\sum_{j \in \mathcal{I}^n \cup \{0\}}q_i^ny_{ijk}^n \leq Q, \space \forall k=\zeta_n^n+1,...,\bar{K} \tag{5}
\\
\\
& u_i(Y^n)\leq L_{max}, \space \forall i \in \mathcal{I}^n \tag{6}
\end{align}
$$



约束1表示了骑手的空闲情况。约束2确保每个棋手都从起点出发，并最终回到起点。约束3和4是流守恒约束和子回路消除约束。约束5确保容量约束不被违反，约束6保证硬配送时间。

而模型的目标在于在计划周期内最小化总的订单配送时间期望。令$l_k^n(Y^n)$表示时期$n$内骑手$k$在路径总时间（包括旅行时间和服务时间）。因此，有限时域随机动态规划的值函数可以表示为



$$
\begin{align}
& \mathcal{H}_n(\mathcal{I}^n,q^n,\zeta^n)=\min_{Y^n \in \mathcal{D}(\mathcal{I}^n,q^n,\zeta^n)} \left \{ \sum_{i\in \mathcal{I}^n} u_i(Y^n)+E_{Q^{n+1}}[\mathcal{H}_{n+1}(\mathcal{I}^{n+1},q^{n+1},\zeta^{n+1})]\right \} \tag{7}
\\
\\
& \mathcal{H}_N(\mathcal{I}^N,q^N,\zeta^N)=\min_{Y^N \in \mathcal{D}(\mathcal{I}^N,q^N,\zeta^N)} \left \{ \sum_{i \in \mathcal{I}^N} u_i(Y^N)\right \} \tag{8}
\end{align}
$$



此外，空闲骑手的转变约束可表示为：



$$
\zeta_{n^\star}^{n+1}=\zeta_{n^\star}^n+\sum_{k=1}^\bar{K} \mathcal{1}(l_k^n(Y^n) \gt t_{n^{\star}}-t_n ), \space \forall n^\star=n+1,...,N,n=1,...,N, \tag{9}
$$



$\mathcal{1}(l_k^n(Y^n)>t_{n^\star}-t_n)$是指示变量，当骑手$k$在时期$n^\star$前通过决策$Y^n$无法回到起点时等于1。

将上述动态规划模型统称为*JDR*。

-----------

**1.4 结构化近似方法**

由于状态空间和动作空间均为高维，*JDR*不能够被精确求解。甚至当需求确定时，多周期分派和路由问题都是NP-hard的。因此，很多公司都采用简单的myopic短视策略来确定决策，只优化当前批次的订单的配送表现，而不考虑对未来订单的影响。然而，在考虑的计划周期内，一个棋手必须完成几次旅行，并在商店和顾客间来回。当前订单批次的分派和路由决策将会影响未来时期的骑手空闲情况，忽略这一相互影响将会极大削弱系统表现。

为了实时得到高质量解，本文提出了一个针对上述随机规划模型的近似框架。在高层，该框架通过结合短视路由、预期调度的参数化分派和路由策略来估计cost-to-go函数。估计的cost-to-go函数用来帮助确定当前状态下最优的分派和路由决策。

估计近似cost-to-go函数的关键在于对当前决策对未来产生的影响进行模型化。分派和路由决策通过限制未来时期的空闲骑手数量来影响未来的配送成本。当当前时期更多的骑手被分派出去时，接下来的时期更少的骑手处于空闲状态。

为了“捕获”这种影响，我们通过短视路由策略下的单周期值函数之和来近似cost-to-go函数。令$\mathcal{H}^s(K^n,\mathcal{I}^n,q^n)$表示$K^n$个分派骑手时顾客地点和订单数量为$\mathcal{I}^n$和$q^n$的单周期最优配送成本。那么，期望cost-to-go函数$E_{Q^n}[\mathcal{H}_n(\mathcal{I^n,q^n,\zeta^n})]$可以通过下述表达式进行估计：


$$
\begin{align}
APT^n(\zeta^n): & \min_{K^{n^\star}\in N}\sum_{n^\star=n}^NE_{Q^{n^{\star}}} \left [ \mathcal{H}^s(K^{n^{\star}},\mathcal{I}^{n^{\star}},q) \right ]
\\
& s.t. \space \sum_{m=n}^{n^{\star}}\omega_m^{n^\star}(K^m) \leq \bar{K}-\zeta_{n^\star}^n, \space \forall n^\star=n,...,N, \tag{10}
\end{align}
$$


此处$E_{Q^n}[\mathcal{H}^s(K^n,\mathcal{I}^n,q^n)]$是期望的单周期最优配送成本，是所有可能的$\mathcal{I}^n$和$q^n$的情况的均值。我们可以通过基于历史数据或拟合概率分布的离线模拟来估计这个值：$E_{Q^n}[\mathcal{H}^s(K^n,\mathcal{I}^n,q^n)]=\sum_{h=1}^H\mathcal{H}^s(K^n,\mathcal{I}_h^n,q_h^n)/H$，对于顾客地点和订单的$H$个样本。约束10保证分派和在路上的骑手数量在每个时期内都不超过$\bar{K}$。

注意到，一个分派出去的骑手的在路上时间至少为一个周期（一个骑手不能在一个周期内被分派出去两次），因此$\omega_{n^\star}^{n^\star}(K^{n^\star})=K^{n^\star},for \space n^\star=n,...,N$。然而，对于$n^\star \gt m$，$\omega_m^{n^\star}(K^m)$不确定，因为需求随机，将其视为可通过离线模拟调整的参数。

上述近似方案通过分解的分派和（短视）路由启发式算法对期望cost-to-go函数进行了估计，也可以视为ADP中基于rollout策略的随机前瞻方法。


$$
\begin{align}
& \min_{Y^n\in \mathcal{D}(\mathcal{I}^n,q^n,\zeta^n)} 
\\
& \left \{ \sum_{i\in \mathcal{I}^n}u_i(Y_n) + \min_{K^{n^\star \in N}} \sum_{n^{\star}=n+1}^NE_{Q^{n^\star}}[\mathcal{H}^s(K^{n^\star},\mathcal{I}^{n^\star},q^{n^\star})] \right \} 
\\
& s.t. \sum_{m=n+1}^{n^\star}\omega_m^{n^\star}(K^m)\leq \bar{K}-\zeta_{n^\star}^{n+1}, \space \forall n^\star = n+1,...,N 
\end{align}
$$


引入0-1变量$x_k^{n^\star}$表示是否$k$个骑手在时期$n^\star$中被分派出去，上式可以写作：


$$
\begin{align}
& \min_{Y^n \in \mathcal{D}(\mathcal{I}^n,q^n,\zeta^n),x_k^{n^\star}\in \{0,1\}} 
\\
&  \sum_{i\in \mathcal{I}^n}u_i(Y_n) + \sum_{n^{\star}=n+1}^N\sum_{k=0}^\bar{K}x_k^{n^\star}E_{Q^{n^\star}}[\mathcal{H}^s(k,\mathcal{I}^{n^\star},q^{n^\star})] \tag{11}
\\
s.t. &\sum_{m=n+1}^{n^\star}\sum_{k=0}^\bar{K}\omega_m^{n^\star}(k)x_k^m \leq \bar{K}-\zeta_{n^\star}^n-\sum_{k=1}^\bar{K}\mathcal{1}(l_k^n(Y^n) \gt t_{n^\star}-t_n), \space \forall n^\star=n+1,...,N, \tag{12}
\\
&\sum_{k=0}^\bar{K}x_k^{n^\star}=1, \space n^\star=n+1,...,N \tag{13}
\end{align}
$$


将上述策略称为*AJRP*.

-----------------

**references**

*Liu S, Luo Z. On-Demand Delivery from Stores: Dynamic Dispatching and Routing with Random Demand[J]. Manufacturing & Service Operations Management, 2022.*
