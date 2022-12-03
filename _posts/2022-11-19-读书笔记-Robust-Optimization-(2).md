---
layout: post
title: "<读书笔记> Robust Optimization (2)"
date:   2022-11-21
tags: [读书笔记,RO]
comments: true
author: Zhang Yue

---



鲁棒优化（robust optimization）是最优化理论中的一类用来寻求在不确定环境中使优化问题具有一定程度的鲁棒性的方法。鲁棒优化的目的是求得这样一个解，对于可能出现的所有情况，约束条件均满足，并且使得最坏情况下的目标函数的函数值最优。 其核心思想是将原始问题以一定的近似程度转化为一个具有多项式计算复杂度的凸优化问题。鲁棒优化的关键是建立相应的鲁棒对等模型。然后利用相关的优化理论将其转化为可求解的“近似”鲁棒对等问题，并给出鲁棒最优解。

> To be uncertain is uncomfortable; but to be certain is ridiculous. (Goethe, 1749-1832)

----------------------

**Chapter 1 Uncertain Linear Optimization Problems and their Robust Counterparts**

**1.1 Data Uncertainty in Linear Optimization**

线性优化问题（LO）的形式如下：



$$
\min_x\left \{ c^Tx+d:Ax\leq b \right \}
$$



此时 $x \in R^n$ 为决策变量，$ c\in R^n, d \in R,A \in R^{m \times n}, b \in R^m$

显然，常量 $d$ 是可以被忽略掉的，因为它并不影响最优解。

这个问题的结构可以归纳为 $(c,d,A,b)$，用数据矩阵 $D\in R^{(m+1)\times(n+1)}$ 表示为



$$
D = 
\begin{bmatrix}
c^T  & d\\
A  & b
\end{bmatrix}
$$



而LO问题中的一些数据在现实世界中常常无法获取。数据不确定性的来源往往分为以下几类：

- 当解决问题时，数据来源并不存在（如未来需求），此时用它们的预测值替代。这些数据容易收到 *prediction error* 的影响；
- 一些数据不能够确切测量，常常用名义上的测量值替代，容易受到 *measurement error* 的影响；
- 一些决策变量不能够按照计算的一样实现，容易受到 *implementation error* 的影响

---------

**1.2 Uncertain Linear Problems and Their Robust Counterparts**

> Def 1.2.1 An uncertain Linear Optimization problem is a collection of LP problems (instances) $\min_{x} \{c^Tx + d:Ax\leq b\}$ of common structure (i.e., with common numbers $m$ of constraints and $n$ of variables) with the data varying in a given uncertainty set $\mathcal{U} \subset R^{(m+1)\times (n+1)}$  
>
> $$
> \left \{ \min_{x} \{ c^Tx + d : Ax \leq b \}\right \}{(c,d,A,b) \in \mathcal{U}} \tag{LOu}
> $$
>

我们总是假定不确定集是参数化的，以一种仿射形式，通过在给定扰动集中变化的扰动向量 $\zeta$ 实现：



$$
\mathcal{U} = 
\left \{ 
\begin{bmatrix}
c^T & d \\
A   & b
\end{bmatrix}
=
\begin{bmatrix}
c_0^T & d_0 \\
A_0   & b_0
\end{bmatrix}
+\sum_{l=1}^L \zeta_l 
\begin{bmatrix}
c_l^T & d_l \\
A_l   & b_l
\end{bmatrix}
:
\zeta \in \mathcal{Z} \subset R^L
\right \}
$$



前一矩阵称为 nominal data $D_0$，后一矩阵称为 basic shifts $D_l$

Pertubation set : $\mathcal{Z}$

> remark 1.2.2. If the perturbation set  $\mathcal{Z}$  itself is represented as the image of another set $\hat{\mathcal{Z}}$ under affine mapping $\xi \mapsto \zeta = p + P \xi$ , then we can pass from perturbations $\zeta$ to pertubations $\xi$ :
> $$
> \begin{align}
> \mathcal{U} &= \left \{ 
> \begin{bmatrix}
> c^T & d \\
> A   & b \\
> \end{bmatrix}
> =
> D_0
> +
> \sum_{l=1}^L \zeta_l D_l : \zeta \in \mathcal{Z}
> \right \}
> \\
> &= \left \{ 
> \begin{bmatrix}
> c^T & d \\
> A   & b \\
> \end{bmatrix}
> =
> D_0
> +
> \sum_{l=1}^L [p_l + \sum_{k=1}^K P_{lk} \xi_k] D_l : \xi \in \hat{\mathcal{Z}}
> \right \}
> \\
> &= \left \{ 
> \begin{bmatrix}
> c^T & d \\
> A   & b \\
> \end{bmatrix}
> =
> [D_0
> +
> \sum_{l=1}^L p_lD_l] + \sum_{k=1}^K \xi_k[\sum_{l=1}^LP_{lk}D_l]  : \xi \in \hat{\mathcal{Z}}
> \right \}
> \\
> &= \left \{ 
> \begin{bmatrix}
> c^T & d \\
> A   & b \\
> \end{bmatrix}
> =
> \hat{D}_0 + \sum_{k=1}^K \xi_k\hat{D}_k  : \xi \in \hat{\mathcal{Z}}
> \right \}
> \end{align}
> $$

因此，当一些扰动集有着简单的几何结构（平行四面体、椭球体等等）时，我们将他们统一化为标准集。

请注意，与单一优化问题相比，像 $LOu$ 这样的优化问题族本身并没有与可行/最优解和最优值的概念相关联。如何定义这些概念当然取决于潜在的决策环境，而在这里，我们关注一个具有以下特征的环境：

- 决策变量 $x$ 中的所有元素都表示为“此时此地”的决策：在问题解决后它们能够取得特定的实数值。
- 当且仅当实际数据在预估的不确定集 $\mathcal{U}$ 中时，决策者要对所作决策的后果全部负责
- 不确定LP问题中的约束均为硬约束，当实际数据处于 $\mathcal{U}$ 中时，决策者不能违反约束

以上的假设规定了什么样的可行解对于不确定问题 $(LOu)$ 是有意义的。根据第一条，可行解首先得是固定向量；根据第二条和第三条，他们需要是鲁棒可行的，即无论数据属于不确定集中的哪个元素，他们都需要满足所有约束。

------------------------

> Def 1.2.3 A vector $x\in R^n$ is a robust feasible solution to $(LOu)$ , if it satisfies all realizations of the constraints from the uncertainty set, that is, 
>
> $$
> Ax \leq b \quad \forall (c,d,A,b) \in \mathcal{U}
> $$

> Def 1.2.4 Given a candidate solution $x$ , the robust value $\hat{c}(x)$ of the objective in $(LOu)$ at $x$ is the largest value of the "true" objective $c^Tx +d$ over all realizations of the data from the uncertainty set:
> 
> $$
> \hat{c}(x) = \sup_{(c,d,A,u)\in \mathcal{U}}[c^Tx +d]
> $$

>Def 1.2.5 The Robust Counterpart of the uncertain LO problem $(LOu)$ is the optimization problem of minimizing the robust value of the objective over all robust feasible solutions to the uncertain problem
> 
> $$
> \min_{x} \left \{
> \hat{c}(x) = \sup_{(c,d,A,u)\in \mathcal{U}} [c^Tx+d]: Ax \leq b \quad \forall (c,d,A,b) \in \mathcal{U}
> \right \}
> $$

Robust Counterpart 问题的最优解称为鲁棒最优解，最优值称为鲁棒最优值

注意到：

- $LOu$ 的 Robust Counterpart 问题可以等同于以下问题。这种情况下，任何一个不确定的LO问题均可以重新规划为一个确定目标的不确定LO问题。重规划问题的Robust Counterpart问题等同于原问题的RC问题。

$$
\min_{x,t}\left\{ t: \begin{matrix} c^Tx - t \leq -d \\Ax\leq b \end{matrix} \quad \forall (c,d,A,b)\in \mathcal{U} \right\}
$$

- 假设 $LOu$ 有确定目标，则其RC问题如下。

$$
\min_x \left \{c^Tx+d:Ax\leq b, \space \forall (A,b) \in \mathcal{U} \right\}
$$

这种情况下，我们将每一个原始约束进行替换：


$$
(Ax)_i\leq b_i \Longleftrightarrow a_i^Tx\leq b_i,\space  \tag{$C_i$} \\
$$

$$
a_i^Tx \leq b_i \space\forall [a_i;b_i]\in \mathcal{U}_i \\
\mathcal{U}_i = \left \{ [a_i;b_i] :[A,b]\in\mathcal{U} \right\}\tag{$RC(C_i)$}
$$


如果$x$ 是问题$(C_i)$的一个鲁棒可行解，则当我们将不确定集 $\mathcal{U}_i$ 延展到它的凸包 $conv(\mathcal{U}_i)$ ，$x$ 仍为鲁棒可行解。实际上，若 $[\bar{a}_i,\bar{b}_i]\in Conv(\mathcal{U}_i)$ ，则


$$
[\bar{a}_i;\bar{b}_i]=\sum_{j=1}^J\lambda_j[a_i^j;b_i^j] \quad \lambda_j \geq0 ,\sum_j \lambda_j = 1
$$


因此，


$$
\bar{a}_i^Tx=\sum_{j=1}^J\lambda_j[a_i^j]^Tx \leq \sum_{j}\lambda _jb_i^j=\bar{b}_i
$$

------------------

**1.3 Tractability of Robust Counterparts**

**1.3.1 The strategy**

我们的策略如下所示：

- 首先，我们将问题限制在具有确定目标的不确定LO问题下
- 其次，我们所需要的是一个具有单一的不确定线性约束的RC的”可解“表示，也就是说，这样一个RC的等价表示是一个显式（且简短的）有效可验证的凸不等式系统
- 对我们不确定问题中的每一条约束都确定一个这样的表示，集成在一起，我们就能够将该问题的RC重新表示为在有限个显式凸约束的系统下最小化原线性目标的问题，因此是个可解问题

> Def 1.3.1 A set $X^+ \subset R_x^n \times R_u^k$  is said to represent a set $X \subset R_x^n$ , if the projection of $X^+$ onto the space of $x$-variables is exactly $X$, i.e., $x \in X$ if and only if there exists $u \in R_u^k$ such that $(x,u)\in X^+$:
> $$
> X =\{x:\exist u:(x,u)\in X^+\}
> $$

举个例子，$x_1+x_2 \leq 1,x_1-x_2 \leq 1,-x_1+x_2 \leq 1,-x_1-x_2 \leq 1$可以转化为$|x_1|+|x_2|\leq 1$，二者都表示了相同的可行域。而$-u_1 \leq x_1 \leq u_1, -u_2 \leq x_2 \leq u_2, u_1+u_2 \leq 1$同样可以表示该可行域。而第二种表述和第三种表述位于不同维度的空间中，因此不能看作相等。

假定我们给定优化问题：


$$
\min_x \left \{ f(x) \space s.t. \space x \space satisfies \space \mathcal{S}_i, i = 1,...,m\right \} \tag{P}
$$


$S_i$ 为变量 $x$ 的约束系统，将其扩展到 $S_i^+$ 上


$$
\min_{x,v^1,...,v^m} \left \{ f(x) \space s.t. \space (x,v^i) \space satisfies \space \mathcal{S}_i^+, i = 1,...,m\right \} \tag{$P^+$}
$$


