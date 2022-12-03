---
layout: post
title: "<读书笔记> IP Chapter 7 Branch and Bound"
date:   2022-11-29
tags: [读书笔记,B&B,B&C]
comments: true
author: Zhang Yue
---

















本笔记为阅读《Integer Programming》中的第7章 Branch and Bound时所作。

------------------------------------------

**7.1 Divide and Conquer**

考虑问题：


$$
Z=\max\{cx:x\in S\}
$$


我们需要将该问题分解为一系列更容易求解的小问题。

> **Proposition 7.1** Let $S=S_1 \cup ...\cup S_K$ be a decomposition of $S$ into smaller sets, and let $Z^k=\max \{cx:x\in S_k\}$ for $k = 1,...,K$. Then $Z = \max_{k} Z^k$ 