# Diagnosing Bottlenecks in Deep Q-learning Algorithms

[Link to the paper]()



## 1. Introduction

- Q-learning algorithms are powerful because it's possible to reduce sample complexity by learning off-policy
- Tabular Q-learning is convergent (the Bellman updates can be shown as a contractive update), but the continuous case has no such guarauantees
- This paper focuses on answering the following questions:
  - *What is the effect of function approximation on convergence?*
  - *What is the effect of sampling error and overfitting?*
  - What is the effect of distribution shift and a moving target? This refers to the different data distribution under which the samples in the replay buffer are draw from
  - *What is the best sampling or weighting distribution?*



## 2. Preliminaries

- Define the Bellman backup operator $$T$$:
  - $$(TQ)(s, a) =  R(s, a) + \gamma E_{s' \sim T}[V(s')]$$
  - $$V(s) = max_{a'} Q(s, a')$$
- Q-learning iterates the Bellman backup $$Q^{t+1} \leftarrow TQ^t$$, and is a $$\gamma$$-contraction in the L-$$\infin$$ norm, and converges as the optimal Q function Q*.
  - The L-$$\infin$$ norm is defined as:
- In nontabular cases, Fitted Q-iteration (FQI) is used to approximate the Q-function over a continuous range of values, performing an L2-projection $$\Pi_\mu$$, such that $$Q^{t+1} \leftarrow \Pi_\mu (TQ^t)$$
  - This has no convergence gaurantees though, since there is a mismatch in the L2-projection and the L-$$\infin$$ contraction.



## 3. Experimental Setup

- Introduce 3 algorithms to use for testing:
  - Exact-FQI: computes packup and projection over all collected state-action tuples
  - Sampled-FQI: computes approximate Bellman error taking a Monte-Carlo estimate from a sampling distribution $$\mu$$
  - Replay-FQI: uses a replay buffer that saves tuples to approximate Bellman error
-  Different sampling distributions used:
  - Unif(s, a): uniform weights over state-action space
  - Random(s, a): "state-action marginal induced by executing uniformly random actions"
  - Prioritized(s, a): weights based on Bellman errors $$|Q(s, a) - TQ(s, a)|$$
  - Replay(s, a), Replay10(s, a): averaged state-action marginal over all policies/last 10



## 4. Function Approximation and Convergence

- 



## 5. Sampling Error and Overfitting

### 5.2 Quantifying Overfitting

- Test this by changing the number of samples 