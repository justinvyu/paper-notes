# Soft Actor-Critic Algorithms and Applications

[Link to paper](https://arxiv.org/pdf/1812.05905.pdf)



## 1. Introduction

- Model-free deep-RL methods are known for having (1) high sample inefficiency (sometimes requiring millions of environment steps or more), (2) drastically different behavior with different hyperparamters (learning rates, exploration constants).
  - Poor sample efficiency in on-policy algorithms (TRPO, PPO, A3C): new samples need to be collected every time the policy gets updated.
    - Hard to scale to more complex tasks, since the number of samples needed for each gradient step increases along with complexity.
  - Off-policy learning focuses on using past experience; Q-learning uses function approximation to learn the Q-function from past data (stored in a replay buffer) that doesn't have to be collected from the latest policy.
    - Downsides are the increased variance in these methods, since returns vary from rollout to rollout. Even harder with continuous state/action spaces, since choosing the action that maximizes reward is not possible with an infinite number of actions (thus an actor network is trained to do this in DDPG).
- SAC is based on a maximum entropy framework
  - This paper introduces a method for gradient-based temperature tuning, where temperature is the scaling factor (tradeoff between maximizing entropy vs. the actual reward signal).
  - Finding a good temperature is pretty task-dependent, and choosing bad value can "drastically degrade performance."



## 2. Related Work

- Maximum entropy RL uses a generalized version of the RL objective (the original objective is the same as setting the temperature to 0).
  - Maximum entropy objective provides improvement in exploration and robustness