# Variational Inverse Control with Events: A General Framework for Data-Driven Reward Definition

[Link to paper](https://arxiv.org/pdf/1805.11686.pdf)



## 1. Introduction

- RL algorithms' success depends heavily on the reward function, but it is difficult to design these rewards. For example, a vacuum agent could continuously collect rewards by dumping out garbage and picking it up again. A badly designed reward function actually encourages this behavior.
  - Sometimes, even difficult to come up with a reward function at all, in high dimensional settings.
- In the past, imitation learning and inverse reinforcement learning have been used to mimic an expert's behavior.
  - Cons: this requires expert demonstrations to show how exactly the task should be solved.
- This paper introduces VICE, generalizing inverse reinforcement learning so that the only thing needed form the expert is what (usually an image) the desired outcome is. 



## 2. Related Work

- **Look into:** connection between control and inference, maximum entropy learning objective



## 3. Preliminaries

### Controls as Inference

- Incorporate an additional variable $$e_t$$ that depends on state (and action): this is the "goal", and in the case of the paper, is a binary value, where 1 means that the goal has been reached.
- We can model a conditional probability for this variable $$p(e_t | s_t, a_t)$$, and we can calculate the probability of the goal being reached, given a reward function: $$p(e_t = 1 | s_t, a_t)=e^{r(s_t, a_t)}$$
  - Requires rewards to be negative to satisfy the definition of a probability.
  - The probability of an action happening in unlikely/low reward regions drops off exponentially, rather than being 0.
- Define a "backward message" as the probability of achieving the goal from time t to T: $$\beta(s_t, a_t) = p(e_{t:T} = 1 | s_t, a_t)$$
  - Using this, we can frame it in an RL context with $$Q(s_t, a_t) = log \beta(s_t, a_t)$$, which can be expanded as: $$Q(s_t, a_t) = r(s_t, a_t) + log E[e^{V(s_{t+1})}]$$
  - We can also define the value function: $$V(s_t) = log \beta(s_t) = log (p(e_{t:T} = 1 | s_t))$$, which can be expanded as: $$V(s_t) = log E_a[e^{Q(s_t, a)}]$$. In the paper, $$V(s_t) = log \int_{a \in A} e^{Q(s_t, a)} da $$.
    - **Question**: Is there a difference here? Definition of expectation?
  - These can be considered "soft" backup equations for two reasons:
    - (1) The value function is a "soft maximum" over actions, in the sense that for large values of Q, the largest terms (with corresponding actions) will dominate. For cases where many actions have equal magnitude Q-values, the value is more of an average rather than ane xact maximum.
    - (2) The q-value function is a "soft maximum" over next states instead of an expectation. Optimistic with respect to the system dynamics, encouraging risk-taking behavior to get to good states (which might have low probability) rather than good/ok states (with high probability).



## 4. Event-based Control

- 3 types of queries:
  - ALL query: $$p(\tau | e_{1:T} = 1)$$, the event should happen at each timestep.
  - AT query: $$p(\tau | e_{t*} = 1)$$, the event should happen at a specific time $$t*$$.
  - ANY query: $$p(\tau | e_1 = 1 \text{ or } e_2=1 ...)$$, the event should happen on at least one time step during each trial.



## 5. Learning event probabilities from data

- **Question**: Confused as to what the discriminator and generator is in this case.

![vice-algorithm](/Users/justinvyu/Developer/paper-notes/reinforcement-learning/images/vice/vice-algorithm.png)



## GANs

Discriminator: $$D(x)=P(x \sim p_{data})$$

We define its loss function to be: $$\mathcal{L}_D(x) = -E_{x \sim p_{data}}[logD(x)] - E_{x \sim p_{generated}}[log(1 - D(x))]$$

- When the discriminator outputs a high probability of the observation being real ($$D(x) \approx 1$$), the loss will be close to 0. When $$D(x) \approx 0$$, the loss will approach positive infinity.

The GAN training procedure:

- Minimax game between a generator (G) and a discriminator (D)
  - $$\min_{G} \max_{D} - \mathcal{L}_D = \min_{G} \max_{D} (E_{x \sim p_{data}}[logD(x)] + E_{x \sim p_{generated}}[log(1 - D(x))])$$
  - The discriminator maximizes the negative objective for a given G, and the generator minimizes the negative objective for the given D, taking turns when training.
  - The discriminator can be trained 

## RL & GANs

Idea: inverse reinforcement learning is similar to the GAN problem.

The discriminator, which can classify a trajectory or a state/state-action, is playing against a policy (the generator), which can generate trajectories and state/state-actions.