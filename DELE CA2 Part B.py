#!/usr/bin/env python
# coding: utf-8

# # **Part B – Deep Q-Network for Pendulum Control Under Different Gravity Settings**
# 
# ### **Background**
# 
# Reinforcement learning allows an agent to learn a control policy through repeated interaction with an environment.
# 
# In the Pendulum environment, the agent must apply torque to move and stabilise the pendulum near the upright position. The observation consists of the pendulum angle representation and angular velocity, while the action determines the applied torque.
# 
# The standard Pendulum action space is continuous, with torque values ranging from `−2.0` to `2.0`. Since Deep Q-Networks (DQN) require a discrete action space, the continuous torque range must first be converted into a finite set of torque actions.
# 
# The quality of the learned policy is influenced not only by the DQN configuration but also by the physical dynamics of the environment. Changing gravity alters how the pendulum responds to the same torque action.
# 
# This investigation therefore evaluates DQN control under four gravity settings:
# 
# | **Environment** | **Gravity** | **Purpose** |
# |---|---:|---|
# | Default | `g = 10` | Standard Pendulum environment and main baseline |
# | Free-fall | `g = 0` | Investigate control without gravitational force |
# | Anti-gravity | `g = −10` | Investigate control under reversed gravitational dynamics |
# | Supergravity | `g = 15` | Investigate control under stronger gravitational force |
# 
# The investigation also examines whether the baseline DQN can be improved systematically through hyperparameter tuning and whether the resulting performance is reliable across repeated trials.

# ### **Problem Statement**
# 
# The main objective is to develop a discretised-action DQN that can learn meaningful Pendulum control under different gravity conditions.
# 
# The investigation focuses on three aspects:
# 
# 1. how gravity affects DQN learning and final control performance;
# 2. whether systematic hyperparameter tuning can improve the default-gravity DQN;
# 3. whether the selected configuration remains reliable when training randomness and action-space resolution are considered.

# ### **Research Questions**
# 
# This investigation addresses the following research questions:
# 
# 1. How does changing gravity affect the learning performance and control behaviour of a DQN agent in the Pendulum environment?
# 
# 2. Can systematic hyperparameter tuning improve the performance, learning speed, and stability of the default-gravity DQN?
# 
# 3. Does the selected epsilon-decay configuration remain reliable across different training seeds, and does reducing the action-space resolution provide any clear advantage?
# 
# 4. Can one common DQN configuration provide effective control across all four gravity environments, or do some gravity settings remain substantially more difficult than others?

# ### **Evaluation Criteria**
# 
# A consistent greedy evaluation procedure is used throughout. Because RL performance varies across runs, the DQN is assessed with multiple criteria rather than one metric.
# 
# The **primary criterion** is **mean greedy evaluation reward**, where values closer to `0` indicate better Pendulum control. Supporting criteria:
# 
# - **evaluation standard deviation** — consistency across episodes
# - **episodes to reach the −300 moving-average threshold** — learning speed
# - **late-training stability** — reward fluctuation near the end of training
# - **training reward and loss** — diagnostic only, since training uses epsilon-greedy exploration
# 
# For the **gravity comparison**, additional physical-control metrics interpret how environmental dynamics affect learned behaviour: mean absolute angle, mean absolute angular velocity, mean absolute torque, stable-upright percentage, and extreme-torque usage.
# 
# For **hyperparameter tuning**, only the core reward-based criteria are used — physical-control metrics are omitted since this stage focuses on final policy performance, learning speed, and training stability under controlled settings.

# ### **Hypotheses**
# 
# Before running the experiments, the following hypotheses are proposed:
# 
# 1. **Gravity Effect** — Changing gravity is expected to affect both learning difficulty and the control strategy learned by the DQN. Free-fall and anti-gravity may reduce some of the difficulty associated with lifting the pendulum against normal gravity, while supergravity is expected to require stronger or more accurately timed control.
# 
# 2. **Hyperparameter Effect** — DQN performance is expected to depend on hyperparameters controlling optimisation, target-network updates, and the exploration–exploitation schedule. Changing these settings may improve learning speed or stability, although an improvement in one metric may not necessarily improve final greedy evaluation performance.
# 
# 3. **Training Variability** — Because DQN training is stochastic, performance is expected to vary across random seeds. A configuration that performs well in one run may therefore require repeated trials before it can be considered reliable.
# 
# 4. **Action Resolution** — The number of discrete torque actions is expected to influence the trade-off between control precision and learning difficulty. A finer action space may provide more precise torque choices, while a coarser action space may simplify the decision problem.

# ### **Experimental Approach**
# 
# The investigation follows a controlled experimental process:
# 
# 1. construct and verify a baseline discretised-action DQN;
# 2. train and evaluate the same baseline configuration under all four gravity settings;
# 3. analyse reward-based and physical-control performance;
# 4. systematically vary selected hyperparameters while keeping other settings fixed;
# 5. evaluate promising configurations using consistent selection criteria;
# 6. perform additional robustness checks using repeated training seeds and action-space resolution;
# 7. evaluate the final selected configuration across the gravity environments.
# 
# This approach is designed to separate the effects of gravity, hyperparameter changes, training randomness, and action-space design as clearly as possible.

# ## **1. Environment Setup**
# ### **1.1 Import Libraries**

# In[1]:


# Standard libraries
import os
import sys
import time
import random
import platform
from collections import deque, namedtuple

# Numerical and visualisation libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Reinforcement learning environment
import gym

from IPython.display import display, Image


# ### **1.2 Reproducibility**
# 
# **Objective**
# 
# Improve the reproducibility of training and enable fair comparisons across different experiments.
# 
# | Question | Decision | Reason |
# |---|---|---|
# | Why use seeds? | Control random initialisation, exploration and replay sampling | DQN results can vary between runs |
# | What is reset? | Python, NumPy, TensorFlow, environment and action space | Makes same-seed comparisons more controlled |
# | Why use one evaluation seed? | All models face the same evaluation initial-state sequence | Prevents evaluation conditions from changing between models |
# 
# **Expectation:** repeated runs may still differ because neural-network training is stochastic and complete hardware-level determinism is not guaranteed.

# In[2]:


# Reproducibility
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Display package and system information
print("Python version:     ", sys.version)
print("Platform:           ", platform.platform())
print("NumPy version:      ", np.__version__)
print("TensorFlow version: ", tf.__version__)
print("Gym version:        ", gym.__version__)
print("GPU devices:        ", tf.config.list_physical_devices("GPU"))


# ### **1.3 Pendulum Environment Inspection**
# 
# Before developing the DQN agent, the `Pendulum-v0` environment from Gym `0.17.3` is inspected to understand the information available to the agent and the requirements of the control task.
# 
# The following components are examined:
# 
# | Component | Purpose of inspection |
# |---|---|
# | Observation space | Determine the state information and input size required by the Q-network |
# | Action space | Identify the available torque range and whether it can be used directly by DQN |
# | Reward function | Understand how angle, angular velocity, and torque affect the agent’s performance |
# 
# This inspection will determine what modifications are required before a standard DQN can be applied to the Pendulum environment.

# In[3]:


# Create and inspect the default Pendulum environment

ENV_NAME = "Pendulum-v0"

env = gym.make(ENV_NAME)
env.seed(SEED)
env.action_space.seed(SEED)

# Convert the state to float32 for consistency with TensorFlow
initial_state = env.reset().astype(np.float32)

print("Environment name:      ", ENV_NAME)
print("Observation space:     ", env.observation_space)
print("Action space:          ", env.action_space)
print("Maximum episode steps: ", env.spec.max_episode_steps)
print("Initial state:         ", initial_state)
print("Initial state shape:   ", initial_state.shape)
print("State data type:       ", initial_state.dtype)


# #### **Analysis**
# 
# | **Item** | **Observation** |
# |----------|-----------------|
# | State Space | The environment provides a **3-dimensional continuous state vector** suitable as input to the DQN. |
# | Action Space | The environment uses a **continuous torque action**, which must be discretised before applying a standard DQN. |
# | Episode Length | Each episode lasts for a maximum of **200 time steps**, providing a consistent training duration. |

# ### **1.3.1 Observation Space**
# 
# **Observation Space**
# 
# The environment returns a **3-dimensional continuous state vector** consisting of:
# 
# - $\cos(\theta)$
# - $\sin(\theta)$
# - Angular velocity ($\dot{\theta}$)
# 
# Using $\cos(\theta)$ and $\sin(\theta)$ instead of the angle itself avoids discontinuities when the angle wraps between $-\pi$ and $\pi$, providing a smooth representation for the neural network.

# In[4]:


print("Observation Space")
print("-----------------")
print(env.observation_space)

print("\nShape :", env.observation_space.shape)
print("Lower :", env.observation_space.low)
print("Upper :", env.observation_space.high)
print("Dtype :", env.observation_space.dtype)


# ### **1.3.2 Action Space**
# 
# **Objective**
# 
# Understand the action space and why discretisation is required for DQN.
# 
# **Action Space**
# 
# - The environment accepts a **single continuous torque** value within the range **[-2, 2]**.
# - Since a standard DQN predicts Q-values for a **finite set of actions**, the continuous torque range cannot be used directly.
# - Therefore, the action space will be discretised before training.

# In[5]:


print("Action Space")
print("------------")
print(env.action_space)

print("\nShape :", env.action_space.shape)
print("Lower :", env.action_space.low)
print("Upper :", env.action_space.high)
print("Dtype :", env.action_space.dtype)


# ### **1.3.3 Reward Function**
# 
# **Objective**
# 
# Understand how the environment evaluates upright position, movement, and control effort.
# 
# At every timestep, the Pendulum reward is:
# 
# $$
# r_t = -\left(\theta_t^2 + 0.1\dot{\theta}_t^2 + 0.001u_t^2\right)
# $$
# 
# where:
# 
# | Term | Meaning |
# |---|---|
# | $\theta_t$ | Angle from the upright position |
# | $\dot{\theta}_t$ | Angular velocity |
# | $u_t$ | Applied torque |
# 
# The reward contains three penalties:
# 
# 1. **Angle penalty:** penalises the pendulum for being far from upright.
# 2. **Angular-velocity penalty:** penalises excessive movement.
# 3. **Torque penalty:** penalises excessive control effort.
# 
# The maximum theoretical reward for one timestep is `0`, which occurs when:
# 
# $$
# \theta_t = 0,\qquad \dot{\theta}_t = 0,\qquad u_t = 0
# $$
# 
# Therefore, a reward closer to `0` indicates better control.
# 
# Although Gym displays the general environment reward range as `(-∞, ∞)`, the Pendulum reward equation is always non-positive because it is the negative of the three penalty terms.
# 
# The total episode reward is the sum of rewards across a maximum of 200 timesteps. An episode reward of `0` would therefore require perfect upright, stationary, and torque-free control throughout the entire episode.
# 
# In practice, the objective is to obtain episode rewards that are progressively closer to `0`, rather than assume that perfect performance will be achieved.

# In[6]:


print("Reward Range")
print("------------")
print(env.reward_range)
print("Maximum Episode Steps:", env.spec.max_episode_steps)


# ### **1.3.4 Environment Interaction**
# 
# **Objective**
# 
# Verify that the Pendulum environment correctly processes an action and returns the information required for DQN training.
# 
# **Method**
# 
# A randomly sampled action from the continuous action space is applied to the environment. The returned next state, reward, termination status, and additional information are inspected.

# In[7]:


# Sample one continuous action from the original environment
sample_action = env.action_space.sample()

# Apply the action for one timestep
next_state, reward, done, info = env.step(sample_action)

# Convert the returned state to float32 for TensorFlow compatibility
next_state = next_state.astype(np.float32)

print("Sample action:    ", sample_action)
print("Next state:       ", next_state)
print("Next-state type:  ", next_state.dtype)
print("Reward:           ", reward)
print("Done:             ", done)
print("Additional info:  ", info)

env.close()


# #### **Analysis**
# 
# | Item | Observation |
# |---|---|
# | Environment step | The sampled torque was successfully applied to the environment. |
# | State transition | A valid three-dimensional next state was returned. |
# | Reward | A numerical reward was returned according to the Pendulum reward function. |
# | Episode status | A valid Boolean `done` value was returned using the Gym `0.17.3` interface. |
# 
# The environment interaction format is therefore suitable for storing DQN transitions in the form $(s, a, r, s', d)$.

# ### **1.3.5 Physical Effect of Gravity and Torque**
# 
# The four environments use the same state representation, neural-network architecture, and available torque actions.
# 
# The gravity setting does not change the action space. All agents can still choose from the same torque range of `−2.0` to `2.0`.
# 
# However, gravity changes how the pendulum responds to each torque action:
# 
# - Default gravity pulls the pendulum downward.
# - Free-fall removes the gravitational force.
# - Anti-gravity reverses the direction of gravity.
# - Supergravity applies a stronger downward force.
# 
# The following diagram illustrates the physical difference between the four environments.

# In[8]:


from matplotlib.patches import FancyArrowPatch
import numpy as np
import matplotlib.pyplot as plt

gravity_settings = [
    ("Default Gravity", 10),
    ("Free-fall", 0),
    ("Anti-gravity", -10),
    ("Supergravity", 15)
]

# Pendulum position used only for illustration
theta = np.radians(40)

pivot_x, pivot_y = 0, 0
bob_x = np.sin(theta)
bob_y = np.cos(theta)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, (label, gravity) in zip(axes, gravity_settings):

    # Pivot, rod and pendulum bob
    ax.scatter(pivot_x, pivot_y, s=80)
    ax.plot(
        [pivot_x, bob_x],
        [pivot_y, bob_y],
        linewidth=3
    )
    ax.scatter(bob_x, bob_y, s=250)

    # Upright reference line
    ax.plot(
        [pivot_x, pivot_x],
        [pivot_y, 1.2],
        linestyle="--",
        alpha=0.5
    )

    # Gravity direction
    if gravity > 0:
        arrow_length = 0.35 * gravity / 10

        ax.arrow(
            bob_x,
            bob_y,
            0,
            -arrow_length,
            width=0.015,
            head_width=0.10,
            length_includes_head=True
        )

        ax.text(
            bob_x + 0.08,
            bob_y - arrow_length / 2,
            f"Gravity\ng = {gravity}"
        )

    elif gravity < 0:
        arrow_length = 0.35 * abs(gravity) / 10

        ax.arrow(
            bob_x,
            bob_y,
            0,
            arrow_length,
            width=0.015,
            head_width=0.10,
            length_includes_head=True
        )

        ax.text(
            bob_x + 0.08,
            bob_y + arrow_length / 2,
            f"Reversed gravity\ng = {gravity}"
        )

    else:
        ax.text(
            bob_x + 0.08,
            bob_y,
            "No gravitational\nforce"
        )

    # Applied control torque
    torque_arrow = FancyArrowPatch(
        (-0.25, 0.15),
        (0.25, 0.15),
        connectionstyle="arc3,rad=-0.8",
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=2
    )

    ax.add_patch(torque_arrow)
    ax.text(-0.30, -0.20, "Agent torque\nu ∈ [−2, 2]")

    ax.set_title(f"{label}\n(g = {gravity})")
    ax.set_xlim(-0.7, 1.3)
    ax.set_ylim(-0.4, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

plt.suptitle(
    "Effect of Gravity and Agent-Applied Torque",
    fontsize=14
)

plt.tight_layout()
plt.show()


# #### **Analysis**
# 
# The available torque range remains unchanged across the four environments. Therefore, later performance differences are not caused by different action limits.
# 
# Instead, gravity changes the physical response produced by each selected torque:
# 
# - Under **default gravity**, torque must counteract the normal downward gravitational influence.
# - Under **free-fall**, the agent controls angular movement without a gravitational force.
# - Under **anti-gravity**, the gravitational direction is reversed, changing the natural stability of the pendulum.
# - Under **supergravity**, the stronger downward influence may make poorly timed actions more difficult to correct.
# 
# Therefore, the same torque value may produce different state transitions under different gravity settings. Separate agents will later be trained under each setting using the same baseline configuration for a controlled comparison.

# ## **2. Action-Space Discretisation**
# 
# ### **2.1 Define the Discrete Torque Actions**
# 
# The original Pendulum environment provides a continuous torque range from `−2.0` to `2.0`. However, a standard DQN produces one Q-value for each available discrete action and therefore cannot directly select from a continuous action space.
# 
# For the baseline model, the continuous torque range is discretised into nine evenly spaced actions:
# 
# $$
# [-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0]
# $$
# 
# Nine actions are initially selected because they:
# 
# - cover the complete permitted torque range;
# - include both positive and negative torque;
# - include zero torque;
# - provide intermediate torque values for smaller corrective actions; and
# - maintain a reasonably small DQN output layer.
# 
# This represents an initial balance between control precision and learning complexity.

# In[9]:


# Discretise the continuous torque range

NUM_DISCRETE_ACTIONS = 9

ACTION_MIN = float(env.action_space.low[0])
ACTION_MAX = float(env.action_space.high[0])

DISCRETE_ACTIONS = np.linspace(
    ACTION_MIN,
    ACTION_MAX,
    NUM_DISCRETE_ACTIONS,
    dtype=np.float32
)

print("Continuous action minimum:", ACTION_MIN)
print("Continuous action maximum:", ACTION_MAX)
print("Number of discrete actions:", NUM_DISCRETE_ACTIONS)
print("Discrete torque values:    ", DISCRETE_ACTIONS)


# ### **2.2 Visualisation of the Discretised Torque Values**
# 
# The following visualisation shows the mapping between the nine action indices and their corresponding torque values across the complete continuous range.

# In[10]:


# Visualise the discretised torque values
plt.figure(figsize=(9, 2.5))

plt.scatter(
    DISCRETE_ACTIONS,
    np.zeros(NUM_DISCRETE_ACTIONS),
    s=100
)

for action_index, torque in enumerate(DISCRETE_ACTIONS):
    plt.text(
        torque,
        0.03,
        str(action_index),
        ha="center"
    )

plt.axhline(0, linewidth=0.8)
plt.yticks([])
plt.xticks(DISCRETE_ACTIONS)
plt.xlabel("Torque")
plt.title("Nine Discrete Actions Across the Continuous Torque Range")
plt.grid(axis="x", linestyle="--", alpha=0.5)

plt.show()


# ### **2.3 Map Action Indices to Torque Values**
# 
# **Objectives**
# 
# * Map each DQN action index (**0–8**) to a corresponding torque value.
# * Convert the selected action into a valid Gym action array.
# * Ensure the agent can interact correctly with the Pendulum environment.

# In[11]:


# Convert a DQN action index into a Gym torque action
def action_index_to_torque(action_index, action_values=None):
    """
    Map a discrete action index to a torque array accepted by Gym.

    action_values may be supplied for experiments with a different
    action-space resolution. When omitted, the baseline 9-action
    discretisation is used.
    """
    if action_values is None:
        action_values = DISCRETE_ACTIONS

    action_values = np.asarray(action_values, dtype=np.float32).reshape(-1)

    if not 0 <= action_index < len(action_values):
        raise ValueError(
            f"Action index must be between 0 and "
            f"{len(action_values) - 1}. "
            f"Received: {action_index}"
        )

    return np.asarray(
        [action_values[action_index]],
        dtype=np.float32
    )


# Display the baseline action-index mapping
for action_index in range(len(DISCRETE_ACTIONS)):
    torque_action = action_index_to_torque(action_index)
    print(
        f"Action index {action_index} "
        f"-> torque {torque_action[0]:.2f}"
    )


# ### **2.4 Verification of Discrete Action Mapping**
# 
# **Objective**
# 
# Verify that a discrete DQN action index is correctly converted into a torque array accepted by the Pendulum environment.
# 
# **Method**
# 
# The action index corresponding to zero torque is passed to `action_index_to_torque()`. The resulting torque array is then applied to the environment, and the returned transition is inspected.

# In[146]:


# Test one environment step using Gym 0.17.3

test_env = gym.make(ENV_NAME)
test_env.seed(SEED)
test_env.action_space.seed(SEED)

test_state = test_env.reset().astype(np.float32)

# Select the action index corresponding to zero torque
test_action_index = 4

# Convert the action index using the mapping function
test_torque = action_index_to_torque(test_action_index)

next_state, reward, done, info = test_env.step(test_torque)
next_state = next_state.astype(np.float32)

print("Initial state:  ", test_state)
print("Action index:   ", test_action_index)
print("Applied torque: ", test_torque)
print("Next state:     ", next_state)
print("Reward:         ", reward)
print("Done:           ", done)

test_env.close()


# **Result**
# 
# Action index `4` was correctly mapped to a torque of `[0.]`. The environment accepted this action and returned a valid next state, reward, and `done` value.
# 
# This confirms that the discrete action mapping is compatible with the Pendulum environment and can be used by the DQN agent.

# ## **3. DQN Agent Development**

# ### **3.1 Baseline DQN Hyperparameters**
# 
# **Objective**
# 
# Define the initial DQN configuration used to establish the baseline. These values remain constant during the first gravity comparison so that gravity is the only environmental factor being changed.
# 
# | Hyperparameter | Baseline value | Rationale | Possible limitation |
# |---|---:|---|---|
# | Training episodes | `500` | Provides a fixed computational budget for all comparisons | May not be sufficient for complete convergence |
# | Batch size | `64` | Uses multiple replay samples for each network update | A very large batch may slow adaptation |
# | Discount factor, $\gamma$ | `0.99` | Gives importance to long-term Pendulum control | A high value may increase target variance |
# | Learning rate | `0.00025` | Provides relatively conservative network updates | Too small may learn slowly; too large may destabilise learning |
# | Replay capacity | `50,000` | Stores varied past transitions | Older experiences may eventually be replaced |
# | Replay warm-up | `2,000` | Prevents training from beginning with very limited experience | Delays the first network updates |
# | Initial epsilon | `1.0` | Begins with full exploration | Early rewards are expected to be poor |
# | Minimum epsilon | `0.01` | Maintains a small possibility of exploration | The minimum may not be reached within 500 episodes |
# | Epsilon decay | `0.995` | Gradually shifts from exploration to exploitation | Exploration may remain relatively high late in training |
# | Target update frequency | Every `10` episodes | Keeps targets stable while periodically incorporating new knowledge | A different interval may produce better learning |
# 
# These values form the baseline rather than a proven optimal configuration. Selected hyperparameters are tested systematically later.

# In[13]:


NUM_EPISODES = 500
BATCH_SIZE = 64
GAMMA = 0.99

LEARNING_RATE = 0.00025

MEMORY_SIZE = 50_000
MIN_MEMORY_SIZE = 2_000

EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

TARGET_UPDATE_FREQUENCY = 10

print("Number of episodes:       ", NUM_EPISODES)
print("Batch size:                ", BATCH_SIZE)
print("Discount factor:           ", GAMMA)
print("Learning rate:             ", LEARNING_RATE)
print("Replay memory capacity:    ", MEMORY_SIZE)
print("Minimum replay memory:     ", MIN_MEMORY_SIZE)
print("Initial epsilon:           ", EPSILON_START)
print("Minimum epsilon:           ", EPSILON_MIN)
print("Epsilon decay rate:        ", EPSILON_DECAY)
print("Target update frequency:   ", TARGET_UPDATE_FREQUENCY)


# ### **3.2 Replay Memory**
# 
# **Objective**
# 
# Store past environment transitions and sample random mini-batches for DQN training.
# 
# Each transition is represented as:
# 
# $$
# (s_t,a_t,r_t,s_{t+1},d_t)
# $$
# 
# where:
# 
# - $s_t$ is the current state;
# - $a_t$ is the selected discrete action index;
# - $r_t$ is the received reward;
# - $s_{t+1}$ is the next state; and
# - $d_t$ indicates whether the episode has ended.
# 
# Consecutive environment transitions are highly related to one another. Randomly sampling transitions from replay memory reduces this temporal correlation and allows past experiences to be reused across multiple network updates.
# 
# Training begins only after the replay memory contains at least `2,000` transitions.

# In[14]:


Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "done"]
)


class ReplayMemory:
    """Store and sample transitions for DQN training."""

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):

        state = np.asarray(
            state,
            dtype=np.float32
        ).reshape(-1)

        next_state = np.asarray(
            next_state,
            dtype=np.float32
        ).reshape(-1)

        action = int(action)

        reward = float(
            np.asarray(reward).reshape(-1)[0]
        )

        done = float(done)

        transition = Transition(
            state,
            action,
            reward,
            next_state,
            done
        )

        self.memory.append(transition)

    def sample(self, batch_size):

        batch = random.sample(
            self.memory,
            batch_size
        )

        states = np.stack(
            [transition.state for transition in batch],
            axis=0
        ).astype(np.float32)

        actions = np.asarray(
            [transition.action for transition in batch],
            dtype=np.int32
        )

        rewards = np.asarray(
            [transition.reward for transition in batch],
            dtype=np.float32
        )

        next_states = np.stack(
            [transition.next_state for transition in batch],
            axis=0
        ).astype(np.float32)

        dones = np.asarray(
            [transition.done for transition in batch],
            dtype=np.float32
        )

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


# #### **3.2.1 Verify Replay Memory**
# 
# **Objective**
# 
# Verify that transitions can be stored and randomly sampled in the format required for DQN training.
# 
# **Method**
# 
# Several environment transitions are stored in the replay memory before a random mini-batch is sampled and inspected.

# In[15]:


# Verify replay-memory storage and sampling
test_memory = ReplayMemory(capacity=10)

test_env = gym.make(ENV_NAME)
test_env.seed(SEED)
test_env.action_space.seed(SEED)

state = test_env.reset().astype(np.float32)

for step in range(6):
    action_index = step % NUM_DISCRETE_ACTIONS
    torque = action_index_to_torque(action_index)

    # Pendulum-v0 expects an action array with shape (1,)
    torque = np.asarray(torque, dtype=np.float32).reshape(1)

    next_state, reward, done, info = test_env.step(torque)
    next_state = next_state.astype(np.float32)

    test_memory.store(
        state=state,
        action=action_index,
        reward=reward,
        next_state=next_state,
        done=done
    )

    state = next_state

    # Reset only if the episode ends
    if done:
        state = test_env.reset().astype(np.float32)

print("Stored transitions:", len(test_memory))

states, actions, rewards, next_states, dones = test_memory.sample(
    batch_size=4
)

print("\nSampled batch shapes")
print("States:      ", states.shape, states.dtype)
print("Actions:     ", actions.shape, actions.dtype)
print("Rewards:     ", rewards.shape, rewards.dtype)
print("Next states: ", next_states.shape, next_states.dtype)
print("Dones:       ", dones.shape, dones.dtype)

print("\nSampled actions:", actions)
print("Sampled rewards:", rewards)
print("Sampled dones:", dones)

test_env.close()


# #### **Analysis**
# 
# | **Item** | **Observation** |
# |----------|-----------------|
# | Transition Storage | All transitions were stored successfully. |
# | Mini-batch Sampling | The requested mini-batch was sampled correctly. |
# | Data Format | States, actions, rewards, next states, and termination flags had the expected shapes and data types. |
# | Verification | Replay memory was correctly configured and ready for DQN training. |

# ### **3.3 Deep Q-Network Architecture**
# 
# **Objective**
# 
# Build a neural network to approximate the Q-function and predict the Q-value for each discretised action.
# 
# **Network Structure**
# 
# | **Component** | **Configuration** |
# |---------------|-------------------|
# | **Input** | 3 Pendulum state features |
# | **Hidden Layer 1** | Dense (128), ReLU |
# | **Hidden Layer 2** | Dense (128), ReLU |
# | **Output** | Dense (9), Linear |
# 
# **Baseline Architecture**
# 
# ```text
# Input (3)
#     ↓
# Dense (128, ReLU)
#     ↓
# Dense (128, ReLU)
#     ↓
# Output (9, Linear)
# ```
# 
# **Design Decisions**
# 
# | **Design Choice** | **Reason** |
# |-------------------|------------|
# | Two hidden layers | Provide sufficient learning capacity while remaining computationally efficient. |
# | 128 neurons | Balance model capacity and computational cost. |
# | ReLU activation | Learns nonlinear relationships efficiently. |
# | Linear output | Produces unrestricted Q-value estimates. |
# | Huber loss | Improves training stability by reducing sensitivity to large TD errors. |
# | Adam optimiser | Provides efficient adaptive gradient updates. |

# In[16]:


# Deep Q-Network Architecture
STATE_SIZE = initial_state.shape[0]
ACTION_SIZE = NUM_DISCRETE_ACTIONS


def build_q_network(
    state_size=STATE_SIZE,
    action_size=ACTION_SIZE,
    learning_rate=LEARNING_RATE
):
    """Build and compile the DQN Q-network."""

    model = keras.Sequential([
        layers.Input(shape=(state_size,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(action_size, activation="linear")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss=keras.losses.Huber()
    )

    return model


# Build one temporary model to inspect the architecture.
architecture_model = build_q_network()
architecture_model.summary()


# ### **3.4 Online and Target Networks**
# 
# **Objective**
# 
# Create and initialise the online and target Q-networks to provide stable learning during DQN training.
# 
# **Network Roles**
# 
# | **Network** | **Purpose** |
# |-------------|-------------|
# | **Online Network** | Selects actions and updates Q-values during training. |
# | **Target Network** | Computes stable target Q-values for the Bellman update. |
# | **Synchronisation** | Periodically copies the online network's weights to the target network, keeping the learning target stable. |
# 
# Both networks use the architecture described in **Section 3.3**, with identical initial weights before training begins.
# 
# The online network is updated during every available training step. The target network is held fixed between synchronisations and is updated by copying the online-network weights every 10 episodes. This reduces rapid changes in the Bellman targets and improves training stability.

# In[17]:


# 3.4 Online and Target Networks

# Temporary networks used only to verify target-network initialisation.
online_network = build_q_network()
target_network = build_q_network()

target_network.set_weights(
    online_network.get_weights()
)

print("Online network output shape:", online_network.output_shape)
print("Target network output shape:", target_network.output_shape)
print("Target network successfully synchronised with online network.")


# #### **3.4.1 Verify the Online and Target Networks**
# 
# **Objective**
# 
# Verify that the online and target Q-networks are correctly configured and synchronised before training.
# 
# **Method**
# 
# A sample Pendulum state is passed through both networks. The verification confirms that both networks accept a **3-dimensional** input state, produce **9 Q-values**, and initially generate identical outputs after synchronisation.

# In[18]:


sample_state = np.array(
    [[1.0, 0.0, 0.0]],
    dtype=np.float32
)

online_q_values = online_network(
    sample_state,
    training=False
).numpy()

target_q_values = target_network(
    sample_state,
    training=False
).numpy()

print("Sample state shape :", sample_state.shape)
print("Online output shape:", online_q_values.shape)
print("Target output shape:", target_q_values.shape)

print("\nOnline Q-values:")
print(online_q_values)

print("\nTarget Q-values:")
print(target_q_values)

print(
    "\nOutputs are initially identical:",
    np.allclose(online_q_values, target_q_values)
)

# Release temporary verification models before creating the actual agent.
del architecture_model, online_network, target_network


# #### **Analysis**
# 
# | **Item** | **Observation** |
# |----------|-----------------|
# | Input and Output | The networks correctly mapped an input state of **(1, 3)** to Q-values of shape **(1, 9)**. |
# | Network Synchronisation | The online and target networks initially produced identical Q-values after synchronisation. |
# | Verification | Both networks were correctly configured and ready for DQN training. |

# ### **3.5 ε-Greedy Action Selection**
# 
# The agent uses an ε-greedy policy to balance exploration and exploitation during training.
# 
# | Condition | Behaviour | Purpose |
# |---|---|---|
# | Exploration with probability $\epsilon$ | Select a random discrete action | Discover different state-action outcomes |
# | Exploitation with probability $1-\epsilon$ | Select the action with the highest predicted Q-value | Use the currently learned policy |
# | Epsilon decay | Multiply epsilon by `0.995` after each episode, towards a minimum of `0.01` | Gradually shift from exploration to exploitation |
# 
# Epsilon begins at `1.0`, meaning that the early policy is fully exploratory.

# ### **3.6 DQN Agent**
# 
# The `DQNAgent` class integrates all components required for DQN training and evaluation.
# 
# **Main Components**
# 
# | **Component** | **Purpose** |
# |---------------|-------------|
# | Replay Memory | Stores transitions for experience replay. |
# | Online Q-Network | Predicts Q-values for action selection and learning. |
# | Target Q-Network | Provides stable target Q-values during training. |
# | ε-Greedy Policy | Balances exploration and exploitation. |
# 
# **Core Methods**
# 
# | **Method** | **Purpose** |
# |------------|-------------|
# | `select_action()` | Selects an action using the ε-greedy policy. |
# | `store_transition()` | Stores a transition in replay memory. |
# | `train_step()` | Updates the online Q-network using a mini-batch. |
# | `update_target_network()` | Synchronises the target network with the online network. |
# | `decay_epsilon()` | Gradually reduces the exploration rate. |
# | `save_weights()` / `load_weights()` | Saves and loads the trained model weights. |
# 
# **Training Process**
# 
# For each environment interaction:
# 
# 1. The agent selects a discrete action using the ε-greedy policy.
# 2. The action index is converted into a torque value.
# 3. The resulting transition is stored in replay memory.
# 4. Once the warm-up requirement is met, a random mini-batch is sampled.
# 5. The target Q-value is calculated using:
# 
# $$
# y_t =
# \begin{cases}
# r_t, & \text{if the episode has ended} \\
# r_t + \gamma \max_a Q_{\text{target}}(s_{t+1},a),
# & \text{otherwise}
# \end{cases}
# $$
# 
# 6. The online network is trained to move its selected-action Q-value towards this target.
# 7. The target network is periodically synchronised with the online network.
# 8. Epsilon is reduced after each episode.
# 
# Through repeated updates, the network learns which discretised torque action is expected to produce the highest long-term reward for each Pendulum state.

# In[19]:


class DQNAgent:

    def __init__(
        self,
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        min_memory_size=MIN_MEMORY_SIZE,
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.min_memory_size = min_memory_size

        # Experience replay memory
        self.memory = ReplayMemory(memory_size)

        # Online network
        self.online_network = build_q_network(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate
        )

        # Target network
        self.target_network = build_q_network(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate
        )

        # Copy weights to target network
        self.update_target_network()

    # --------------------------------------------------------
    # Select an action using epsilon-greedy exploration
    # --------------------------------------------------------
    def select_action(self, state):

        # Exploration
        if np.random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # Convert state into a clean batch with shape (1, state_size)
        state = np.asarray(
            state,
            dtype=np.float32
        ).reshape(1, self.state_size)

        # Predict Q-values
        q_values = self.online_network(state, training=False).numpy()

        # Return the action index with the highest Q-value
        return int(np.argmax(q_values[0]))

    # --------------------------------------------------------
    # Store one experience in replay memory
    # --------------------------------------------------------
    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.store(
            state,
            action,
            reward,
            next_state,
            done
        )

    # --------------------------------------------------------
    # Train the online network
    # --------------------------------------------------------
    def train_step(self):

        if len(self.memory) < self.min_memory_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = self.memory.sample(self.batch_size)

        current_q = self.online_network(states, training=False).numpy()
        next_q = self.target_network(next_states, training=False).numpy()

        target_q = current_q.copy()

        for i in range(self.batch_size):

            if dones[i]:

                target = rewards[i]

            else:

                target = rewards[i] + self.gamma * np.max(next_q[i])

            target_q[i][actions[i]] = target

        loss = self.online_network.train_on_batch(states, target_q)
        return float(loss)

    # --------------------------------------------------------
    # Update target network
    # --------------------------------------------------------
    def update_target_network(self):

        self.target_network.set_weights(
            self.online_network.get_weights()
        )

    # --------------------------------------------------------
    # Decay epsilon
    # --------------------------------------------------------
    def decay_epsilon(self):

        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay

            self.epsilon = max(
                self.epsilon,
                self.epsilon_min
            )

    # --------------------------------------------------------
    # Save model weights
    # --------------------------------------------------------
    def save_weights(self, filepath):

        self.online_network.save_weights(filepath)

    # --------------------------------------------------------
    # Load model weights
    # --------------------------------------------------------
    def load_weights(self, filepath):

        self.online_network.load_weights(filepath)

        self.update_target_network()


# ### **3.7 Agent and ε-Greedy Policy Verification**
# 
# **Objective**
# 
# Verify that the DQN agent is correctly initialised and that the ε-greedy action-selection mechanism behaves as expected before training.
# 
# **Method**
# 
# A fixed sample state is passed repeatedly to the agent.
# 
# - **ε = 1.0:** The agent should select different random actions, confirming exploration.
# - **ε = 0.0:** The agent should repeatedly select the same action with the highest predicted Q-value, confirming greedy action selection.
# 
# After the verification, epsilon is restored to its initial value before training begins.

# In[20]:


agent = DQNAgent()

print("State size:             ", agent.state_size)
print("Action size:            ", agent.action_size)
print("Replay memory capacity: ", MEMORY_SIZE)
print("Initial epsilon:        ", agent.epsilon)

sample_state = np.array(
    [1.0, 0.0, 0.0],
    dtype=np.float32
)

print("\nExploration test (ε = 1.0)")
agent.epsilon = 1.0
exploration_actions = [
    agent.select_action(sample_state)
    for _ in range(5)
]
print("Selected actions:", exploration_actions)

print("\nExploitation test (ε = 0.0)")
agent.epsilon = 0.0
exploitation_actions = [
    agent.select_action(sample_state)
    for _ in range(5)
]
print("Selected actions:", exploitation_actions)

# Restore the initial epsilon before the training section.
agent.epsilon = EPSILON_START


# #### **Analysis**
# 
# - The agent was successfully initialised with the required state size, action count, replay memory, online network, target network, and exploration settings.
# - With $\epsilon=1.0$, different random action indices were selected, verifying exploration behaviour.
# - With $\epsilon=0.0$, the same highest-Q action was selected repeatedly for the fixed sample state, verifying deterministic greedy selection.
# - At this stage, the network is untrained, so this test verifies the action-selection mechanism rather than the quality of the selected action.
# - Epsilon was restored before the baseline training experiment.

# ## **4. Default-Gravity Baseline Experiment**
# 
# The first complete DQN agent is trained under the default Pendulum gravity of `g = 10`.
# 
# This experiment establishes a baseline before changing gravity or tuning hyperparameters. The baseline provides a controlled reference for comparing:
# 
# - learning progress;
# - final greedy evaluation performance;
# - training stability;
# - exploration behaviour; and
# - later improvements.
# 
# No conclusions about the best configuration are made at this stage.

# In[21]:


# Master results list -- every train_agent() + evaluate_experiment() pair
# for every gravity setting appends one merged record here, so Section 5.4
# can compare all of them from a single place.
all_results = []


# In[22]:


COMMON_EVALUATION_SEED = SEED


def set_experiment_seed(seed):
    """Reset the random generators used by each training experiment."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_agent(
    gravity,
    label,
    hyperparams=None,
    num_episodes=NUM_EPISODES,
    seed=SEED,
    verbose=True,
    discrete_actions=None
):

    hyperparams = hyperparams or {}

    action_values = np.asarray(
        DISCRETE_ACTIONS if discrete_actions is None else discrete_actions,
        dtype=np.float32
    ).reshape(-1)

    if len(action_values) < 2:
        raise ValueError("At least two discrete torque actions are required.")

    if np.any(action_values < ACTION_MIN) or np.any(action_values > ACTION_MAX):
        raise ValueError(
            f"All torque actions must be within [{ACTION_MIN}, {ACTION_MAX}]."
        )

    # Reset all experiment RNGs before the environment and networks are built.
    # This makes comparisons more controlled when only one setting is changed.
    set_experiment_seed(seed)

    # ---- Build the training environment with the requested gravity ----
    train_env = gym.make(ENV_NAME)
    train_env.seed(seed)
    train_env.action_space.seed(seed)
    train_env.unwrapped.g = float(gravity)

    max_steps = train_env.spec.max_episode_steps

    # ---- Build the agent (hyperparameters overridable per experiment) ----
    agent = DQNAgent(
        action_size=len(action_values),
        learning_rate=hyperparams.get("learning_rate", LEARNING_RATE),
        gamma=hyperparams.get("gamma", GAMMA),
        epsilon=hyperparams.get("epsilon_start", EPSILON_START),
        epsilon_min=hyperparams.get("epsilon_min", EPSILON_MIN),
        epsilon_decay=hyperparams.get("epsilon_decay", EPSILON_DECAY),
        batch_size=hyperparams.get("batch_size", BATCH_SIZE),
        memory_size=hyperparams.get("memory_size", MEMORY_SIZE),
        min_memory_size=hyperparams.get("min_memory_size", MIN_MEMORY_SIZE),
    )
    target_update_frequency = hyperparams.get(
        "target_update_frequency", TARGET_UPDATE_FREQUENCY
    )

    # ================================================================
    # Tier 1 -- Training (diagnostic metrics)
    # ================================================================
    episode_rewards, episode_losses, epsilon_history = [], [], []

    for episode in range(1, num_episodes + 1):

        state = np.asarray(train_env.reset(), dtype=np.float32).reshape(-1)
        episode_reward = 0.0
        losses = []

        for step in range(max_steps):

            action_index = agent.select_action(state)
            torque = action_index_to_torque(action_index, action_values)

            next_state, reward, done, info = train_env.step(torque)
            next_state = np.asarray(next_state, dtype=np.float32).reshape(-1)
            reward = float(reward)
            done = bool(done)

            agent.store_transition(state, action_index, reward, next_state, done)

            loss = agent.train_step()
            if loss is not None:
                losses.append(float(loss))

            state = next_state
            episode_reward += reward

            if done:
                break

        if episode % target_update_frequency == 0:
            agent.update_target_network()

        agent.decay_epsilon()

        average_loss = float(np.mean(losses)) if losses else np.nan
        episode_rewards.append(float(episode_reward))
        episode_losses.append(average_loss)
        epsilon_history.append(float(agent.epsilon))

        if verbose and (episode == 1 or episode % 10 == 0):

            loss_text = (
                f"{average_loss:>10.4f}"
                if not np.isnan(average_loss)
                else "       N/A"
            )

            print(
                f"[{label}] Episode {episode:>3}/{num_episodes} | "
                f"Reward: {episode_reward:>8.2f} | "
                f"Average loss: {loss_text} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Memory: {len(agent.memory)}"
            )

    if verbose:
        print(f"\n[{label}] Training completed.")

        final_window = min(20, len(episode_rewards))

        final_mean = float(np.mean(episode_rewards[-final_window:]))
        final_std = float(np.std(episode_rewards[-final_window:]))
        final_best = float(np.max(episode_rewards[-final_window:]))
        final_worst = float(np.min(episode_rewards[-final_window:]))
        final_score = final_mean - final_std

        print(
            f"[{label}] Training Summary | "
            f"Mean: {final_mean:.2f} | "
            f"Std: {final_std:.2f} | "
            f"Best: {final_best:.2f} | "
            f"Worst: {final_worst:.2f} | "
            f"Score: {final_score:.2f}"
        )

    train_env.close()

    final_window = min(20, len(episode_rewards))

    return {
        "Label": label,
        "Gravity": gravity,
        "Hyperparams": hyperparams,
        "Agent": agent,
        "Seed": seed,
        "Discrete Actions": action_values.copy(),
        "Action Count": int(len(action_values)),

        # Tier 1 -- training diagnostics
        "Episode Rewards": episode_rewards,
        "Episode Losses": episode_losses,
        "Epsilon History": epsilon_history,
        "Final Training Avg": round(float(np.mean(episode_rewards[-final_window:])), 2),
        "Best Training Reward": round(float(np.max(episode_rewards)), 2),
        "Worst Training Reward": round(float(np.min(episode_rewards)), 2),
        "Final Training Reward": round(float(episode_rewards[-1]), 2),
        "Final Epsilon": round(float(agent.epsilon), 4),
        "Replay Buffer": len(agent.memory),
    }


# ### **4.1 Train the Default-Gravity Agent**
# 
# | Controlled item | Setting |
# |---|---|
# | Gravity | `10.0` |
# | Episodes | `500` |
# | Training seed | Fixed experiment seed |
# | Architecture and hyperparameters | Baseline values |
# 
# **Evidence collected:** episode reward, moving-average reward, loss, epsilon and final training statistics.
# 

# In[23]:


default_result = train_agent(
    gravity=10.0,
    label="Default",
    verbose=True
)


# ### **4.2 Evaluate the Default-Gravity Agent (20 Episodes, Greedy Policy)**
# 
# **Objective**
# 
# Evaluate the learned policy without exploration and measure its overall performance and consistency.
# 
# **Method**
# 
# During evaluation, **ε = 0**, so the agent always selects the action with the highest predicted Q-value (greedy policy). The trained agent is evaluated over **20 episodes** with different initial pendulum states, and the cumulative reward of each episode is recorded.
# 
# **Evaluation Metrics**
# 
# | **Metric** | **Purpose** |
# |------------|-------------|
# | **Mean Reward** | Measures the overall performance of the learned policy. Higher (less negative) values indicate better control. |
# | **Standard Deviation** | Measures the consistency of the learned policy across different initial states. Lower values indicate more consistent performance. |
# | **Best Reward** | Highest reward achieved in a single evaluation episode. |
# | **Worst Reward** | Lowest reward achieved in a single evaluation episode. |
# | **Episode Rewards** | Shows the cumulative reward for each evaluation episode and the variation in performance. |
# 
# Mean greedy evaluation reward is treated as the main final-policy metric, while standard deviation and the remaining measures provide supporting evidence.

# In[24]:


def evaluate_agent(
    agent,
    env,
    num_episodes=20,
    max_steps=200,
    action_values=None
):
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    if action_values is None:
        action_values = DISCRETE_ACTIONS

    action_values = np.asarray(
        action_values,
        dtype=np.float32
    ).reshape(-1)

    evaluation_rewards = []

    try:
        for episode in range(1, num_episodes + 1):

            state = np.asarray(
                env.reset(),
                dtype=np.float32
            ).reshape(-1)

            total_reward = 0.0

            for step in range(max_steps):

                action_index = agent.select_action(state)

                torque = action_index_to_torque(
                    action_index,
                    action_values
                )

                next_state, reward, done, info = env.step(torque)

                state = np.asarray(
                    next_state,
                    dtype=np.float32
                ).reshape(-1)

                total_reward += float(reward)

                if done:
                    break

            evaluation_rewards.append(float(total_reward))

            print(
                f"Evaluation episode {episode:>2}/{num_episodes} | "
                f"Reward: {total_reward:>8.2f}"
            )

    finally:
        agent.epsilon = original_epsilon

    return evaluation_rewards


# In[25]:


def evaluate_experiment(
    result,
    eval_episodes=20,
    seed=COMMON_EVALUATION_SEED,
    verbose=True
):

    agent = result["Agent"]
    gravity = result["Gravity"]
    label = result["Label"]

    # Use one common evaluation seed for every trained agent so that
    # repeated trials are tested on the same sequence of initial states.
    evaluation_seed = COMMON_EVALUATION_SEED if seed is None else int(seed)

    eval_env = gym.make(ENV_NAME)
    eval_env.seed(evaluation_seed)
    eval_env.action_space.seed(evaluation_seed)
    eval_env.unwrapped.g = float(gravity)

    action_values = np.asarray(
        result.get("Discrete Actions", DISCRETE_ACTIONS),
        dtype=np.float32
    ).reshape(-1)

    evaluation_rewards = evaluate_agent(
        agent=agent,
        env=eval_env,
        num_episodes=eval_episodes,
        max_steps=eval_env.spec.max_episode_steps,
        action_values=action_values
    )
    eval_env.close()

    evaluation_rewards = np.asarray(evaluation_rewards, dtype=np.float32)

    mean_reward = float(np.mean(evaluation_rewards))
    std_reward = float(np.std(evaluation_rewards))
    best_reward = float(np.max(evaluation_rewards))
    worst_reward = float(np.min(evaluation_rewards))

    # Stability-adjusted ranking score: rewards high performance AND low variance.
    score = mean_reward - std_reward

    # How much greedy-evaluation performance differs from the agent's own
    # final training-time average (positive = evaluation outperforms training).
    training_evaluation_gap = mean_reward - result["Final Training Avg"]

    result.update({
        "Evaluation Seed": evaluation_seed,
        "Evaluation Rewards": evaluation_rewards.tolist(),
        "Mean Eval Reward": round(mean_reward, 2),
        "Std Eval Reward": round(std_reward, 2),
        "Best Eval Reward": round(best_reward, 2),
        "Worst Eval Reward": round(worst_reward, 2),
        "Score (Mean - Std)": round(score, 2),
        "Training-Evaluation Gap": round(training_evaluation_gap, 2),
    })

    if verbose:
        print(
            f"[{label}] Evaluation | Seed: {evaluation_seed} | "
            f"Mean: {mean_reward:.2f} | Std: {std_reward:.2f} | "
            f"Best: {best_reward:.2f} | Worst: {worst_reward:.2f} | "
            f"Score: {score:.2f}"
        )

    return result


# In[26]:


default_result = evaluate_experiment(default_result, verbose=True)
all_results.append(default_result)


# ### **4.3 Visualise the Training Results**
# 
# **Objective**
# 
# Visualise the agent’s learning progress, stability, and exploration behaviour.
# 
# **Training Metrics**
# 
# * **Episode Reward:** Shows performance per episode; a **20-episode moving average** highlights the overall learning trend.
# * **Average Huber Loss:** Indicates Q-network prediction error and training stability.
# * **Epsilon:** Shows how exploration decreases as exploitation increases.
# 
# Together, these metrics summarise the agent’s training performance.
# 

# In[136]:


import numpy as np
import matplotlib.pyplot as plt


def moving_average(values, window=20):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    window = max(1, min(window, len(values)))

    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid"
    )


def compare_training_results(
    results,
    labels=None,
    moving_average_window=20,
    figsize=(18, 5),
    title=None,
    reward_ylim=None,
    loss_ylim=None,
    epsilon_ylim=(0, 1.05)
):
    """
    Compare one or more DQN training results using smoothed curves.

    Parameters
    ----------
    results : list or tuple
        Result dictionaries returned by train_agent().

    labels : list, optional
        Labels for each configuration.

    moving_average_window : int
        Window used to smooth reward and loss curves.

    figsize : tuple
        Figure size.

    title : str, optional
        Overall title.

    reward_ylim : tuple, optional
        Fixed y-axis limits for reward, e.g. (-1800, 0).

    loss_ylim : tuple, optional
        Fixed y-axis limits for loss, e.g. (0, 0.65).

    epsilon_ylim : tuple
        Fixed y-axis limits for epsilon.
    """

    if not isinstance(results, (list, tuple)) or len(results) < 1:
        raise ValueError(
            "Provide at least one result dictionary inside a list or tuple."
        )

    if labels is None:
        labels = [
            result.get("Label", f"Configuration {i + 1}")
            for i, result in enumerate(results)
        ]

    if len(labels) != len(results):
        raise ValueError(
            "The number of labels must match the number of results."
        )

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # ================================================================
    # Reward comparison
    # ================================================================
    all_smoothed_rewards = []

    for result, label in zip(results, labels):
        rewards = np.asarray(
            result["Episode Rewards"],
            dtype=float
        )

        if len(rewards) == 0:
            continue

        reward_window = min(
            moving_average_window,
            len(rewards)
        )

        smoothed_rewards = moving_average(
            rewards,
            window=reward_window
        )

        reward_episodes = np.arange(
            reward_window,
            reward_window + len(smoothed_rewards)
        )

        all_smoothed_rewards.append(smoothed_rewards)

        axes[0].plot(
            reward_episodes,
            smoothed_rewards,
            linewidth=2,
            label=label
        )

    axes[0].set_title(
        f"Training Reward\n"
        f"({moving_average_window}-Episode Moving Average)"
    )
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if reward_ylim is not None:
        axes[0].set_ylim(reward_ylim)
    elif all_smoothed_rewards:
        reward_values = np.concatenate(all_smoothed_rewards)
        padding = 0.05 * (
            np.nanmax(reward_values) - np.nanmin(reward_values)
        )
        axes[0].set_ylim(
            np.nanmin(reward_values) - padding,
            np.nanmax(reward_values) + padding
        )

    # ================================================================
    # Loss comparison
    # ================================================================
    all_smoothed_losses = []

    for result, label in zip(results, labels):
        losses = np.asarray(
            result["Episode Losses"],
            dtype=float
        )

        if len(losses) == 0:
            continue

        valid_mask = ~np.isnan(losses)
        valid_losses = losses[valid_mask]

        valid_episodes = np.arange(
            1,
            len(losses) + 1
        )[valid_mask]

        if len(valid_losses) == 0:
            continue

        loss_window = min(
            moving_average_window,
            len(valid_losses)
        )

        smoothed_losses = moving_average(
            valid_losses,
            window=loss_window
        )

        loss_episodes = valid_episodes[
            loss_window - 1:
        ]

        all_smoothed_losses.append(smoothed_losses)

        axes[1].plot(
            loss_episodes,
            smoothed_losses,
            linewidth=2,
            label=label
        )

    axes[1].set_title(
        f"Training Loss\n"
        f"({moving_average_window}-Episode Moving Average)"
    )
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Average Huber Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    if loss_ylim is not None:
        axes[1].set_ylim(loss_ylim)
    elif all_smoothed_losses:
        loss_values = np.concatenate(all_smoothed_losses)
        padding = 0.05 * (
            np.nanmax(loss_values) - np.nanmin(loss_values)
        )
        axes[1].set_ylim(
            max(0, np.nanmin(loss_values) - padding),
            np.nanmax(loss_values) + padding
        )

    # ================================================================
    # Epsilon comparison
    # ================================================================
    for result, label in zip(results, labels):
        epsilon_history = np.asarray(
            result["Epsilon History"],
            dtype=float
        )

        if len(epsilon_history) == 0:
            continue

        episodes = np.arange(
            1,
            len(epsilon_history) + 1
        )

        axes[2].plot(
            episodes,
            epsilon_history,
            linewidth=2,
            label=label
        )

    axes[2].set_title("Exploration Rate")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Epsilon")
    axes[2].set_ylim(epsilon_ylim)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    if title is not None:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    if title is not None:
        plt.subplots_adjust(top=0.82)

    plt.show()


# In[137]:


compare_training_results(
    results=[default_result],
    labels=["Default"],
    title="Default Gravity Training"
)


# **Default-Gravity Training Analysis**
# 
# * **Reward:** Rewards were highly negative during early training because the agent was exploring. The 20-episode moving average began improving noticeably after approximately **120 episodes** and reached **−137.69** at the end of training.
# * **Replay Memory Warm-Up:** Q-network training began after approximately **10 episodes**, when the replay buffer exceeded the minimum size of **2,000 transitions**.
# * **Stability:** Episode rewards continued to fluctuate throughout training, although performance became more stable during the later episodes.
# * **Loss:** The Huber loss decreased rapidly from approximately **0.62** to below **0.05** during early training. It later increased slightly and stabilised at approximately **0.08**, indicating relatively stable Q-network learning.
# * **Exploration:** Epsilon decreased smoothly from **1.0** to approximately **0.0816**, gradually shifting the agent from exploration to exploitation.
# * **Overall Performance:** The final 20-episode moving-average reward was **−137.69**, while the best episode reward was close to **0**, showing that the agent achieved near-optimal pendulum control in some episodes.
# 
# **Conclusion**
# 
# The default-gravity agent successfully learned to control the pendulum and achieved several strong episode rewards. However, the remaining reward fluctuations indicate that the policy had not fully converged and could still be improved.
# 

# ### **4.4 Visualise the Greedy Evaluation Results**
# 
# The episode rewards from the 20 greedy evaluation runs are visualised to show the variation in performance across different initial pendulum states.

# In[29]:


def plot_evaluation_curve(result):
    """
    Plot per-episode greedy evaluation reward for one run_experiment()
    result, with the mean reward marked. Same layout for every experiment.
    """
    evaluation_rewards = result["Evaluation Rewards"]
    mean_reward = result["Mean Eval Reward"]
    label = result["Label"]

    plt.figure(figsize=(10, 5))
    episode_numbers = range(1, len(evaluation_rewards) + 1)

    plt.plot(episode_numbers, evaluation_rewards, marker="o", label="Evaluation reward")
    plt.axhline(mean_reward, linestyle="--", label=f"Mean reward = {mean_reward:.2f}")

    plt.xlabel("Evaluation Episode")
    plt.ylabel("Total Reward")
    plt.title(f"Greedy-Policy Evaluation -- {label}")
    plt.xticks(episode_numbers)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# In[30]:


plot_evaluation_curve(default_result)


# **Evaluation Analysis**
# 
# * **Mean reward:** **−192.28** over 20 greedy evaluation episodes, showing that the agent learned a useful control policy.
# * **Standard deviation:** **66.86**, indicating noticeable variation in performance across episodes.
# * **Reward range:** Approximately **−119 to −332**, showing that the policy was sensitive to different initial states.
# 
# **Conclusion**
# 
# The agent successfully controlled the pendulum without exploration in several episodes. However, the relatively high standard deviation and wide reward range show that its performance and consistency can still be improved.
# 

# ### **4.5 Compare Training and Evaluation Performance**
# 
# | Metric | Definition |
# |---|---|
# | Final training average | Mean reward over the final 20 training episodes while epsilon remains above zero |
# | Greedy evaluation mean | Mean reward across 20 evaluation episodes with epsilon set to zero |
# | Evaluation standard deviation | Standard deviation of the 20 greedy evaluation rewards |
# | Training–evaluation gap | Evaluation mean minus final training average; a negative value means evaluation performed worse |
# | Episodes to `−300` | First episode where the 20-episode rolling-average training reward reaches or exceeds `−300` |
# | Late-training stability | Standard deviation of the final 100 available values of the 20-episode rolling-average reward; lower is more stable |
# 
# The training–evaluation gap is a descriptive comparison rather than a formal generalisation measurement because training and evaluation use different episode samples and different exploration settings.

# In[97]:


import numpy as np
import pandas as pd


def get_training_rewards(result):
    """
    Retrieve training rewards from the result dictionary.
    """
    possible_keys = [
        "Training Rewards",
        "Rewards",
        "Episode Rewards",
        "Training Episode Rewards",
        "reward_history"
    ]

    for key in possible_keys:
        if key in result:
            return np.asarray(result[key], dtype=float)

    raise KeyError(
        "Training rewards could not be found. "
        f"Available keys: {list(result.keys())}"
    )


def episodes_to_threshold(
    result,
    threshold=-300.0,
    window=20
):
    """
    Return the first episode where the rolling-average reward
    reaches or exceeds the threshold.
    """
    rewards = get_training_rewards(result)

    if len(rewards) < window:
        return np.nan

    rolling_average = pd.Series(rewards).rolling(
        window=window
    ).mean()

    reached = rolling_average >= threshold

    if not reached.any():
        return np.nan

    # Pandas index starts at 0, so add 1 for the episode number.
    return int(reached[reached].index[0] + 1)


def late_training_stability(
    result,
    window=20,
    last_n=100
):
    """
    Calculate the standard deviation of the final last_n rolling-average
    reward values.

    A lower value means more stable late-training performance.
    """
    rewards = get_training_rewards(result)

    if len(rewards) == 0:
        return np.nan

    rolling_average = pd.Series(rewards).rolling(
        window=window
    ).mean()

    # Remove the initial NaN values created by rolling().
    rolling_average = rolling_average.dropna()

    if len(rolling_average) == 0:
        return np.nan

    late_rolling_average = rolling_average.tail(last_n)

    # Pandas std() uses ddof=1, matching the original result.
    return float(late_rolling_average.std())

def build_gap_summary(
    results,
    configuration_label=None,
    threshold=-300.0,
    window=20,
    last_n=100
):
    """
    Build a compact training-versus-evaluation comparison table.
    """
    if isinstance(results, dict):
        results = [results]

    threshold_column = f"Episodes to {threshold:g}"

    rows = []

    for result in results:
        rows.append({
            "Configuration": configuration_label or result["Label"],
            "Seed": result.get("Seed", np.nan),
            "Final Training Avg": result["Final Training Avg"],
            "Evaluation Mean": result["Mean Eval Reward"],
            "Evaluation Std": result["Std Eval Reward"],
            "Training-Evaluation Gap": result["Training-Evaluation Gap"],
            threshold_column: episodes_to_threshold(result, threshold=threshold, window=window),
            "Late Training Stability": late_training_stability(result, window=window, last_n=last_n)
        })

    return pd.DataFrame(rows)


# **Summary Table**

# In[32]:


display(build_gap_summary(default_result))


# **Analysis**
# 
# * The evaluation reward was **54.59 points lower** than the final training average.
# * This difference may be caused by variations in the initial states encountered during evaluation, even though evaluation uses a fully greedy policy (**ε = 0**).
# * Despite the lower evaluation reward, the learned policy performed substantially better than during the early stages of training, indicating successful learning and reasonable generalisation.
# 
# **Problems Identified**
# 
# * Training rewards still fluctuated, suggesting that the policy had not fully converged.
# * The evaluation reward remained below the optimal value of **0**, indicating that perfect pendulum control was not achieved.
# * The relatively high evaluation standard deviation of **66.86** indicates that performance varied across different initial states.
# * Additional runs with multiple random seeds are required to further evaluate the reliability of the learned policy.
# 

# ### **4.6 Transition to the Gravity Experiments: Natural Pendulum Movement**
# 
# Before training separate DQN agents under the four required gravity settings, the uncontrolled movement of the Pendulum is examined in each environment.
# 
# All environments use:
# 
# - the same initial angle;
# - the same initial angular velocity;
# - zero applied torque; and
# - the same number of timesteps.
# 
# Therefore, differences in the trajectories are caused by gravity rather than agent behaviour.
# 
# This visualisation is not a DQN performance evaluation. It provides physical context for understanding why the same torque actions may produce different outcomes under different gravity settings.

# In[33]:


def simulate_without_torque(
    gravity,
    initial_theta=np.pi / 2,
    initial_angular_velocity=0.0,
    max_steps=200
):
    """
    Simulate the Pendulum environment with zero applied torque.
    The same initial state is used for every gravity setting.
    """

    test_env = gym.make(ENV_NAME)
    test_env.seed(SEED)
    test_env.action_space.seed(SEED)

    test_env.unwrapped.g = float(gravity)

    # Initialise the environment before manually setting its state
    test_env.reset()

    # Pendulum's internal state is [theta, theta_dot]
    test_env.unwrapped.state = np.array(
        [initial_theta, initial_angular_velocity],
        dtype=np.float32
    )

    angles = []
    angular_velocities = []

    zero_torque = np.array([0.0], dtype=np.float32)

    for _ in range(max_steps):

        next_state, reward, done, info = test_env.step(zero_torque)

        # Observation is [cos(theta), sin(theta), theta_dot]
        angle = np.arctan2(
            next_state[1],
            next_state[0]
        )

        angles.append(angle)
        angular_velocities.append(next_state[2])

        if done:
            break

    test_env.close()

    return np.array(angles), np.array(angular_velocities)


# In[34]:


gravity_values = {
    "Default (g = 10)": 10,
    "Free-fall (g = 0)": 0,
    "Anti-gravity (g = −10)": -10,
    "Supergravity (g = 15)": 15
}

plt.figure(figsize=(11, 5))

for label, gravity in gravity_values.items():

    angles, angular_velocities = simulate_without_torque(
        gravity=gravity
    )

    plt.plot(
        range(len(angles)),
        angles,
        label=label
    )

plt.axhline(
    0,
    linestyle="--",
    label="Upright position"
)

plt.xlabel("Timestep")
plt.ylabel("Pendulum Angle, θ (radians)")
plt.title(
    "Pendulum Movement Under Different Gravity Settings\n"
    "Same Initial State and Zero Applied Torque"
)
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# #### **Visualisation Analysis**
# 
# - Under **free-fall**, the angle remains constant because the initial angular velocity and applied torque are both zero and no gravity acts on the pendulum.
# - Under **anti-gravity**, reversed gravity pulls the pendulum towards the upright position, causing movement around $\theta=0$.
# - Under **default gravity**, the pendulum moves around the naturally stable downward position.
# - Under **supergravity**, the stronger gravitational influence produces faster angular movement and more frequent oscillation.
# - Apparent jumps between $+\pi$ and $-\pi$ are caused by angle wrapping and do not represent instantaneous physical movement.
# 
# The visualisation confirms that gravity changes the Pendulum dynamics even when the action is held constant. This supports training separate agents under each gravity setting while keeping the DQN configuration unchanged for the initial comparison.

# ## **5. Different Gravity Experiments**
# 
# The DQN agent is trained and evaluated under different gravity settings while keeping the network architecture, hyperparameters, training episodes, and evaluation method unchanged. This ensures that any performance differences are due only to the change in gravity.
# 
# | **Setup** | **Gravity (g)** | **Objective** |
# |------------|:---------------:|---------------|
# | **Default** | **10** | Baseline. |
# | **Free-fall** | **0** | No gravity. |
# | **Anti-gravity** | **−10** | Reversed gravity. |
# | **Supergravity** | **15** | Stronger gravity. |
# 
# The same functions from Section 4 are reused for each experiment by changing only the `gravity` and `label` arguments.

# ### **5.1 Train Free-fall (g = 0.0)**

# **Step 1 --  Train (g = 0.0)**
# 

# In[35]:


freefall_result = train_agent(
    gravity=0.0,
    label="Free-fall",
    verbose=True
)


# **Step 2 — Evaluate the Free-Fall Agent**

# In[36]:


freefall_result = evaluate_experiment(freefall_result, verbose=True)
all_results.append(freefall_result)


# **Step 3 - Training curves**

# In[132]:


compare_training_results(
    results=[freefall_result],
    labels=["Free Fall"],
    title="Freefall Gravity Training"
)


# **Free-Fall Training Analysis**
# 
# * **Reward:** The 20-episode moving-average reward improved rapidly during the first **100 episodes** and later stabilised at approximately **−40 to −50**, showing fast and effective learning.
# * **Stability:** Episode rewards became less variable after approximately **150 episodes**, indicating that the learned policy became more consistent.
# * **Loss:** The Huber loss dropped sharply from approximately **0.31** to below **0.02** during early training and gradually decreased to around **0.005**, suggesting stable Q-network learning.
# * **Exploration:** Epsilon decreased steadily from **1.0** to approximately **0.0816**, allowing the agent to shift gradually from exploration to exploitation.
# 
# **Conclusion**
# 
# The free-fall agent learned quickly and achieved relatively stable performance because the absence of gravity made the pendulum easier to control.

# **Step 4 - Greedy evaluation curve**
# 

# In[38]:


plot_evaluation_curve(freefall_result)


# 
# **Evaluation Comparison**
# 
# * **Mean reward:** Free-fall achieved a much better mean reward of **−54.21**, compared with **−192.28** under default gravity.
# * **Difference:** The free-fall mean reward was **138.07 points higher** than the default-gravity mean reward.
# * **Episode performance:** Free-fall produced several rewards close to **0**, while the default-gravity rewards remained considerably lower.
# * **Variation:** Both settings showed fluctuations across different initial states, but free-fall generally maintained much stronger performance.
# 
# **Conclusion**
# 
# The greedy policy performed substantially better in the free-fall environment, confirming that the pendulum was easier to control without normal gravitational force.

# **Step 5 - Training-vs-evaluation gap**
# 

# In[39]:


display(build_gap_summary(freefall_result))


# **Free-Fall Training–Evaluation Analysis**
# 
# * **Evaluation standard deviation:** **35.21**, indicating moderate variation across different initial states.
# * **Training–evaluation gap:** **−15.79**, showing that evaluation performance remained relatively close to the final training average.
# 
# **Problems Identified**
# 
# * The policy was not fully consistent across evaluation episodes.
# * Evaluation performance was **15.79 points lower** than the final training average.
# * The agent remained somewhat sensitive to different initial states, as reflected by the evaluation standard deviation of **35.21**.
# 
# **Conclusion**
# 
# The free-fall agent achieved strong and reasonably consistent performance. Only a relatively small decrease was observed during greedy evaluation, suggesting that the learned policy generalised well beyond the training episodes.
# 

# ### **5.2 Anti-gravity (g = -10.0)**

# **Step 1 -- Train**
# 

# In[40]:


antigravity_result = train_agent(
    gravity=-10.0,
    label="Anti-gravity",
    verbose=True
)


# **Step 2 -- Evaluate (20 episodes, greedy policy)**
# 

# In[41]:


antigravity_result = evaluate_experiment(antigravity_result, verbose=True)
all_results.append(antigravity_result)


# **Step 3 - Training curves**
# 

# In[138]:


compare_training_results(
    results=[antigravity_result],
    labels=["Anti Gravity"],
    title="Anti Gravity Training"
)


# **Anti-Gravity Training Analysis**
# 
# * **Reward:** The 20-episode moving-average reward improved rapidly during the first **100 episodes** and later stabilised at approximately **−50**.
# * **Stability:** Episode rewards became less variable after approximately **150 episodes**, although occasional fluctuations remained throughout training.
# * **Loss:** The Huber loss dropped sharply from approximately **0.20** to below **0.02** during early training and gradually decreased to around **0.005**, indicating stable Q-network learning.
# * **Exploration:** Epsilon decreased steadily from **1.0** to approximately **0.0816**, gradually shifting the agent from exploration to exploitation.
# * **Overall Performance:** Several individual episode rewards were close to **0**, showing that the agent occasionally achieved strong control from favourable starting states. This does not establish consistently near-optimal performance.
# 
# **Conclusion**
# 
# The anti-gravity agent learned quickly and achieved relatively strong performance. However, the remaining reward fluctuations indicate that the learned policy was not fully stable or consistent.

# **Step 4 - Greedy evaluation curve**
# 

# In[43]:


plot_evaluation_curve(antigravity_result)


# **Anti-Gravity Evaluation Analysis**
# 
# * **Evaluation mean:** **−62.53** over 20 greedy evaluation episodes, showing that the learned policy generally maintained relatively strong control without exploration.
# * **Reward range:** Approximately **−2 to −142**, indicating noticeable variation across different initial states.
# * **Episode performance:** Several episodes achieved rewards close to **0**, while some episodes produced considerably lower rewards.
# * **Consistency:** The wide reward range shows that the agent’s performance was not fully consistent across evaluation episodes.
# 
# **Conclusion**
# 
# The anti-gravity greedy policy achieved generally strong performance, but its results varied considerably across initial states. This suggests that the agent learned an effective control policy, although further training or tuning may be needed to improve consistency.
# 

# **Step 5 - Training-vs-evaluation gap**
# 

# In[44]:


display(build_gap_summary(antigravity_result))


# **Anti-Gravity Training–Evaluation Analysis**
# 
# * **Training average:** **−44.23**
# * **Evaluation mean:** **−62.53**
# * **Evaluation standard deviation:** **41.03**, indicating moderate variation across different initial states.
# * **Training–evaluation gap:** **−18.30**, showing that greedy evaluation performance was lower than the final training average.
# 
# **Problems Identified**
# 
# * Training rewards continued to fluctuate even after the moving average had stabilised.
# * The policy did not consistently achieve rewards close to **0**.
# * Performance remained sensitive to different episode conditions and initial states.
# * Evaluation rewards varied considerably, as shown by the standard deviation of **41.03**.
# * The greedy policy performed **18.30 points lower** than the final training average.
# 
# **Conclusion**
# 
# The anti-gravity agent achieved relatively strong performance during both training and evaluation. However, the remaining reward fluctuations, moderate evaluation variation, and training–evaluation gap show that the learned policy was not fully stable or consistent.
# 

# ### **5.3 Supergravity (g = 15.0)**
# 

# **Step 1 -- Train**
# 

# In[45]:


supergravity_result = train_agent(
    gravity=15.0,
    label="Supergravity",
    verbose=True
)


# **Step 2 -- Evaluate (20 episodes, greedy policy)**
# 

# In[46]:


supergravity_result = evaluate_experiment(supergravity_result, verbose=True)
all_results.append(supergravity_result)


# **Step 3 - Training curves**
# 

# In[139]:


compare_training_results(
    results=[supergravity_result],
    labels=["supergravity"],
    title="Super Gravity Training"
)


# **Supergravity Training Analysis**
# 
# * **Reward:** The 20-episode moving-average reward improved gradually from approximately **−1,500** during early training to a final 20-episode average of **−246.91**, showing that the agent learned despite the more difficult environment.
# * **Stability:** Episode rewards remained highly variable throughout training, with frequent large decreases and occasional strong episodes.
# * **Loss:** The Huber loss initially dropped sharply and later increased, indicating more difficult and less stable Q-value learning.
# * **Exploration:** Epsilon decreased steadily from **1.0** to approximately **0.0816**, using the same exploration schedule as the other gravity settings.
# 
# **Overall Performance:** Although the agent improved substantially, its final performance remained weaker and less stable than under the other gravity settings.
# 

# **Step 4 - Greedy evaluation curve**
# 

# In[48]:


plot_evaluation_curve(supergravity_result)


#  **Supergravity Evaluation Analysis**
# 
# * **Evaluation mean:** **−305.38** over 20 greedy evaluation episodes, indicating relatively weak control performance.
# * **Evaluation standard deviation:** Approximately **116.30**, showing substantial variation across different initial states.
# * **Reward range:** Approximately **−120 to −518**, demonstrating that performance was highly sensitive to the evaluation conditions.
# * **Consistency:** A few episodes achieved rewards close to **−120**, but many episodes produced rewards below **−300**.

# **Step 5 - Training Evaluation Analysis**

# In[49]:


display(build_gap_summary(supergravity_result))


# **Supergravity Training–Evaluation Analysis**
# 
# * **Final training average:** **−246.91**
# * **Evaluation mean:** **−305.38**
# * **Evaluation standard deviation:** **116.36**
# * **Training–evaluation gap:** **−58.47**, showing a noticeable performance decrease during greedy evaluation.
# 
# **Problems Identified**
# 
# * Training rewards remained highly variable even after substantial improvement.
# * The policy did not consistently achieve rewards close to `0`.
# * The loss increased during later training, suggesting that Q-value estimation remained difficult.
# * Evaluation performance varied considerably across initial states.
# * The greedy policy performed noticeably worse than the final training average.
# * The large evaluation standard deviation indicates that the learned policy was unreliable.
# 
# **Conclusion**
# 
# The supergravity agent learned some useful pendulum-control behaviour, but it achieved weaker and less consistent performance than the other gravity settings. The high reward variation, increased later-stage loss, and negative training–evaluation gap show that the stronger gravitational force made the environment substantially more difficult to control.
# 

# ### **5.4 Compare All Gravity Settings**

# In[129]:


compare_training_results(
    results=[
        default_result,
        freefall_result,
        antigravity_result,
        supergravity_result
    ],
    labels=[
        "Default",
        "Free-fall",
        "Anti-gravity",
        "Supergravity"
    ],
    title="Gravity Setting Comparison"
)


# #### **Visualisation Analysis**
# 
# * Free-fall and anti-gravity improved rapidly and reached high, relatively stable rewards early in training.
# * Default gravity showed a slower but consistent improvement throughout training.
# * Supergravity learned the slowest and experienced larger reward fluctuations, indicating less stable training.
# * The loss curves were lowest for free-fall and anti-gravity, while supergravity produced the highest loss.
# * All exploration-rate curves overlapped because the same epsilon settings were used.

# **Summary Table**

# In[143]:


print("\nFull Evaluation Comparison -- All Gravity Settings")

display(build_gap_summary(all_results))


# #### **Training and Evaluation Analysis**
# For the gravity comparison, the best-performing environment is defined as the one with the strongest **greedy evaluation performance**.
# 
# Mean evaluation reward is used as the primary criterion because evaluation is performed with exploration disabled, so it reflects the quality of the learned policy rather than exploratory actions during training. Since Pendulum rewards are negative, a value closer to zero indicates better control.
# 
# Evaluation standard deviation is used to assess how consistently the policy performs across different initial states. Learning speed and late-training stability are used as supporting evidence to show how efficiently and reliably the agent learns. The training–evaluation gap is considered only as additional evidence because training includes epsilon-greedy exploration and therefore is not directly comparable with greedy evaluation.
# 
# Therefore, the best gravity setting is determined mainly by **final greedy policy performance**, with consistency, learning speed, and stability used to support the conclusion.
# 
# | Evaluation Criterion             | Best Gravity Setting |               Supporting Result |
# | -------------------------------- | -------------------- | ------------------------------: |
# | Highest final training average   | Free-fall            |                        `−38.42` |
# | Highest evaluation mean          | Free-fall            |                        `−54.21` |
# | Most consistent evaluation       | Free-fall            |         Evaluation std: `35.21` |
# | Fastest learning                 | Free-fall            |  Reached `−300` at episode `49` |
# | Most stable late training        | Free-fall            |               Stability: `6.93` |
# | Smallest training–evaluation gap | Free-fall            |                  `15.79` points |
# | Slowest learning                 | Supergravity         | Reached `−300` at episode `374` |
# | Most variable evaluation         | Supergravity         |        Evaluation std: `116.36` |
# 
# - **Free-fall performed best under the shared baseline configuration**, achieving the strongest greedy evaluation result together with the fastest and most stable learning.
# 
# - **Anti-gravity followed closely**, reaching the `−300` threshold at episode `50` and maintaining relatively stable late-training performance.
# 
# - **Default gravity learned more slowly** and showed greater reward variation during training and evaluation.
# 
# - **Supergravity was the most difficult setting for the shared baseline configuration**, producing the slowest learning and greatest variation.
# 
# All agents used the same architecture, hyperparameters, action space, training budget, seed (`42`), and evaluation procedure. Therefore, gravity was the main experimental factor that changed.
# 
# Overall, this comparison shows that gravity has a clear effect on environmental difficulty. Free-fall produced the easiest control conditions, while supergravity was the most challenging under the shared baseline DQN configuration.

# ### **5.5 Quantitative Control-Behaviour Evaluation**
# 
# Total episode reward combines angle error, angular velocity, and torque into a single value. Therefore, additional physical-control metrics are calculated across 20 greedy evaluation episodes to better understand how each agent controls the pendulum.
# 
# | Metric | Interpretation |
# |---|---|
# | Mean absolute angle | Average distance from the upright position |
# | Mean absolute angular velocity | Average rotational movement |
# | Mean absolute torque | Average control effort |
# | Stable-upright percentage | Percentage of steps where the pendulum is near upright and moving slowly |
# | Extreme-torque percentage | Percentage of steps using the strongest torque values, `−2` or `+2` |
# 
# A timestep is classified as stable and upright when both conditions are satisfied:
# 
# $$
# |\theta| < 0.2 \text{ rad} \approx 11.5^\circ
# $$
# 
# $$
# |\dot{\theta}| < 1.0 \text{ rad/s}
# $$
# 
# Both conditions are required because being near upright does not necessarily indicate stable control if the pendulum is moving quickly. Similarly, slow movement does not indicate good control if the pendulum is far from upright.
# 
# These thresholds are used as practical indicators of a visually balanced and slow-moving pendulum rather than as formally derived stability boundaries.
# 
# All agents are evaluated across 20 greedy episodes using the same evaluation procedure and seed settings as the earlier comparison. These metrics are used to support the reward-based evaluation by showing the physical behaviour of each learned policy.

# In[53]:


UPRIGHT_ANGLE_THRESHOLD = 0.2
STABLE_VELOCITY_THRESHOLD = 1.0


def evaluate_control_behaviour(
    result,
    num_episodes=20,
    seed=COMMON_EVALUATION_SEED,
    angle_threshold=UPRIGHT_ANGLE_THRESHOLD,
    velocity_threshold=STABLE_VELOCITY_THRESHOLD
):
    """
    Evaluate reward, angle, angular velocity, and torque usage
    over multiple greedy-policy episodes.
    """
    agent = result["Agent"]
    gravity = float(result["Gravity"])
    label = result["Label"]
    action_values = np.asarray(
        result.get("Discrete Actions", DISCRETE_ACTIONS), dtype=np.float32
    ).reshape(-1)
    max_torque = float(np.max(np.abs(action_values)))

    env = gym.make(ENV_NAME)
    env.seed(seed)
    env.action_space.seed(seed)
    env.unwrapped.g = gravity
    max_steps = env.spec.max_episode_steps

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    episode_rows, step_rows = [], []

    try:
        for episode in range(1, num_episodes + 1):
            state = np.asarray(env.reset(), dtype=np.float32).reshape(-1)
            episode_reward = 0.0
            angles, velocities, torques, stable_flags = [], [], [], []

            for timestep in range(1, max_steps + 1):
                action_index = agent.select_action(state)
                torque_action = action_index_to_torque(action_index, action_values)
                torque_value = float(torque_action[0])

                next_state, reward, done, info = env.step(torque_action)
                next_state = np.asarray(next_state, dtype=np.float32).reshape(-1)

                theta = float(np.arctan2(next_state[1], next_state[0]))
                angular_velocity = float(next_state[2])
                stable_upright = (
                    abs(theta) < angle_threshold
                    and abs(angular_velocity) < velocity_threshold
                )

                angles.append(theta)
                velocities.append(angular_velocity)
                torques.append(torque_value)
                stable_flags.append(stable_upright)
                episode_reward += float(reward)

                step_rows.append({
                    "Gravity": label, "g": gravity,
                    "Episode": episode, "Timestep": timestep,
                    "Angle": theta, "Absolute Angle": abs(theta),
                    "Angular Velocity": angular_velocity,
                    "Absolute Angular Velocity": abs(angular_velocity),
                    "Torque": torque_value, "Absolute Torque": abs(torque_value),
                    "Reward": float(reward), "Stable Upright": stable_upright
                })

                state = next_state
                if done:
                    break

            angles = np.asarray(angles, dtype=float)
            velocities = np.asarray(velocities, dtype=float)
            torques = np.asarray(torques, dtype=float)
            stable_flags = np.asarray(stable_flags, dtype=bool)
            extreme_torque_flags = np.isclose(np.abs(torques), max_torque)

            episode_rows.append({
                "Gravity": label, "g": gravity, "Episode": episode,
                "Episode Reward": episode_reward,
                "Mean Absolute Angle": np.mean(np.abs(angles)),
                "Mean Absolute Angular Velocity": np.mean(np.abs(velocities)),
                "Mean Absolute Torque": np.mean(np.abs(torques)),
                "Stable Upright (%)": np.mean(stable_flags) * 100,
                "Extreme Torque (%)": np.mean(extreme_torque_flags) * 100
            })

    finally:
        agent.epsilon = original_epsilon
        env.close()

    return pd.DataFrame(episode_rows), pd.DataFrame(step_rows)


# In[54]:


gravity_results = [default_result, freefall_result, antigravity_result, supergravity_result]
gravity_order = [result["Label"] for result in gravity_results]

all_episode_data, all_step_data = [], []
for result in gravity_results:
    episode_data, step_data = evaluate_control_behaviour(
        result=result, num_episodes=20, seed=COMMON_EVALUATION_SEED
    )
    all_episode_data.append(episode_data)
    all_step_data.append(step_data)

control_episode_df = pd.concat(all_episode_data, ignore_index=True)
control_step_df = pd.concat(all_step_data, ignore_index=True)

for df in (control_episode_df, control_step_df):
    df["Gravity"] = pd.Categorical(df["Gravity"], categories=gravity_order, ordered=True)

AGG_MAP = {
    "Mean Evaluation Reward": ("Episode Reward", "mean"),
    "Evaluation Reward Std": ("Episode Reward", lambda v: np.std(v, ddof=0)),
    "Mean Absolute Angle": ("Mean Absolute Angle", "mean"),
    "Mean Absolute Angular Velocity": ("Mean Absolute Angular Velocity", "mean"),
    "Mean Absolute Torque": ("Mean Absolute Torque", "mean"),
    "Stable Upright (%)": ("Stable Upright (%)", "mean"),
    "Extreme Torque (%)": ("Extreme Torque (%)", "mean"),
}

control_summary = (
    control_episode_df
    .groupby(["Gravity", "g"], observed=True, sort=False)
    .agg(**AGG_MAP)
    .reset_index()
)

display(control_summary.round(3))

def best_setting(col, mode="min"):
    idx = col.idxmin() if mode == "min" else col.idxmax()
    return control_summary.loc[idx, "Gravity"]

print("Lowest mean absolute angle:", best_setting(control_summary["Mean Absolute Angle"]))
print("Lowest mean absolute angular velocity:", best_setting(control_summary["Mean Absolute Angular Velocity"]))
print("Lowest mean absolute torque:", best_setting(control_summary["Mean Absolute Torque"]))
print("Highest stable-upright percentage:", best_setting(control_summary["Stable Upright (%)"], "max"))
print("Highest extreme-torque usage:", best_setting(control_summary["Extreme Torque (%)"], "max"))


# ### **5.6 Visualisation of Control-Behaviour Comparison Across Gravity Settings**

# In[55]:


fig, axes = plt.subplots(2, 2, figsize=(15, 10))

graph_settings = [
    ("Mean Absolute Angle", "Mean Absolute Angle (radians)", "Distance from Upright"),
    ("Mean Absolute Angular Velocity", "Mean Absolute Angular Velocity", "Movement and Stability"),
    ("Mean Absolute Torque", "Mean Absolute Torque", "Control Effort"),
    ("Stable Upright (%)", "Percentage of Timesteps (%)", "Stable-Upright Time")
]

for ax, (metric, y_axis_label, title) in zip(axes.flatten(), graph_settings):
    ax.bar(control_summary["Gravity"].astype(str), control_summary[metric])
    ax.set_xlabel("Gravity Setting")
    ax.set_ylabel(y_axis_label)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Physical Control Performance Across Gravity Settings", fontsize=15)
plt.tight_layout()
plt.show()


# #### **Control-Behaviour Analysis**
# 
# | Control Criterion | Best / Highest Setting | Supporting Result |
# |---|---|---:|
# | Closest to upright | Anti-gravity | Mean absolute angle: `0.181` |
# | Smoothest movement | Free-fall | Mean absolute angular velocity: `0.278` |
# | Lowest average control effort | Default gravity | Mean absolute torque: `0.995` |
# | Highest stable-upright percentage | Free-fall | `87.68%` |
# | Lowest extreme-torque usage | Free-fall | `5.90%` |
# | Highest extreme-torque usage | Supergravity | `37.90%` |
# 
# - **Free-fall produced the smoothest and most stable control behaviour.** It recorded the lowest angular velocity, highest stable-upright percentage, and lowest use of extreme torque.
# 
# - **Anti-gravity kept the pendulum closest to upright**, but it used extreme torque in `32.75%` of the evaluation steps. This suggests accurate angle control with more aggressive actions.
# 
# - **Default gravity used the lowest average torque**, showing the lowest overall control effort. However, its angle and stable-upright results were weaker than free-fall and anti-gravity.
# 
# - **Supergravity required the most aggressive control behaviour.** It recorded the highest angular velocity and extreme-torque usage, together with the lowest stable-upright percentage.
# 
# Overall, free-fall achieved the strongest balance between smooth movement, stability, and limited extreme-torque usage. The results also show that lower average torque alone does not necessarily produce the best control performance.

# ### **5.7 Torque-Action Usage Across Gravity Settings**
# 
# The reward function penalises excessive torque, and different gravity settings may require different levels of control effort.
# 
# The following graph shows the percentage of greedy evaluation actions assigned to each discrete torque value.
# 
# This allows the analysis to determine:
# 
# - whether free-fall uses zero or smaller torque values more frequently;
# - whether supergravity relies more heavily on extreme torque;
# - whether anti-gravity produces a different action pattern; and
# - whether default gravity requires repeated corrective actions.

# In[56]:


control_step_df["Torque Level"] = control_step_df["Torque"].astype(float).round(2)

torque_usage = (
    control_step_df
    .groupby(["Gravity", "Torque Level"], observed=True)
    .size()
    .rename("Action Count")
    .reset_index()
)

torque_usage["Action Percentage"] = (
    torque_usage.groupby("Gravity", observed=True)["Action Count"]
    .transform(lambda counts: counts / counts.sum() * 100)
)

torque_usage_pivot = torque_usage.pivot(
    index="Torque Level", columns="Gravity", values="Action Percentage"
)

action_order = np.round(DISCRETE_ACTIONS.astype(float), 2)
torque_usage_pivot = torque_usage_pivot.reindex(action_order, fill_value=0)

torque_usage_pivot.plot(kind="bar", figsize=(14, 6))
plt.xlabel("Discrete Torque Value")
plt.ylabel("Percentage of Selected Actions (%)")
plt.title("Greedy-Policy Torque Usage Across Gravity Settings")
plt.xticks(rotation=0)
plt.legend(title="Gravity Setting")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

print("Most frequently selected torque for each gravity setting:")

for gravity_label in gravity_order:
    gravity_data = torque_usage[torque_usage["Gravity"] == gravity_label]
    dominant_row = gravity_data.loc[gravity_data["Action Percentage"].idxmax()]
    print(
        f"{gravity_label}: torque = {dominant_row['Torque Level']:.2f}, "
        f"percentage = {dominant_row['Action Percentage']:.2f}%"
    )


# #### **Greedy-Policy Torque Usage Analysis**
# 
# * **Default gravity** mainly used moderate torques, especially `+0.50`, showing balanced corrections.
# * **Free-fall** frequently used both `+1.50` and `−1.50`, because torque was the only force controlling movement.
# * **Anti-gravity** strongly preferred `−1.00`, indicating a dominant directional correction strategy.
# * **Supergravity** selected zero torque most often, but also used extreme torques frequently for strong corrections.
# 
# Overall, gravity caused each agent to learn a different torque-selection strategy.
# 

# ### **5.8 Greedy-Policy Trajectory Under a Common Evaluation Seed**
# 
# The previous results summarise performance across 20 evaluation episodes. The following trajectory provides a qualitative timestep-level comparison.
# 
# All four agents are evaluated using:
# 
# - the same seed;
# - one greedy-policy episode;
# - the same maximum episode length; and
# - no exploration.
# 
# The graph records angle, angular velocity, and selected torque at every timestep.

# In[57]:


def collect_greedy_trajectory(result, seed=0, max_steps=200):
    """
    Collect one greedy-policy trajectory containing angle,
    angular velocity, torque, and reward.
    """
    agent = result["Agent"]
    gravity = float(result["Gravity"])
    label = result["Label"]
    action_values = np.asarray(
        result.get("Discrete Actions", DISCRETE_ACTIONS), dtype=np.float32
    ).reshape(-1)

    env = gym.make(ENV_NAME)
    env.seed(seed)
    env.action_space.seed(seed)
    env.unwrapped.g = gravity
    state = np.asarray(env.reset(), dtype=np.float32).reshape(-1)

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    trajectory_rows = []
    total_reward = 0.0

    try:
        for timestep in range(1, max_steps + 1):
            action_index = agent.select_action(state)
            torque_action = action_index_to_torque(action_index, action_values)
            torque_value = float(torque_action[0])

            next_state, reward, done, info = env.step(torque_action)
            next_state = np.asarray(next_state, dtype=np.float32).reshape(-1)

            theta = float(np.arctan2(next_state[1], next_state[0]))
            angular_velocity = float(next_state[2])
            total_reward += float(reward)

            trajectory_rows.append({
                "Gravity": label, "g": gravity, "Timestep": timestep,
                "Angle": theta, "Angular Velocity": angular_velocity,
                "Torque": torque_value, "Reward": float(reward)
            })

            state = next_state
            if done:
                break

    finally:
        agent.epsilon = original_epsilon
        env.close()

    print(f"{label} trajectory reward: {total_reward:.2f}")

    return pd.DataFrame(trajectory_rows)


# In[58]:


trajectory_frames = [
    collect_greedy_trajectory(result=result, seed=0, max_steps=200)
    for result in gravity_results
]
gravity_trajectory_df = pd.concat(trajectory_frames, ignore_index=True)

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

for gravity_label in gravity_order:
    gravity_data = gravity_trajectory_df[gravity_trajectory_df["Gravity"] == gravity_label]
    axes[0].plot(gravity_data["Timestep"], gravity_data["Angle"], label=gravity_label)
    axes[1].plot(gravity_data["Timestep"], gravity_data["Angular Velocity"], label=gravity_label)
    axes[2].step(gravity_data["Timestep"], gravity_data["Torque"], where="post", label=gravity_label)

# Angle graph
axes[0].axhline(0, linestyle="--", label="Upright position")
axes[0].axhline(UPRIGHT_ANGLE_THRESHOLD, linestyle=":", alpha=0.6)
axes[0].axhline(-UPRIGHT_ANGLE_THRESHOLD, linestyle=":", alpha=0.6)
axes[0].set_ylabel("Angle (radians)")
axes[0].set_title("Pendulum Angle")
axes[0].grid(alpha=0.3)
axes[0].legend()

# Angular velocity graph
axes[1].axhline(0, linestyle="--")
axes[1].set_ylabel("Angular Velocity")
axes[1].set_title("Pendulum Angular Velocity")
axes[1].grid(alpha=0.3)

# Torque graph
axes[2].axhline(0, linestyle="--")
axes[2].set_xlabel("Timestep")
axes[2].set_ylabel("Selected Torque")
axes[2].set_title("Agent-Applied Torque")
axes[2].set_yticks(DISCRETE_ACTIONS)
axes[2].grid(alpha=0.3)

plt.suptitle(
    "Greedy-Policy Behaviour Under Different Gravity Settings\nCommon Evaluation Seed = 0",
    fontsize=15
)
plt.tight_layout()
plt.show()


# #### **Trajectory Analysis**
# 
# Using the same evaluation seed, all four agents eventually moved the pendulum towards the upright position at `θ = 0` and reduced angular velocity close to zero.
# 
# - **Free-fall** approached the upright position smoothly with relatively small oscillations.
# - **Anti-gravity** initially overshot the target but stabilised quickly.
# - **Default gravity** required larger swings before reaching the upright position.
# - **Supergravity** showed the largest angular-velocity changes and took longer to stabilise, reflecting the stronger gravitational force.
# 
# The torque graph shows that the agents continued switching between discrete torque actions even after the pendulum became stable. This action switching is partly caused by the discretised action space.
# 
# Since this graph represents only one evaluation episode, it is used as qualitative evidence. The 20-episode evaluation rewards and physical-control metrics provide the main quantitative comparison.

# ### **5.9 Visualising Agent Behaviour Across Four Gravity Settings**
# 
# The four gravity-trained agents are visualised using the same evaluation seed. This provides a qualitative comparison of how the learned control behaviour changes under different gravitational conditions.
# 
# The animations are presented in the following order:
# 
# 1. Default gravity
# 2. Free-fall
# 3. Anti-gravity
# 4. Supergravity

# In[88]:


def render_agent_episode(result, max_steps=200, seed=0):
    """
    Run one greedy-policy episode for a trained agent and record frames
    for animation. Restores the agent's original epsilon afterward.
    """
    agent = result["Agent"]
    gravity = result["Gravity"]
    label = result["Label"]
    action_values = np.asarray(
        result.get("Discrete Actions", DISCRETE_ACTIONS),
        dtype=np.float32
    ).reshape(-1)

    render_env = gym.make(ENV_NAME)
    render_env.seed(seed)
    render_env.action_space.seed(seed)
    render_env.unwrapped.g = float(gravity)

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # force greedy policy for the animation

    frames = []
    state = render_env.reset().astype(np.float32)
    frames.append(render_env.render(mode="rgb_array"))

    total_reward = 0.0
    for _ in range(max_steps):
        action_index = agent.select_action(state)
        torque = action_index_to_torque(
            action_index,
            action_values
        )
        next_state, reward, done, _ = render_env.step(torque)
        frames.append(render_env.render(mode="rgb_array"))
        total_reward += reward
        state = next_state.astype(np.float32)
        if done:
            break

    render_env.close()
    agent.epsilon = original_epsilon  # restore trained agent's state

    print(f"[{label}] Rendered episode total reward: {total_reward:.2f}")
    return frames

def animate_frames(frames, title=""):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis("off")
    im = ax.imshow(frames[0])
    ax.set_title(title)

    def update(i):
        im.set_data(frames[i])
        return [im]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=50, blit=True
    )
    plt.close(fig)
    return anim


# In[89]:


import matplotlib.animation as animation
from IPython.display import HTML, display

gravity_agents = [
    ("Default Gravity (g = 10)", default_result),
    ("Free-fall (g = 0)", freefall_result),
    ("Anti-gravity", antigravity_result),
    ("Supergravity", supergravity_result),
]

for title, result in gravity_agents:
    frames = render_agent_episode(
        result,
        seed=0
    )

    animation_html = HTML(
        animate_frames(
            frames,
            title=title
        ).to_jshtml()
    )

    display(animation_html)


# ### **Analysis of the Four Gravity Settings**
# 
# - **Default gravity (`g = 10`)** created the standard Pendulum task. The agent had to overcome downward gravity, generate momentum, and apply corrective torque to remain upright.
# 
# - **Free-fall (`g = 0`)** removed the gravitational force. It achieved the best training and evaluation results, the lowest mean absolute angular velocity, and the highest stable-upright percentage.
# 
# - **Anti-gravity (`g = −10`)** reversed the gravitational direction, causing the agent to learn a different control strategy. It performed strongly and maintained the smallest mean absolute angle.
# 
# - **Supergravity (`g = 15`)** produced faster and less stable movement. It achieved the weakest evaluation performance and required more aggressive corrective actions.
# 
# Overall, free-fall performed best under the common baseline configuration, followed by anti-gravity, default gravity, and supergravity.

# ### **5.10 Summary of Gravity Setting Analysis**
# 
# #### **Purpose**
# 
# The same baseline DQN architecture, nine-action torque space, and hyperparameters were used to train separate agents under all four gravity settings. This provided a fair comparison of whether one common DQN setup could handle different environmental dynamics.
# 
# Hyperparameter tuning was conducted only under **default gravity**, since it represents the standard Pendulum environment.
# 
# #### **Overall Performance**
# 
# | **Gravity Setting** | **Evaluation Mean** | **Episodes to −300** | **Late Stability** | **Stable Upright (%)** | **Result** |
# |---|---:|---:|---:|---:|---|
# | **Free-fall** | **−54.21** | **49** | **6.93** | **87.68%** | Best overall |
# | **Anti-gravity** | −62.53 | 50 | 9.78 | 86.00% | Strong performance |
# | **Default gravity** | −192.28 | 258 | 28.85 | 81.70% | Moderate performance |
# | **Supergravity** | −305.38 | 374 | 83.12 | 71.18% | Weakest performance |
# 
# Lower late-stability values indicate less reward fluctuation near the end of training.
# 
# #### **Purpose of the Visualisations**
# 
# | **Visualisation** | **What It Shows** | **Main Finding** |
# |---|---|---|
# | **Zero-torque movement** | Natural pendulum motion under each gravity setting without agent control | The same initial state behaved differently under each gravity setting, confirming that gravity directly changes the environment dynamics |
# | **Physical-control metrics** | Angle error, angular velocity, control effort, and stable-upright time | Anti-gravity stayed slightly closest to upright, free-fall produced the smoothest movement and highest stable-upright percentage, while supergravity was least stable |
# | **Torque-action distribution** | Frequency of the nine discrete torque actions selected by each policy | Although all agents had the same action space, they learned substantially different torque-selection strategies |
# | **Common-seed trajectory** | Angle, angular velocity, and applied torque during the same evaluation episode | Free-fall stabilised with smaller movement, while supergravity showed the largest initial deviations and stronger corrective behaviour |
# 
# #### **Key Findings**
# 
# - **Free-fall** achieved the highest reward, fastest learning and smoothest movement.
# - **Anti-gravity** stayed closest to upright but used extreme torque relatively frequently.
# - **Default gravity** used the lowest average torque but required more episodes to learn.
# - **Supergravity** achieved the weakest reward, greatest movement and highest extreme-torque usage.
# 
# The common-seed trajectory represents only one episode and is therefore used as qualitative evidence. The multi-episode evaluation rewards and physical-control metrics remain the main quantitative evidence.
# 
# #### **Evaluation**
# 
# **Does one common DQN setup work well across all four gravity environments, or could some gravity settings benefit from separate tuning?**
# 
# The shared baseline DQN learned meaningful control behaviour in all four gravity environments, but performance differed substantially between them.
# 
# Free-fall and anti-gravity achieved strong results, default gravity was more challenging, and supergravity produced the weakest and least stable performance. The visualisations also showed that gravity changed both the pendulum dynamics and the torque-selection strategies learned by the agents.
# 
# Therefore, the common baseline configuration was suitable for comparing the four gravity settings fairly, but it was not necessarily optimal for every environment. In particular, the weaker supergravity result suggests that more difficult gravity conditions could benefit from separate tuning.
# 
# However, this project does not attempt to optimise every gravity setting individually. Hyperparameter tuning is conducted under **default gravity** because it represents the standard Pendulum task and provides a controlled benchmark for systematically improving the DQN configuration.

# ## **6. Hyperparameter Tuning**
# 
# **Objective**
# 
# Investigate how different hyperparameter settings affect DQN learning under the standard default-gravity environment.
# 
# Learning rate, target update frequency, and epsilon decay were changed **one at a time**, while the network architecture, gravity, training budget, and remaining hyperparameters were kept fixed.
# 
# Using default gravity ensured that differences in performance were caused by the hyperparameter being tested rather than by changes in the environment dynamics.

# ### **6.1 Tuning Approach**
# 
# Each hyperparameter was changed one at a time while all other settings remained at their baseline values.
# 
# | **Hyperparameter** | **Low (new run)** | **Baseline (`default_result`)** | **High (new run)** |
# |---|---:|---:|---:|
# | Learning rate | `0.00005` | `0.00025` | `0.001` |
# | Target update frequency | `5` | `10` | `25` |
# | Epsilon decay | `0.990` | `0.995` | `0.999` |
# 
# These hyperparameters were selected because they affect weight updates, target-network stability, and the balance between exploration and exploitation.
# 
# ### **Selection Criteria**
# 
# The preferred setting from each sweep was selected in the following order:
# 
# 1. **Highest mean greedy evaluation reward**
# 2. **Lower evaluation standard deviation when the means were similar**
# 3. **Faster and more stable reward progression**
# 4. **Smoother loss behaviour**
# 5. **Smaller training–evaluation gap as supporting evidence**
# 
# Mean greedy evaluation reward was the main criterion because it measures final policy performance without exploration. Since Pendulum rewards are negative, values closer to zero indicate better control.
# 
# Training reward, loss curves, and the training–evaluation gap were used only as supporting evidence because training includes epsilon-greedy exploration and uses different episode samples from evaluation.
# 
# Therefore, the preferred configuration was the one with the strongest and most consistent greedy evaluation performance.

# ### **6.2 Tune Learning Rate**
# 
# - Train the two new learning rates, **0.00005** and **0.001**, under the same conditions as `default_result`.
# - Compare whether a lower rate improves stability or a higher rate learns faster without destabilising training.

# In[59]:


tuning_results = []

LR_CANDIDATES = [0.00005, 0.001]
TARGET_UPDATE_CANDIDATES = [5, 25]
EPSILON_DECAY_CANDIDATES = [0.99, 0.999]


# **Step 1 - Train and Evaluate (learning rate =0.00005)**

# In[60]:


lr_low_result = train_agent(
    gravity=10.0,
    label="LR = 0.00005",
    hyperparams={"learning_rate": 0.00005},
    verbose=True
)
lr_low_result = evaluate_experiment(lr_low_result, verbose=True)
tuning_results.append(lr_low_result)


# **Step 2 - Train and Evaluate (learning rate =0.001)**

# In[61]:


lr_high_result = train_agent(
    gravity=10.0,
    label="LR = 0.001",
    hyperparams={"learning_rate": 0.001},
    verbose=True
)
lr_high_result = evaluate_experiment(lr_high_result, verbose=True)
tuning_results.append(lr_high_result)


# **Step 3 - Plot Training Curves**

# In[147]:


compare_training_results(
    results=[
        default_result,
        lr_low_result,
        lr_high_result
    ],
    labels=[
        "Baseline LR = 0.00025",
        "LR = 0.00005",
        "LR = 0.001"
    ],
    title="Learning Rate Comparison"
)


# **Step 4 - Summary Table**

# In[63]:


print("Learning Rate Sweep -- vs. Default-Gravity Baseline")
display(build_gap_summary([default_result, lr_low_result, lr_high_result]))


# #### **Learning Rate Analysis**
# 
# | **Factor** | **Best Setting** | **Observation** |
# |---|---|---|
# | **Final training performance** | **Default** | Highest final training average (**−137.69**) |
# | **Evaluation performance** | **Default** | Highest evaluation mean (**−192.28**) |
# | **Learning speed** | **Default** | Reached −300 fastest, in **258 episodes** |
# | **Late-training stability** | **0.00005** | Lowest stability value (**20.30**), indicating fewer late-training fluctuations |
# | **Loss behaviour** | **Default / 0.001** | Both maintained lower smoothed loss than `0.00005` during later training |
# | **Evaluation consistency** | **0.001** | Lowest evaluation standard deviation (**63.93**) |
# 
# - **Default:** Learned fastest and achieved the strongest training and evaluation performance.
# - **0.00005:** Had the most stable late-training rewards, but showed a noticeable mid-training performance drop and recovered more slowly.
# - **0.001:** Produced the most consistent evaluation results, but reached the −300 threshold slowest and had the largest late-training reward fluctuation (**56.51**).
# - **Epsilon:** All three settings used the same epsilon schedule, so differences were caused by the learning-rate change rather than exploration decay.
# 
# **Selected value:** **`0.00025` (Default)** because it achieved the highest mean greedy evaluation reward and fastest learning. Although `0.00005` was more stable late in training and `0.001` had a slightly lower evaluation standard deviation, neither improved the primary evaluation result.

# ### **6.3 Tune Target Update Frequency**
# 
# - default target update frequency = 10
# - Train the two new target update frequencies, **5** and **25**, under the same conditions as `default_result`.
# - Compare whether more frequent updates improve learning stability or less frequent updates produce more stable target values.

# **Step 1 - Train and Evaluate (target update frequency = 5)**

# In[65]:


tu_low_result = train_agent(
    gravity=10.0,
    label="Target Update = 5",
    hyperparams={"target_update_frequency": 5},
    verbose=True
)
tu_low_result = evaluate_experiment(tu_low_result, verbose=True)
tuning_results.append(tu_low_result)


# **Step 2 - Train and Evaluate (target update frequency = 25)**

# In[66]:


tu_high_result = train_agent(
    gravity=10.0,
    label="Target Update = 25",
    hyperparams={"target_update_frequency": 25},
    verbose=True
)
tu_high_result = evaluate_experiment(tu_high_result, verbose=True)
tuning_results.append(tu_high_result)


# **Step 3 - Plot Training Curves**

# In[148]:


compare_training_results(
    results=[
        default_result,
        tu_low_result,
        tu_high_result
    ],
    labels=[
        "Baseline Target Update = 10",
        "Target Update = 5",
        "Target Update = 25"
    ],
    title="Target Update Frequency Comparison"
)


# **Step 4 - Summary Table**

# In[68]:


print("Target Update Frequency Sweep -- vs. Default-Gravity Baseline")
display(build_gap_summary([default_result, tu_low_result, tu_high_result]))


# #### **Target Update Frequency Analysis**
# 
# | **Factor** | **Best Setting** | **Why** |
# |---|---|---|
# | **Final training performance** | **10 (Default)** | Highest final training average (**−137.69**) |
# | **Evaluation performance** | **10 (Default)** | Highest evaluation mean (**−192.28**) |
# | **Learning speed** | **5** | Reached −300 fastest, in **248 episodes** |
# | **Late-training stability** | **25** | Lowest stability value (**14.14**), indicating fewer late-training reward fluctuations |
# | **Loss behaviour** | **25** | Lowest and flattest smoothed loss curve during later training |
# | **Evaluation consistency** | **10 (Default)** | Lowest evaluation standard deviation (**66.86**) |
# | **Training–evaluation gap** | **25** | Smallest gap (**−21.85**), mainly because its training performance was weaker |
# 
# - **Target update = 5:** Learned fastest, reaching −300 in **248 episodes**, but its final training average (**−138.24**) and evaluation mean (**−194.03**) were slightly weaker than the baseline. It also had the highest evaluation variation (**69.18**).
# 
# - **Target update = 25:** Improved more slowly and reached −300 at episode **309**. It produced the most stable late-training rewards (**14.14**) and the lowest, flattest loss curve, but had the weakest final training average (**−177.80**) and evaluation mean (**−199.65**).
# 
# - **Target update = 10:** Achieved the strongest final training average (**−137.69**), best evaluation mean (**−192.28**), and lowest evaluation standard deviation (**66.86**).
# 
# - **Epsilon:** All settings used the same epsilon schedule, so exploration behaviour was controlled.
# 
# **Selected value:** **10 (Default)**
# 
# **Reason:** The baseline achieved the strongest greedy evaluation performance and most consistent evaluation results. Although `5` learned slightly faster and `25` produced more stable training and smoother loss behaviour, neither alternative improved the primary evaluation criterion.

# ### **6.4 Tune Epsilon Decay**
# 
# - Train the two new epsilon-decay values, **0.99** and **0.999**, under the same conditions as `default_result`.
# - Compare whether faster decay improves exploitation or slower decay provides more useful exploration.
# - Reuse **0.995** as the baseline without retraining.

# **Train and Evaluate (epsilon decay = 0.99)**

# In[69]:


eps_low_result = train_agent(
    gravity=10.0,
    label="Epsilon Decay = 0.99",
    hyperparams={"epsilon_decay": 0.99},
    verbose=True
)
eps_low_result = evaluate_experiment(eps_low_result, verbose=True)
tuning_results.append(eps_low_result)


# **Train and Evaluate (epsilon decay = 0.999)**

# In[70]:


eps_high_result = train_agent(
    gravity=10.0,
    label="Epsilon Decay = 0.999",
    hyperparams={"epsilon_decay": 0.999},
    verbose=True
)
eps_high_result = evaluate_experiment(eps_high_result, verbose=True)
tuning_results.append(eps_high_result)


# In[149]:


compare_training_results(
    results=[
        default_result,
        eps_low_result,
        eps_high_result
    ],
    labels=[
        "Baseline Epsilon Decay = 0.995",
        "Epsilon Decay = 0.99",
        "Epsilon Decay = 0.999"
    ],
    title="Epsilon Decay Comparison"
)


# **Summary Table**

# In[72]:


print("Epsilon Decay Sweep -- vs. Default-Gravity Baseline")
display(build_gap_summary([default_result, eps_low_result, eps_high_result]))


# #### **Epsilon Decay Analysis**
# 
# | **Factor** | **Best Setting** | **Why** |
# |---|---|---|
# | **Final training performance** | **Default (0.995)** | Highest final training average (**−137.69**) |
# | **Evaluation performance** | **0.99** | Highest evaluation mean (**−191.22**) |
# | **Learning speed** | **0.99** | Reached −300 fastest, in **155 episodes** |
# | **Late-training stability** | **0.99** | Lowest stability value (**16.79**), indicating fewer late-training reward fluctuations |
# | **Evaluation consistency** | **0.99** | Lowest evaluation standard deviation (**64.69**) |
# | **Training–evaluation gap** | **0.99** | Smaller gap (**−48.68**) than the baseline among configurations with comparable final evaluation performance |
# 
# - **Epsilon decay = 0.99:** Learned fastest, reaching −300 in **155 episodes**. It achieved the best evaluation mean (**−191.22**), lowest evaluation standard deviation (**64.69**), and more stable late-training rewards (**16.79**).
# 
# - **Epsilon decay = 0.999:** Exploration remained high throughout training, with epsilon still around `0.61` after 500 episodes. It did not reach the −300 training threshold and had a weak final training average (**−599.16**). Its large positive training–evaluation gap (**407.63**) occurred because training included substantial exploration and should not be interpreted as better performance.
# 
# - **Epsilon decay = 0.995:** Achieved the strongest final training average (**−137.69**), but learned more slowly and produced a slightly weaker evaluation mean than `0.99`.
# 
# **Selected value:** **0.99**
# 
# **Reason:** The improvement in mean evaluation reward over the baseline was small, but `0.99` also learned substantially faster, produced lower evaluation variation, and had more stable late-training rewards. Therefore, it provided the strongest overall trade-off across the prioritised evaluation criteria.

# ### **6.5 Overall Tuning Summary**

# In[74]:


print("Overall Hyperparameter Comparison -- vs. Default-Gravity Baseline")
display(build_gap_summary([default_result, lr_low_result, lr_high_result, tu_low_result, tu_high_result, eps_low_result, eps_high_result]))


# #### **Overall Tuning Conclusion**
# 
# The baseline used the following hyperparameters:
# 
# | **Baseline Hyperparameter** | **Value** |
# |---|---:|
# | Learning rate | **0.00025** |
# | Target update frequency | **10** |
# | Epsilon decay | **0.995** |
# 
# | **Criterion** | **Best Performer** | **Evidence** |
# |---|---|---|
# | **Highest evaluation reward** | **Epsilon decay = 0.99** | Best evaluation mean: **−191.22** |
# | **Best final training performance** | **Baseline** | Highest final training average: **−137.69** |
# | **Most consistent evaluation** | **Learning rate = 0.001** | Lowest evaluation standard deviation: **63.93** |
# | **Fastest learning** | **Epsilon decay = 0.99** | Reached −300 in **155 episodes** |
# | **Most stable late training** | **Target update = 25** | Lowest late-training stability value: **14.14** |
# | **Lowest and smoothest loss** | **Target update = 25** | Lowest and flattest late-training loss curve |
# 
# #### **Key Findings**
# 
# - Learning rate `0.00025` performed best among the tested learning-rate settings.
# - Target update `25` improved training stability and loss behaviour but weakened final evaluation performance.
# - Epsilon decay `0.99` achieved the best evaluation mean and fastest learning.
# - Epsilon decay `0.999` failed to reach the −300 threshold because exploration remained high.
# 
# #### **Final Selection**
# 
# | **Hyperparameter** | **Baseline Value** | **Selected Value** |
# |---|---:|---:|
# | Learning rate | **0.00025** | **0.00025** |
# | Target update frequency | **10** | **10** |
# | Epsilon decay | **0.995** | **0.99** |
# 
# #### **Conclusion**
# 
# The baseline learning rate and target update frequency were retained because their alternatives did not improve the primary greedy evaluation criterion.
# 
# Epsilon decay was changed from `0.995` to `0.99` because it achieved the highest mean greedy evaluation reward (**−191.22**) among the tuning configurations. It also produced lower evaluation variation (**64.69**), reached the −300 threshold much faster (**155 vs 258 episodes**), and showed more stable late-training rewards (**16.79 vs 28.85**).
# 
# Although the improvement in evaluation mean over the baseline was small, epsilon decay `0.99` performed better across several of the prioritised criteria. Therefore, learning rate `0.00025`, target update frequency `10`, and epsilon decay `0.99` formed the **best-tested default-gravity configuration**.
# 
# The selection is based primarily on greedy evaluation performance, with evaluation consistency, learning speed, and training stability used as supporting evidence.
# 
# These results should still be interpreted cautiously because each tuning setting was initially tested using only one random seed. Additional combined and repeated-seed experiments are therefore conducted to further validate the selected configuration.

# ### **6.6 Combined Interaction Test**
# 
# The tuning results selected **epsilon decay = 0.99**, while the baseline learning rate and target update frequency remained best for final reward performance.
# 
# However, **target update frequency = 25** produced the most stable late-training rewards and the smoothest loss curve. A combined test is therefore conducted to check whether this stability benefit can complement the faster learning of epsilon decay `0.99`.
# 
# | **Hyperparameter**      | **Combined Value** |
# | ----------------------- | -----------------: |
# | Learning rate           |          `0.00025` |
# | Target update frequency |               `25` |
# | Epsilon decay           |             `0.99` |
# 
# The combined model will be  compared with the selected epsilon decay = 0.99 model.

# In[75]:


combined_result = train_agent(
    gravity=10.0,
    label="Target Update = 25, Epsilon Decay = 0.99",
    hyperparams={
        "target_update_frequency": 25,
        "epsilon_decay": 0.99
    },
    verbose=True
)

combined_result = evaluate_experiment(
    combined_result,
    verbose=True
)

tuning_results.append(combined_result)


# **Plot Training Curves**

# In[150]:


compare_training_results(
    results=[
        eps_low_result,
        combined_result
    ],
    labels=[
        "Epsilon Decay = 0.99",
        "Target Update = 25 + Epsilon Decay = 0.99"
    ],
    title="Selected vs Combined Configuration"
)


# **Summary Table**

# In[152]:


print("Combined Hyperparameter Test -- Epsilon Decay 0.99 vs Combined Configuration")
display(build_gap_summary([eps_low_result, combined_result]))


# #### **Combined Hyperparameter Analysis**
# 
# | **Factor** | **Best Setting** | **Why** |
# |---|---|---|
# | **Final training performance** | **Combined setting** | Slightly higher final training average (**−137.50** vs **−142.54**) |
# | **Evaluation performance** | **Epsilon decay = 0.99** | Higher evaluation mean (**−191.22** vs **−195.71**) |
# | **Learning speed** | **Epsilon decay = 0.99** | Reached −300 much faster, in **155 episodes** compared with **264** |
# | **Late-training stability** | **Epsilon decay = 0.99** | Lower stability value (**16.79** vs **21.83**) |
# | **Loss behaviour** | **Combined setting** | Produced a much lower and flatter smoothed loss curve |
# | **Evaluation consistency** | **Epsilon decay = 0.99** | Lower evaluation standard deviation (**64.69** vs **67.64**) |
# | **Training–evaluation gap** | **Epsilon decay = 0.99** | Smaller negative gap (**−48.68** vs **−58.21**) |
# 
# The combined setting achieved a slightly better final training average and clearly lower loss. However, it learned substantially more slowly and produced weaker and less consistent greedy evaluation performance.
# 
# This shows that a lower loss does not necessarily correspond to a better control policy. The combined setting may have produced more stable Q-value updates, but this did not translate into stronger pendulum-control performance.
# 
# **Selected configuration:** **Epsilon decay = `0.99` with target update frequency = `10`**
# 
# **Reason:** It achieved the stronger greedy evaluation mean, faster learning, lower evaluation variation, and more stable late-training rewards. Therefore, the additional change to target update frequency `25` was not retained.

# ## **7. Robustness and Action-Resolution Experiments**
# 
# After the individual hyperparameter tuning experiments, epsilon decay `0.99` was selected as the best-tested setting under default gravity. A combined test also showed that adding target update frequency `25` did not improve the final greedy policy.
# 
# However, the improvement in evaluation mean over the original baseline was small:
# 
# | **Configuration** | **Evaluation Mean** | **Evaluation Std** | **Episodes to −300** | **Late-Training Stability** |
# |---|---:|---:|---:|---:|
# | **Baseline (`0.995`)** | −192.28 | 66.86 | 258 | 28.85 |
# | **Epsilon decay `0.99`** | **−191.22** | **64.69** | **155** | **16.79** |
# 
# Epsilon decay `0.99` reached the `−300` threshold **103 episodes earlier** and produced more stable late-training rewards. This indicates a clear improvement in learning speed and stability.
# 
# However, the evaluation mean improved by only **1.06 points**, which is small compared with the evaluation standard deviations of about `65`. Therefore, additional robustness experiments are conducted before confirming the final configuration.
# 
# This stage has two purposes:
# 
# 1. **Repeated-seed validation** — compare baseline decay `0.995` and selected decay `0.99` using the same training seeds to determine whether the differences in evaluation performance, learning speed, and stability are reproducible.
# 2. **Action-resolution sensitivity check** — compare five and nine discrete torque actions under the selected configuration to determine whether reducing the action-space resolution provides any clear advantage.
# 
# The repeated-seed experiments use seeds `42`, `123`, and `2026`. All models are evaluated greedily using the same sequence of evaluation initial states, providing a fairer comparison across training runs.

# ### **7.1 Baseline Epsilon Decay `0.995` Across Seeds**
# 

# **Train and evaluate baseline (seed = 123)**

# In[78]:


baseline_seed_123_result = train_agent(
    gravity=10.0,
    label="Baseline 0.995 (seed=123)",
    seed=123,
    hyperparams={"epsilon_decay": 0.995},
    verbose=True
)

baseline_seed_123_result = evaluate_experiment(
    baseline_seed_123_result,
    verbose=True
)


# **Train and evaluate baseline (seed = 2026)**

# In[79]:


baseline_seed_2026_result = train_agent(
    gravity=10.0,
    label="Baseline 0.995 (seed=2026)",
    seed=2026,
    hyperparams={"epsilon_decay": 0.995},
    verbose=True
)

baseline_seed_2026_result = evaluate_experiment(
    baseline_seed_2026_result,
    verbose=True
)

baseline_seed_results = [
    default_result,
    baseline_seed_123_result,
    baseline_seed_2026_result
]

print("Baseline Epsilon Decay 0.995 -- Repeated Seeds")
display(build_gap_summary(baseline_seed_results))


# ### **7.2 Tuned Epsilon Decay `0.99` Across Seeds**
# 
# The selected epsilon decay `0.99` is evaluated using the same three training seeds. The existing seed-42 run is reused, while seeds `123` and `2026` are trained below.
# 

# **Train and evaluate tuned configuration (seed = 123)**

# In[80]:


seed_trial_results = [eps_low_result]  # Existing seed 42 result

seed_123_result = train_agent(
    gravity=10.0,
    label="Epsilon Decay = 0.99 (seed=123)",
    seed=123,
    hyperparams={"epsilon_decay": 0.99},
    verbose=True
)

seed_123_result = evaluate_experiment(
    seed_123_result,
    verbose=True
)

seed_trial_results.append(seed_123_result)


# **Train and evaluate tuned configuration (seed = 2026)**

# In[81]:


seed_2026_result = train_agent(
    gravity=10.0,
    label="Epsilon Decay = 0.99 (seed=2026)",
    seed=2026,
    hyperparams={"epsilon_decay": 0.99},
    verbose=True
)

seed_2026_result = evaluate_experiment(
    seed_2026_result,
    verbose=True
)

seed_trial_results.append(seed_2026_result)


# ### **7.3 Baseline `0.995` vs Tuned `0.99` Across the Same Seeds**
# 
# The following comparison uses the same training seeds for both epsilon-decay settings. The evaluation mean is the main final-policy measure, while episodes to `−300` and late-training stability are supporting measures.
# 

# In[159]:


baseline_seed_results = [
    default_result,
    baseline_seed_123_result,
    baseline_seed_2026_result
]

tuned_seed_results = [
    eps_low_result,
    seed_123_result,
    seed_2026_result
]


print("Baseline epsilon decay 0.995")
display(build_gap_summary(baseline_seed_results))

print("Tuned epsilon decay 0.99")
display(build_gap_summary(tuned_seed_results))


def stats(results):
    means = [r["Mean Eval Reward"] for r in results]
    eval_stds = [r["Std Eval Reward"] for r in results]
    episodes = [episodes_to_threshold(r) for r in results]
    stability = [late_training_stability(r) for r in results]

    return (
        np.mean(means),
        np.std(means),
        np.mean(eval_stds),
        np.nanmean(episodes),
        np.mean(stability)
    )

baseline_stats = stats(baseline_seed_results)
tuned_stats = stats(tuned_seed_results)

summary_table = pd.DataFrame({
    "Configuration": [
        "Baseline epsilon decay 0.995",
        "Tuned epsilon decay 0.99"
    ],
    "Mean Evaluation Reward": [
        baseline_stats[0],
        tuned_stats[0]
    ],
    "SD Across Seeds": [
        baseline_stats[1],
        tuned_stats[1]
    ],
    "Mean Evaluation Std": [
        baseline_stats[2],
        tuned_stats[2]
    ],
    "Mean Episodes to -300": [
        baseline_stats[3],
        tuned_stats[3]
    ],
    "Mean Late-Training Stability": [
        baseline_stats[4],
        tuned_stats[4]
    ]
})

display(summary_table.round(2))


# #### **Across-Seed Comparison Analysis**
# 
# Across the three training seeds, epsilon decay `0.99` produced better overall results than the baseline decay `0.995`.
# 
# - **Mean evaluation reward:** `0.99` achieved **−190.94**, compared with **−195.36** for the baseline.
# - **Variation across training seeds:** `0.99` had a much lower SD across seeds (**0.56** vs **2.48**), indicating that its final evaluation means were more consistent between training runs.
# - **Evaluation consistency:** `0.99` also achieved a slightly lower mean evaluation standard deviation (**65.21** vs **66.50**), indicating slightly more consistent performance across evaluation episodes.
# - **Learning speed:** `0.99` reached the `−300` threshold in an average of **147.67 episodes**, compared with **218.00 episodes** for the baseline.
# - **Late-training stability:** `0.99` achieved a lower mean stability value (**19.47** vs **28.21**), indicating fewer reward fluctuations late in training.
# 
# The largest advantage of epsilon decay `0.99` was in **learning speed and late-training stability**. Its mean final evaluation reward was also better, although the difference remained relatively small compared with the variation within individual evaluation runs.
# 
# **Conclusion:** The repeated-seed results support retaining epsilon decay `0.99`. It achieved a better mean greedy evaluation reward, slightly better evaluation consistency, substantially faster learning, and more stable late-training performance across the tested seeds.

# ### **7.4 Five-Action vs Nine-Action Sensitivity Check**
# 
# After validating the selected hyperparameter configuration across multiple random seeds, an additional sensitivity experiment was conducted to examine whether action-space resolution affected DQN performance.
# 
# DQN requires a discrete action space, so the continuous Pendulum torque range from `−2.0` to `2.0` was discretised into:
# 
# - **5 actions:** `[-2, -1, 0, 1, 2]`
# - **9 actions:** `[-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]`
# 
# This experiment is not another hyperparameter-tuning stage. Instead, it checks whether reducing the original nine-action resolution provides any clear performance or control advantage.
# 
# | **Controlled Item** | **Setting** |
# |---|---|
# | Gravity | `10.0` |
# | Learning rate | `0.00025` |
# | Target update frequency | `10` |
# | Epsilon decay | `0.99` |
# | Training episodes | `500` |
# | Training seed | `42` |
# | Evaluation episodes | `20`, greedy |
# 
# The existing nine-action tuned model is reused, while only the five-action model requires a new training run.
# 
# The comparison considers greedy evaluation reward, learning speed, late-training stability, and physical-control behaviour. Physical metrics are included because changing the action count directly changes the torque choices available to the agent.

# In[85]:


FIVE_DISCRETE_ACTIONS = np.linspace(
    ACTION_MIN,
    ACTION_MAX,
    5,
    dtype=np.float32
)

nine_action_result = dict(eps_low_result)
nine_action_result["Label"] = "9 Actions"
nine_action_result["Discrete Actions"] = DISCRETE_ACTIONS.copy()
nine_action_result["Action Count"] = 9

five_action_result = train_agent(
    gravity=10.0,
    label="5 Actions",
    seed=42,
    hyperparams={
        "learning_rate": 0.00025,
        "target_update_frequency": 10,
        "epsilon_decay": 0.99
    },
    discrete_actions=FIVE_DISCRETE_ACTIONS,
    verbose=True
)

five_action_result = evaluate_experiment(
    five_action_result,
    verbose=True
)

action_resolution_results = [
    five_action_result,
    nine_action_result
]


# In[165]:


print("Action-Resolution Test -- 5 Actions vs 9 Actions")
display(
    build_gap_summary([
        five_action_result,
        eps_low_result
    ])
)


# In[167]:


action_results = [
    ("5 Actions", 5, five_action_result),
    ("9 Actions", 9, eps_low_result)
]

action_order = [item[0] for item in action_results]


all_episode_data = []
all_step_data = []

for configuration, action_count, result in action_results:
    episode_data, step_data = evaluate_control_behaviour(
        result=result,
        num_episodes=20,
        seed=COMMON_EVALUATION_SEED
    )

    # Replace gravity label with action-space configuration
    episode_data["Configuration"] = configuration
    episode_data["Action Count"] = action_count

    step_data["Configuration"] = configuration
    step_data["Action Count"] = action_count

    all_episode_data.append(episode_data)
    all_step_data.append(step_data)


action_episode_df = pd.concat(all_episode_data, ignore_index=True)
action_step_df = pd.concat(all_step_data, ignore_index=True)


for df in (action_episode_df, action_step_df):
    df["Configuration"] = pd.Categorical(
        df["Configuration"],
        categories=action_order,
        ordered=True
    )


ACTION_AGG_MAP = {
    "Mean Evaluation Reward": ("Episode Reward", "mean"),
    "Evaluation Reward Std": ("Episode Reward", lambda v: np.std(v, ddof=0)),
    "Mean Absolute Angle": ("Mean Absolute Angle", "mean"),
    "Mean Absolute Angular Velocity": ("Mean Absolute Angular Velocity", "mean"),
    "Mean Absolute Torque": ("Mean Absolute Torque", "mean"),
    "Stable Upright (%)": ("Stable Upright (%)", "mean"),
    "Extreme Torque (%)": ("Extreme Torque (%)", "mean"),
}


action_control_summary = (
    action_episode_df
    .groupby(
        ["Configuration", "Action Count"],
        observed=True,
        sort=False
    )
    .agg(**ACTION_AGG_MAP)
    .reset_index()
)


display(action_control_summary.round(3))


# ### **Action-Resolution Sensitivity Check**
# 
# A five-action version of the selected DQN was compared with the original nine-action design using the same training seed and hyperparameter configuration.
# 
# - **Final training performance:** 5 actions achieved a slightly better final training average (**−136.74** vs **−142.54**).
# - **Evaluation performance:** 9 actions achieved a better greedy evaluation mean (**−191.22** vs **−194.33**).
# - **Evaluation consistency:** 9 actions produced a lower evaluation standard deviation (**64.69** vs **72.53**).
# - **Learning speed:** Both configurations learned at almost the same rate, reaching the `−300` threshold in **154** and **155 episodes** respectively.
# - **Late-training stability:** 9 actions produced more stable late-training rewards (**16.79** vs **20.26**).
# - **Control accuracy:** Both configurations achieved nearly identical mean absolute angle (**0.342**) and stable-upright percentages (**81.1%** vs **81.2%**).
# - **Control effort:** 5 actions used substantially lower mean absolute torque (**0.444** vs **1.370**) and less extreme-torque usage (**12.78%** vs **23.65%**).
# 
# The five-action configuration therefore reduced control effort, but this did not improve final greedy evaluation performance or late-training stability. The nine-action configuration achieved better evaluation reward, lower evaluation variation, and more stable learning while maintaining almost identical upright-control performance.
# 
# **Conclusion:** The original nine-action discretisation was retained because it performed better on the main selection criteria of greedy evaluation performance, evaluation consistency, and late-training stability. The five-action experiment was treated as a sensitivity check rather than a separate tuning stage, so no additional repeated-seed tuning was performed for the five-action configuration.

# ### **8.1 Selected Configuration Across Gravity Settings**
# 
# The robustness experiments showed that epsilon decay `0.99` consistently outperformed the baseline decay `0.995` across three training seeds. It achieved a higher mean evaluation reward, faster learning, lower variation across seeds, and better late-training stability.
# 
# The nine-action configuration was retained because it achieved a slightly higher evaluation reward and more stable late training than the five-action configuration.
# 
# The selected configuration is therefore:
# 
# - Learning rate: `0.00025`
# - Target update frequency: `10`
# - Epsilon decay: `0.99`
# - Number of discrete torque actions: `9`
# 
# This configuration was selected using default gravity. It is now applied unchanged across the four gravity settings to evaluate whether the tuned setup generalises to different environmental dynamics.
# 
# A new agent is trained from scratch for each gravity setting using the same seed. This is a generalisation experiment rather than separate hyperparameter tuning for each environment.

# In[115]:


SELECTED_HYPERPARAMS = {
    "learning_rate": 0.00025,
    "target_update_frequency": 10,
    "epsilon_decay": 0.99
}

SELECTED_SEED = 42


# **1. Training and Evaluation of Tuned Free fall**

# In[116]:


tuned_freefall_result = train_agent(
    gravity=0.0,
    label="Tuned Free-fall",
    hyperparams=SELECTED_HYPERPARAMS,
    seed=SELECTED_SEED,
    verbose=True
)

tuned_freefall_result = evaluate_experiment(
    tuned_freefall_result,
    verbose=True
)


# **2. Training and Evaluation of Tuned Anti Gravity**

# In[117]:


tuned_antigravity_result = train_agent(
    gravity=-10.0,
    label="Tuned Anti-gravity",
    hyperparams=SELECTED_HYPERPARAMS,
    seed=SELECTED_SEED,
    verbose=True
)

tuned_antigravity_result = evaluate_experiment(
    tuned_antigravity_result,
    verbose=True
)


# **3. Training and Evaluation of Tuned Supergravity**

# In[118]:


tuned_supergravity_result = train_agent(
    gravity=15.0,
    label="Tuned Supergravity",
    hyperparams=SELECTED_HYPERPARAMS,
    seed=SELECTED_SEED,
    verbose=True
)

tuned_supergravity_result = evaluate_experiment(
    tuned_supergravity_result,
    verbose=True
)


# In[173]:


compare_training_results(
    results=[
        eps_low_result,
        tuned_freefall_result,
        tuned_antigravity_result,
        tuned_supergravity_result
    ],
    labels=[
        "Tuned Default Gravity",
        "Tuned Free-fall",
        "Tuned Anti-gravity",
        "Tuned Supergravity"
    ],
    title="Tuned Gravity Comparison"
)


# ### **8.2 Training and Evaluation Summary Table**

# In[121]:


tuned_gravity_results = [
    eps_low_result,                 # Tuned default gravity
    tuned_freefall_result,
    tuned_antigravity_result,
    tuned_supergravity_result
]

display(build_gap_summary(tuned_gravity_results))

baseline_gravity_results = [
    default_result,
    freefall_result,
    antigravity_result,
    supergravity_result
]

display(build_gap_summary(baseline_gravity_results))


# #### **Tuned Configuration Across Gravity Settings — Evaluation Performance**
# 
# After selecting epsilon decay `0.99` under default gravity, the same tuned configuration was applied to the other gravity settings to examine whether the improvement generalised across different environmental dynamics.
# 
# | **Gravity** | **Baseline Eval Mean** | **Tuned Eval Mean** | **Baseline Eval Std** | **Tuned Eval Std** | **Baseline Episodes to −300** | **Tuned Episodes to −300** |
# |---|---:|---:|---:|---:|---:|---:|
# | Default | −192.28 | **−191.22** | 66.86 | **64.69** | 258 | **155** |
# | Free-fall | −54.21 | **−53.65** | 35.21 | **34.91** | 49 | **40** |
# | Anti-gravity | −62.53 | **−61.62** | 41.03 | **40.87** | 50 | **40** |
# | Supergravity | −305.38 | **−233.27** | 116.36 | **97.48** | 374 | **246** |
# 
# Evaluation mean improved for all four gravity settings, though by a small margin for default gravity, free-fall, and anti-gravity. The gain under supergravity was far larger — from **−305.38** to **−233.27** — with evaluation variance also dropping (**116.36 → 97.48**). Learning speed improved across the board too, most notably under supergravity, where the `−300` threshold was reached in **246 episodes** instead of **374**.

# ### **8.3 Quantitative Control-Behaviour Evaluation**

# In[169]:


def build_control_summary(results):
    order = [r["Label"] for r in results]
    episode_dfs, step_dfs = [], []
    for r in results:
        ep, step = evaluate_control_behaviour(result=r, num_episodes=20, seed=COMMON_EVALUATION_SEED)
        episode_dfs.append(ep)
        step_dfs.append(step)

    episode_df = pd.concat(episode_dfs, ignore_index=True)
    step_df = pd.concat(step_dfs, ignore_index=True)
    for df in (episode_df, step_df):
        df["Gravity"] = pd.Categorical(df["Gravity"], categories=order, ordered=True)

    AGG_MAP = {
        "Mean Evaluation Reward": ("Episode Reward", "mean"),
        "Evaluation Reward Std": ("Episode Reward", lambda v: np.std(v, ddof=0)),
        "Mean Absolute Angle": ("Mean Absolute Angle", "mean"),
        "Mean Absolute Angular Velocity": ("Mean Absolute Angular Velocity", "mean"),
        "Mean Absolute Torque": ("Mean Absolute Torque", "mean"),
        "Stable Upright (%)": ("Stable Upright (%)", "mean"),
        "Extreme Torque (%)": ("Extreme Torque (%)", "mean"),
    }

    summary = (
        episode_df.groupby(["Gravity", "g"], observed=True, sort=False)
        .agg(**AGG_MAP)
        .reset_index()
    )
    return episode_df, step_df, summary


def best_setting(summary, col, mode="min"):
    idx = summary[col].idxmin() if mode == "min" else summary[col].idxmax()
    return summary.loc[idx, "Gravity"]


def report_best_settings(summary):
    print("Lowest mean absolute angle:", best_setting(summary, "Mean Absolute Angle"))
    print("Lowest mean absolute angular velocity:", best_setting(summary, "Mean Absolute Angular Velocity"))
    print("Lowest mean absolute torque:", best_setting(summary, "Mean Absolute Torque"))
    print("Highest stable-upright percentage:", best_setting(summary, "Stable Upright (%)", "max"))
    print("Highest extreme-torque usage:", best_setting(summary, "Extreme Torque (%)", "max"))


# --- Baseline ---
baseline_gravity_results = [default_result, freefall_result, antigravity_result, supergravity_result]
control_episode_df, control_step_df, control_summary = build_control_summary(baseline_gravity_results)
print("Baseline gravity control summary")
display(control_summary.round(3))
report_best_settings(control_summary)

# --- Tuned ---
tuned_gravity_results = [eps_low_result, tuned_freefall_result, tuned_antigravity_result, tuned_supergravity_result]
tuned_control_episode_df, tuned_control_step_df, tuned_control_summary = build_control_summary(tuned_gravity_results)
print("Tuned gravity control summary")
display(tuned_control_summary.round(3))
report_best_settings(tuned_control_summary)


# 
# #### **Tuned Gravity Control-Behaviour Analysis**
# 
# Beyond reward and learning speed, tuning also changed how the agents physically controlled the pendulum:
# 
# - **Default gravity:** Control accuracy remained similar, but the tuned agent used higher mean torque (**1.370 vs 0.995**) and more extreme torque (**23.65% vs 10.83%**).
# - **Free-fall:** The tuned agent achieved lower angular velocity (**0.245 vs 0.278**), a higher stable-upright percentage (**89.15% vs 87.68%**), and lower mean torque (**0.772 vs 1.316**).
# - **Anti-gravity:** Mean absolute angle decreased from **0.181** to **0.147**, indicating more accurate upright control.
# - **Supergravity:** The largest behavioural improvement occurred here — mean absolute angle decreased from **0.474** to **0.356**, mean absolute angular velocity decreased from **1.127** to **0.859**, and stable-upright time increased from **71.18%** to **80.60%**. This came at a cost: mean absolute torque rose from **1.159** to **1.489**, and extreme-torque usage increased from **37.90%** to **63.68%**.
# 
# **Conclusion:** Epsilon decay `0.99`, tuned on default gravity, generalised across all four gravity environments — improving reward, learning speed, and stability everywhere, and additionally improving control accuracy under free-fall, anti-gravity, and especially supergravity. The strongest gains, in both reward and control quality, appeared under supergravity, but came at the cost of greater torque usage — a trade-off between control performance and control effort.

# ### **8.4 Tuned Gravity Animations**

# In[174]:


from IPython.display import HTML, display

tuned_gravity_results = [
    ("Tuned Default Gravity (g = 10)", eps_low_result),
    ("Tuned Free-fall (g = 0)", tuned_freefall_result),
    ("Tuned Anti-gravity (g = -10)", tuned_antigravity_result),
    ("Tuned Supergravity (g = 15)", tuned_supergravity_result)
]

for title, result in tuned_gravity_results:

    frames = render_agent_episode(
        result,
        seed=0
    )

    display(
        HTML(
            animate_frames(
                frames,
                title=title
            ).to_jshtml()
        )
    )


# #### **Qualitative Animation Analysis**
# 
# The final tuned configuration with epsilon decay `0.99` was visualised under all four gravity settings using the same evaluation seed.
# 
# - **Default gravity:** The tuned agent demonstrated successful pendulum control. This is consistent with the quantitative results, although the improvement over the earlier baseline evaluation was relatively small.
# - **Free-fall:** The tuned agent maintained relatively smooth control, consistent with its high stable-upright percentage and low angular velocity.
# - **Anti-gravity:** The tuned agent successfully adapted to the reversed gravitational dynamics and maintained accurate upright control.
# - **Supergravity:** The agent was still exposed to larger disturbances, but it recovered and stabilised the pendulum effectively. This agrees with the substantial improvements in evaluation reward, angular velocity, and stable-upright percentage after tuning.
# 
# Overall, the animations show that the selected configuration produced meaningful control across all four gravity environments. The strongest quantitative improvement occurred under supergravity.
# 
# However, each animation represents only one evaluation episode. Therefore, the multi-episode reward metrics and physical-control measurements remain the primary evidence for evaluating performance.

# ### **8.5 Final Model Selection**
# 
# The final submitted weights correspond to the tuned default-gravity DQN with:
# 
# - learning rate `0.00025`;
# - target update frequency `10`;
# - epsilon decay `0.99`;
# - nine discrete torque actions.
# 
# This configuration was selected because it achieved slightly better mean greedy evaluation performance, substantially faster learning, and more stable late-training rewards than the baseline. The repeated-seed experiment further supported these advantages.
# 
# Although free-fall achieved the highest reward across the gravity experiments, it represents an easier environment rather than the standard default-gravity benchmark used for hyperparameter tuning. Therefore, the tuned default-gravity model was selected as the final submitted model.

# In[ ]:


selected_result = eps_low_result

agent = selected_result["Agent"]

agent.online_network.save_weights(
    "final_dqn_selected_model.weights.h5"
)

print(
    "Weights saved successfully from:",
    selected_result["Label"]
)


# # **9. Final Conclusion**
# 
# This investigation examined whether a discretised-action Deep Q-Network could learn effective Pendulum control under different gravity conditions, whether systematic hyperparameter tuning could improve the default-gravity agent, and whether the selected configuration would remain reliable under additional robustness checks.
# 
# The initial hypotheses predicted that gravity would affect task difficulty and learned control behaviour, that DQN hyperparameters would influence learning speed and stability, that training outcomes would vary across random seeds, and that action-space resolution could affect the trade-off between control precision and learning difficulty.
# 
# The experimental results are used below to answer each research question and assess these hypotheses.
# 
# ---
# 
# ## **9.1 Research Question 1**
# 
# ### **How does changing gravity affect DQN learning and control behaviour?**
# 
# The results support the **gravity-effect hypothesis**. Gravity had a clear effect on both learning difficulty and the physical behaviour of the learned policy.
# 
# Under the shared baseline configuration:
# 
# - **Free-fall** achieved the strongest overall performance, with an evaluation mean of **−54.21**, fast learning, and the highest stable-upright percentage.
# - **Anti-gravity** also performed strongly, with an evaluation mean of **−62.53** and the lowest mean absolute angle.
# - **Default gravity** was more challenging, achieving an evaluation mean of **−192.28**.
# - **Supergravity** was the most difficult environment, producing the weakest evaluation mean (**−305.38**), greater evaluation variation, slower learning, and less stable control.
# 
# The physical-control analysis also showed differences in angle, angular velocity, torque usage, and stable-upright behaviour across the four environments.
# 
# Therefore, changing gravity affects not only the reward obtained by the DQN but also the control strategy required to stabilise the pendulum.
# 
# ---
# 
# ## **9.2 Research Question 2**
# 
# ### **Can systematic hyperparameter tuning improve the performance, learning speed, and stability of the default-gravity DQN?**
# 
# The results support the **hyperparameter-effect hypothesis**.
# 
# Learning rate, target update frequency, and epsilon decay were varied systematically while other experimental settings were kept fixed.
# 
# The baseline learning rate of `0.00025` and target update frequency of `10` were retained because their alternatives did not improve the primary greedy evaluation criterion.
# 
# Epsilon decay was changed from `0.995` to `0.99`.
# 
# For the initial seed, the improvement in evaluation mean was relatively small:
# 
# - evaluation mean: **−192.28 → −191.22**
# - evaluation standard deviation: **66.86 → 64.69**
# 
# However, the tuned configuration showed much stronger improvements in learning behaviour:
# 
# - episodes to the `−300` threshold: **258 → 155**
# - late-training stability: **28.85 → 16.79**
# 
# A combined configuration using target update frequency `25` and epsilon decay `0.99` produced a lower training loss but weaker greedy evaluation performance. This demonstrates that a lower loss does not necessarily correspond to a better final control policy.
# 
# Therefore, systematic tuning improved the overall balance of final policy performance, learning speed, and training stability.
# 
# ---
# 
# ## **9.3 Research Question 3**
# 
# ### **Does the selected epsilon-decay configuration remain reliable across different training seeds, and does reducing the action-space resolution provide any clear advantage?**
# 
# The repeated-seed results support the **training-variability hypothesis** and provide additional evidence for retaining epsilon decay `0.99`.
# 
# Across seeds `42`, `123`, and `2026`:
# 
# | **Metric** | **Baseline 0.995** | **Tuned 0.99** |
# |---|---:|---:|
# | Mean evaluation reward | −195.36 | **−190.94** |
# | SD across seeds | 2.48 | **0.56** |
# | Mean evaluation std | 66.50 | **65.21** |
# | Mean episodes to −300 | 218.00 | **147.67** |
# | Mean late-training stability | 28.21 | **19.47** |
# 
# Epsilon decay `0.99` therefore achieved slightly better and more consistent greedy evaluation performance while learning substantially faster and showing more stable late-training behaviour.
# 
# The **action-resolution hypothesis** was also investigated using five and nine discrete torque actions.
# 
# The five-action configuration produced lower mean torque usage and a slightly better final training average. However, the nine-action configuration achieved:
# 
# - a better greedy evaluation mean;
# - lower evaluation variation;
# - better late-training stability; and
# - almost identical stable-upright performance.
# 
# Therefore, reducing the action space to five actions did not provide a clear overall advantage, and the original **nine-action discretisation** was retained.
# 
# ---
# 
# ## **9.4 Research Question 4**
# 
# ### **Can one common DQN configuration provide effective control across all four gravity environments, or do some gravity settings remain substantially more difficult than others?**
# 
# The baseline experiment showed that one common DQN configuration could learn meaningful control behaviour across all four gravity settings, although performance differed substantially between environments.
# 
# The selected epsilon decay `0.99`, tuned under default gravity, was then applied unchanged to all four gravity settings.
# 
# | **Gravity** | **Baseline Evaluation Mean** | **Tuned Evaluation Mean** |
# |---|---:|---:|
# | Default | −192.28 | **−191.22** |
# | Free-fall | −54.21 | **−53.65** |
# | Anti-gravity | −62.53 | **−61.62** |
# | Supergravity | −305.38 | **−233.27** |
# 
# The selected configuration improved evaluation performance in all four environments.
# 
# The largest improvement occurred under **supergravity**. Its mean absolute angle and angular velocity decreased, while stable-upright time increased from **71.18%** to **80.60%**.
# 
# However, the improved supergravity control also required more aggressive torque usage, showing a trade-off between control performance and control effort.
# 
# Therefore, a common tuned DQN configuration can provide effective control across all four gravity environments, but gravity still has a substantial effect on task difficulty. Supergravity remained the most challenging condition and could potentially benefit from gravity-specific tuning.
# 
# ---
# 
# ## **9.5 Limitations and Future Improvements**
# 
# ### **Limitations**
# 
# - **Limited repeated trials:** Repeated-seed testing was performed for the selected epsilon-decay comparison, while most individual hyperparameter configurations were initially evaluated using one training seed. Some tuning results may therefore still be influenced by training randomness.
# 
# - **Discrete approximation of continuous torque:** DQN requires discrete actions, so the continuous torque range was approximated using fixed torque values. This limits the control precision available to the agent.
# 
# - **Limited hyperparameter search:** The investigation focused on learning rate, target update frequency, and epsilon decay using selected candidate values. Other settings such as discount factor, batch size, replay-memory size, and network architecture were not extensively tuned.
# 
# - **Default-gravity-based tuning:** Hyperparameter tuning was performed primarily under default gravity before the selected configuration was applied to the other environments. Each gravity condition may have its own optimal hyperparameters.
# 
# - **Single-episode qualitative animations:** Each animation represents one evaluation initial state. The animations therefore provide qualitative supporting evidence rather than a complete measure of policy performance.
# 
# ### **Future Improvements**
# 
# Future work could:
# 
# - repeat additional hyperparameter configurations across more random seeds to provide stronger statistical evidence;
# - perform gravity-specific tuning, particularly for supergravity;
# - test additional or non-uniform torque discretisations;
# - evaluate policies using more evaluation episodes and initial states;
# - investigate further DQN improvements such as Double DQN or Dueling DQN; and
# - compare the discretised DQN with continuous-action reinforcement-learning approaches such as DDPG, TD3, or SAC after establishing the DQN baseline.
# 
# ---
# 
# ## **9.6 Overall Conclusion**
# 
# Overall, the investigation supports the initial expectation that both **environmental dynamics and DQN design choices influence reinforcement-learning performance**.
# 
# Gravity strongly affected task difficulty and learned control behaviour. Free-fall and anti-gravity provided easier control conditions, while supergravity remained the most difficult environment.
# 
# Systematic hyperparameter tuning showed that improving one training metric does not necessarily improve the final greedy policy. Epsilon decay `0.99` was selected because it provided the strongest overall trade-off across the prioritised criteria, particularly learning speed, late-training stability, evaluation consistency, and mean greedy evaluation performance. The repeated-seed experiment further supported this selection.
# 
# The action-resolution sensitivity check showed that reducing the action space from nine to five actions did not provide a clear overall advantage. Therefore, nine discrete torque actions were retained.
# 
# The final submitted model is the **tuned default-gravity DQN** using:
# 
# | **Component** | **Final Setting** |
# |---|---:|
# | Gravity | `10.0` |
# | Learning rate | `0.00025` |
# | Target update frequency | `10` |
# | Epsilon decay | `0.99` |
# | Number of discrete actions | `9` |
# 
# Although free-fall achieved the highest reward across the gravity experiments, it represents an easier environmental condition rather than the standard Pendulum benchmark used for systematic optimisation. The tuned default-gravity model was therefore selected as the final submitted model, and its neural-network weights were saved for reproducibility.
# 
# Overall, the results demonstrate that a discretised-action DQN can learn meaningful Pendulum control across substantially different gravity conditions, while systematic experimentation, consistent evaluation, and repeated trials are important for identifying a configuration that learns efficiently and behaves reliably.

# ## **10. References**
# 
# OpenAI. (2026). *ChatGPT* [Large language model]. Used to assist with explanation refinement, code structuring, debugging, and interpretation of experimental results. All generated suggestions were reviewed and adapted to the final implementation.
