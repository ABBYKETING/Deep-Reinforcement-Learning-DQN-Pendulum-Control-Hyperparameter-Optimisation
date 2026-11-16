# Deep Reinforcement Learning — DQN Pendulum Control

**Deep Learning | Reinforcement Learning | TensorFlow | Python**

Academic Pair Project — ST1504 Deep Learning, Singapore Polytechnic

## Project Overview

This project investigates the application of Deep Q-Networks (DQN) to a continuous-control problem using the OpenAI Gym Pendulum environment.

The objective is to develop an agent capable of applying torque to stabilise a pendulum near its upright position, and to investigate how changes in gravity, hyperparameters, and training randomness affect the learned control policy.

## Tools & Technologies

- Python
- TensorFlow / Keras
- OpenAI Gym
- NumPy, pandas, Matplotlib
- Jupyter Notebook
- Deep Q-Learning, Experience Replay, Target Networks

## DQN Architecture

The neural network estimates action values using:

| Layer | Configuration |
|---|---|
| Input | 3 state features |
| Hidden Layer 1 | 128 neurons, ReLU |
| Hidden Layer 2 | 128 neurons, ReLU |
| Output | 9 Q-values, linear |

The continuous torque range of −2 to +2 is discretised into nine actions.

The agent incorporates experience replay, an online Q-network, a target network, and epsilon-greedy exploration.

## Baseline Training Configuration

| Parameter | Value |
|---|---|
| Training episodes | 500 |
| Batch size | 64 |
| Discount factor | 0.99 |
| Learning rate | 0.00025 |
| Replay buffer capacity | 50,000 |
| Target update frequency | 10 episodes |
| Initial epsilon | 1.0 |
| Baseline epsilon decay | 0.995 |

## Experimental Design

### 1. Gravity Comparison

Trained and evaluated agents under four gravity conditions:

- Default gravity: g = 10
- Free-fall: g = 0
- Anti-gravity: g = −10
- Supergravity: g = 15

### 2. Hyperparameter Optimisation

Compared learning rates, target-network update frequencies, and epsilon-decay schedules while controlling other experimental settings.

### 3. Robustness Evaluation

Tested configurations across multiple random seeds and compared five versus nine discrete torque actions.

### 4. Evaluation

Evaluated policies using 20 greedy episodes per experiment, with mean evaluation reward as the primary criterion.

Additional metrics included evaluation standard deviation, training convergence speed, and late-training stability.

## Key Results

| Gravity | Baseline Reward | Selected Reward |
|---|---:|---:|
| Default | −192.28 | −191.22 |
| Free-fall | −54.21 | −53.65 |
| Anti-gravity | −62.53 | −61.62 |
| Supergravity | −305.38 | −233.27 |

The selected epsilon-decay configuration improved mean evaluation rewards across all four tested gravity environments.

Under default gravity:

- Episodes to reach the −300 training threshold decreased from 258 to 155.
- Late-training stability improved from 28.85 to 16.79.
- Across three training seeds, the mean evaluation reward improved from −195.36 to −190.94.

## Final Model

The selected model uses:

- Default gravity: g = 10
- Learning rate: 0.00025
- Target update frequency: 10
- Epsilon decay: 0.99
- Nine discrete torque actions

The trained model weights are provided in `models/final_dqn_selected_model.weights.h5`.

## How to Run

1. Set up a Python environment compatible with the dependencies used in the original notebook.
2. Install the required packages.
3. Open `notebooks/dqn_pendulum_experiments.ipynb`.
4. Run the notebook cells in order to reproduce the experiments.

**Compatibility note:** The original implementation uses the legacy Gym 0.17.3 / Pendulum-v0 API. Newer Gym or Gymnasium releases require code changes. The full experimental workflow is computationally intensive, and successful execution on a fresh environment has not been independently verified.

## Limitations & Future Improvements

- DQN requires a discrete action space, limiting torque precision.
- Training results vary with random initialisation.
- Hyperparameter tuning was performed primarily under default gravity.
- Additional seeds and evaluation trials would strengthen robustness conclusions.
- Double DQN, Dueling DQN, or continuous-control reinforcement learning algorithms could be explored in future work.

## Project Collaboration

Developed collaboratively as a two-person academic project, including DQN implementation, controlled experiments, performance evaluation, visualisation, and interpretation.

## Conclusion

The investigation demonstrated that environmental dynamics substantially affect the difficulty of reinforcement learning control.

Hyperparameter tuning improved learning speed and training stability, while the best evaluation reward did not always correspond to the most stable or fastest-learning configuration.
