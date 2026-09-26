# DeZero-RL

Reinforcement learning algorithms implemented using DeZero.

This repository is a personal implementation project for studying reinforcement learning through both papers and code.

While studying deep learning frameworks, I implemented DeZero by following *Deep Learning from Scratch 3*.  
Based on that experience, this project uses DeZero as the underlying deep learning framework to implement and study reinforcement learning algorithms.

The main goal is to understand how mathematical formulations in reinforcement learning papers are translated into actual implementations.

---

## Goals

- Study reinforcement learning algorithms from their papers
- Understand the mathematical ideas behind each algorithm
- Translate equations and algorithm descriptions into working code
- Implement RL algorithms directly using DeZero
- Organize implementations and experiments in a reusable structure

---

## Algorithms and Papers

| Algorithm | Paper / Reference | Study | Implementation | Experiments |
|---|---|---:|---:|---:|
| DQN | *Human-level Control through Deep Reinforcement Learning* | Done | In Progress | Planned |
| A3C | *Asynchronous Methods for Deep Reinforcement Learning* | Done | Planned | Planned |
| TRPO | *Trust Region Policy Optimization* | Done | Planned | Planned |
| PPO | *Proximal Policy Optimization Algorithms* | Done | Planned | Planned |
| SAC | *Soft Actor-Critic* | Done | Planned | Planned |

---

## Documentation

Detailed notes for each algorithm are organized under `docs/`.

- [DQN](docs/dqn.md)
- [A3C](docs/a3c.md)
- [TRPO](docs/trpo.md)
- [PPO](docs/ppo.md)
- [SAC](docs/sac.md)

Each document may include:

- Motivation
- Core ideas
- Important equations
- Algorithm structure
- Implementation details
- Connections with related algorithms

---

## Project Structure

```text
dezero-rl/
├── dezero_rl/
│   ├── algorithms/
│   │   ├── dqn.py
│   │   ├── ppo.py
│   │   └── sac.py
│   ├── buffers/
│   ├── networks/
│   ├── trainers/
│   └── utils/
│
├── experiments/
│   ├── dqn/
│   ├── ppo/
│   └── sac/
│
├── configs/
├── results/
├── examples/
├── docs/
│   ├── dqn.md
│   ├── a3c.md
│   ├── trpo.md
│   ├── ppo.md
│   └── sac.md
│
├── README.md
├── LICENSE
└── requirements.txt
```

---

## DeZero

DeZero is a lightweight deep learning framework introduced in *Deep Learning from Scratch 3* by Koki Saitoh.

I implemented DeZero while studying the internal structure of deep learning frameworks, including automatic differentiation, computational graphs, neural network layers, and optimization.

This repository builds on that implementation and uses DeZero as the foundation for reinforcement learning algorithms.

Official repository:

https://github.com/oreilly-japan/deep-learning-from-scratch-3

---

## License

DeZero is distributed under the MIT License.

Original DeZero copyright:

```text
Copyright (c) 2019 Koki Saitoh
```

When redistributing copies or substantial portions of DeZero, the original copyright and license notice must be preserved.

The reinforcement learning implementations in this repository are released under the MIT License unless otherwise noted.

See [LICENSE](LICENSE) for details.

---

## References

- Mnih et al., *Human-level Control through Deep Reinforcement Learning*
- Mnih et al., *Asynchronous Methods for Deep Reinforcement Learning*
- Schulman et al., *Trust Region Policy Optimization*
- Schulman et al., *Proximal Policy Optimization Algorithms*
- Haarnoja et al., *Soft Actor-Critic*
- Koki Saitoh, *Deep Learning from Scratch 3*
