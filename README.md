# MALLVi: A Multi-Agent Framework for Integrated Generalized Robotics Manipulation

[![Paper](https://img.shields.io/badge/paper-arXiv.2602.16898-red)](https://arxiv.org/abs/2602.16898)

**MALLVi** (Multi-Agent Large Language and Vision) is a closed-loop, feedback-driven framework that coordinates specialized LLM and VLM agents for generalized robotic manipulation. Unlike monolithic systems, MALLVi uses a distributed multi-agent architecture to handle task decomposition, visual perception, object localization, motion planning, and error recovery – enabling robust zero-shot performance in dynamic, unstructured environments.

> **Paper:** [MALLVi: A Multi-Agent Framework for Integrated Generalized Robotics Manipulation](https://arxiv.org/abs/2602.16898) (arXiv:2602.16898)  
> **Authors:** Mehrshad Taji, Arad Mahdinezhad Kashani, Iman Ahmadi, AmirHossein Jadidi, Saina Kashani, Babak Khalaj

## 🧠 Key Features

- **Multi-Agent Collaboration** – Specialized agents (Decomposer, Descriptor, Localizer, Thinker, Actor, Reflector) each handle a distinct part of the manipulation pipeline.
- **Closed-Loop Feedback** – The Reflector agent provides continuous visual verification and triggers targeted recovery without costly global replanning.
- **Zero-Shot Generalization** – Handles novel objects, instructions, and environments without task-specific fine-tuning.
- **Modular & Extensible** – Each agent can be independently improved or replaced (e.g., swapping the LLM backbone).
- **Visual Memory** – Descriptor agent builds a spatial graph of the scene, enabling memory‑aware reasoning.
- **Real-Time Error Recovery** – The Reflector detects failures, explains them, and reactivates only the relevant agent.

## 🏗️ Architecture Overview

MALLVi processes a human instruction and a current environment image through six core agents:

| Agent | Role |
|-------|------|
| **Decomposer** | Breaks high‑level instruction into a queue of atomic subtasks (e.g., `move`, `reach`, `push`). |
| **Descriptor** | Uses a VLM to build a spatial graph of objects and their relationships – the visual memory. |
| **Localizer** | Performs open‑vocabulary detection (GroundingDINO + OWLv2) and extracts 3D grasp points via SAM + depth projection. |
| **Thinker** | Translates each subtask into actionable parameters (target objects, positions, rotations) using memory tags. |
| **Actor** | Executes the low‑level motion primitives via a robot API (simulation or real hardware). |
| **Reflector** | A VLM that checks success/failure, produces explanations, and selectively reactivates the failing agent. |

The Reflector’s **targeted feedback loop** is a key contribution: instead of global replanning, it re‑calls only the agent that failed (e.g., localizer or thinker), then re‑executes the same subtask. This dramatically improves efficiency and success rates.

## 📊 Evaluation Highlights

Extensive experiments on **real-world tasks**, **VIMABench** (12 tasks), and **RLBench** (9 tasks) show MALLVi outperforming prior methods (MALMM, VoxPoser, ReKep, PerAct, Wonderful Team, etc.):

- **Real-world** (8 tasks, 20 reps): up to **100%** success (Place Food), average **87%**.
- **VIMABench** (4 categories, 100 reps): **95%** on Novel Concepts, **90%** on Visual Reasoning.
- **RLBench** (5 tasks, 100 reps): **92–96%** success rates.

Ablations prove the necessity of each agent: removing the Reflector drops performance by 15–30%, and a single‑agent baseline fails on most complex tasks.


## 📚 Citation
If you use MALLVi code or paper in your research, please cite the original paper:

```bibtex
@misc{taji2026mallvimultiagentframeworkintegrated,
      title={MALLVI: A Multi-Agent Framework for Integrated Generalized Robotics Manipulation}, 
      author={Mehrshad Taji and Arad Mahdinezhad Kashani and Iman Ahmadi and AmirHossein Jadidi and Saina Kashani and Babak Khalaj},
      year={2026},
      eprint={2602.16898},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.16898}, 
}
```
