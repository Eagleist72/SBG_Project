# CSE588 Term Project: AI Game Agent using Deep Learning

## 1. Project Overview
This project implements an AI agent capable of playing a strategic board game (7x7). 
Unlike traditional search-based agents (e.g., Minimax, Alpha-Beta Pruning), this agent uses **Supervised Deep Learning (Behavioral Cloning)**.

The AI does **not** use any search trees during the tournament execution. Instead, it relies entirely on a pre-trained **Neural Network (Multi-Layer Perceptron)** to evaluate board states and make decisions instantly.

### Key Features
* **Approach:** Supervised Learning (Behavioral Cloning).
* **Model:** Deep Neural Network (PyTorch) with 3 Hidden Layers (128, 64, 32).
* **Training Data:** Generated via Self-Play using a "Pro Teacher" algorithm.
* **Teacher Logic:** Uses **2-Step Lookahead** and **Weighted Mobility Heuristic** (Killer Heuristic) to generate high-quality move data.
* **Tournament Mode:** Supports file-based communication (`move.txt`) as per requirements.

---

## 2. Installation & Prerequisites
The project requires **Python 3.x** and the following libraries.

To install the dependencies, run:

```bash
pip install torch torchvision pandas numpy