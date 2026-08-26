# 🧠 Strategic Board Game AI - Deep Learning Edition

## 📖 Overview
This project implements an intelligent AI agent for a 7x7 strategic board game. Instead of relying on traditional runtime search algorithms like Minimax or Alpha-Beta Pruning, this AI is powered by **Supervised Deep Learning (Behavioral Cloning)**. The model instantly evaluates board states using a learned Value Function, making optimal decisions in milliseconds.

## 🚀 Technical Highlights
* **Methodology:** Behavioral Cloning via Supervised Learning.
* **Data Generation:** Synthetic, high-quality training data generated through Self-Play using a custom "Pro Teacher" algorithm.
* **Teacher Logic:** Employs a **2-Step Lookahead** and a **Weighted Mobility (Killer) Heuristic** (heavily penalizing opponent mobility while securing central control).
* **Model Architecture:** A deep Multi-Layer Perceptron (MLP) built with **PyTorch**. Configured with 4 hidden layers (256 -> 128 -> 64 -> 32) to capture complex positional patterns.
* **Inference Engine:** Generates all valid (Move + Remove) combinations, evaluates the resulting board states through the Neural Network, and selects the optimal move instantly—without any recursive tree search at runtime.
* **Architecture Design:** Fully decoupled logic, modular configuration via YAML, and highly optimized for rapid inference.

## 📁 Project Structure
* `collect_data.py`: Generates the dataset (`game_data.csv`) by running the Teacher bot against itself.
* `Train_Model.py`: Trains the PyTorch model using the generated dataset (supports CUDA for GPU acceleration and early stopping).
* `NeuralNet.py`: Contains the `BoardEvaluator` MLP architecture.
* `MainSBG.py`: The playable GUI game loop utilizing the trained `ai_brain.pth` model.
* `config.yaml` / `ConfigLoader.py`: Centralized configuration for hyperparameters, network size, and file paths.

## 🛠️ How to Run

1. **Install Dependencies:**
   ```bash
   pip install torch pandas numpy pyyaml tqdm
