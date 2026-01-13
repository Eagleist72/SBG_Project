# How to Run the Project

## 1. Prerequisites

Ensure you have Python installed. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

*Dependencies include: `torch`, `pandas`, `numpy`, `tqdm`, `PyYAML`.*

## 2. Configuration (`config.yaml`)

All project settings are centralized in `config.yaml`. You can modify:

*   **Training**: `epochs`, `batch_size`, `learning_rate`.
*   **Model**: `hidden_layers` architecture.
*   **Data**: `num_games` to generate.
*   **Device**: Set to `"auto"`, `"cuda"`, or `"cpu"`.

## 3. Workflow

### Step 1: Collect Training Data
Generate high-quality game data using the "Pro Teacher" logic.

```bash
python collect_data.py
```
*   **Progress**: Shows a progress bar.
*   **Pause**: Press `Ctrl+C` to safely stop and save your current progress.
*   **Resume**: Simply run the command again; it appends to `game_data.csv`.

### Step 2: Train the Model
Train the PyTorch Neural Network using the collected data.

```bash
python Train_Model.py
```
*   **Resume Training**: To resume from a checkpoint, use:
    ```bash
    python Train_Model.py --resume
    ```
*   **Output**: Saves the best model to `ai_brain.pth`.

### Step 3: Play the Game
Run the game interface and play against the AI.

```bash
python MainSBG.py
```

## Troubleshooting

*   **"Torch not compiled with CUDA enabled"**: Open `config.yaml` and set `device: "cpu"` or `"auto"`.
*   **Missing `ai_brain.pth`**: You must run `Train_Model.py` first, or the AI will play randomly.
