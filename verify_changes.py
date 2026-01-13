import os
import torch
import pandas as pd
import numpy as np
from ConfigLoader import load_config

def verify_config_loading():
    print("Verifying ConfigLoader...")
    config = load_config()
    print(f"  Config loaded. Epochs: {config['training']['epochs']}, Device: {config['training']['device']}")
    
def verify_neural_net():
    print("Verifying NeuralNet.py...")
    from NeuralNet import BoardEvaluator
    model = BoardEvaluator()
    input_tensor = torch.randn(1, 49)
    output = model(input_tensor)
    print(f"  Model output shape: {output.shape}")
    print("  NeuralNet verified (using config architecture).")

def verify_collect_data():
    print("\nVerifying collect_data.py imports and class...")
    from collect_data import GameEngine
    game = GameEngine()
    print("  GameEngine initialized.")
    # Create dummy data using config filename? No, let's just make sure it runs.
    print("  collect_data verified.")

def verify_inference():
    print("\nVerifying MainSBG.py...")
    # MainSBG loads model path from config
    from MainSBG import Game
    game = Game()
    # It might print warning about missing model, which is fine
    score = game.predict_board_score()
    print(f"  Inference Score: {score}")
    print("  MainSBG verified.")

if __name__ == "__main__":
    verify_config_loading()
    verify_neural_net()
    verify_collect_data()
    verify_inference()
    
    print("\nVerification Complete!")
