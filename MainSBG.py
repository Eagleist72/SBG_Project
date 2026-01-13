"""
Strategic Board Game - Deep Learning Edition
Description: 7x7 Board Game using a Pre-trained Neural Network (MLP) for AI decisions.
Note: This version does NOT use Minimax or Alpha-Beta pruning during runtime.
      It uses a learned Value Function to evaluate board states.
"""

import copy
import tkinter as tk
from tkinter import messagebox
import numpy as np
import warnings
import torch
from NeuralNet import BoardEvaluator
import os
from ConfigLoader import load_config

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Load Configuration
config = load_config()
MODEL_PATH = config['paths']['model_checkpoint']
DEVICE_CONFIG = config['training']['device']

# --- 1. GAME LOGIC ---

class Game:
    def __init__(self):
        self.board_size = 7 # Hardcoded as per request
        # 0: Empty, 1: AI (Blue), 2: Human (Red), -1: Removed
        self.board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]

        # Initial positions
        self.p1_pos = (0, 3)  # AI
        self.board[0][3] = 1
        self.p2_pos = (6, 3)  # Human
        self.board[6][3] = 2

        # LOAD BRAIN: Load the pre-trained neural network
        self.brain_loaded = False
        
        if DEVICE_CONFIG == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(DEVICE_CONFIG)

        try:
            # Model architecture is loaded via config in NeuralNet
            self.model = BoardEvaluator().to(self.device)
            if os.path.exists(MODEL_PATH):
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
                self.model.eval() # Set to evaluation mode
                self.brain_loaded = True
                print(f"AI Brain loaded successfully from {MODEL_PATH}.")
            else:
                print(f"WARNING: '{MODEL_PATH}' not found! AI will play randomly.")
        except Exception as e:
            print(f"Error loading brain: {e}")

    def is_within_bounds(self, r, c):
        """Checks if coordinates are within the board limits."""
        return 0 <= r < self.board_size and 0 <= c < self.board_size

    def is_valid_move(self, current_pos, target_pos):
        """Checks if a move is valid (neighboring cell, empty, within bounds)."""
        r1, c1 = current_pos
        r2, c2 = target_pos

        if not self.is_within_bounds(r2, c2): return False
        if self.board[r2][c2] != 0: return False

        dr = abs(r1 - r2)
        dc = abs(c1 - c2)

        # Must move to a neighbor (distance <= 1)
        if dr <= 1 and dc <= 1 and dr + dc > 0:
            return True
        return False

    def make_turn(self, player_id, move_pos, remove_pos):
        """Executes a turn: Move Piece -> Remove Square."""
        if player_id == 1:
            current_pos = self.p1_pos
        else:
            current_pos = self.p2_pos

        # 1. Move
        r_old, c_old = current_pos
        r_new, c_new = move_pos
        self.board[r_old][c_old] = 0
        self.board[r_new][c_new] = player_id

        if player_id == 1:
            self.p1_pos = (r_new, c_new)
        else:
            self.p2_pos = (r_new, c_new)

        # 2. Remove
        r_rem, c_rem = remove_pos
        self.board[r_rem][c_rem] = -1

    def clone(self):
        """Creates a deep copy of the game state for simulation."""
        new_game = Game()
        new_game.board = copy.deepcopy(self.board)
        new_game.p1_pos = self.p1_pos
        new_game.p2_pos = self.p2_pos
        
        # Pass the brain reference to the clone
        # We don't need to deepcopy the model, just reference it
        new_game.model = self.model
        new_game.brain_loaded = self.brain_loaded
        new_game.device = self.device
        return new_game

    def get_neighbors(self, pos):
        """Returns valid neighbor coordinates for movement."""
        r, c = pos
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if self.is_within_bounds(nr, nc) and self.board[nr][nc] == 0:
                    neighbors.append((nr, nc))
        return neighbors

    def get_all_valid_moves(self, player_id):
        """Generates all possible (Move, Remove) combinations."""
        moves = []
        current_pos = self.p1_pos if player_id == 1 else self.p2_pos
        possible_moves = self.get_neighbors(current_pos)

        for move_pos in possible_moves:
            # Check all cells for valid removal
            for r in range(self.board_size):
                for c in range(self.board_size):
                    # Valid Remove: Empty OR Old Position, BUT NOT New Position
                    is_empty = self.board[r][c] == 0
                    is_old_spot = (r, c) == current_pos
                    is_new_spot = (r, c) == move_pos

                    if (is_empty or is_old_spot) and not is_new_spot:
                        moves.append((move_pos, (r, c)))
        return moves

    # --- DEEP LEARNING INFERENCE ---

    def predict_board_score(self):
        """Uses the Neural Network to predict the value of the current board."""
        if not self.brain_loaded:
            return 0

        # Flatten board to match training format
        flat_board = [cell for row in self.board for cell in row]

        # Prepare Input Tensor
        input_tensor = torch.tensor(flat_board, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            score = self.model(input_tensor).item()
            
        return score

    def get_best_move_dl(self):
        """
        Strategy:
        1. Generate all possible next states.
        2. Ask the Neural Network to evaluate each state.
        3. Pick the state with the highest score.
        """
        moves = self.get_all_valid_moves(1)
        if not moves: return None

        best_score = -float('inf')
        best_move = None

        # Optional: Shuffle moves to add variety if scores are equal
        import random
        random.shuffle(moves)

        # Limit moves to prevent UI freezing if there are too many (optimization)
        if len(moves) > 50:
            moves = moves[:50]

        for move_pos, remove_pos in moves:
            # 1. Simulate the move
            temp_game = self.clone()
            temp_game.make_turn(1, move_pos, remove_pos)

            # 2. Predict Score
            score = temp_game.predict_board_score()

            # 3. Select Max
            if score > best_score:
                best_score = score
                best_move = (move_pos, remove_pos)

        return best_move


# --- 2. GUI CLASS ---

class GameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Strategic Board Game - AI (Deep Learning - PyTorch)")
        self.game = Game()

        # Game State Control
        self.turn_phase = 0
        self.selected_move_target = None

        # UI Colors
        self.color_bg = "#ffffff"
        self.color_p1 = "#3498db"  # AI Color
        self.color_p2 = "#e74c3c"  # Human Color
        self.color_removed = "#34495e"
        self.color_highlight = "#f1c40f"

        # Layout
        self.info_label = tk.Label(root, text="Welcome! AI is Thinking...", font=("Arial", 14))
        self.info_label.pack(pady=10)

        self.board_frame = tk.Frame(root)
        self.board_frame.pack()

        self.buttons = [[None for _ in range(7)] for _ in range(7)]
        self.create_board_grid()

        self.update_board_ui()

        # Start the game loop (AI goes first)
        self.root.after(1000, self.run_ai_turn)

    def create_board_grid(self):
        for r in range(7):
            for c in range(7):
                btn = tk.Button(self.board_frame, width=4, height=2, font=("Arial", 12, "bold"),
                                command=lambda row=r, col=c: self.on_cell_click(row, col))
                btn.grid(row=r, column=c, padx=2, pady=2)
                self.buttons[r][c] = btn

                # Coordinate Labels
                if c == 0:
                    lbl = tk.Label(self.board_frame, text=chr(97 + r))
                    lbl.grid(row=r, column=7, sticky="w")
                if r == 6:
                    lbl = tk.Label(self.board_frame, text=str(c + 1))
                    lbl.grid(row=7, column=c)

    def update_board_ui(self):
        for r in range(7):
            for c in range(7):
                cell_val = self.game.board[r][c]
                btn = self.buttons[r][c]

                text = ""
                bg_color = self.color_bg
                state = "normal"

                if cell_val == 1:
                    bg_color = self.color_p1
                    text = "AI"
                elif cell_val == 2:
                    bg_color = self.color_p2
                    text = "YOU"
                elif cell_val == -1:
                    bg_color = self.color_removed
                    state = "disabled"

                btn.config(text=text, bg=bg_color, state=state, relief="raised")

    def on_cell_click(self, r, c):
        # Ignore clicks if it's AI's turn
        if self.turn_phase == 3: return

        # PHASE 1: Move Selection
        if self.turn_phase == 1:
            current_pos = self.game.p2_pos
            target_pos = (r, c)

            if self.game.is_valid_move(current_pos, target_pos):
                self.selected_move_target = target_pos
                self.buttons[r][c].config(bg=self.color_highlight, text="MOV")
                self.info_label.config(text="Step 2: Click a cell to REMOVE.")
                self.turn_phase = 2
            else:
                messagebox.showwarning("Invalid Move", "You can only move to empty neighbors!")

        # PHASE 2: Remove Selection
        elif self.turn_phase == 2:
            remove_pos = (r, c)
            target_pos = self.selected_move_target
            current_pos = self.game.p2_pos

            r_rem, c_rem = remove_pos
            is_empty = (self.game.board[r_rem][c_rem] == 0)
            is_my_old_spot = (remove_pos == current_pos)
            is_target_spot = (remove_pos == target_pos)

            if (is_empty or is_my_old_spot) and not is_target_spot:
                # Execute Human Turn
                self.game.make_turn(2, target_pos, remove_pos)
                self.update_board_ui()

                # Switch to AI
                self.turn_phase = 3
                self.info_label.config(text="AI is thinking...")
                self.root.update()
                self.root.after(100, self.run_ai_turn)
            else:
                messagebox.showwarning("Invalid Remove", "You cannot remove an occupied cell or your new position.")

    def run_ai_turn(self):
        # Check Win Condition
        if not self.game.get_all_valid_moves(1):
            messagebox.showinfo("Game Over", "AI has no moves. YOU WIN!")
            self.root.quit()
            return

        # AI DECISION (Deep Learning)
        ai_move = self.game.get_best_move_dl()

        if ai_move:
            move_pos, remove_pos = ai_move
            self.game.make_turn(1, move_pos, remove_pos)
            self.update_board_ui()

            # Check Loss Condition
            if not self.game.get_all_valid_moves(2):
                messagebox.showinfo("Game Over", "You have no moves left. AI WINS!")
                self.root.quit()
            else:
                self.turn_phase = 1
                self.info_label.config(text="Your Turn: Select a neighbor to MOVE.")
        else:
            messagebox.showinfo("Game Over", "AI is stuck. YOU WIN!")
            self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    gui = GameGUI(root)
    root.mainloop()