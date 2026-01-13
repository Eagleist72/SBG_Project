"""
Data Collector (Ultimate Edition)
Description: Generates High-Quality training data for the Deep Learning model.
Features:
  - Append Mode: Adds new games to existing file without deleting old ones.
  - Pro Teacher: Uses 'Killer Heuristic' (Aggressive play).
  - 2-Step Lookahead: Anticipates opponent's counter-moves to avoid traps.
  - Progress Bar (TQDM)
  - Graceful Exit (Ctrl+C)
  - Configurable via YAML
"""
import os
import sys
import pandas as pd
import copy
import random
import time
from tqdm import tqdm
import signal
from ConfigLoader import load_config

# Load Configuration
config = load_config()

# --- GAME ENGINE (PRO TEACHER) ---
class GameEngine:
    """
    Standalone Game Logic used for generating high-quality training data.
    """
    def __init__(self):
        self.board_size = 7 # Hardcoded as per user request
        self.board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]

        # Initial positions
        self.p1_pos = (0, 3) # Blue
        self.board[0][3] = 1
        self.p2_pos = (6, 3) # Red
        self.board[6][3] = 2

    def is_within_bounds(self, r, c):
        return 0 <= r < self.board_size and 0 <= c < self.board_size

    def make_turn(self, player_id, move_pos, remove_pos):
        if player_id == 1: current_pos = self.p1_pos
        else: current_pos = self.p2_pos

        r_old, c_old = current_pos
        r_new, c_new = move_pos
        self.board[r_old][c_old] = 0
        self.board[r_new][c_new] = player_id

        if player_id == 1: self.p1_pos = (r_new, c_new)
        else: self.p2_pos = (r_new, c_new)

        r_rem, c_rem = remove_pos
        self.board[r_rem][c_rem] = -1

    def clone(self):
        new_game = GameEngine()
        new_game.board = copy.deepcopy(self.board)
        new_game.p1_pos = self.p1_pos
        new_game.p2_pos = self.p2_pos
        return new_game

    def get_neighbors(self, pos):
        r, c = pos
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if self.is_within_bounds(nr, nc) and self.board[nr][nc] == 0:
                    neighbors.append((nr, nc))
        return neighbors

    def evaluate(self, player_id):
        """
        ADVANCED EVALUATION (Killer Heuristic)
        Focus: Restrict opponent's moves more than maximizing own moves.
        """
        p1_moves = len(self.get_neighbors(self.p1_pos))
        p2_moves = len(self.get_neighbors(self.p2_pos))

        # Critical Win/Loss Detection
        if player_id == 1:
            if p2_moves == 0: return 10000  # WIN (Opponent Trapped)
            if p1_moves == 0: return -10000 # LOSS (I am Trapped)
        else:
            if p1_moves == 0: return 10000
            if p2_moves == 0: return -10000

        # Tuning Weights
        w_my_moves = 1.0
        w_opp_moves = 2.5  # High penalty for opponent mobility (Aggressive)
        w_center = 0.5

        if player_id == 1:
            mobility = (p1_moves * w_my_moves) - (p2_moves * w_opp_moves)

            center = (3, 3)
            d1 = abs(self.p1_pos[0] - center[0]) + abs(self.p1_pos[1] - center[1])
            d2 = abs(self.p2_pos[0] - center[0]) + abs(self.p2_pos[1] - center[1])
            center_score = (d2 - d1) * w_center

            return mobility + center_score
        else:
            mobility = (p2_moves * w_my_moves) - (p1_moves * w_opp_moves)

            center = (3, 3)
            d1 = abs(self.p1_pos[0] - center[0]) + abs(self.p1_pos[1] - center[1])
            d2 = abs(self.p2_pos[0] - center[0]) + abs(self.p2_pos[1] - center[1])
            center_score = (d1 - d2) * w_center

            return mobility + center_score

    def get_all_valid_moves(self, player_id):
        moves = []
        current_pos = self.p1_pos if player_id == 1 else self.p2_pos
        possible_moves = self.get_neighbors(current_pos)

        for move_pos in possible_moves:
            for r in range(self.board_size):
                for c in range(self.board_size):
                    is_empty = self.board[r][c] == 0
                    is_old_spot = (r, c) == current_pos
                    is_new_spot = (r, c) == move_pos
                    if (is_empty or is_old_spot) and not is_new_spot:
                        moves.append((move_pos, (r, c)))
        return moves

    def get_teacher_move(self, player_id):
        """
        PRO TEACHER LOGIC (2-Step Lookahead)
        Anticipates the opponent's best counter-attack to avoid traps.
        """
        best_score = -float('inf')
        best_move = None

        moves = self.get_all_valid_moves(player_id)
        random.shuffle(moves)

        # Optimization: Check top 8 moves to save time (Lookahead is expensive)
        if len(moves) > 8:
            moves = moves[:8]

        if not moves: return None

        # STEP 1: Simulate My Move
        for move_pos, remove_pos in moves:
            sim_game = self.clone()
            sim_game.make_turn(player_id, move_pos, remove_pos)

            # STEP 2: Simulate Opponent's Best Response (Worst case for me)
            opponent_id = 2 if player_id == 1 else 1
            opp_moves = sim_game.get_all_valid_moves(opponent_id)

            if not opp_moves:
                # If opponent has NO moves, this is a winning move. Pick immediately.
                return (move_pos, remove_pos)

            min_score_after_opponent = float('inf')

            # Check a subset of opponent moves for speed
            random.shuffle(opp_moves)
            subset_opp = opp_moves[:5] if len(opp_moves) > 5 else opp_moves

            for opp_m, opp_r in subset_opp:
                sim_game_2 = sim_game.clone()
                sim_game_2.make_turn(opponent_id, opp_m, opp_r)

                score = sim_game_2.evaluate(player_id)
                if score < min_score_after_opponent:
                    min_score_after_opponent = score

            # The value of my move is determined by the board state AFTER opponent reacts
            final_score = min_score_after_opponent

            if final_score > best_score:
                best_score = final_score
                best_move = (move_pos, remove_pos)

        return best_move

def flatten_board(board):
    return [cell for row in board for cell in row]

def generate_dataset(num_games=None):
    if num_games is None:
        num_games = config['data']['num_games']
    
    file_name = config['data']['file_name']
    
    # Calculate how many already done
    games_done = 0
    if os.path.isfile(file_name):
        try:
            with open(file_name, 'r') as f:
                # Subtract 1 for header
                games_done = sum(1 for line in f) - 1
            games_done = max(0, games_done)
            print(f"File exists with {games_done} samples. Appending...")
        except:
            pass
    
    print(f"--- STARTING HIGH-QUALITY DATA COLLECTION: {num_games} GAMES ---")
    print("Logic: Pro Teacher with 2-Step Lookahead (Slower but Smarter)")
    print("Press Ctrl+C to stop and save progress.")

    batch_data = []
    # Save to file every N games (configurable)
    BATCH_SIZE = config['data'].get('save_interval', 50) 
    
    try:
        # Use TQDM implementation
        pbar = tqdm(range(num_games), desc="Generating Games", unit="game")
        
        for i in pbar:
            game = GameEngine()
            current_player = 1
            game_samples = []

            # Play max 40 turns
            for _ in range(40):
                best_move = game.get_teacher_move(current_player)

                if not best_move: break

                move_pos, remove_pos = best_move

                # Save Data
                features = flatten_board(game.board)
                score = game.evaluate(1)
                game_samples.append(features + [score])

                # Execute
                game.make_turn(current_player, move_pos, remove_pos)
                current_player = 2 if current_player == 1 else 1
            
            # Add game samples to batch
            batch_data.extend(game_samples)
            
            # Save batch periodically
            if (i + 1) % BATCH_SIZE == 0:
                save_to_csv(batch_data, file_name)
                batch_data = [] # Clear batch

    except KeyboardInterrupt:
        print("\n\n!! Interrupted by User !!")
        print("Saving current batch before exiting...")
    finally:
        # Save any remaining data
        if batch_data:
            save_to_csv(batch_data, file_name)
            
    print("\n--- Data Collection Stopped ---")

def save_to_csv(data, file_name):
    if not data: return
    
    file_exists = os.path.isfile(file_name)
    columns = [f'cell_{i}' for i in range(49)] + ['score']
    df = pd.DataFrame(data, columns=columns)
    
    try:
        df.to_csv(file_name, mode='a', index=False, header=not file_exists)
    except Exception as e:
        print(f"ERROR Saving File: {e}")

if __name__ == "__main__":
    generate_dataset()