import time
import math
from config import *
from board import Move

# ===================== AI AGENT =====================


class B23CM1046:
    """
    Minimax + Alpha-Beta chess agent for the 4x8 mini-chess.
    - Material (from PIECE_VALUES)
    - Piece-Square Tables (from config.py)
    - +2 for giving check, -2 for being in check
    - +/-300 for checkmate (king captured)
    """

    def __init__(self, board):
        self.board = board
        self.nodes_expanded = 0
        self.depth = 3  # default depth
        self.time_budget = 1.2  # seconds per move
        self.start_time = 0.0
        self.root_is_white = True

    # =============== Required API =================

    def get_best_move(self):
        """Return the best move for the current board state."""
        self.nodes_expanded = 0
        self.start_time = time.time()
        self.root_is_white = self.board.white_to_move

        legal_moves = self.board.get_legal_moves()
        if not legal_moves:
            return None

        best_move = legal_moves[0]
        best_val = -math.inf
        alpha, beta = -math.inf, math.inf

        for mv in legal_moves:
            if self._out_of_time():
                break
            self.board.make_move(mv)
            self.nodes_expanded += 1
            val = self._alphabeta(self.depth - 1, alpha, beta, maximizing=False)
            self.board.undo_move()
            if val > best_val:
                best_val = val
                best_move = mv
            alpha = max(alpha, val)
        return best_move

    def evaluate_board(self):
        """Heuristic evaluation from the root side's perspective."""
        score = 0
        b = self.board.board

        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                p = b[r][c]
                if p == EMPTY_SQUARE:
                    continue

                # Material value
                score += PIECE_VALUES.get(p, 0)

                # Positional bonus via PST
                pst_val = self._pst_value(p, r, c)
                score += pst_val

        # Check bonuses
        if self.board.is_in_check():
            # If current player (to move) is in check, that’s bad for them
            if self.board.white_to_move == self.root_is_white:
                score -= 2
            else:
                score += 2

        # King capture = checkmate
        if not self._has_king(True):
            score -= 300
        if not self._has_king(False):
            score += 300

        return score if self.root_is_white else -score

    # =============== Core Search =================

    def _alphabeta(self, depth, alpha, beta, maximizing):
        if self._out_of_time():
            return self.evaluate_board()

        state = self.board.get_game_state()
        if state == "checkmate":
            return (
                -1_000_000
                if self.board.white_to_move == self.root_is_white
                else 1_000_000
            )
        if state == "stalemate" or depth == 0:
            return self.evaluate_board()

        moves = self.board.get_legal_moves()
        if not moves:
            return self.evaluate_board()

        if maximizing:
            value = -math.inf
            for mv in moves:
                if self._out_of_time():
                    break
                self.board.make_move(mv)
                self.nodes_expanded += 1
                value = max(
                    value, self._alphabeta(depth - 1, alpha, beta, maximizing=False)
                )
                self.board.undo_move()
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = math.inf
            for mv in moves:
                if self._out_of_time():
                    break
                self.board.make_move(mv)
                self.nodes_expanded += 1
                value = min(
                    value, self._alphabeta(depth - 1, alpha, beta, maximizing=True)
                )
                self.board.undo_move()
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    # =============== Helpers =================

    def _pst_value(self, piece, r, c):
        """Return PST value (positive for white, negative for black)."""
        if piece == WHITE_PAWN:
            return PAWN_PST[r][c]
        if piece == BLACK_PAWN:
            return -PAWN_PST[BOARD_HEIGHT - 1 - r][c]
        if piece == WHITE_KNIGHT:
            return KNIGHT_PST[r][c]
        if piece == BLACK_KNIGHT:
            return -KNIGHT_PST[BOARD_HEIGHT - 1 - r][c]
        if piece == WHITE_BISHOP:
            return BISHOP_PST[r][c]
        if piece == BLACK_BISHOP:
            return -BISHOP_PST[BOARD_HEIGHT - 1 - r][c]
        if piece == WHITE_KING:
            return KING_PST_LATE_GAME[r][c]
        if piece == BLACK_KING:
            return -KING_PST_LATE_GAME[BOARD_HEIGHT - 1 - r][c]
        return 0

    def _has_king(self, white=True):
        target = WHITE_KING if white else BLACK_KING
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                if self.board.board[r][c] == target:
                    return True
        return False

    def _out_of_time(self):
        return (time.time() - self.start_time) >= self.time_budget
