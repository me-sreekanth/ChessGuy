import chess
import chess.engine
import os

class ChessLogic:
    def __init__(self):
        self.board = chess.Board()
        self.engine = None
        
    def initialize_engine(self, stockfish_path=None):
        """
        Initialize the chess engine.
        If stockfish_path is not provided, it will look for stockfish in the system PATH.
        """
        try:
            if stockfish_path and os.path.exists(stockfish_path):
                self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            else:
                self.engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        except Exception as e:
            print(f"Error initializing chess engine: {e}")
            self.engine = None

    def generate_fen(self, piece_positions):
        """
        Generate FEN string from piece positions.
        piece_positions: Dictionary mapping squares to (piece_type, color)
        """
        # Create empty board
        self.board.clear()
        
        # Place pieces on the board
        for square, (piece_type, is_white) in piece_positions.items():
            piece = chess.Piece(piece_type, chess.WHITE if is_white else chess.BLACK)
            self.board.set_piece_at(square, piece)
            
        return self.board.fen()

    def get_best_move(self, time_limit=1.0):
        """
        Get the best move for the current position.
        time_limit: Time limit in seconds for the engine to think
        """
        if not self.engine:
            return None
            
        try:
            result = self.engine.play(self.board, chess.engine.Limit(time=time_limit))
            return result.move
        except Exception as e:
            print(f"Error getting best move: {e}")
            return None

    def close_engine(self):
        """
        Close the chess engine.
        """
        if self.engine:
            self.engine.quit()

    def convert_square_to_notation(self, row, col):
        """
        Convert board coordinates to chess notation.
        """
        file = chr(ord('a') + col)
        rank = str(8 - row)
        return file + rank

    def convert_notation_to_square(self, notation):
        """
        Convert chess notation to board coordinates.
        """
        file = ord(notation[0]) - ord('a')
        rank = 8 - int(notation[1])
        return rank, file 