import sys
import chess
import chess.engine

def best_move_from_fen(fen: str, engine_path: str, time_limit: float = 0.1) -> str:
    """
    Given a FEN and a path to a UCI engine, return the best move.
    
    :param fen: FEN string of the position
    :param engine_path: path to the Stockfish (or other UCI) engine executable
    :param time_limit: thinking time in seconds
    :return: best move in UCI notation (e.g. "e2e4")
    """
    # 1. Parse the board
    board = chess.Board(fen)
    
    # 2. Launch engine
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        # 3. Ask engine for the move
        result = engine.play(board, chess.engine.Limit(time=time_limit))
    
    return result.move.uci()

def main():
    if len(sys.argv) < 3:
        print("Usage: python best_move.py <FEN> <path/to/stockfish> [time_limit_sec]")
        sys.exit(1)
    
    fen = sys.argv[1]
    engine_path = sys.argv[2]
    time_limit = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.1
    
    move = best_move_from_fen(fen, engine_path, time_limit)
    print(f"Best move: {move}")

if __name__ == "__main__":
    main()
