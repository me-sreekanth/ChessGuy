from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from chess_detector import ChessDetector
from chess_logic import ChessLogic
import chess
import os

app = FastAPI(title="Chess Position Analyzer")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chess components
detector = ChessDetector()
chess_logic = ChessLogic()

# Initialize Stockfish engine
STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "stockfish")
chess_logic.initialize_engine(STOCKFISH_PATH)

@app.post("/analyze-position")
async def analyze_position(image: UploadFile = File(...)):
    """
    Analyze a chess position from an uploaded image.
    Returns the detected board, pieces, last move, FEN, and best move.
    """
    try:
        # Read the uploaded image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 1. Detect chessboard
        success, corners = detector.detect_chessboard(img)
        if not success:
            raise HTTPException(status_code=400, detail="Could not detect chessboard in the image")
        
        # 2. Detect pieces and their positions
        piece_positions = {}
        for row in range(8):
            for col in range(8):
                square_roi = detector.get_square_roi(img, row, col)
                has_piece, is_white = detector.detect_piece(square_roi)
                
                if has_piece:
                    # TODO: Implement piece type detection
                    # For now, we'll assume it's a pawn
                    piece_type = chess.PAWN
                    square = chess_logic.convert_square_to_notation(row, col)
                    piece_positions[square] = (piece_type, is_white)
        
        # 3. Detect last move
        from_square, to_square = detector.detect_last_move(img)
        
        # 4. Generate FEN
        fen = chess_logic.generate_fen(piece_positions)
        
        # 5. Get best move
        best_move = chess_logic.get_best_move()
        
        return {
            "fen": fen,
            "last_move": {
                "from": from_square,
                "to": to_square
            } if from_square and to_square else None,
            "best_move": str(best_move) if best_move else None,
            "piece_positions": piece_positions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Chess Position Analyzer API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 