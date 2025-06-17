# Chess Position Analyzer

This API service analyzes chess positions from images. It can detect the chessboard, pieces, last move, generate FEN notation, and suggest the best move.

## Features

1. Chessboard detection from images
2. Piece detection and color identification
3. Position detection
4. Last move detection (from highlighted squares)
5. FEN notation generation
6. Best move suggestion using Stockfish

## Prerequisites

- Python 3.8 or higher
- Stockfish chess engine
- OpenCV
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd chess-position-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Stockfish:
   - Windows: Download from [Stockfish website](https://stockfishchess.org/download/) and add to PATH
   - Linux: `sudo apt-get install stockfish`
   - macOS: `brew install stockfish`

## Usage

1. Start the server:
```bash
python main.py
```

2. The API will be available at `http://localhost:8000`

3. Send a POST request to `/analyze-position` with an image file:
```bash
curl -X POST -F "image=@chess_position.jpg" http://localhost:8000/analyze-position
```

## API Response Format

```json
{
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "last_move": {
        "from": "e2",
        "to": "e4"
    },
    "best_move": "e4",
    "piece_positions": {
        "a1": [4, true],
        "b1": [2, true],
        // ... other piece positions
    }
}
```

## Environment Variables

- `STOCKFISH_PATH`: Path to Stockfish executable (optional, defaults to "stockfish")

## Limitations

- Current implementation assumes standard chess piece shapes
- Piece type detection is simplified (currently assumes all pieces are pawns)
- Last move detection requires highlighted squares
- Image quality and lighting conditions may affect detection accuracy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 