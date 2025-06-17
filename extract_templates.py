import cv2
import os
import numpy as np

# Path to the reference chessboard image
REFERENCE_IMAGE = 'chessboard-template.png'  # Change if your file is named differently
TEMPLATE_DIR = 'templates'

# Standard chess starting order for each side
PIECE_ORDER = [
    'rook', 'knight', 'bishop', 'queen', 'king', 'bishop', 'knight', 'rook'
]

def get_tile_color(row, col):
    # Chessboard alternates: top-left is always a light square
    return 'white' if (row + col) % 2 == 0 else 'black'

# Create output directory if it doesn't exist
os.makedirs(TEMPLATE_DIR, exist_ok=True)

def extract_templates():
    img = cv2.imread(REFERENCE_IMAGE)
    if img is None:
        print(f"Could not read {REFERENCE_IMAGE}")
        return

    height, width = img.shape[:2]
    square_height = height // 8
    square_width = width // 8

    # Extract black pieces (top two rows)
    for col, piece in enumerate(PIECE_ORDER):
        # Row 0: major pieces
        x = col * square_width
        y = 0
        roi = img[y:y+square_height, x:x+square_width]
        tile_color = get_tile_color(0, col)
        cv2.imwrite(os.path.join(TEMPLATE_DIR, f'black_{piece}_{tile_color}.png'), roi)
        # Row 1: pawns
        y = square_height
        roi = img[y:y+square_height, x:x+square_width]
        tile_color = get_tile_color(1, col)
        cv2.imwrite(os.path.join(TEMPLATE_DIR, f'black_pawn_{col+1}_{tile_color}.png'), roi)

    # Extract white pieces (bottom two rows)
    for col, piece in enumerate(PIECE_ORDER):
        # Row 7: major pieces
        x = col * square_width
        y = 7 * square_height
        roi = img[y:y+square_height, x:x+square_width]
        tile_color = get_tile_color(7, col)
        cv2.imwrite(os.path.join(TEMPLATE_DIR, f'white_{piece}_{tile_color}.png'), roi)
        # Row 6: pawns
        y = 6 * square_height
        roi = img[y:y+square_height, x:x+square_width]
        tile_color = get_tile_color(6, col)
        cv2.imwrite(os.path.join(TEMPLATE_DIR, f'white_pawn_{col+1}_{tile_color}.png'), roi)

    print(f"Templates for both tile colors extracted to '{TEMPLATE_DIR}' directory.")

if __name__ == '__main__':
    extract_templates() 