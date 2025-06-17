import cv2
import numpy as np
import os
from datetime import datetime
import chess
import chess.engine

def get_square_roi(img, x, y, width, height):
    """Get the region of interest for a specific square."""
    return img[y:y+height, x:x+width]

def validate_piece_counts(pieces):
    """
    Validate and adjust piece counts according to chess rules.
    Returns a list of valid pieces.
    """
    # Initialize counters
    piece_counts = {
        'white': {'king': 0, 'queen': 0, 'bishop': 0, 'knight': 0, 'rook': 0, 'pawn': 0},
        'black': {'king': 0, 'queen': 0, 'bishop': 0, 'knight': 0, 'rook': 0, 'pawn': 0}
    }
    
    # Count pieces
    for piece in pieces:
        color = piece['color']
        piece_type = piece['type']
        piece_counts[color][piece_type] += 1
    
    # Validate and adjust pieces
    valid_pieces = []
    for piece in pieces:
        color = piece['color']
        piece_type = piece['type']
        
        # Check if piece count exceeds maximum
        max_count = {
            'king': 1,
            'queen': 1,
            'bishop': 2,
            'knight': 2,
            'rook': 2,
            'pawn': 8
        }
        
        if piece_counts[color][piece_type] <= max_count[piece_type]:
            valid_pieces.append(piece)
        else:
            # If exceeded, try to reclassify as a pawn if it's not already a pawn
            if piece_type != 'pawn' and piece_counts[color]['pawn'] < 8:
                piece['type'] = 'pawn'
                piece_counts[color]['pawn'] += 1
                valid_pieces.append(piece)
    
    return valid_pieces

def detect_piece(square_roi):
    """
    Detect if there's a piece in the square and determine its color and type.
    Returns: (has_piece, is_white, piece_type)
    """
    if square_roi is None or square_roi.size == 0:
        return False, None, None

    # Convert to grayscale
    gray = cv2.cvtColor(square_roi, cv2.COLOR_BGR2GRAY)
    
    # Calculate average brightness and standard deviation
    avg_brightness = np.mean(gray)
    std_dev = np.std(gray)
    
    # Threshold for piece detection
    has_piece = std_dev > 20  # Increased threshold for better noise rejection
    
    if has_piece:
        # Determine piece color based on average brightness
        is_white = avg_brightness > 127
        
        # Get contours for shape analysis
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Calculate shape features
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Get bounding box and its features
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = float(w)/h if h > 0 else 0
            
            # Calculate relative area
            relative_area = area / (square_roi.shape[0] * square_roi.shape[1])
            
            # Calculate convex hull features
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Calculate Hu moments for shape matching
            moments = cv2.moments(largest_contour)
            hu_moments = cv2.HuMoments(moments)
            
            # Piece type classification based on multiple features
            if relative_area < 0.1:  # Too small to be a piece
                return False, None, None
                
            # King detection
            if circularity > 0.85 and relative_area > 0.3 and solidity > 0.9:
                piece_type = "king"
            
            # Queen detection
            elif circularity > 0.75 and relative_area > 0.25 and solidity > 0.85:
                piece_type = "queen"
            
            # Bishop detection
            elif circularity > 0.7 and relative_area > 0.2 and solidity > 0.8:
                piece_type = "bishop"
            
            # Knight detection
            elif circularity > 0.6 and relative_area > 0.2 and solidity > 0.75:
                piece_type = "knight"
            
            # Rook detection
            elif circularity > 0.65 and relative_area > 0.2 and solidity > 0.8:
                piece_type = "rook"
            
            # Pawn detection
            elif circularity > 0.7 and relative_area > 0.15 and solidity > 0.8:
                piece_type = "pawn"
            
            else:
                # Default to pawn if no clear match
                piece_type = "pawn"
            
            # Color-based refinement
            if is_white:
                # White pieces tend to have higher brightness
                if piece_type in ["king", "queen"] and avg_brightness < 180:
                    piece_type = "bishop"  # Downgrade to bishop if not bright enough
                elif piece_type in ["bishop", "knight"] and avg_brightness < 160:
                    piece_type = "pawn"  # Downgrade to pawn if not bright enough
            else:
                # Black pieces tend to have lower brightness
                if piece_type in ["king", "queen"] and avg_brightness > 80:
                    piece_type = "bishop"  # Downgrade to bishop if too bright
                elif piece_type in ["bishop", "knight"] and avg_brightness > 100:
                    piece_type = "pawn"  # Downgrade to pawn if too bright
            
            return True, is_white, piece_type
    
    return False, None, None

def log_board_state(pieces, empty_squares):
    """Log the board state to the console."""
    # Validate piece counts
    valid_pieces = validate_piece_counts(pieces)
    
    print("\n" + "=" * 50)
    print("CHESS BOARD STATE ANALYSIS")
    print("=" * 50)
    
    # Group pieces by color
    white_pieces = [p for p in valid_pieces if p['color'] == 'white']
    black_pieces = [p for p in valid_pieces if p['color'] == 'black']
    
    print("\nWHITE PIECES:")
    print("-" * 30)
    if white_pieces:
        # Group by piece type
        for piece_type in ['king', 'queen', 'bishop', 'knight', 'rook', 'pawn']:
            type_pieces = [p for p in white_pieces if p['type'] == piece_type]
            if type_pieces:
                print(f"\n{piece_type.upper()}:")
                for piece in type_pieces:
                    print(f"  Position: {piece['position']}")
    else:
        print("No white pieces detected")
    
    print("\nBLACK PIECES:")
    print("-" * 30)
    if black_pieces:
        # Group by piece type
        for piece_type in ['king', 'queen', 'bishop', 'knight', 'rook', 'pawn']:
            type_pieces = [p for p in black_pieces if p['type'] == piece_type]
            if type_pieces:
                print(f"\n{piece_type.upper()}:")
                for piece in type_pieces:
                    print(f"  Position: {piece['position']}")
    else:
        print("No black pieces detected")
    
    print("\nEMPTY SQUARES:")
    print("-" * 30)
    if empty_squares:
        print(", ".join(empty_squares))
    else:
        print("No empty squares detected")
    
    print("\n" + "=" * 50)

def load_templates(template_dir='templates'):
    """Load all piece templates from the templates directory, including tile color variants."""
    templates = []
    for fname in os.listdir(template_dir):
        if fname.endswith('.png'):
            path = os.path.join(template_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                # Parse piece code and tile color from filename
                base = fname.replace('.png', '')
                parts = base.split('_')
                if len(parts) >= 3:
                    color = 'W' if parts[0] == 'white' else 'B'
                    if parts[1] == 'pawn':
                        code = color + 'P'
                    elif parts[1] == 'rook':
                        code = color + 'R'
                    elif parts[1] == 'knight':
                        code = color + 'N'
                    elif parts[1] == 'bishop':
                        code = color + 'B'
                    elif parts[1] == 'queen':
                        code = color + 'Q'
                    elif parts[1] == 'king':
                        code = color + 'K'
                    else:
                        continue
                    tile_color = parts[-1]  # 'white' or 'black'
                    templates.append({'img': img, 'code': code, 'name': fname, 'tile_color': tile_color})
    return templates

def match_piece(square_roi, templates):
    """Match the square ROI against all templates and return the best match code and score."""
    best_score = -1
    best_code = None
    h, w = square_roi.shape[:2]
    if h == 0 or w == 0:
        return None, -1
    for tpl in templates:
        tpl_img = tpl['img']
        if tpl_img is None or tpl_img.shape[0] == 0 or tpl_img.shape[1] == 0:
            continue
        # Resize template to match square if needed
        try:
            tpl_resized = cv2.resize(tpl_img, (w, h))
        except Exception as e:
            continue
        # Convert both to grayscale
        roi_gray = cv2.cvtColor(square_roi, cv2.COLOR_BGR2GRAY)
        tpl_gray = cv2.cvtColor(tpl_resized, cv2.COLOR_BGR2GRAY)
        # Template matching
        res = cv2.matchTemplate(roi_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = max_val
            best_code = tpl['code']
    # Threshold to avoid false positives
    if best_score > 0.55:
        return best_code, best_score
    else:
        return None, best_score

def piece_code_to_fen_char(code):
    mapping = {
        'WP': 'P', 'WN': 'N', 'WB': 'B', 'WR': 'R', 'WQ': 'Q', 'WK': 'K',
        'BP': 'p', 'BN': 'n', 'BB': 'b', 'BR': 'r', 'BQ': 'q', 'BK': 'k'
    }
    return mapping.get(code, '')

def board_dict_to_fen(board_dict):
    fen_rows = []
    for rank in range(8, 0, -1):
        row = ''
        empty = 0
        for file in 'abcdefgh':
            square = f'{file}{rank}'
            code = board_dict.get(square)
            if code:
                if empty > 0:
                    row += str(empty)
                    empty = 0
                row += piece_code_to_fen_char(code)
            else:
                empty += 1
        if empty > 0:
            row += str(empty)
        fen_rows.append(row)
    fen = '/'.join(fen_rows) + ' w KQkq - 0 1'
    return fen

def best_move_from_fen(fen: str, engine_path: str, time_limit: float = 0.1) -> str:
    board = chess.Board(fen)
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        result = engine.play(board, chess.engine.Limit(time=time_limit))
    return result.move.uci()

def detect_chessboard_with_templates(image_path, template_dir='templates', output_path='detected-board.png', engine_path='stockfish'):
    """
    Detect chessboard and pieces using template matching, mark detected pieces on the output image.
    Fallback to edge/contour-based board detection if chessboard corners are not found.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    result_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
    board_found = False
    if ret:
        # Standard chessboard detection
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        corners = corners.reshape(-1, 2)
        square_size = np.mean([
            np.linalg.norm(corners[i+1] - corners[i])
            for i in range(len(corners)-1)
        ])
        min_x = int(np.min(corners[:, 0]) - square_size)
        max_x = int(np.max(corners[:, 0]) + square_size)
        min_y = int(np.min(corners[:, 1]) - square_size)
        max_y = int(np.max(corners[:, 1]) + square_size)
        square_width = (max_x - min_x) // 8
        square_height = (max_y - min_y) // 8
        board_found = True
    else:
        # Fallback: edge/contour-based detection
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        best_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w)/h
                if 0.8 < aspect_ratio < 1.2:
                    if area > max_area:
                        max_area = area
                        best_contour = contour
        if best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)
            min_x, min_y, max_x, max_y = x, y, x+w, y+h
            square_width = w // 8
            square_height = h // 8
            board_found = True
        else:
            print("Chessboard not found!")
            return
    # Proceed with template matching if board is found
    templates = load_templates(template_dir)
    board_dict = {}
    for row in range(8):
        for col in range(8):
            x = min_x + col * square_width
            y = min_y + row * square_height
            square_roi = img[y:y+square_height, x:x+square_width]
            code, score = match_piece(square_roi, templates)
            square = f"{chr(97+col)}{8-row}"
            if code:
                board_dict[square] = code
                cv2.rectangle(result_img, (x, y), (x+square_width, y+square_height), (0, 0, 255), 2)
                cv2.putText(result_img, code, (x+5, y+square_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.imwrite(output_path, result_img)
    print(f"Detection result saved to {output_path}")
    # Generate FEN
    fen = board_dict_to_fen(board_dict)
    print(f"FEN: {fen}")
    # Get best move from Stockfish
    try:
        best_move = best_move_from_fen(fen, engine_path)
        print(f"Best move: {best_move}")
    except Exception as e:
        print(f"Error getting best move: {e}")

def main():
    # Input and output file paths
    input_image = 'chessboard.png'
    output_image = 'detected-board.png'
    template_dir = 'templates'
    try:
        detect_chessboard_with_templates(input_image, template_dir=template_dir, output_path=output_image)
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == '__main__':
    main() 