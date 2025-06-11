import cv2
import numpy as np
import os
from datetime import datetime

def get_square_roi(img, x, y, width, height):
    """Get the region of interest for a specific square."""
    return img[y:y+height, x:x+width]

def detect_piece(square_roi):
    """
    Detect if there's a piece in the square and determine its color.
    Returns: (has_piece, is_white, piece_type)
    """
    if square_roi is None or square_roi.size == 0:
        return False, None, None

    # Convert to grayscale
    gray = cv2.cvtColor(square_roi, cv2.COLOR_BGR2GRAY)
    
    # Calculate average brightness
    avg_brightness = np.mean(gray)
    
    # Calculate standard deviation to detect piece presence
    std_dev = np.std(gray)
    
    # Threshold for piece detection - lowered for better sensitivity
    has_piece = std_dev > 20  # Adjusted threshold
    
    if has_piece:
        # Determine piece color based on average brightness
        is_white = avg_brightness > 127
        
        # Simple piece type detection based on shape analysis
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Calculate circularity
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Basic piece type classification
            if circularity > 0.8:  # More circular
                piece_type = "pawn"  # or queen/king
            else:
                piece_type = "other"  # rook, knight, bishop
        else:
            piece_type = "unknown"
            
        return True, is_white, piece_type
    
    return False, None, None

def log_board_state(pieces, empty_squares):
    """Log the board state to the console."""
    print("\n" + "=" * 50)
    print("CHESS BOARD STATE ANALYSIS")
    print("=" * 50)
    
    print("\nDETECTED PIECES:")
    print("-" * 30)
    if pieces:
        for piece in pieces:
            print(f"Position: {piece['position']}")
            print(f"Color: {piece['color']}")
            print(f"Type: {piece['type']}")
            print("-" * 30)
    else:
        print("No pieces detected")
    
    print("\nEMPTY SQUARES:")
    print("-" * 30)
    if empty_squares:
        print(", ".join(empty_squares))
    else:
        print("No empty squares detected")
    
    print("\n" + "=" * 50)

def detect_chessboard(image_path):
    """
    Detect chessboard in the image and draw a rectangle around it.
    Returns the image with the detected board highlighted and the crop coordinates.
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Create a copy for drawing
    result_img = img.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Try to detect the full 8x8 board
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

    if ret:
        print("Found internal corners of the chessboard")
        
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Get the corners array
        corners = corners.reshape(-1, 2)

        # Calculate the average square size
        square_size = np.mean([
            np.linalg.norm(corners[i+1] - corners[i])
            for i in range(len(corners)-1)
        ])

        # Get the top-left and bottom-right corners of the full board
        min_x = int(np.min(corners[:, 0]) - square_size)
        max_x = int(np.max(corners[:, 0]) + square_size)
        min_y = int(np.min(corners[:, 1]) - square_size)
        max_y = int(np.max(corners[:, 1]) + square_size)

        # Calculate square dimensions
        square_width = (max_x - min_x) // 8
        square_height = (max_y - min_y) // 8

        # Draw rectangle around the full board with thicker lines
        cv2.rectangle(result_img, (min_x, min_y), (max_x, max_y), (0, 0, 255), 3)

        # Draw grid lines and detect pieces
        pieces = []
        empty_squares = []
        
        for row in range(8):
            for col in range(8):
                # Calculate square coordinates
                x = min_x + col * square_width
                y = min_y + row * square_height
                
                # Draw grid lines with thicker lines
                if col < 7:  # Vertical lines
                    cv2.line(result_img, (x + square_width, y), 
                            (x + square_width, y + square_height), (0, 255, 0), 2)
                if row < 7:  # Horizontal lines
                    cv2.line(result_img, (x, y + square_height), 
                            (x + square_width, y + square_height), (0, 255, 0), 2)
                
                # Get square ROI and detect piece
                square_roi = get_square_roi(img, x, y, square_width, square_height)
                has_piece, is_white, piece_type = detect_piece(square_roi)
                
                # Get square notation
                square_notation = f"{chr(97+col)}{8-row}"
                
                if has_piece:
                    # Draw piece indicator with thicker lines and filled circle
                    color = (255, 255, 255) if is_white else (0, 0, 0)
                    # Draw filled circle
                    cv2.circle(result_img, 
                             (x + square_width//2, y + square_height//2),
                             square_width//3, color, -1)  # Filled circle
                    # Draw outline
                    cv2.circle(result_img, 
                             (x + square_width//2, y + square_height//2),
                             square_width//3, (0, 0, 255), 2)  # Red outline
                    
                    # Add piece information
                    pieces.append({
                        'position': square_notation,
                        'color': 'white' if is_white else 'black',
                        'type': piece_type
                    })
                    
                    # Add text label with background
                    label = f"{piece_type[0].upper()}"  # First letter of piece type
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    text_x = x + (square_width - text_size[0]) // 2
                    text_y = y + (square_height + text_size[1]) // 2
                    
                    # Draw text background with padding
                    padding = 4
                    cv2.rectangle(result_img, 
                                (text_x - padding, text_y - text_size[1] - padding),
                                (text_x + text_size[0] + padding, text_y + padding),
                                (255, 255, 255), -1)
                    
                    # Draw text with thicker lines
                    cv2.putText(result_img, label, 
                              (text_x, text_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    empty_squares.append(square_notation)

        # Add text with thicker lines
        cv2.putText(result_img, "8x8 Chessboard Detected", (min_x, min_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Log the board state to console
        log_board_state(pieces, empty_squares)

        return result_img, (min_x, min_y, max_x, max_y), pieces

    else:
        print("No chessboard detected. Trying alternative method...")
        # Try edge detection
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour that could be a rectangle
        max_area = 0
        best_contour = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter out small contours
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w)/h
                if 0.8 < aspect_ratio < 1.2:  # Look for roughly square shapes
                    if area > max_area:
                        max_area = area
                        best_contour = contour
        
        if best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 3)
            
            # Calculate square dimensions
            square_width = w // 8
            square_height = h // 8
            
            # Draw grid lines and detect pieces
            pieces = []
            empty_squares = []
            
            for row in range(8):
                for col in range(8):
                    # Calculate square coordinates
                    square_x = x + col * square_width
                    square_y = y + row * square_height
                    
                    # Draw grid lines
                    if col < 7:  # Vertical lines
                        cv2.line(result_img, (square_x + square_width, square_y), 
                                (square_x + square_width, square_y + square_height), (0, 255, 0), 2)
                    if row < 7:  # Horizontal lines
                        cv2.line(result_img, (square_x, square_y + square_height), 
                                (square_x + square_width, square_y + square_height), (0, 255, 0), 2)
                    
                    # Get square ROI and detect piece
                    square_roi = get_square_roi(img, square_x, square_y, square_width, square_height)
                    has_piece, is_white, piece_type = detect_piece(square_roi)
                    
                    # Get square notation
                    square_notation = f"{chr(97+col)}{8-row}"
                    
                    if has_piece:
                        # Draw piece indicator
                        color = (255, 255, 255) if is_white else (0, 0, 0)
                        cv2.circle(result_img, 
                                 (square_x + square_width//2, square_y + square_height//2),
                                 square_width//3, color, -1)  # Filled circle
                        cv2.circle(result_img, 
                                 (square_x + square_width//2, square_y + square_height//2),
                                 square_width//3, (0, 0, 255), 2)  # Red outline
                        
                        # Add piece information
                        pieces.append({
                            'position': square_notation,
                            'color': 'white' if is_white else 'black',
                            'type': piece_type
                        })
                        
                        # Add text label
                        label = f"{piece_type[0].upper()}"
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                        text_x = square_x + (square_width - text_size[0]) // 2
                        text_y = square_y + (square_height + text_size[1]) // 2
                        
                        # Draw text background
                        padding = 4
                        cv2.rectangle(result_img, 
                                    (text_x - padding, text_y - text_size[1] - padding),
                                    (text_x + text_size[0] + padding, text_y + padding),
                                    (255, 255, 255), -1)
                        
                        # Draw text
                        cv2.putText(result_img, label, 
                                  (text_x, text_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        empty_squares.append(square_notation)
            
            cv2.putText(result_img, "Possible 8x8 Chessboard", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Log the board state
            log_board_state(pieces, empty_squares)
            
            print("Found a possible chessboard using edge detection")
            return result_img, (x, y, x+w, y+h), pieces
        else:
            print("No chessboard detected using any method")
            # Draw a message on the image
            cv2.putText(result_img, "No Chessboard Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Log empty state
            log_board_state([], [])
            return result_img, None, []

def main():
    # Input and output file paths
    input_file = "chessboard.png"
    output_file = "detected-board.jpg"

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        return

    try:
        # Detect chessboard and get the processed image and crop coordinates
        result_img, crop_coords, pieces = detect_chessboard(input_file)

        if crop_coords is not None:
            # Crop the image to the detected board area
            min_x, min_y, max_x, max_y = crop_coords
            cropped_img = result_img[min_y:max_y, min_x:max_x]
            
            # Save the cropped result
            cv2.imwrite(output_file, cropped_img)
            print(f"\nCropped chessboard saved as '{output_file}'")
        else:
            # Save the full image if no board was detected
            cv2.imwrite(output_file, result_img)
            print(f"\nFull image saved as '{output_file}' (no board detected)")

    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main() 