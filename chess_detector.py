import cv2
import numpy as np

class ChessDetector:
    def __init__(self):
        self.board_size = 8
        self.square_size = None
        self.board_corners = None

    def detect_chessboard(self, image):
        """
        Detect the chessboard in the image and return the corners.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Store the corners
            self.board_corners = corners
            
            # Calculate square size
            self.square_size = np.mean([
                np.linalg.norm(corners[i+1] - corners[i])
                for i in range(len(corners)-1)
            ])
            
            return True, corners
        
        return False, None

    def get_square_roi(self, image, row, col):
        """
        Get the region of interest for a specific square.
        """
        if self.board_corners is None:
            return None
            
        # Calculate the corners of the square
        top_left = self.board_corners[row * 7 + col]
        top_right = self.board_corners[row * 7 + col + 1]
        bottom_left = self.board_corners[(row + 1) * 7 + col]
        bottom_right = self.board_corners[(row + 1) * 7 + col + 1]
        
        # Get the minimum and maximum coordinates
        min_x = int(min(top_left[0][0], bottom_left[0][0]))
        max_x = int(max(top_right[0][0], bottom_right[0][0]))
        min_y = int(min(top_left[0][1], top_right[0][1]))
        max_y = int(max(bottom_left[0][1], bottom_right[0][1]))
        
        # Extract the square
        square = image[min_y:max_y, min_x:max_x]
        return square

    def detect_piece(self, square_roi):
        """
        Detect if there's a piece in the square and its color.
        Returns: (has_piece, is_white)
        """
        if square_roi is None:
            return False, None
            
        # Convert to grayscale
        gray = cv2.cvtColor(square_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate average brightness
        avg_brightness = np.mean(gray)
        
        # Simple threshold to determine if there's a piece
        has_piece = avg_brightness < 200  # Adjust threshold as needed
        
        if has_piece:
            # Determine piece color based on average brightness
            is_white = avg_brightness > 100  # Adjust threshold as needed
            return True, is_white
            
        return False, None

    def detect_last_move(self, image):
        """
        Detect the last move by looking for highlighted squares.
        Returns: (from_square, to_square) in chess notation
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for highlighted squares (usually green or yellow)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        
        # Create mask for highlighted squares
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Find contours of highlighted squares
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) >= 2:
            # Get the two largest contours
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            
            # Convert contour centers to board coordinates
            squares = []
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Convert to board coordinates
                    # TODO: Implement coordinate conversion
                    squares.append((cx, cy))
            
            if len(squares) == 2:
                # TODO: Convert coordinates to chess notation
                return "e2", "e4"  # Example return
                
        return None, None 