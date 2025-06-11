import cv2
import numpy as np
import os

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
    # We use (7,7) for internal corners, then expand to get the full board
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

    if ret:
        print("Found internal corners of the chessboard")
        
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw the internal corners
        cv2.drawChessboardCorners(result_img, (7, 7), corners, ret)

        # Get the corners array
        corners = corners.reshape(-1, 2)

        # Calculate the average square size
        square_size = np.mean([
            np.linalg.norm(corners[i+1] - corners[i])
            for i in range(len(corners)-1)
        ])

        # Get the top-left and bottom-right corners of the full board
        # We need to extend beyond the internal corners to include the outer squares
        min_x = int(np.min(corners[:, 0]) - square_size)
        max_x = int(np.max(corners[:, 0]) + square_size)
        min_y = int(np.min(corners[:, 1]) - square_size)
        max_y = int(np.max(corners[:, 1]) + square_size)

        # Draw rectangle around the full board
        cv2.rectangle(result_img, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

        # Draw grid lines to show the 8x8 squares
        square_width = (max_x - min_x) / 8
        square_height = (max_y - min_y) / 8

        # Draw vertical lines
        for i in range(1, 8):
            x = int(min_x + i * square_width)
            cv2.line(result_img, (x, min_y), (x, max_y), (0, 255, 0), 1)

        # Draw horizontal lines
        for i in range(1, 8):
            y = int(min_y + i * square_height)
            cv2.line(result_img, (min_x, y), (max_x, y), (0, 255, 0), 1)

        # Add text
        cv2.putText(result_img, "8x8 Chessboard Detected", (min_x, min_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return result_img, (min_x, min_y, max_x, max_y)

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
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            # Draw grid lines
            square_width = w / 8
            square_height = h / 8
            
            # Draw vertical lines
            for i in range(1, 8):
                line_x = int(x + i * square_width)
                cv2.line(result_img, (line_x, y), (line_x, y+h), (0, 255, 0), 1)
            
            # Draw horizontal lines
            for i in range(1, 8):
                line_y = int(y + i * square_height)
                cv2.line(result_img, (x, line_y), (x+w, line_y), (0, 255, 0), 1)
            
            cv2.putText(result_img, "Possible 8x8 Chessboard", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            print("Found a possible chessboard using edge detection")
            return result_img, (x, y, x+w, y+h)
        else:
            print("No chessboard detected using any method")
            # Draw a message on the image
            cv2.putText(result_img, "No Chessboard Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return result_img, None

def main():
    # Input and output file paths
    input_file = "chessboard.jpg"
    output_file = "detected-board.jpg"

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        return

    try:
        # Detect chessboard and get the processed image and crop coordinates
        result_img, crop_coords = detect_chessboard(input_file)

        if crop_coords is not None:
            # Crop the image to the detected board area
            min_x, min_y, max_x, max_y = crop_coords
            cropped_img = result_img[min_y:max_y, min_x:max_x]
            
            # Save the cropped result
            cv2.imwrite(output_file, cropped_img)
            print(f"Cropped chessboard saved as '{output_file}'")
        else:
            # Save the full image if no board was detected
            cv2.imwrite(output_file, result_img)
            print(f"Full image saved as '{output_file}' (no board detected)")

    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main() 