
# Hough Circle Transform Classifier Script (OpenCV)
## Description
This project is designed to detect a specific shape and color in a set of traffic light images. The program processes images, segments them by color using the HSV colour model, and identifies the shape of the traffic light images using contour detection and Hough Circle Transform to check if the contour is circular. The script also classifies arrow shapes by using circular line computer vision detection.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation
Install [PyCharm Community Edition](https://www.jetbrains.com/pycharm/download/?section=windows)

## Usage
To use the script, you need to provide the path to the reference shape specimen image and the test folder containing images. You can download the shape specimen image from the internet (e.g., Circle).

    1. Update the ref_image and test_folder variables in the script with the appropriate paths.
    2. Run the script:

### Example
    
    import cv2
    import numpy as np
    import os

    # Load the user-defined circular reference image
    ref_image = cv2.imread('/path/to/your/shape/specimen/image', 0)

    # Define the contour of the reference image
    ref_contour = cv2.findContours(ref_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Load the test image folder
    test_folder = "/path/to/your/test/folder"

    # Loop through all images in the test folder
    for filename in os.listdir(test_folder):
        # Load the image
        image_path = os.path.join(test_folder, filename)
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to segment out the shapes
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

        # Initialize green_mask as None
        green_mask = None

        # Classify arrow vs round shape using circular line computer vision detection
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h

            # Check if the contour is circular using Hough Circle Transform
            circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=50)
            if circles is not None:
                shape = 'Circular'
                print('Circular shape detected')

                # Crop the circular shape to classify the color (for round traffic light)
                x, y, r = circles[0, 0]
                crop = image[int(y-r):int(y+r), int(x-r):int(x+r)]

                # Display the cropped image
                cv2.imshow('Crop', crop)

                # Classify the color of the circular shape
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                cv2.imshow('HSV', hsv)

                green_range = (70, 90, 100), (100, 120, 140)
                yellow_range = (20, 100, 100), (30, 120, 120)
                red_range = (0, 100, 100), (10, 120, 120)
                green_mask = cv2.inRange(hsv, *green_range)
                yellow_mask = cv2.inRange(hsv, *yellow_range)
                red_mask = cv2.inRange(hsv, *red_range)

                green_pixels = cv2.countNonZero(green_mask)
                yellow_pixels = cv2.countNonZero(yellow_mask)
                red_pixels = cv2.countNonZero(red_mask)

                print('Green pixels:', green_pixels)
                print('Yellow pixels:', yellow_pixels)
                print('Red pixels:', red_pixels)

                # Check if the cropped shape is green
                if green_pixels > yellow_pixels and green_pixels > red_pixels:
                    color = 'Green'
                    print(f"Green {shape} detected in {filename}")
                elif yellow_pixels > green_pixels and yellow_pixels > red_pixels:
                    color = 'Yellow'
                    print(f"Yellow {shape} detected in {filename}")
                else:
                    color = 'Red'
                    print(f"Red {shape} detected in {filename}")
            else:
                shape = 'Arrow'
                print('Arrow shape detected')

        # Resize the green mask to match the image dimensions
        if green_mask is not None:
            green_mask_resized = cv2.resize(green_mask, (image.shape[1], image.shape[0]))
        else:
            green_mask_resized = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Display the original image and the masked image side by side
        combined_image = np.hstack([image, cv2.cvtColor(green_mask_resized, cv2.COLOR_GRAY2BGR)])
        cv2.imshow('Image and Mask', combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

## License
This project is licensed under the [MIT License](https://www.mit.edu/~amini/LICENSE.md).



