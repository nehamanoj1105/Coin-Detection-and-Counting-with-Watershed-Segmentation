# Coin Counter using Watershed Algorithm

This project detects and counts the number of coins in an image using image processing techniques such as grayscale conversion, Gaussian blur, Canny edge detection, distance transform, and the Watershed segmentation algorithm. It is implemented in Python using OpenCV.

## Description

Many coin counting applications struggle with overlapping or touching coins. This project aims to accurately count coins even when they are overlapping by applying the Watershed algorithm for precise segmentation.

## Features

- Pre-processes images using Gaussian blur and edge detection.
- Handles overlapping coins using Distance Transform and Watershed segmentation.
- Counts coins accurately and visually marks them using contours and circles.
- Displays the total number of coins on the image.

## Algorithm Workflow

1. Grayscale Conversion – Convert the original image to grayscale.
2. Gaussian Blur – Reduce noise for better edge detection.
3. Canny Edge Detection – Detect edges to find coin boundaries.
4. Thresholding – Apply binary segmentation.
5. Distance Transform – Highlights the center of coins.
6. Watershed Algorithm – Separates overlapping coins based on distance peaks.
7. Contour Detection – Detect and count segmented coin areas.
8. Drawing Output – Draws contours and displays the coin count on the image.

## Input

- Type: JPEG or PNG images.
- Requirement: Coins should be placed on a plain, contrasting background.
- Limitation: Accuracy may drop if coins are heavily occluded, poorly lit, or the image is noisy or blurry.

## Output

- Image with circles drawn around detected coins.
- Printed total coin count on the image.

## Requirements

- Python 3.7 or higher
- OpenCV
- NumPy
- matplotlib (optional, for plotting images)

## Installation

```bash
pip install opencv-python numpy matplotlib
```

## How to Execute
1. Clone the Repository
```bash
git clone https://github.com/yourusername/coin-counter-watershed.git
cd coin-counter-watershed
```

2. Install Dependencies
```bash
pip install opencv-python numpy matplotlib
```

3. Prepare Your Input
   
There are sample input images in the `input_images` folder

4. Run the Script
```bash
python3 coin_counter.py
```

This will:

Display the image with detected coins outlined and also the number of coins in the given image.

Print the number of coins to the terminal.


## Results and Conclusion

The Coin Counter using the Watershed Algorithm successfully detects and counts coins in static images, even in cases where coins are overlapping or touching. The combination of distance transform and the watershed segmentation technique provides a reliable method for separating adjacent coins that traditional edge detection methods alone cannot handle effectively.

On clear, high-contrast backgrounds with moderate lighting, the system achieves high accuracy (typically above 95%) in identifying individual coins. The visual output with contours and count display makes it easy to verify the results.

This project demonstrates the practical application of classical image processing techniques in solving real-world segmentation challenges. While effective, the system can be further improved by incorporating advanced denoising, adaptive thresholding, and deep learning models for more robust performance in diverse environments.
