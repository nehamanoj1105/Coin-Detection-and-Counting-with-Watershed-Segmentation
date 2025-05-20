# Coin Counter using Watershed Algorithm

This project detects and counts the number of coins in an image using image processing techniques such as grayscale conversion, Gaussian blur, Canny edge detection, distance transform, and the Watershed segmentation algorithm. It is implemented in Python using OpenCV.


## Description

Many coin counting applications struggle with overlapping or touching coins. This project aims to accurately count coins even when they are overlapping by applying the Watershed algorithm for precise segmentation.


## Features

- Pre-processes images using Gaussian blur and edge detection.
- Handles overlapping coins using Distance Transform and Watershed segmentation.
- Counts coins accurately and visually marks them using contours and circles.
- Displays the total number of coins on the image.


## Input

- Type: JPEG or PNG images.
- Requirement: Coins should be placed on a plain, contrasting background.
- Limitation: Accuracy may drop if coins are heavily occluded, poorly lit, or the image is noisy or blurry.



## Output

- Image with circles drawn around detected coins.
- Printed total coin count on the image.

