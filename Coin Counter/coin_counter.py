import cv2
import numpy as np
import os

def process_image(image_path):
    img = cv2.imread(image_path)
    original = img.copy()

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Step 3: Apply Canny edge detection (tune thresholds if needed)
    edges = cv2.Canny(blur, 30, 100)

    # Step 4: Threshold + Morphology to remove noise
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Step 5: Sure background via dilation
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Step 6: Sure foreground via distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)

    # Step 7: Unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Step 8: Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add 1 so background is not 0
    markers = markers + 1
    markers[unknown == 255] = 0

    # Step 9: Apply watershed
    markers = cv2.watershed(img, markers)

    # Draw boundaries
    img[markers == -1] = [0, 0, 255]

    # Step 10: Find contours for coins
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coin_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:  # Adjust based on your image
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(original, center, radius, (0, 255, 0), 2)
            coin_count += 1

    # Show result
    print(f"Image: {os.path.basename(image_path)} | Coins Detected: {coin_count}")
    cv2.putText(original, f"Coins Detected: {coin_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Save output
    output_path = os.path.join("output_images", os.path.basename(image_path))
    os.makedirs("output_images", exist_ok=True)
    cv2.imwrite(output_path, original)

    # Optional: Display output
    # cv2.imshow("Result", original)
    # cv2.waitKey(0)

def process_all_images():
    input_dir = "input_images"
    for file in os.listdir(input_dir):
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
            process_image(os.path.join(input_dir, file))

if __name__ == "__main__":
    process_all_images()

