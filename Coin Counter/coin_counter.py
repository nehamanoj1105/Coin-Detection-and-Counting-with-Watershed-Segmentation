import cv2
import numpy as np
import os

# Function to process a single image and count coins
def process_image(image_path):
    # Create directory for intermediate outputs
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    intermediate_dir = os.path.join("intermediate_outputs", image_name)
    os.makedirs(intermediate_dir, exist_ok=True)

    # Read the image
    img = cv2.imread(image_path)
    original = img.copy()

    # Step 1: Convert to grayscale and apply CLAHE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    cv2.imwrite(os.path.join(intermediate_dir, "1_gray_clahe.png"), gray)

    # Step 2: Gaussian blur
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    cv2.imwrite(os.path.join(intermediate_dir, "2_blur.png"), blur)

    # Step 3: Canny edge detection
    edges = cv2.Canny(blur, 30, 100)
    cv2.imwrite(os.path.join(intermediate_dir, "3_edges.png"), edges)

    # Step 4: Threshold + Morph open
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cv2.imwrite(os.path.join(intermediate_dir, "4_thresh_opening.png"), opening)

    # Step 5: Sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    cv2.imwrite(os.path.join(intermediate_dir, "5_sure_background.png"), sure_bg)

    # Step 6: Distance transform and sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dist_display = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(intermediate_dir, "6_distance_transform.png"), dist_display)
    
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    cv2.imwrite(os.path.join(intermediate_dir, "7_sure_foreground.png"), sure_fg)

    # Step 7: Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    cv2.imwrite(os.path.join(intermediate_dir, "8_unknown.png"), unknown)

    # Step 8: Connected components and watershed
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers_before = np.uint8(markers.copy())
    cv2.imwrite(os.path.join(intermediate_dir, "9_markers_before_watershed.png"), markers_before)

    markers = cv2.watershed(img, markers)
    watershed_result = img.copy()
    watershed_result[markers == -1] = [0, 0, 255]  # red boundary
    cv2.imwrite(os.path.join(intermediate_dir, "10_watershed_result.png"), watershed_result)

    # Step 10: Count and draw contours
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coin_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(original, center, radius, (0, 255, 0), 2)
            coin_count += 1

    print(f"Image: {os.path.basename(image_path)} | Coins Detected: {coin_count}")

    # Text overlay
    cv2.putText(original, f"Coins Detected: {coin_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Save final image
    output_path = os.path.join("output_images", os.path.basename(image_path))
    os.makedirs("output_images", exist_ok=True)
    cv2.imwrite(output_path, original)


# Process all images in input_images folder
def process_all_images():
    input_dir = "input_images"
    for file in os.listdir(input_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            process_image(os.path.join(input_dir, file))

# Main
if __name__ == "__main__":
    process_all_images()
