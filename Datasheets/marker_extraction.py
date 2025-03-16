



import cv2
import numpy as np

def auto_color_correction(image):
    """Apply histogram equalization on the L channel to correct colors."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better results
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    corrected_lab = cv2.merge([l, a, b])
    return cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)

def brighten_image_lab(image, factor=1.3):
    """Increase brightness using LAB color space multiplication."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Increase brightness in the L channel
    l = np.clip(l.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    brightened_lab = cv2.merge([l, a, b])
    return cv2.cvtColor(brightened_lab, cv2.COLOR_LAB2BGR)

def extract_white_mask(image):
    """Extract white areas from the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([230, 50, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Apply Dilation to close small gaps
    kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size if needed
    mask_dilated = cv2.dilate(mask, kernel, iterations=2)
    
    return mask_dilated

def draw_bounding_boxes_on_black(mask):
    
    """Fill detected white regions with white on a black background and draw bounding boxes."""
    black_background = np.zeros_like(mask)  # Create black image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 25000:  # Ignore small noise
            cv2.drawContours(black_background, [cnt], -1, (255, 255, 255), thickness=cv2.FILLED)  # Fill with white
            # cv2.rectangle(black_background, (x, y), (x + w, y + h), (255, 255, 255), 2)  # White bounding box
    
    return black_background

def add_label(image, text):
    """Add a text label above the image."""
    height, width = image.shape[:2]
    label_height = 40  # Space for label
    labeled_image = np.zeros((height + label_height, width, 3), dtype=np.uint8)  # Black background for text
    labeled_image[label_height:, :] = image  # Place image below the text
    cv2.putText(labeled_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # White text
    return labeled_image

# Load the image
image = cv2.imread("All Marker.jpg")
image =~ image
if image is None:
    print("Error: Could not load image.")
    exit(1)

# Processing Steps
corrected = auto_color_correction(image)  # Auto color correction
brightened = brighten_image_lab(corrected, factor=1.1)  # Brightness enhancement
mask = extract_white_mask(brightened)

# Convert mask to 3-channel for concatenation
mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# Fill white regions while keeping the rest black
bbox_filled_black = draw_bounding_boxes_on_black(mask)

# Prepare for display
image_titles = ["Original Image", "Color Corrected", "Brightened", "White Mask (Dilated)", "White Filled on Black"]
processed_images = [image, corrected, brightened, mask_colored, cv2.cvtColor(bbox_filled_black, cv2.COLOR_GRAY2BGR)]

# Ensure all images have the same aspect ratio
original_height, original_width = image.shape[:2]
resize_width = 600  # Fixed width for display
resize_height = int((resize_width / original_width) * original_height)  # Maintain aspect ratio

resized_images = [cv2.resize(img, (resize_width, resize_height)) for img in processed_images]

# Add labels
labeled_images = [add_label(img, title) for img, title in zip(resized_images, image_titles)]

# Concatenate images horizontally
final_output = cv2.hconcat(labeled_images)

# Display the result
cv2.imshow("Marker Extraction Pipeline", final_output)
cv2.waitKey(0)
cv2.destroyAllWindows()

















