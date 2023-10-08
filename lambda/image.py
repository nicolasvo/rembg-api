import cv2
import numpy as np
from rembg import remove


def rescale_image(image, px=512, padding=0):
    height, width, _ = image.shape
    if [height, width].index(max([height, width])) == 0:
        factor = px / height
        height = px
        width = int(width * factor)
    else:
        factor = px / width
        width = px
        height = int(height * factor)

    image_resized = cv2.resize(
        image, dsize=(width, height), interpolation=cv2.INTER_LINEAR
    )

    # Create a larger canvas with the same number of channels as the input image
    padded_height = height + 2 * padding
    padded_width = width + 2 * padding
    padded_image = np.zeros(
        (padded_height, padded_width, image.shape[2]), dtype=np.uint8
    )

    # Calculate the position to place the resized image in the center
    x_offset = (padded_width - width) // 2
    y_offset = (padded_height - height) // 2

    # Place the resized image in the center of the padded canvas
    padded_image[
        y_offset : y_offset + height, x_offset : x_offset + width
    ] = image_resized

    return padded_image


def add_outline(image, stroke_size, outline_color):
    # Ensure the image has an alpha channel for transparency
    if image.shape[-1] != 4:
        raise ValueError("Input image must have an alpha channel (4 channels).")

    # Create a copy of the original image
    outlined_image = image.copy()

    # Create a mask for fully transparent parts of the image
    mask = (image[:, :, 3] == 0).astype(np.uint8)

    # Calculate the kernel size based on the desired stroke size
    kernel_size = int(stroke_size * 0.2) * 2 + 1  # Ensure it's an odd number
    if kernel_size < 1:
        kernel_size = 1

    # Create a kernel for dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply erosion to round the outline
    outline = cv2.erode(mask, kernel, iterations=1)

    # Use the eroded mask to get the outline of the fully transparent parts
    outline = mask - outline

    # Apply Gaussian blur to smooth the outline
    outline = cv2.GaussianBlur(
        outline.astype(np.float32), (kernel_size, kernel_size), 0
    )

    # Threshold the blurred outline to make it binary
    _, outline = cv2.threshold(outline, 0.5, 1, cv2.THRESH_BINARY)

    # Apply the outline color only to the outline region
    for c in range(4):  # Loop through RGBA channels
        outlined_image[:, :, c] = (
            outlined_image[:, :, c] * (1 - outline) + outline_color[c] * outline
        )

    return outlined_image


def extract_bounding_box(image, bbox):
    if bbox:
        min_x, min_y, max_x, max_y = bbox
        bounding_box_image = image[min_y : max_y + 1, min_x : max_x + 1]
        return bounding_box_image
    else:
        return None


def get_bbox_from_mask(mask):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    x1, y1, w, h = cv2.boundingRect(contours[0])
    x2, y2 = x1 + w, y1 + h
    if len(contours) > 1:
        for b in contours:
            x_t, y_t, w_t, h_t = cv2.boundingRect(b)
            x1 = min(x1, x_t)
            y1 = min(y1, y_t)
            x2 = max(x2, x_t + w_t)
            y2 = max(y2, y_t + h_t)
        h = y2 - y1
        w = x2 - x1
    return [x1, y1, x2, y2]


def segment(input_path, output_path):
    with open(input_path, "rb") as i:
        input = i.read()
        output = remove(
            input,
            alpha_matting=True,
            alpha_matting_erode_size=0,
            alpha_matting_foreground_threshold=30,
        )
    image_array = np.frombuffer(output, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
    mask = image[:, :, 3] != 0
    bbox = get_bbox_from_mask(mask)
    image = extract_bounding_box(image, bbox)
    image = rescale_image(image, padding=13)
    image = add_outline(image, 40, (255, 255, 255, 255))
    image = rescale_image(image, padding=0)

    cv2.imwrite(output_path, image)
    print("donezo")
