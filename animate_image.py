# animate_image.py

import numpy as np
import cv2

def generate_frame_at_time(img_rgb, depth_map, t, duration, max_disp=50, initial_zoom=1.2):
    h, w, _ = img_rgb.shape

    # Adjust the speed of the animation to be 10 times quicker
    # Calculate the number of loops needed to fill the duration
    animation_cycle_duration = duration / 10  # Each cycle is 1/10th of the total duration
    cycles = int(duration / animation_cycle_duration)

    # Compute the time within the current cycle
    cycle_time = t % animation_cycle_duration
    norm_t = cycle_time / animation_cycle_duration  # Normalized time from 0 to 1 within the cycle

    # Determine the direction of the animation within the cycle
    # If we're in an even-numbered cycle, we zoom out; if odd, we zoom in
    cycle_number = int(t / animation_cycle_duration)
    direction = 1 if cycle_number % 2 == 0 else -1

    # Zoom in and out within each cycle
    zoom_range = initial_zoom - 1.0  # The total range of zoom
    zoom = initial_zoom - zoom_range * norm_t * direction

    # Loop zoom to stay within the bounds [1.0, initial_zoom]
    if zoom > initial_zoom:
        zoom = initial_zoom
    elif zoom < 1.0:
        zoom = 1.0

    # Pan back and forth within each cycle
    x_shift = -max_disp * np.sin(2 * np.pi * norm_t)
    y_shift = -max_disp * np.cos(2 * np.pi * norm_t)

    # No rotation
    angle = 0

    # Apply zoom
    img_zoomed = cv2.resize(img_rgb, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
    depth_zoomed = cv2.resize(depth_map, (img_zoomed.shape[1], img_zoomed.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Calculate displacement based on depth
    disp_x = x_shift * depth_zoomed
    disp_y = y_shift * depth_zoomed

    # Create meshgrid for pixel coordinates
    X, Y = np.meshgrid(np.arange(img_zoomed.shape[1]), np.arange(img_zoomed.shape[0]))

    # Map coordinates
    map_x = (X + disp_x).astype(np.float32)
    map_y = (Y + disp_y).astype(np.float32)

    # Since angle is 0, rotation is not applied
    img_rotated = img_zoomed
    map_x_rotated = map_x
    map_y_rotated = map_y

    # Warp the image using the displacement maps
    warped_img = cv2.remap(img_rotated, map_x_rotated, map_y_rotated, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)

    # Crop or pad to original size
    start_x = (warped_img.shape[1] - w) // 2
    start_y = (warped_img.shape[0] - h) // 2
    warped_img_cropped = warped_img[start_y:start_y + h, start_x:start_x + w]

    return warped_img_cropped
