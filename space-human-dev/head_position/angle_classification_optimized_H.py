import cv2
import numpy as np
from src.mediapipe import GooMedia
from src.visualize import draw_face_one_color

# Define the landmark indices for different facial features
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

# Combine all indices
ALL_INDICES = set(LIPS + LEFT_EYE + LEFT_EYEBROW + RIGHT_EYE + RIGHT_EYEBROW)

def angle(left_hemisphere_ratio, right_hemisphere_ratio):
    under_limit_centralized = 0.9
    upper_limit_centralized = 1.1
    upper_limit_angled = 1.3

    if under_limit_centralized <= left_hemisphere_ratio <= upper_limit_centralized and under_limit_centralized <= right_hemisphere_ratio <= upper_limit_centralized:
        return 'The face is centralized'
    elif upper_limit_centralized < left_hemisphere_ratio < upper_limit_angled:
        return 'The face is a little bit more on the left hemisphere.'
    elif upper_limit_centralized < right_hemisphere_ratio < upper_limit_angled:
        return 'The face is a little bit more on the right hemisphere.'
    elif left_hemisphere_ratio >= upper_limit_angled:
        return 'The face is more angled on the left hemisphere.'
    elif right_hemisphere_ratio >= upper_limit_angled:
        return 'The face is more angled on the right hemisphere.'

def filter_points(all_points, indices):
    return [all_points[i] for i in indices if i < len(all_points)]

def process_frame(frame, points):
    if points:
        max_y = max(point[1] for point in points)
        min_y = min(point[1] for point in points)
        max_x = max(point[0] for point in points)
        min_x = min(point[0] for point in points)

        # Calculate midpoint
        mid_x = (max_x + min_x) // 2

        # Count points in each hemisphere
        left_hemisphere_count = sum(1 for point in points if point[0] < mid_x)
        right_hemisphere_count = len(points) - left_hemisphere_count

        # Calculate area and densities
        area = (max_x - min_x) * (max_y - min_y)
        points_density = len(points) / area if area > 0 else 0
        left_hemisphere_density = left_hemisphere_count / (area/2) if area > 0 else 0
        right_hemisphere_density = right_hemisphere_count / (area/2) if area > 0 else 0

        # Calculate ratios
        left_hemisphere_ratio = left_hemisphere_density / points_density if points_density > 0 else 0
        right_hemisphere_ratio = right_hemisphere_density / points_density if points_density > 0 else 0

        # Get angle message
        check_angle = angle(left_hemisphere_ratio, right_hemisphere_ratio)

        # Draw face points
        for point in points:
            cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)

        # Draw bounding box and midline
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        cv2.line(frame, (mid_x, min_y), (mid_x, max_y), (0, 0, 255), 2)

        # Add message to image
        cv2.putText(frame, check_angle, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame