import argparse
import cv2

from mediapipe.solutions import pose as mp_pose


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cam",
        help="Index of the camera to use",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    # Capture a frame from your camera
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise IOError("Cannot open camera")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with the Pose solution
        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                # Get the nose landmark
                nose_landmark = results.pose_landmarks.landmark[
                    mp_pose.PoseLandmark.NOSE
                ]

                # Get the depth value of the nose landmark
                depth = nose_landmark.z

                # Convert the depth value to meters
                distance = depth * 1000  # in millimeters

                print(f"Distance from camera to person: {distance:.2f} mm")


if __name__ == "__main__":
    main()
