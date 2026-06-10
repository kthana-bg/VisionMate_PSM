import argparse
import os
import sys
import csv
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Landmark indices
NOSE_IDX           = 0
LEFT_EAR_IDX       = 7
RIGHT_EAR_IDX      = 8
LEFT_SHOULDER_IDX  = 11
RIGHT_SHOULDER_IDX = 12

# Output path 
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "data", "posture_dataset", "posture_data.csv"
)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

NUM_FEATURES = 10
FIELDNAMES   = [f"feature_{i}" for i in range(NUM_FEATURES)] + ["label"]

# Pose model asset path
POSE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "assets", "pose_landmarker_lite.task"
)


def extract_landmarks_new_api(detection_result, image_width, image_height):
    """
    Extract key landmark pixel positions from a MediaPipe Tasks
    PoseLandmarkerResult.  Returns a dict of named points or None.
    """
    if not detection_result or not detection_result.pose_landmarks:
        return None

    lm = detection_result.pose_landmarks[0]   # first (only) person

    def to_pixel(idx):
        return (
            int(lm[idx].x * image_width),
            int(lm[idx].y * image_height),
        )

    return {
        "nose":            to_pixel(NOSE_IDX),
        "left_ear":        to_pixel(LEFT_EAR_IDX),
        "right_ear":       to_pixel(RIGHT_EAR_IDX),
        "left_shoulder":   to_pixel(LEFT_SHOULDER_IDX),
        "right_shoulder":  to_pixel(RIGHT_SHOULDER_IDX),
    }


def extract_landmark_feature_vector(landmarks):
    """
    Build a flat 10-element feature vector normalised to shoulder width.
    """
    shoulder_mid_x = (landmarks["left_shoulder"][0] + landmarks["right_shoulder"][0]) / 2.0
    shoulder_mid_y = (landmarks["left_shoulder"][1] + landmarks["right_shoulder"][1]) / 2.0
    shoulder_width = max(
        abs(landmarks["left_shoulder"][0] - landmarks["right_shoulder"][0]), 1
    )

    points = [
        landmarks["nose"],
        landmarks["left_ear"],
        landmarks["right_ear"],
        landmarks["left_shoulder"],
        landmarks["right_shoulder"],
    ]

    features = []
    for (x, y) in points:
        features.append((x - shoulder_mid_x) / shoulder_width)
        features.append((y - shoulder_mid_y) / shoulder_width)

    return np.array(features, dtype=np.float32)


def main(label_str: str):
    label      = 0 if label_str == "good" else 1
    label_name = "Good" if label == 0 else "Slouching"

    # Build PoseLandmarker
    if not os.path.exists(POSE_MODEL_PATH):
        print(f"ERROR: Pose model not found at:\n  {POSE_MODEL_PATH}")
        print("Make sure pose_landmarker_lite.task is in the assets/ folder.")
        sys.exit(1)

    base_options    = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    landmarker_opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(landmarker_opts)

    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Open CSV in append mode
    file_exists = os.path.exists(OUTPUT_PATH)
    csv_file    = open(OUTPUT_PATH, "a", newline="")
    writer      = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()

    sample_count = 0
    frame_ts_ms  = 0   # monotonically increasing timestamp for VIDEO mode

    print(f"\nCapturing [{label_name}] posture samples.")
    print("SPACE = capture sample  |  Q = quit & save\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        h, w      = frame.shape[:2]
        frame_ts_ms += 33          # ~30 fps cadence

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result    = landmarker.detect_for_video(mp_image, frame_ts_ms)
        landmarks = extract_landmarks_new_api(result, w, h)

        if landmarks:
            for pt in landmarks.values():
                cv2.circle(frame, pt, 5, (0, 255, 0), -1)

        cv2.putText(
            frame,
            f"Label: {label_name} | Captured: {sample_count}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        cv2.putText(
            frame,
            "SPACE=capture  Q=quit",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )
        cv2.imshow("Posture Data Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            if landmarks:
                feat = extract_landmark_feature_vector(landmarks)
                row  = {f"feature_{i}": float(feat[i]) for i in range(NUM_FEATURES)}
                row["label"] = label
                writer.writerow(row)
                sample_count += 1
                print(f"  Sample {sample_count} captured ({label_name})")
            else:
                print("  No pose detected — move into frame and try again.")

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    landmarker.close()
    print(f"\nDone. {sample_count} samples saved to:\n  {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        choices=["good", "slouch"],
        required=True,
        help="Label for captured samples: 'good' or 'slouch'",
    )
    args = parser.parse_args()
    main(args.label)