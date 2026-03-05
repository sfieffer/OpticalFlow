## Same as VideoCropping.py but for cropping multiple videos in bulk so I can walk away from my laptop for 6 hours.
import cv2
import os
import numpy as np

input_folder = "input_videos/uneditted"
output_folder = "cropped_videos/untrimmed"
os.makedirs(output_folder, exist_ok=True)

# Secondary crop margins (AFTER left-eye crop)
CROP_LEFT = 70
CROP_RIGHT = 25
CROP_BOTTOM = 45
CROP_TOP = 0

# Acceptable video formats
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(VIDEO_EXTENSIONS)]

print("Found", len(video_files), "videos")

for video_file in video_files:

    input_path = os.path.join(input_folder, video_file)

    name_no_ext = os.path.splitext(video_file)[0]

    # Example: P3_ConditionCenter_varjo_capture...
    parts = name_no_ext.split("_")

    participant = parts[0]  # P3
    condition_raw = parts[1]  # ConditionCenter

    # Remove "Condition"
    condition = condition_raw.replace("Condition", "")

    clean_name = f"{participant}_{condition}"

    output_filename = f"{clean_name}_cropped.avi"
    output_path = os.path.join(output_folder, output_filename)
    if os.path.exists(output_path):
        print("Skipping (already processed):", video_file)
        continue

    print("\n==============================")
    print("Processing:", video_file)
    print("Saving as:", output_filename)

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Skipping (cannot open):", video_file)
        continue

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Resolution:", width, "x", height)
    print("Total frames:", total_frames)

    if fps is None or fps <= 1e-6:
        fps = 30.0

    # Left-eye crop
    left_width = width // 2

    left_width_even = left_width - (left_width % 2)
    height_even = height - (height % 2)

    # Final size after secondary crop
    out_w = left_width_even - CROP_LEFT - CROP_RIGHT
    out_h = height_even - CROP_TOP - CROP_BOTTOM

    out_w_even = out_w - (out_w % 2)
    out_h_even = out_h - (out_h % 2)

    extra_w = out_w - out_w_even
    extra_h = out_h - out_h_even

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w_even, out_h_even))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame is None:
            continue

        # Crop left eye
        left_eye = frame[:height_even, :left_width_even]

        h, w = left_eye.shape[:2]

        x1 = CROP_LEFT
        x2 = w - CROP_RIGHT - extra_w
        y1 = CROP_TOP
        y2 = h - CROP_BOTTOM - extra_h

        cropped = left_eye[y1:y2, x1:x2]

        out.write(cropped)

        frame_count += 1

        if total_frames > 0 and frame_count % 500 == 0:
            percent = (frame_count / total_frames) * 100
            print(f"{video_file}: {percent:.1f}%")

    cap.release()
    out.release()

    print("Finished:", video_file)