## File used to crop the two-eyed VR videos from Amin's pendulum chair study
## Crops to the left eye and then rectangular crop to remove black bars from warped VR recording
import cv2
import os
import numpy as np

input_path = "input_videos/uneditted/P2_Whole.mp4"
output_folder = "cropped_videos/untrimmed"
os.makedirs(output_folder, exist_ok=True)

# --- Build output filename automatically ---
base_name = os.path.basename(input_path)              # input.mp4
name_no_ext, ext = os.path.splitext(base_name)       # ("input", ".mp4")

output_filename = f"{name_no_ext}_cropped.avi"
output_path = os.path.join(output_folder, output_filename)

print("Input file:", base_name)
print("Output file:", output_filename)

# Secondary crop margins (AFTER left-eye crop)
CROP_LEFT = 70
CROP_RIGHT = 25
CROP_BOTTOM = 45
CROP_TOP = 0  # keep top as-is

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError(f"Could not open input video: {input_path}")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

print("Input width,height,fps:", width, height, fps)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames:", total_frames)
# FPS sometimes returns 0.0 -> use a safe default
if fps is None or fps <= 1e-6:
    fps = 30.0
    print("WARNING: FPS was 0/invalid; defaulting to", fps)

# Left-eye crop (assumes side-by-side stereo)
left_width = width // 2

# Some codecs are picky about even dimensions -> force even
left_width_even = left_width - (left_width % 2)
height_even = height - (height % 2)

print("Left-eye base crop to:", left_width_even, "x", height_even)

# Compute final output size AFTER secondary cropping
out_w = left_width_even - CROP_LEFT - CROP_RIGHT
out_h = height_even - CROP_TOP - CROP_BOTTOM

if out_w <= 0 or out_h <= 0:
    raise ValueError(
        f"Invalid crop margins: would produce non-positive size ({out_w}x{out_h}). "
        f"Check CROP_LEFT/RIGHT/TOP/BOTTOM."
    )

# Many codecs prefer even output dimensions -> force even
out_w_even = out_w - (out_w % 2)
out_h_even = out_h - (out_h % 2)

# If we forced evenness, we need to adjust the right/bottom edge accordingly
# (i.e., shave off the extra 1 pixel on the right/bottom if needed)
extra_w = out_w - out_w_even
extra_h = out_h - out_h_even

print("Final output (after secondary crop):", out_w_even, "x", out_h_even)

fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # reliable in .avi
out = cv2.VideoWriter(output_path, fourcc, fps, (out_w_even, out_h_even))

print("VideoWriter opened:", out.isOpened())
if not out.isOpened():
    raise RuntimeError("VideoWriter failed to open. Try a different codec/container.")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ensure frame is expected dtype
    if frame is None:
        continue
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    # 1) Crop left half (left eye) and enforce even height
    left_eye = frame[:height_even, :left_width_even]

    # Sanity check: base crop size
    if left_eye.shape[1] != left_width_even or left_eye.shape[0] != height_even:
        raise RuntimeError(
            f"Base frame size mismatch: got {left_eye.shape[1]}x{left_eye.shape[0]}, "
            f"expected {left_width_even}x{height_even}"
        )

    # 2) Secondary crop (remove borders)
    h, w = left_eye.shape[:2]
    x1 = CROP_LEFT
    x2 = w - CROP_RIGHT - extra_w
    y1 = CROP_TOP
    y2 = h - CROP_BOTTOM - extra_h

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(
            f"Secondary crop produced invalid ROI: "
            f"x1={x1}, x2={x2}, y1={y1}, y2={y2} (w={w}, h={h})"
        )

    cropped = left_eye[y1:y2, x1:x2]

    # Final sanity check: must match VideoWriter dimensions exactly
    if cropped.shape[1] != out_w_even or cropped.shape[0] != out_h_even:
        raise RuntimeError(
            f"Final frame size mismatch: got {cropped.shape[1]}x{cropped.shape[0]}, "
            f"expected {out_w_even}x{out_h_even}"
        )

    out.write(cropped)
    frame_count += 1

    if total_frames > 0 and frame_count % 500 == 0:
        percent = (frame_count / total_frames) * 100
        print(f"Processed {frame_count}/{total_frames} "
              f"({percent:.1f}%)")

cap.release()
out.release()

print("Wrote frames:", frame_count)
print("Saved to:", output_path)