import csv
import os
import cv2
import numpy as np


VIDEO_FOLDER = "cropped_videos/trimmed"
OUTPUT_FOLDER = "Output/bulk"

SAVE_VISUALIZATION = False

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

SAMPLE_FPS = 4.0
RESIZE_WIDTH = 640
RESIZE_HEIGHT = None

USE_GAUSSIAN = False
MAG_CLIP = None

RUN_SELF_TEST = False

# FARNEBÄCK PARAMETERS
PYR_SCALE = 0.5
LEVELS = 3
WINSIZE = 15
ITERATIONS = 3
POLY_N = 5
POLY_SIGMA = 1.2
FLAGS = cv2.OPTFLOW_FARNEBACK_GAUSSIAN if USE_GAUSSIAN else 0


os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if SAVE_VISUALIZATION:
    VIZ_FOLDER = os.path.join(OUTPUT_FOLDER, "visualizations")
    os.makedirs(VIZ_FOLDER, exist_ok=True)


# ----------------------------
# HELPER FUNCTIONS
# ----------------------------

def to_gray_u8(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def resize_if_needed(gray):
    if RESIZE_WIDTH is None and RESIZE_HEIGHT is None:
        return gray

    h, w = gray.shape

    if RESIZE_WIDTH is None:
        scale = RESIZE_HEIGHT / h
        new_w = int(w * scale)
        new_h = RESIZE_HEIGHT
    elif RESIZE_HEIGHT is None:
        scale = RESIZE_WIDTH / w
        new_w = RESIZE_WIDTH
        new_h = int(h * scale)
    else:
        new_w = RESIZE_WIDTH
        new_h = RESIZE_HEIGHT

    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)


def flow_stats(flow):
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return {
        "mean_mag": float(np.mean(mag)),
        "sum_mag": float(np.sum(mag)),
        "median_mag": float(np.median(mag)),
        "p90_mag": float(np.percentile(mag, 90)),
        "max_mag": float(np.max(mag)),
        "mean_u": float(np.mean(flow[..., 0])),
        "mean_v": float(np.mean(flow[..., 1])),
    }


def flow_to_bgr(flow):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    if MAG_CLIP is not None:
        mag = np.clip(mag, 0, MAG_CLIP)

    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = mag.astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# -------------------------------------
# PROCESS A SINGLE VIDEO
# -------------------------------------

def process_video(video_path):

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, f"{video_name}_flow_timeseries.csv")

    if SAVE_VISUALIZATION:
        OUTPUT_VIZ = os.path.join(VIZ_FOLDER, f"{video_name}_flow_visualization.mp4")
    else:
        OUTPUT_VIZ = None

    if os.path.exists(OUTPUT_CSV):
        print("Skipping existing:", video_name)
        return

    print("\n=============================")
    print("Processing:", video_name)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open video")
        return

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 1e-6:
        native_fps = 30.0

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step_frames = max(1, int(round(native_fps / SAMPLE_FPS)))

    print("Resolution:", orig_width, "x", orig_height)
    print("FPS:", native_fps)
    print("Frames:", total_frames)

    ok, frame = cap.read()
    if not ok:
        print("Could not read first frame")
        return

    prev_gray = resize_if_needed(to_gray_u8(frame))
    resized_height, resized_width = prev_gray.shape

    prev_frame_idx = 0

    viz_writer = None
    if OUTPUT_VIZ:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        viz_writer = cv2.VideoWriter(
            OUTPUT_VIZ,
            fourcc,
            SAMPLE_FPS,
            (resized_width, resized_height)
        )

    with open(OUTPUT_CSV, "w", newline="") as f:

        f.write("# Optical Flow Metadata\n")
        f.write(f"# video_name: {video_name}\n")
        f.write(f"# original_resolution: {orig_width}x{orig_height}\n")
        f.write(f"# resized_resolution: {resized_width}x{resized_height}\n")
        f.write(f"# native_fps: {native_fps}\n")
        f.write(f"# sample_fps: {SAMPLE_FPS}\n")
        f.write(f"# step_frames: {step_frames}\n")
        f.write("# -----------------------------------\n\n")

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time_sec",
                "frame_idx",
                "dt_frames",
                "mean_mag",
                "sum_mag",
                "median_mag",
                "p90_mag",
                "max_mag",
                "mean_u",
                "mean_v",
                "mean_mag_px_per_sec",
            ],
        )
        writer.writeheader()

        frame_idx = 0
        last_print_pct = -1

        while True:

            for _ in range(step_frames):
                ok = cap.grab()
                frame_idx += 1
                if not ok:
                    break

            if total_frames > 0:
                pct = int((frame_idx / total_frames) * 100)
                if pct != last_print_pct and pct % 5 == 0:
                    print(f"{video_name}: {pct}%")
                    last_print_pct = pct

            ok, frame = cap.retrieve()
            if not ok:
                break

            gray = resize_if_needed(to_gray_u8(frame))

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                PYR_SCALE,
                LEVELS,
                WINSIZE,
                ITERATIONS,
                POLY_N,
                POLY_SIGMA,
                FLAGS
            )

            if viz_writer:
                viz_writer.write(flow_to_bgr(flow))

            stats = flow_stats(flow)

            dt = frame_idx - prev_frame_idx
            if dt == 0:
                dt = 1

            mean_px_per_sec = (stats["mean_mag"] / dt) * native_fps

            writer.writerow({
                "time_sec": frame_idx / native_fps,
                "frame_idx": frame_idx,
                "dt_frames": dt,
                "mean_mag": stats["mean_mag"],
                "sum_mag": stats["sum_mag"],
                "median_mag": stats["median_mag"],
                "p90_mag": stats["p90_mag"],
                "max_mag": stats["max_mag"],
                "mean_u": stats["mean_u"],
                "mean_v": stats["mean_v"],
                "mean_mag_px_per_sec": mean_px_per_sec
            })

            prev_gray = gray
            prev_frame_idx = frame_idx

    cap.release()

    if viz_writer:
        viz_writer.release()

    print("Finished:", video_name)


# -----------------------
# BULK PROCESS ALL VIDEOS
# -----------------------

def bulk_main():

    video_files = sorted([
        f for f in os.listdir(VIDEO_FOLDER)
        if f.lower().endswith(VIDEO_EXTENSIONS)
    ])

    print("Found", len(video_files), "videos")

    for video in video_files:

        video_path = os.path.join(VIDEO_FOLDER, video)

        process_video(video_path)


if __name__ == "__main__":
    bulk_main()