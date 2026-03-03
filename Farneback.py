import csv
import os
import cv2
import numpy as np


# USER SETTINGS (DIRECT THIS TO YOUR VIDEO!)
# VIDEO_FOLDER = "input_videos"
VIDEO_FOLDER = "cropped_videos"
VIDEO_NAME = "left_eye.avi"
VIDEO_PATH = f"{VIDEO_FOLDER}/{VIDEO_NAME}"
VIDEO_NAME = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_CSV = f"Output/{VIDEO_NAME}_flow_timeseries.csv"
OUTPUT_VIZ = f"Output/{VIDEO_NAME}_flow_visualization.mp4"

SAMPLE_FPS = 4.0                   # optical flow sampling rate (Hz)
# HORIZONTAL_FOV_DEG = 115.0         # Adjust to headset horizontal FOV - VARJO = 115
RESIZE_WIDTH = 640                 # None = no resize
RESIZE_HEIGHT = None               # None = keep aspect ratio

USE_GAUSSIAN = False               # Farnebäck Gaussian window
MAG_CLIP = None                    # e.g., 10.0 to cap visualization scale

RUN_SELF_TEST = False              # Toggle the self test for code validation check

# FARNEBÄCK PARAMETERS
PYR_SCALE = 0.5
LEVELS = 3
WINSIZE = 15
ITERATIONS = 3
POLY_N = 5
POLY_SIGMA = 1.2
FLAGS = cv2.OPTFLOW_FARNEBACK_GAUSSIAN if USE_GAUSSIAN else 0



# HELPER FUNCTIONS
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


# SELF TEST FOR VALIDATION
def run_self_test():
    """
    Synthetic validation: known translation test.
    Confirms Farnebäck behaves as expected.
    """
    h, w = 240, 320
    rng = np.random.default_rng(0)

    # Create a textured image
    img = (rng.random((h, w)) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (9, 9), 0)

    # Known translation in pixels
    dx, dy = 5, -3  # right 5 px, up 3 px

    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    flow = cv2.calcOpticalFlowFarneback(
        img,
        shifted,
        None,
        PYR_SCALE,
        LEVELS,
        WINSIZE,
        ITERATIONS,
        POLY_N,
        POLY_SIGMA,
        FLAGS
    )

    mean_u = np.mean(flow[..., 0])
    mean_v = np.mean(flow[..., 1])

    print("SELF TEST RESULTS")
    print(f"Expected mean_u ≈ {dx}, mean_v ≈ {dy}")
    print(f"Measured mean_u = {mean_u:.2f}, mean_v = {mean_v:.2f}")

    if abs(mean_u - dx) > 1.5 or abs(mean_v - dy) > 1.5:
        raise RuntimeError("Self-test failed")
    else:
        print("Self-test passed")


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    step_frames = max(1, int(round(native_fps / SAMPLE_FPS)))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames reported: {total_frames}")

    print(f"Original resolution: {orig_width} x {orig_height}")
    print(f"Video FPS: {native_fps:.2f}")
    print(f"Sampling every {step_frames} frames (~{SAMPLE_FPS} Hz)")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame")

    prev_gray = resize_if_needed(to_gray_u8(frame))
    resized_height, resized_width = prev_gray.shape

    print(f"Resized resolution: {resized_width} x {resized_height}")
    print(f"Due to resizing, all magnitudes reduced by a factor of {orig_width/RESIZE_WIDTH}.")

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

        # --------- METADATA HEADER ----------
        f.write("# Optical Flow Metadata\n")
        f.write(f"# video_name: {VIDEO_NAME}\n")
        f.write(f"# original_resolution: {orig_width}x{orig_height}\n")
        f.write(f"# resized_resolution: {resized_width}x{resized_height}\n")
        f.write(f"# native_fps: {native_fps}\n")
        f.write(f"# sample_fps: {SAMPLE_FPS}\n")
        f.write(f"# step_frames: {step_frames}\n")
        f.write(f"# pyr_scale: {PYR_SCALE}\n")
        f.write(f"# levels: {LEVELS}\n")
        f.write(f"# winsize: {WINSIZE}\n")
        f.write(f"# iterations: {ITERATIONS}\n")
        f.write(f"# poly_n: {POLY_N}\n")
        f.write(f"# poly_sigma: {POLY_SIGMA}\n")
        f.write(f"# gaussian_flag: {USE_GAUSSIAN}\n")
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
                # "mean_mag_deg_per_sec",
            ]
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

            # Progress (percent of original frames)
            if total_frames > 0:
                pct = int((frame_idx / total_frames) * 100)
                if pct != last_print_pct and (pct % 5 == 0):  # prints at 0,5,10,...,100
                    print(f"Progress: {pct}% ({frame_idx}/{total_frames} frames)")
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

            stats = flow_stats(flow)

            dt = frame_idx - prev_frame_idx

            # ---- Convert to pixels/sec ----
            mean_px_per_sec = (stats["mean_mag"] / dt) * native_fps

            # ---- Convert to degrees/sec ----
            # deg_per_pixel = HORIZONTAL_FOV_DEG / resized_width
            # mean_deg_per_sec = mean_px_per_sec * deg_per_pixel

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
                "mean_mag_px_per_sec": mean_px_per_sec,
                # "mean_mag_deg_per_sec": mean_deg_per_sec,
            })

            if viz_writer:
                viz_writer.write(flow_to_bgr(flow))

            prev_gray = gray
            prev_frame_idx = frame_idx

    cap.release()
    if viz_writer:
        viz_writer.release()

    print("Done.")
    print(f"Wrote: {OUTPUT_CSV}")
    if OUTPUT_VIZ:
        print(f"Wrote: {OUTPUT_VIZ}")

if __name__ == "__main__":
    if RUN_SELF_TEST:
        run_self_test()
    else:
        main()

