import csv
import os
import cv2
import numpy as np


# USER SETTINGS (DIRECT THIS TO YOUR VIDEO!)
VIDEO_FOLDER = "input_videos"
VIDEO_NAME = "DJI_0012.mp4"
VIDEO_PATH = f"{VIDEO_FOLDER}/{VIDEO_NAME}"
VIDEO_NAME = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_CSV = f"Output/{VIDEO_NAME}_flow_timeseries.csv"
OUTPUT_VIZ = f"Output/{VIDEO_NAME}_flow_visualization.mp4"

SAMPLE_FPS = 4.0                   # optical flow sampling rate (Hz)
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
    step_frames = max(1, int(round(native_fps / SAMPLE_FPS)))

    print(f"Video FPS: {native_fps:.2f}")
    print(f"Sampling every {step_frames} frames (~{SAMPLE_FPS} Hz)")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame")

    prev_gray = resize_if_needed(to_gray_u8(frame))
    prev_frame_idx = 0

    writer = None
    viz_writer = None

    if OUTPUT_VIZ:
        h, w = prev_gray.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        viz_writer = cv2.VideoWriter(
            OUTPUT_VIZ,
            fourcc,
            SAMPLE_FPS,
            (w, h)
        )

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time_sec",
                "frame_idx",
                "dt_frames",
                "mean_mag",
                "median_mag",
                "p90_mag",
                "max_mag",
                "mean_u",
                "mean_v",
            ]
        )
        writer.writeheader()

        frame_idx = 0

        while True:
            for _ in range(step_frames):
                ok = cap.grab()
                frame_idx += 1
                if not ok:
                    break

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

            writer.writerow({
                "time_sec": frame_idx / native_fps,
                "frame_idx": frame_idx,
                "dt_frames": frame_idx - prev_frame_idx,
                **stats
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

