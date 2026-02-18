import cv2
import numpy as np
from pathlib import Path

# ============================================================
# FARNEBÄCK PARAMETERS (match your pipeline)
# ============================================================
PYR_SCALE = 0.5
LEVELS = 3
WINSIZE = 15
ITERATIONS = 3
POLY_N = 5
POLY_SIGMA = 1.2
FLAGS = 0

# ============================================================
# HELPERS
# ============================================================

def flow_to_rgb(flow, max_mag=None):
    """HSV flow visualization (BGR output)."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    if max_mag is None:
        max_mag = np.percentile(mag, 99) + 1e-6

    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.float32)
    hsv[..., 0] = ang * 180 / np.pi / 2   # hue = angle
    hsv[..., 1] = 1.0                     # saturation
    hsv[..., 2] = np.clip(mag / max_mag, 0, 1)  # value = normalized magnitude

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return (bgr * 255).astype(np.uint8)

def make_textured_image(H, W, blur_ksize=5, noise_sigma=10.0, seed=None):
    """
    Create a textured grayscale image with enough structure for OF.
    Uses random field + blur + noise.
    """
    rng = np.random.default_rng(seed)
    base = rng.uniform(0, 255, (H, W)).astype(np.float32)

    if blur_ksize and blur_ksize > 1:
        base = cv2.GaussianBlur(base, (blur_ksize, blur_ksize), 0)

    if noise_sigma and noise_sigma > 0:
        base += rng.normal(0, noise_sigma, (H, W)).astype(np.float32)

    return np.clip(base, 0, 255).astype(np.uint8)

def translate_image(img, dx, dy):
    H, W = img.shape[:2]
    M = np.array([[1, 0, dx],
                  [0, 1, dy]], dtype=np.float32)
    # Border constant to avoid wraparound cheating
    out = cv2.warpAffine(
        img, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return out

# ============================================================
# TEST
# ============================================================

def run_translation_test(
    dx, dy,
    H=256, W=384,
    trials=30,
    blur_ksize=5,
    noise_sigma=10.0,
    save_dir="fb_synth_translation",
    save_example=True
):
    """
    Unit test for optical flow: known translation (dx, dy).
    Reports recovered median flow and EPE on interior ROI.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Avoid border artifacts from warping
    border = int(max(abs(dx), abs(dy)) + 15)
    y0, y1 = border, H - border
    x0, x1 = border, W - border

    expected = np.array([dx, dy], dtype=np.float32)

    epe_means = []
    epe_medians = []
    flow_meds = []

    example_saved = False

    for t in range(trials):
        img1 = make_textured_image(H, W, blur_ksize=blur_ksize, noise_sigma=noise_sigma, seed=t)
        img2 = translate_image(img1, dx, dy)

        fb_flow = cv2.calcOpticalFlowFarneback(
            img1, img2, None,
            PYR_SCALE, LEVELS, WINSIZE,
            ITERATIONS, POLY_N, POLY_SIGMA, FLAGS
        )

        roi = fb_flow[y0:y1, x0:x1, :]
        epe = np.sqrt((roi[..., 0] - expected[0])**2 + (roi[..., 1] - expected[1])**2)

        epe_means.append(float(np.mean(epe)))
        epe_medians.append(float(np.median(epe)))

        # robust central tendency for the flow field
        flow_med = np.median(roi.reshape(-1, 2), axis=0)
        flow_meds.append(flow_med)

        if save_example and not example_saved:
            # save a qualitative example
            cv2.imwrite(str(save_dir / f"img1_dx{dx}_dy{dy}.png"), img1)
            cv2.imwrite(str(save_dir / f"img2_dx{dx}_dy{dy}.png"), img2)
            cv2.imwrite(str(save_dir / f"fb_flow_rgb_dx{dx}_dy{dy}.png"), flow_to_rgb(fb_flow))

            # EPE heatmap (normalized)
            epe_map = np.zeros((H, W), dtype=np.float32)
            epe_map[y0:y1, x0:x1] = epe
            epe_vis = cv2.normalize(epe_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            epe_vis = cv2.applyColorMap(epe_vis, cv2.COLORMAP_JET)
            cv2.imwrite(str(save_dir / f"epe_heat_dx{dx}_dy{dy}.png"), epe_vis)

            example_saved = True

    flow_meds = np.vstack(flow_meds)
    flow_med_avg = np.mean(flow_meds, axis=0)

    print("\n========== SYNTHETIC TRANSLATION TEST ==========")
    print(f"Expected (dx, dy)        : ({dx:.2f}, {dy:.2f})")
    print(f"Trials                  : {trials}")
    print(f"ROI border              : {border}px")
    print(f"Recovered median flow   : ({flow_med_avg[0]:.2f}, {flow_med_avg[1]:.2f})")
    print(f"Mean EPE (avg)          : {np.mean(epe_means):.4f} px")
    print(f"Median EPE (avg)        : {np.mean(epe_medians):.4f} px")
    print(f"Example outputs in      : {save_dir.resolve()}")
    print("===============================================\n")


if __name__ == "__main__":
    # A few standard translations (edit as you like)
    run_translation_test(dx=5, dy=0)
    run_translation_test(dx=0, dy=5)
    run_translation_test(dx=8, dy=3)
    run_translation_test(dx=-6, dy=2)
