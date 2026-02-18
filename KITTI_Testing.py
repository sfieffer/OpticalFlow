import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Website: https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow
# Farneback models have F1 of 58-68% (Items 233-235)
# ============================================================
# USER SETTINGS
# ============================================================

KITTI_ROOT = Path("kitti_flow_2015")

USE_FLOW_OCC = True      # True = flow_occ, False = flow_noc
MAX_PAIRS = None         # None = all
SAVE_VIZ = True
VIZ_COUNT = 10           # number of qualitative samples to save
OUT_DIR = Path("fb_kitti_results")

# ============================================================
# FARNEBÄCK PARAMETERS (MATCH MAIN PIPELINE)
# ============================================================

PYR_SCALE = 0.5
LEVELS = 3
WINSIZE = 15
ITERATIONS = 3
POLY_N = 5
POLY_SIGMA = 1.2
FLAGS = 0

# ============================================================
# KITTI FLOW DECODING
# ============================================================

def read_kitti_flow_png(path):
    """
    KITTI 2015 flow PNG decoding.
    OpenCV loads as BGR:
      B = valid
      G = v
      R = u
    """
    flow_png = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if flow_png is None:
        raise RuntimeError(f"Could not read {path}")

    valid = flow_png[:, :, 0] > 0
    v = flow_png[:, :, 1].astype(np.float32)
    u = flow_png[:, :, 2].astype(np.float32)

    u = (u - 32768.0) / 64.0
    v = (v - 32768.0) / 64.0

    flow = np.dstack((u, v))
    return flow, valid

# ============================================================
# METRICS
# ============================================================

def compute_metrics(fb_flow, gt_flow, valid):
    u_fb = fb_flow[..., 0][valid]
    v_fb = fb_flow[..., 1][valid]
    u_gt = gt_flow[..., 0][valid]
    v_gt = gt_flow[..., 1][valid]

    epe = np.sqrt((u_fb - u_gt)**2 + (v_fb - v_gt)**2)
    gt_mag = np.sqrt(u_gt**2 + v_gt**2)

    outlier = (epe > 3.0) & (epe / np.maximum(gt_mag, 1e-6) > 0.05)

    return {
        "epe_mean": np.mean(epe),
        "epe_median": np.median(epe),
        "epe_p95": np.percentile(epe, 95),
        "fl_percent": 100.0 * np.mean(outlier),
    }

# ============================================================
# VISUALIZATION
# ============================================================

def flow_to_rgb(flow, max_mag=None):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    if max_mag is None:
        max_mag = np.percentile(mag, 99)

    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.float32)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 1.0
    hsv[..., 2] = np.clip(mag / max_mag, 0, 1)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return (rgb * 255).astype(np.uint8)

# ============================================================
# MAIN BENCHMARK
# ============================================================

def run():
    training = KITTI_ROOT / "training"
    img_dir = training / "image_2"
    flow_dir = training / ("flow_occ" if USE_FLOW_OCC else "flow_noc")

    assert img_dir.exists(), "image_2 not found"
    assert flow_dir.exists(), "flow dir not found"

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    if SAVE_VIZ:
        (OUT_DIR / "viz").mkdir(exist_ok=True)

    flow_files = sorted(flow_dir.glob("*_10.png"))
    if MAX_PAIRS:
        flow_files = flow_files[:MAX_PAIRS]

    metrics_all = []

    print(f"\nRunning Farnebäck on KITTI-2015 ({flow_dir.name})")
    print(f"Pairs: {len(flow_files)}\n")

    for i, flow_path in enumerate(tqdm(flow_files)):
        stem = flow_path.stem.replace("_10", "")
        img1_path = img_dir / f"{stem}_10.png"
        img2_path = img_dir / f"{stem}_11.png"

        img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(img2_path), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            continue

        fb_flow = cv2.calcOpticalFlowFarneback(
            img1, img2, None,
            PYR_SCALE, LEVELS, WINSIZE,
            ITERATIONS, POLY_N, POLY_SIGMA, FLAGS
        )

        gt_flow, valid = read_kitti_flow_png(flow_path)
        m = compute_metrics(fb_flow, gt_flow, valid)
        metrics_all.append(m)

        if SAVE_VIZ and i < VIZ_COUNT:
            fb_rgb = flow_to_rgb(fb_flow)
            gt_rgb = flow_to_rgb(gt_flow)

            epe_map = np.zeros(img1.shape, dtype=np.float32)
            epe_map[valid] = np.sqrt(
                (fb_flow[..., 0][valid] - gt_flow[..., 0][valid])**2 +
                (fb_flow[..., 1][valid] - gt_flow[..., 1][valid])**2
            )
            epe_vis = cv2.normalize(epe_map, None, 0, 255, cv2.NORM_MINMAX)
            epe_vis = cv2.applyColorMap(epe_vis.astype(np.uint8), cv2.COLORMAP_JET)

            cv2.imwrite(str(OUT_DIR / "viz" / f"{stem}_fb.png"), fb_rgb)
            cv2.imwrite(str(OUT_DIR / "viz" / f"{stem}_gt.png"), gt_rgb)
            cv2.imwrite(str(OUT_DIR / "viz" / f"{stem}_epe.png"), epe_vis)

    # ========================================================
    # SUMMARY
    # ========================================================

    mean_epe = np.mean([m["epe_mean"] for m in metrics_all])
    median_epe = np.mean([m["epe_median"] for m in metrics_all])
    p95_epe = np.mean([m["epe_p95"] for m in metrics_all])
    fl = np.mean([m["fl_percent"] for m in metrics_all])

    print("\n================ SUMMARY ================")
    print(f"Dataset        : KITTI-2015 {flow_dir.name}")
    print(f"Pairs          : {len(metrics_all)}")
    print(f"Mean EPE       : {mean_epe:.2f}")
    print(f"Median EPE     : {median_epe:.2f}")
    print(f"95th %ile EPE  : {p95_epe:.2f}")
    print(f"Fl (%)         : {fl:.2f}")
    print("========================================")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run()
