import cv2 as cv
import numpy as np
import glob, pickle, os

IMAGE_GLOB = "calibration_images/*.png"   # path to chessboard images
CANDIDATE_SIZES = [(9,5), (5,9)]          # 10x6 squares -> 9x5 inner corners (and rotated)
SQUARE_MM = 50.0                          # square size in mm (only scales t, not K)
USE_SB = True                             # try findChessboardCornersSB first if available

def detect(gray, size):
    """Find corners; try SB first (if present), then classic + SubPix."""
    if USE_SB and hasattr(cv, "findChessboardCornersSB"):
        ok, corners = cv.findChessboardCornersSB(
            gray, size, flags=cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY
        )
        if ok:
            return True, corners
    ok, corners = cv.findChessboardCorners(
        gray, size,
        flags=cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE | cv.CALIB_CB_FAST_CHECK
    )
    if ok:
        corners = cv.cornerSubPix(
            gray, corners, (11,11), (-1,-1),
            (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        )
    return ok, corners

def main():
    imgs = sorted(glob.glob(IMAGE_GLOB))
    assert imgs, f"No images found for pattern: {IMAGE_GLOB}"

    # 1) pick orientation (9x5 vs 5x9)
    hits = {}
    for sz in CANDIDATE_SIZES:
        cnt = 0
        for fn in imgs:
            img = cv.imread(fn)
            if img is None: continue
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ok,_ = detect(gray, sz)
            cnt += int(ok)
        hits[sz] = cnt
    best_size = max(hits, key=hits.get)
    print("candidate hits:", hits, "=> chosen:", best_size)

    # 2) collect points (Z=0 plane style of OpenCV)
    w,h = best_size
    objp = np.zeros((w*h,3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1,2).astype(np.float32) * SQUARE_MM

    objpoints, imgpoints, used = [], [], []
    imsize = None
    for fn in imgs:
        img = cv.imread(fn)
        if img is None: continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if imsize is None:
            imsize = gray.shape[::-1]  # (W,H)
        ok, corners = detect(gray, best_size)
        if not ok:
            continue
        objpoints.append(objp)
        imgpoints.append(corners)
        used.append(fn)

    print(f"frames used: {len(used)}/{len(imgs)}")
    assert used, "No valid frames — check lighting/focus/pattern size."

    # 3) calibrate 
    rms, K, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, imsize, None, None
    )

    print("\n=== CALIBRATION RESULTS ===")
    print("K (cameraMatrix):\n", K)
    print("dist coefficients:", dist.ravel())

    # 4) save outputs 
    pickle.dump((K, dist), open("calibration.pkl", "wb"))
    pickle.dump(K, open("cameraMatrix.pkl", "wb"))
    pickle.dump(dist, open("dist.pkl", "wb"))
    print("Saved: cameraMatrix.pkl, dist.pkl, calibration.pkl")

    # 5) average reprojection error over all frames
    errs = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        e = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        errs.append(float(e))
    mean_err = float(np.mean(errs)) if errs else float('nan')
    print(f"\nTotal reprojection error (avg over frames): {mean_err:.6f} px")

    # 6) choose best frame and output extrinsics (R,t)
    errs_np = np.array(errs)
    best_idx = int(np.argmin(errs_np))
    best_err = errs_np[best_idx]
    best_img = used[best_idx]

    rvec_best = rvecs[best_idx]
    tvec_best = tvecs[best_idx]
    R_best, _ = cv.Rodrigues(rvec_best)

    print("\n=== EXTRINSICS (best frame) ===")
    print("best frame:", best_img)
    print("reproj error (best) =", f"{best_err:.6f}", "px")
    print("\nR =\n", R_best)
    print("\nt =\n", tvec_best.reshape(-1,1))

    np.savetxt("R.txt", R_best, fmt="%.9f")
    np.savetxt("t.txt", tvec_best.reshape(-1,1), fmt="%.9f")
    pickle.dump(
        {"R": R_best, "t": tvec_best, "rvec": rvec_best,
         "best_frame": best_img, "reproj_err_px": float(best_err),
         "K": K, "dist": dist, "image_size": imsize,
         "pattern": best_size, "square_mm": float(SQUARE_MM)},
        open("extrinsics_ref.pkl", "wb")
    )
    print("Saved: R.txt, t.txt, extrinsics_ref.pkl")

if __name__ == "__main__":
    main()
