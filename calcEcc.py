import numpy as np
import cv2

def _to_gray32(img):
    """Convert BGR/RGB/GRAY uint8/float images to single-channel float32 in [0,1]."""
    if img.ndim == 3:
        # assume BGR/RGB; OpenCV treats it as BGR typically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = gray.astype(np.float32)
    # normalize if likely in 0..255
    if gray.max() > 1.5:
        gray /= 255.0
    return gray

def _coerce_points(pts):
    """
    Accepts (N,2) or (N,1,2) arrays. Returns (N,2) float32.
    """
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim == 3 and pts.shape[1] == 1 and pts.shape[2] == 2:
        pts = pts[:, 0, :]
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Points must be shape (N,2) or (N,1,2).")
    return pts.astype(np.float32, copy=False)

def calcEcc(
    prev_img,
    next_img,
    prev_pts,
    next_pts,
    prev_rois,
    term_criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4),
    gaussFiltSize=5,
    local_search = 100, 
):
    """
    ECC-based per-point tracker (translation only), roughly analogous to cv2.calcOpticalFlowPyrLK
    but using cv2.findTransformECC on per-point templates with per-point ROI sizes.

    Parameters
    ----------
    prev_img : ndarray
        Previous frame (gray or color). Any dtype supported by OpenCV; converted internally to float32 gray in [0,1].
    next_img : ndarray
        Next frame (same size as prev_img). Converted internally to float32 gray in [0,1].
    prev_pts : ndarray
        (N,2) or (N,1,2) array of point locations in prev_img (x,y).
    next_pts : ndarray
        (N,2) or (N,1,2) array of *initial guesses* for the point locations in next_img (x,y).
        Will be refined and returned in the same (N,2) shape.
    prev_rois : ndarray
        (N,4) int array of per-point template ROIs in prev_img: [x0, y0, w, h] (upper-left origin).
        Each point can have a different template size.
    term_criteria : tuple
        Termination criteria for ECC (count, eps).
    gaussFiltSize : int
        Gaussian smoothing size for ECC (odd positive integer; 0 disables).

    Returns
    -------
    next_pts_out : ndarray (N,2) float32
        Refined next-frame point locations.
    err : ndarray (N,) float32
        ECC correlation coefficients (retval) per point; -1 for failures or invalid templates.

    Notes
    -----
    * ECC finds a warp that maps the *input image* (our next-frame patch) to the *template* (our prev-frame ROI).
      For pure translation, if the optimal warp is [dx, dy], it means: input(x+dx, y+dy) ≈ template(x, y).
      Therefore the *point location in next_img* is updated as:
          new_next_pt = initial_next_pt - [dx, dy]
      (i.e., subtract the ECC warp translation).
    * We use cv2.getRectSubPix to extract a same-size next-frame patch centered at the current guess.
      If the true motion is very large, consider doing a coarse-to-fine (pyramids) or increase search strategies.
    * If an ROI is invalid (empty/out-of-bounds) or low-texture (zero variance), we keep the initial guess and set err=-1.
    """

    # Prepare grayscale float32 images in [0,1]
    prev_gray = _to_gray32(prev_img)
    next_gray = _to_gray32(next_img)

    # Coerce point shapes to (N,2)
    prev_pts = _coerce_points(prev_pts)
    next_pts_init = _coerce_points(next_pts)

    prev_rois = np.asarray(prev_rois)
    if prev_rois.ndim != 2 or prev_rois.shape[1] != 4:
        raise ValueError("prev_rois must be an (N,4) array of [x0, y0, w, h].")

    N = prev_pts.shape[0]
    if prev_rois.shape[0] != N or next_pts_init.shape[0] != N:
        raise ValueError("prev_pts, next_pts, and prev_rois must have the same length N.")

    next_pts_out = next_pts_init.copy().astype(np.float32)
    ecc_corr = np.full((N,), -1.0, dtype=np.float32)  # ECC score per point (retval), -1 on failure
    ecc_best_filtSize = np.full((N,), 0, dtype=np.int32)  # best gaussFiltSize per point

    h_img, w_img = prev_gray.shape[:2]

    # for each point
    for i in range(N):
        x0, y0, w, h = map(int, prev_rois[i])
        if w <= 1 or h <= 1:
            # Degenerate ROI
            continue

        # Clip ROI to image bounds
        x1 = max(0, x0)
        y1 = max(0, y0)
        x2 = min(w_img, x0 + w)
        y2 = min(h_img, y0 + h)

        if x2 - x1 < 2 or y2 - y1 < 2:
            # Invalid ROI after clipping
            continue

        # Template from prev frame (copy to ensure contiguous float32 block)
        template = prev_gray[y1:y2, x1:x2].copy()

        # Reject low-texture templates: ECC diverges on flat patches
        if float(template.var()) < 1e-6:
            continue

        # if local_search is positive
        if local_search <= 0:
            local_search = 0
            search_x1 = 0; search_y1 = 0
        # Extract next-frame patch at current guess (same size as template)
        if local_search > 0:
            search_x1 = max(x1 - local_search, 0)
            search_x2 = min(x2 + local_search, w_img)
            search_y1 = max(y1 - local_search, 0)
            search_y2 = min(y2 + local_search, h_img)
            next_gray_local_search = next_gray[search_y1:search_y2, search_x1:search_x2].copy()
        else:
            next_gray_local_search = next_gray
 
        # Initialize translation-only warp (including initial guess)
        warp = np.eye(2, 3, dtype=np.float32)
        warp[0,2] = x1 + next_pts_init[i, 0] - prev_pts[i, 0] - search_x1
        warp[1,2] = y1 + next_pts_init[i, 1] - prev_pts[i, 1] - search_y1

        try:
            # try the best result from different gaussFiltSize. If gaussFiltSize is 5, then try 5,3,1
            for tryGaussFiltSize in range(gaussFiltSize, 0, -2):
                # ECC returns correlation coefficient (higher is better), and updates 'warp' in-place
                cc, warp = cv2.findTransformECC(
                    template,          # template (prev ROI)
                    next_gray_local_search,  # input image (next gray image)
                    warp,              # initial 2x3 warp (translation-only)
                    cv2.MOTION_TRANSLATION,
                    term_criteria,
                    None,              # input mask (None)
                    tryGaussFiltSize
                )
                # cc, warp = cv2.findTransformECC(template,next_gray,warp,cv2.MOTION_TRANSLATION,term_criteria,None,gaussFiltSize)
                # Translation components that warp the *next patch* to the *template*
                dx = warp[0, 2] + search_x1
                dy = warp[1, 2] + search_y1
                if cc > ecc_corr[i]:
                    # Update point: subtract the ECC warp translation (see note above)
                    next_pts_out[i, 0] = dx + (prev_pts[i, 0] - prev_rois[i, 0])
                    next_pts_out[i, 1] = dy + (prev_pts[i, 1] - prev_rois[i, 1])
                    ecc_corr[i] = float(cc)
                    ecc_best_filtSize[i] = tryGaussFiltSize
            # end of for tryGaussFiltSize
        except cv2.error:
            # Keep initial next point guess and leave ecc_corr[i] as -1.0
            pass
        # end of try-except
    # end of for each point

    return next_pts_out, ecc_corr, ecc_best_filtSize
