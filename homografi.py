import logging

import cv2
import numpy as np

from errors import PanoramaError

# Minimum number of ratio-test matches required for a reliable homography.
# Shared with panorama_pipeline so both stages use the very same gate.
MIN_MATCH_COUNT = 10

_logger = logging.getLogger(__name__)


def donusum_matrisi_hesapla(
    kp1,
    kp2,
    good_matches,
    sol_resim,
    sag_resim,
    ransac_cikti_yolu=None,
):
    """Estimate the left -> right homography with RANSAC.

    Returns (H, mask). Raises PanoramaError when there are too few matches or
    when RANSAC cannot fit a model. The RANSAC inlier visualization is only
    drawn and saved when ransac_cikti_yolu is given (nothing is written to the
    current working directory by default).
    """
    # 1. Safety gate: require at least MIN_MATCH_COUNT matches for a reliable result
    if len(good_matches) < MIN_MATCH_COUNT:
        raise PanoramaError(
            f"Yeterli eslesme bulunamadi! Gerekli: {MIN_MATCH_COUNT}, "
            f"Bulunan: {len(good_matches)}"
        )

    # Collect point coordinates
    sol_noktalar = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    sag_noktalar = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography (warp matrix) with RANSAC.
    # RANSAC draws random minimal sample sets, so without a fixed seed the
    # inlier mask -- and every image derived from it -- changes from run to
    # run. Seeding right before the call keeps outputs repeatable for tests.
    cv2.setRNGSeed(0)
    H, mask = cv2.findHomography(sol_noktalar, sag_noktalar, cv2.RANSAC, 5.0)

    if H is None or mask is None:
        raise PanoramaError("Homografi matrisi hesaplanamadi.")

    saglam_nokta_sayisi = int(np.sum(mask))

    # 2. Visualization: draw only matches that passed RANSAC.
    # mask holds values like [1, 0, 1, 1...]; 1 = inlier, 0 = outlier
    if ransac_cikti_yolu:
        matchesMask = mask.ravel().tolist()

        # Draw inliers (mask == 1) in green
        draw_params = dict(matchColor=(0, 255, 0),  # Inlier match lines in green
                           singlePointColor=None,
                           matchesMask=matchesMask,  # Draw only RANSAC inliers
                           flags=2)

        # Draw filtered matches on the image
        ransac_sonrasi_resim = cv2.drawMatches(
            sol_resim, kp1, sag_resim, kp2, good_matches, None, **draw_params
        )

        # Save the filtered match visualization
        cv2.imwrite(ransac_cikti_yolu, ransac_sonrasi_resim)
        _logger.info(
            "RANSAC %d noktadan %d tanesini kusursuz buldu. Temizlenmis resim kaydedildi: %s",
            len(good_matches), saglam_nokta_sayisi, ransac_cikti_yolu,
        )
    else:
        _logger.info(
            "RANSAC %d noktadan %d tanesini kusursuz buldu.",
            len(good_matches), saglam_nokta_sayisi,
        )

    return H, mask
