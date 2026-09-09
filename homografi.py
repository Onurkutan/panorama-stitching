import cv2
import numpy as np

def donusum_matrisi_hesapla(
    kp1,
    kp2,
    good_matches,
    sol_resim,
    sag_resim,
    ransac_cikti_yolu="ransac_temizlenmis_eslesmeler.jpg",
):
    # 1. Safety gate: require at least 10 matches for a reliable result
    MIN_MATCH_COUNT = 10
    
    if len(good_matches) < MIN_MATCH_COUNT:
        print(f"HATA: Yeterli eşleşme bulunamadı kanka! Gerekli: {MIN_MATCH_COUNT}, Bulunan: {len(good_matches)}")
        return None, None  # Return empty values so the program does not crash

    # Collect point coordinates
    sol_noktalar = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    sag_noktalar = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography (warp matrix) with RANSAC
    H, mask = cv2.findHomography(sol_noktalar, sag_noktalar, cv2.RANSAC, 5.0)

    if H is None or mask is None:
        print("HATA: Homografi matrisi hesaplanamadı.")
        return None, None
    
    # 2. Visualization: draw only matches that passed RANSAC
    # mask holds values like [1, 0, 1, 1...]; 1 = inlier, 0 = outlier
    matchesMask = mask.ravel().tolist() 

    # Draw inliers (mask == 1) in green
    draw_params = dict(matchColor=(0, 255, 0),  # Inlier match lines in green
                       singlePointColor=None,
                       matchesMask=matchesMask,  # Draw only RANSAC inliers
                       flags=2)

    # Draw filtered matches on the image
    ransac_sonrasi_resim = cv2.drawMatches(sol_resim, kp1, sag_resim, kp2, good_matches, None, **draw_params)
    
    # Save the filtered match visualization
    if ransac_cikti_yolu:
        cv2.imwrite(ransac_cikti_yolu, ransac_sonrasi_resim)
    
    saglam_nokta_sayisi = np.sum(mask)
    print(f"RANSAC {len(good_matches)} noktadan {saglam_nokta_sayisi} tanesini kusursuz buldu. Temizlenmiş resim kaydedildi!")

    return H, mask
