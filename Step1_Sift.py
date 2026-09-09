import cv2
import os

import matcher  # import matching helpers from matcher.py
import homografi
import birlestirme

# Avoid blocking in terminal/CI: HEADLESS=1 python3 Step1_Sift.py
HEADLESS = os.environ.get("HEADLESS", "").lower() in ("1", "true", "yes")

# Three image sets under images/: each folder has left + right source photos
GORUNTU_SETLERI = [
    ("images/Clock", "sol1.jpg", "sag1.jpg"),
    ("images/SchoolImage", "sol2.jpg", "sag2.jpg"),
    ("images/test1", "s1.jpg", "s2.jpg"),
]


def detect_features(image_path):
    # 1. Load the image from disk
    # cv2.imread returns None if the file is missing
    img = cv2.imread(image_path)

    if img is None:
        print(f"HATA: {image_path} bulunamadı! Dosya adını kontrol et.")
        return None, None, None

    # 2. Convert to grayscale
    # Feature detectors work on single-channel images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Create SIFT detector
    sift = cv2.SIFT_create()

    # 4. Detect keypoints and compute descriptors
    # kp: coordinates of salient points (e.g. x=150, y=200)
    # des: descriptor vectors used for matching
    kp, des = sift.detectAndCompute(gray, None)

    # 5. Draw keypoints on the image for visualization
    # DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS also draws scale and orientation
    img_with_keypoints = cv2.drawKeypoints(gray, kp, img.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    print(f"{image_path} içinde {len(kp)} adet nokta bulundu!")

    return kp, des, img_with_keypoints


def _bir_klasor_isle(klasor, sol_ad, sag_ad, gorsel_goster):
    """Run SIFT → matching → homography → panorama for one image pair."""
    sol_resim_yolu = os.path.join(klasor, sol_ad)
    sag_resim_yolu = os.path.join(klasor, sag_ad)

    print(f"\n=== {klasor} ({sol_ad} + {sag_ad}) ===")

    kp1, des1, sol_cizimli = detect_features(sol_resim_yolu)
    kp2, des2, sag_cizimli = detect_features(sag_resim_yolu)

    if sol_cizimli is None or sag_cizimli is None:
        print("Bu set atlandı (okuma hatası).")
        return False

    cv2.imwrite(os.path.join(klasor, "sol_noktalar.jpg"), sol_cizimli)
    cv2.imwrite(os.path.join(klasor, "sag_noktalar.jpg"), sag_cizimli)

    if gorsel_goster:
        cv2.imshow("Sol Resim Noktalari", sol_cizimli)
        cv2.imshow("Sag Resim Noktalari", sag_cizimli)

    good_matches = matcher.match_features(kp1, des1, kp2, des2)

    sol_bgr_eslesme = cv2.imread(sol_resim_yolu)
    sag_bgr_eslesme = cv2.imread(sag_resim_yolu)
    img_matches = cv2.drawMatches(
        sol_bgr_eslesme,
        kp1,
        sag_bgr_eslesme,
        kp2,
        good_matches[:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    if gorsel_goster:
        cv2.imshow("Eslesme Sonuclari", img_matches)
    cv2.imwrite(os.path.join(klasor, "eslesme_final.jpg"), img_matches)

    print("4. Aşama (Homografi ve RANSAC) hesaplanıyor...")
    H_matrisi, maske = homografi.donusum_matrisi_hesapla(
        kp1, kp2, good_matches, sol_bgr_eslesme, sag_bgr_eslesme
    )
    print("4. Aşama da tamam! Dönüşüm Matrisi (Homografi) başarıyla hesaplandı.")

    if H_matrisi is not None:
        print("5. Aşama (Birleştirme / Warping & Stitching) çalışıyor...")
        sol_bgr = cv2.imread(sol_resim_yolu)
        sag_bgr = cv2.imread(sag_resim_yolu)
        if sol_bgr is None or sag_bgr is None:
            print("HATA: Birleştirme için renkli görüntüler okunamadı.")
        else:
            panorama = birlestirme.panorama_birlestir(sol_bgr, sag_bgr, H_matrisi)
            if panorama is not None:
                cikti_yolu = os.path.join(klasor, "panorama_birlestirme.jpg")
                cv2.imwrite(cikti_yolu, panorama)
                print(f"5. Aşama tamam! Panorama kaydedildi: {cikti_yolu}")
                if gorsel_goster:
                    cv2.imshow("Panorama (5. Asama)", panorama)

    return True


# --- Main entry point ---
if __name__ == "__main__":
    for idx, (klasor, sol_ad, sag_ad) in enumerate(GORUNTU_SETLERI):
        # Process all three sets; show GUI only for the last set to avoid window clutter
        son_set = idx == len(GORUNTU_SETLERI) - 1
        _bir_klasor_isle(klasor, sol_ad, sag_ad, gorsel_goster=son_set and not HEADLESS)

    if HEADLESS:
        print("\nİşlem tamam! Üç görüntü seti işlendi (HEADLESS, GUI yok).")
    else:
        print("\nİşlem tamam! Üç görüntü seti işlendi. Çıkmak için herhangi bir tuşa bas.")
        cv2.waitKey(0)
    cv2.destroyAllWindows()
