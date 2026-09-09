import cv2

def match_features(kp1, des1, kp2, des2):
    # 1. FLANN parameters (standard settings for SIFT)
    # FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)

    # 2. Create the matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 3. k-nearest neighbor matching (k=2)
    # For each point, find the two best matches
    matches = flann.knnMatch(des1, des2, k=2)

    # 4. Lowe's ratio test (quality filter)
    # Keep a match only if the best distance is at least 30% smaller than the second best
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"Toplam {len(matches)} eşleşmeden {len(good_matches)} tanesi kaliteli bulundu.")
    return good_matches