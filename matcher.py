import cv2

def match_features(kp1, des1, kp2, des2):
    # 1. Set FLANN parameters (Standard settings for SIFT)
    # FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)

    # 2. Define the Matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 3. Perform k-Nearest Neighbors (k-NN) matching
    # Find the top 2 best matches (k=2) for each keypoint
    matches = flann.knnMatch(des1, des2, k=2)

    # 4. Lowe's Ratio Test (Quality Filter)
    # If the closest match is significantly closer than the second best (70%),
    # it is considered a robust match.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"Found {len(good_matches)} high-quality matches out of {len(matches)} total matches.")
    return good_matches