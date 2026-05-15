import cv2

def match_features(kp1, des1, kp2, des2):
    # 1. FLANN Parametrelerini Ayarla (SIFT için standart ayarlar)
    # FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)

    # 2. Eşleştiriciyi (Matcher) Tanımla
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 3. k-En Yakın Komşu (k-Nearest Neighbors) Eşleştirmesi Yap
    # Her nokta için en iyi 2 eşleşmeyi (k=2) buluruz.
    matches = flann.knnMatch(des1, des2, k=2)

    # 4. Lowe's Ratio Test (Kalite Filtresi)
    # Eğer en iyi eşleşme, ikinci en iyiden %70 daha yakınsa o nokta "gerçekten" eşleşmiştir.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"Toplam {len(matches)} eşleşmeden {len(good_matches)} tanesi kaliteli bulundu.")
    return good_matches