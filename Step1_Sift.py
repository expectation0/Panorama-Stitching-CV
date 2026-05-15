import cv2
import os
import matcher  # Integration with Onur's matcher module
import homografi  # Integration with the homography module
import birlestirme  # Integration with the stitching/blending module

# Environment check for headless systems (GUI-less)
HEADLESS = os.environ.get("HEADLESS", "").lower() in ("1", "true", "yes")

# Dataset definitions: (Folder, Left Image, Right Image)
# DO NOT CHANGE these names as they match the project directory structure
GORUNTU_SETLERI = [
    ("images/Clock", "sol1.jpg", "sag1.jpg"),
    ("images/SchoolImage", "sol2.jpg", "sag2.jpg"),
    ("images/test1", "s1.jpg", "s2.jpg"),
]


def detect_features(image_path):
    """
    Reads an image and extracts SIFT keypoints and descriptors.
    Developed by: Ümit Sevil
    """
    # 1. Load the image from the disk
    img = cv2.imread(image_path)

    if img is None:
        print(f"ERROR: {image_path} not found! Check the filename.")
        return None, None, None

    # 2. Convert to grayscale for better feature detection performance
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Create SIFT detector object
    sift = cv2.SIFT_create()

    # 4. Detect keypoints (kp) and compute descriptors (des)
    # kp: spatial location, des: local texture identity
    kp, des = sift.detectAndCompute(gray, None)

    # 5. Draw detected features on the image for visualization purposes
    img_with_keypoints = cv2.drawKeypoints(gray, kp, img.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    print(f"{image_path}: Found {len(kp)} keypoints.")

    return kp, des, img_with_keypoints


def _bir_klasor_isle(klasor, sol_ad, sag_ad, gorsel_goster):
    """
    Executes the full CV pipeline for a single image set:
    SIFT -> Matching -> Homography -> Warping -> Panorama
    """
    sol_resim_yolu = os.path.join(klasor, sol_ad)
    sag_resim_yolu = os.path.join(klasor, sag_ad)

    print(f"\n=== Processing: {klasor} ({sol_ad} + {sag_ad}) ===")

    # Step 1: Feature Detection (SIFT)
    kp1, des1, sol_cizimli = detect_features(sol_resim_yolu)
    kp2, des2, sag_cizimli = detect_features(sag_resim_yolu)

    if sol_cizimli is None or sag_cizimli is None:
        print("Set skipped due to reading error.")
        return False

    # Save visualization results to the folder
    cv2.imwrite(os.path.join(klasor, "sol_noktalar.jpg"), sol_cizimli)
    cv2.imwrite(os.path.join(klasor, "sag_noktalar.jpg"), sag_cizimli)

    if gorsel_goster:
        cv2.imshow("Left Image Keypoints", sol_cizimli)
        cv2.imshow("Right Image Keypoints", sag_cizimli)

    # Step 2 & 3: Feature Matching (Integration with matcher.py)
    good_matches = matcher.match_features(kp1, des1, kp2, des2)

    sol_bgr_eslesme = cv2.imread(sol_resim_yolu)
    sag_bgr_eslesme = cv2.imread(sag_resim_yolu)

    # Draw matches between the two images
    img_matches = cv2.drawMatches(
        sol_bgr_eslesme, kp1, sag_bgr_eslesme, kp2, good_matches[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    if gorsel_goster:
        cv2.imshow("Matching Results", img_matches)
    cv2.imwrite(os.path.join(klasor, "eslesme_final.jpg"), img_matches)

    # Step 4: Homography and RANSAC Calculation (Integration with homografi.py)
    print("Calculating Homography and RANSAC...")
    H_matrisi, maske = homografi.donusum_matrisi_hesapla(
        kp1, kp2, good_matches, sol_bgr_eslesme, sag_bgr_eslesme
    )

    if H_matrisi is not None:
        print("Homography Matrix successfully computed.")

        # Step 5: Final Warping and Stitching (Integration with birlestirme.py)
        print("Executing Warping & Stitching...")
        sol_bgr = cv2.imread(sol_resim_yolu)
        sag_bgr = cv2.imread(sag_resim_yolu)

        if sol_bgr is not None and sag_bgr is not None:
            panorama = birlestirme.panorama_birlestir(sol_bgr, sag_bgr, H_matrisi)
            if panorama is not None:
                cikti_yolu = os.path.join(klasor, "panorama_birlestirme.jpg")
                cv2.imwrite(cikti_yolu, panorama)
                print(f"Panorama saved successfully to: {cikti_yolu}")
                if gorsel_goster:
                    cv2.imshow("Final Panorama", panorama)

    return True


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # Loop through each defined image set
    for idx, (klasor, sol_ad, sag_ad) in enumerate(GORUNTU_SETLERI):
        # Only show GUI for the final set to keep the screen clean, unless Headless mode is active
        is_last_set = idx == len(GORUNTU_SETLERI) - 1
        _bir_klasor_isle(klasor, sol_ad, sag_ad, gorsel_goster=is_last_set and not HEADLESS)

    if HEADLESS:
        print("\nProcess finished! All image sets processed (Headless Mode).")
    else:
        print("\nProcess finished! All image sets processed. Press any key to close windows.")
        cv2.waitKey(0)

    cv2.destroyAllWindows()