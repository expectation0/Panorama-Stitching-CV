import cv2
import numpy as np


def donusum_matrisi_hesapla(kp1, kp2, good_matches, sol_resim, sag_resim):
    # 1. Safety Lock: Minimum match count required for a reliable result
    MIN_MATCH_COUNT = 10

    if len(good_matches) < MIN_MATCH_COUNT:
        print(f"ERROR: Insufficient matches found! Required: {MIN_MATCH_COUNT}, Found: {len(good_matches)}")
        return None, None  # Return None to prevent system crash

    # Extract coordinates of the matched keypoints
    sol_noktalar = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    sag_noktalar = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute Homography matrix using the RANSAC algorithm
    H, mask = cv2.findHomography(sol_noktalar, sag_noktalar, cv2.RANSAC, 5.0)

    # 2. Visualization: Drawing only the robust inliers identified by RANSAC
    # The mask contains 1 for inliers (strong matches) and 0 for outliers (weak matches)
    matchesMask = mask.ravel().tolist()

    # Configuration for drawing inliers in green color
    draw_params = dict(matchColor=(0, 255, 0),  # Green lines for valid matches
                       singlePointColor=None,
                       matchesMask=matchesMask,  # Display only points passing the RANSAC test
                       flags=2)

    # Draw the filtered matches on the visualization image
    ransac_sonrasi_resim = cv2.drawMatches(sol_resim, kp1, sag_resim, kp2, good_matches, None, **draw_params)

    # Save the processed image and log results
    cv2.imwrite("ransac_filtered_matches.jpg", ransac_sonrasi_resim)

    inlier_count = np.sum(mask)
    print(f"RANSAC identified {inlier_count} robust points out of {len(good_matches)}. Filtered image saved.")

    return H, mask