from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray


def determine_skew(
        image_array: NDArray[np.uint8],
        sigma: float = 3.0,
        num_peaks: int = 20,
        angle_pm_90: bool = False,
        min_angle: Optional[float] = None,
        max_angle: Optional[float] = None,
        min_deviation: Optional[float] = 1.0,
        canny_threshold_low: Optional[float] = 100,
        canny_threshold_high: Optional[float] = 200,
        hough_rho: Optional[int] = 1,
        hough_threshold: Optional[int] = 500,
) -> Optional[np.float64]:
    """

    :param image_array:
    :param sigma:
    :param num_peaks:
    :param angle_pm_90:
    :param min_angle:
    :param max_angle: the max supported
    :param min_deviation: strictly positive minimum angle tolerance
    :param canny_threshold_low: the low threshold for canny edge detection
    :param canny_threshold_high: the high threshold for canny edge detection
    :param hough_threshold: the hough threshold for hough line transform
    :return:
    """
    # Convert float angles to radian
    min_deviation_rad = np.deg2rad(min_deviation)
    min_angle_rad = np.deg2rad(min_angle)
    max_angle_rad = np.deg2rad(max_angle)
    num_angles = round(np.pi / min_deviation_rad)

    blurred_image = cv2.GaussianBlur(image_array, (0, 0), sigma)
    edges = cv2.Canny(blurred_image,canny_threshold_low,canny_threshold_high)
    lines = cv2.HoughLines(
        edges,
        hough_rho,
        threshold=hough_threshold,
        min_theta=min_angle_rad,
        max_theta=max_angle_rad
    )
    if lines is None:
        return None

    rho = lines[:, 0, 0]
    theta = lines[:, 0, 1]
    angles_space = np.linspace(min_angle_rad, max_angle_rad, num=num_angles, endpoint=False)
    distances_space = np.linspace(-image_array.shape[1], image_array.shape[1], lines.shape[0])
    # fill the hough space before computing the peak
    rho_distance = np.abs(rho[:, np.newaxis] - distances_space[np.newaxis, :])
    rho_index = np.argmin(rho_distance, axis=1)
    angle_distance = np.abs(theta[:, np.newaxis] - angles_space[np.newaxis, :])
    angle_index = np.argmin(angle_distance, axis=1)
    # pair the index before counting
    concatenated_index = np.vstack((angle_index, rho_index))
    # count
    pair_index, count = np.unique(concatenated_index, axis=1, return_counts=True)
    kept_peak = count > (max(count) * 0.05)
    sort_index = np.argsort(-count[kept_peak])
    index_select = pair_index[:, kept_peak][:, sort_index]
    kept_index = index_select[:, :num_peaks]
    # finally, get the peaks
    peaks = theta[np.logical_and(np.isin(angle_index, kept_index[0, :]), np.isin(rho_index, kept_index[1, :]))]
    angles, count = np.unique(peaks, return_counts=True)
    best_angle_by_count = angles[np.argmax(count)]
    angle = (best_angle_by_count % np.pi - np.pi / 2) if angle_pm_90 else (
                (best_angle_by_count + np.pi / 4) % (np.pi / 2) - np.pi / 4)

    return None if angle is None else np.rad2deg(angle)
