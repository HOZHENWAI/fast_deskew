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
    min_deviation: float = 1.0,
    gradient_percentile: float = 99.0,
    canny_threshold_low: Optional[float] = None,
    canny_threshold_high: Optional[float] = None,
    hough_rho: float = 1.0,
    hough_threshold: int = 30,
) -> Optional[np.float64]:
    """Estimate the skew angle (in degrees) of a (grayscale) document image.

    The pipeline follows the classical Hough-transform skew-detection method:
    blur -> Canny edges -> Hough line transform -> fold each line orientation
    into the canonical skew interval -> return the most frequent (modal) angle.
    The rho (offset) dimension carries no skew information and is ignored.

    Thresholds are derived relative to the image (rather than fixed constants) so
    the same defaults work across document sizes and contrasts: the Canny
    thresholds come from a percentile of the gradient magnitude, and the dominant
    orientation is taken from the strongest ``num_peaks`` Hough lines.

    :param image_array: 8-bit single-channel image.
    :param sigma: standard deviation of the Gaussian pre-blur.
    :param num_peaks: number of strongest Hough lines to vote with.
    :param angle_pm_90: fold into [-90, 90) instead of [-45, 45).
    :param min_angle: lower bound (degrees) on the reported skew; None disables it.
    :param max_angle: upper bound (degrees) on the reported skew; None disables it.
    :param min_deviation: strictly positive angular resolution in degrees; also the
        voting bin width used to find the modal angle.
    :param gradient_percentile: percentile of the gradient magnitude used as the
        high Canny threshold when the Canny thresholds are auto-derived.
    :param canny_threshold_low: explicit low Canny threshold; None auto-derives it.
    :param canny_threshold_high: explicit high Canny threshold; None auto-derives it.
    :param hough_rho: distance resolution of the accumulator in pixels.
    :param hough_threshold: minimum Hough votes for a line to be considered. It acts
        as a floor only; the dominant orientation is chosen from the strongest
        ``num_peaks`` lines, so peak selection stays relative to the image.
    :return: the skew angle in degrees, or None if no dominant line is found.
    """
    if min_deviation <= 0:
        raise ValueError("min_deviation must be strictly positive")

    theta_resolution = np.deg2rad(min_deviation)
    blurred_image = cv2.GaussianBlur(image_array, (0, 0), sigma)

    if canny_threshold_low is None or canny_threshold_high is None:
        # Derive Canny thresholds from this image's gradient magnitude instead of
        # using fixed constants, so faint or heavily blurred documents still yield
        # edges (mirrors scikit-image's gradient-relative Canny).
        grad_x = cv2.Sobel(blurred_image, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred_image, cv2.CV_16S, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x.astype(np.float32), grad_y.astype(np.float32))
        high = max(float(np.percentile(magnitude, gradient_percentile)), 1.0)
        edges = cv2.Canny(grad_x, grad_y, 0.5 * high, high)
    else:
        edges = cv2.Canny(blurred_image, canny_threshold_low, canny_threshold_high)

    lines = cv2.HoughLines(edges, hough_rho, theta_resolution, hough_threshold)
    if lines is None:
        return None

    # cv2.HoughLines returns lines ordered by descending vote count, so the first
    # `num_peaks` are the strongest peaks. theta is the orientation of each line's
    # normal in [0, pi).
    theta = lines[:num_peaks, 0, 1]

    # Fold every orientation into the canonical skew interval BEFORE counting, so
    # that text baselines and vertical strokes (which differ by ~90 deg) reinforce
    # the same skew estimate.
    if angle_pm_90:
        folded = theta % np.pi - np.pi / 2  # [-pi/2, pi/2)
    else:
        folded = (theta + np.pi / 4) % (np.pi / 2) - np.pi / 4  # [-pi/4, pi/4)

    folded_deg = np.rad2deg(folded)

    # Restrict the reported skew to the requested output range, if any.
    if min_angle is not None:
        folded_deg = folded_deg[folded_deg >= min_angle]
    if max_angle is not None:
        folded_deg = folded_deg[folded_deg <= max_angle]
    if folded_deg.size == 0:
        return None

    # Most frequent angle: quantise to the min_deviation grid to group nearby
    # peaks, pick the densest bin, then average its members for sub-bin precision.
    bins = np.round(folded_deg / min_deviation)
    values, counts = np.unique(bins, return_counts=True)
    best_bin = values[np.argmax(counts)]
    best_angle = folded_deg[bins == best_bin].mean()

    return np.float64(best_angle)
