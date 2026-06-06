import cv2
import numpy as np
import pytest

from deskew import determine_skew


def _document_page(size: int = 600) -> np.ndarray:
    """White page with rows of black word-like blocks (skew 0), mimicking text.

    Solid blocks survive the default Gaussian pre-blur and produce strong,
    consistent horizontal edges for the Hough transform to lock onto.
    """
    img = np.full((size, size), 255, dtype=np.uint8)
    for y in range(40, size - 40, 28):
        x = 40
        for width in (60, 40, 90, 30, 70, 50, 80):
            if x + width > size - 40:
                break
            cv2.rectangle(img, (x, y), (x + width, y + 12), 0, -1)
            x += width + 18
    return img


def _rotate(img: np.ndarray, degrees: float) -> np.ndarray:
    height, width = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    return cv2.warpAffine(img, matrix, (width, height), borderValue=255)


def test_default_call_runs_and_returns_float():
    # Regression for the README usage `determine_skew(img)`, which used to crash
    # on np.deg2rad(None) and on cv2.HoughLines missing its `theta` argument.
    angle = determine_skew(_document_page())
    assert angle is not None
    assert isinstance(angle, (float, np.floating))


def test_unrotated_page_has_near_zero_skew():
    assert determine_skew(_document_page()) == pytest.approx(0.0, abs=1.0)


@pytest.mark.parametrize("applied", [5.0, 10.0, -7.0, -12.0])
def test_recovers_known_skew_magnitude(applied):
    angle = determine_skew(_rotate(_document_page(), applied))
    assert angle is not None
    assert abs(angle) == pytest.approx(abs(applied), abs=1.5)


def test_sign_is_responsive_to_rotation_direction():
    pos = determine_skew(_rotate(_document_page(), 8.0))
    neg = determine_skew(_rotate(_document_page(), -8.0))
    assert pos is not None and neg is not None
    assert np.sign(pos) == -np.sign(neg)


def test_angle_pm_90_agrees_for_small_skew():
    # For small skews the +/-90 folding and the default +/-45 folding coincide.
    img = _rotate(_document_page(), 9.0)
    default = determine_skew(img)
    pm90 = determine_skew(img, angle_pm_90=True)
    assert pm90 is not None
    assert -90.0 <= pm90 < 90.0
    assert pm90 == pytest.approx(default, abs=0.5)


def test_explicit_canny_thresholds_are_used():
    # Passing both Canny thresholds bypasses the gradient-percentile auto path.
    angle = determine_skew(
        _rotate(_document_page(), 6.0), canny_threshold_low=100, canny_threshold_high=200
    )
    assert angle is not None
    assert abs(angle) == pytest.approx(6.0, abs=1.5)


def test_blank_image_returns_none():
    blank = np.full((300, 300), 255, dtype=np.uint8)
    assert determine_skew(blank) is None


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_min_deviation_raises(bad):
    with pytest.raises(ValueError):
        determine_skew(_document_page(), min_deviation=bad)


def test_output_angle_range_filter_is_respected():
    # A ~10 deg skew restricted to +/-2 deg must never be reported as 10 deg:
    # the out-of-range dominant angle is filtered, yielding None or an in-range value.
    result = determine_skew(_rotate(_document_page(), 10.0), min_angle=-2.0, max_angle=2.0)
    assert result is None or abs(result) <= 2.0
