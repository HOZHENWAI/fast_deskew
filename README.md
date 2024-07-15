# Fast-deskew
An alternative to the excellent deskew library by [sbrunner](https://github.com/sbrunner/deskew) using opencv for faster computation.

# Installation
- Using pip:
```
pip install fast-deskew-cv
```
- From source:
```
pip install  git+https://git@github.com/HOZHENWAI/fast_deskew.git
```

# Usage
```
import cv2
from deskew import determine_skew

image_grayscale = cv2.imread('input_image.png', cv2.IMREAD_GRAYSCALE)
skew_angle = determine_skew(image_grayscale)
```

# Note
I do not recommend you to use this library directly when dealing with angle or more than 90 but to use it after fix the global rotation.
