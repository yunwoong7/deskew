import cv2

from .hough_transform import get_angle_hough_transform
from .fast_fourier_transform import get_angle_fft
from .bounding_box import get_angle_bbox


def rotate(img, angle, resize=True, border_mode=cv2.BORDER_REPLICATE, border_value=None):
    '''
    :param img: (numpy)
    :param angle: (float)
    :param resize: (bool)
    :param border_mode:
    :param border_value:
    :return: (numpy.array)
    '''
    if border_mode == cv2.BORDER_REPLICATE:
        border_value = None

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D(center=(w // 2, h // 2), angle=angle, scale=1.0)
    output_image = cv2.warpAffine(src=img, M=M, dsize=(w, h), flags=cv2.INTER_CUBIC, borderMode=border_mode,
                                  borderValue=border_value, )

    if resize is True:
        output_image = cv2.resize(output_image, (w, h))

    return output_image
