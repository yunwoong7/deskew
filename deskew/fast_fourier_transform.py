import cv2
import numpy as np


def get_angle_fft(img, angle_max=360, angle_min=0):
    """
    image : np.ndarray
    vertical_image_shape : int
    angle_max : float
    """
    stat_end_point = 10

    if len(img.shape) < 3:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    nw = nh = cv2.getOptimalDFTSize(max(gray.shape[:2]))
    opt_gray = cv2.copyMakeBorder(src=gray, top=0, bottom=nh - gray.shape[0], left=0, right=nw - gray.shape[1],
                                  borderType=cv2.BORDER_CONSTANT, value=255, )

    # thresh
    opt_gray = cv2.adaptiveThreshold(~opt_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)

    # perform fft
    dft = np.fft.fft2(opt_gray)
    shifted_dft = np.fft.fftshift(dft)

    # get the magnitude (module)
    magnitude = np.abs(shifted_dft)

    r = c = magnitude.shape[0] // 2

    num = 20

    tr = np.linspace(-1 * stat_end_point, stat_end_point, int(stat_end_point * num * 2)) / 180 * np.pi
    profile_arr = tr.copy()

    def f(t):
        _f = np.vectorize(lambda x: magnitude[c + int(x * np.cos(t)), c + int(-1 * x * np.sin(t))])
        _l = _f(range(0, r))
        val_init = np.sum(_l)
        return val_init

    vf = np.vectorize(f)
    li = vf(profile_arr)

    angle = tr[np.argmax(li)] / np.pi * 180

    if abs(angle) > angle_max or abs(angle) < angle_min:
        angle = 0

    return angle
