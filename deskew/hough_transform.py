import numpy as np
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks


def _get_max_freq_elem(peaks):
    freqs = {}
    for peak in peaks:
        if peak in freqs:
            freqs[peak] += 1
        else:
            freqs[peak] = 1

    sorted_keys = sorted(freqs.keys(), key=freqs.get, reverse=True)  # type: ignore
    max_freq = freqs[sorted_keys[0]]

    max_arr = []
    for sorted_key in sorted_keys:
        if freqs[sorted_key] == max_freq:
            max_arr.append(sorted_key)

    return max_arr


def _compare_sum(value):
    return 44 <= value <= 46


def _calculate_deviation(angle):
    angle_in_degrees = np.abs(angle)
    deviation: np.float64 = np.abs(np.pi / 4 - angle_in_degrees)

    return deviation


def get_angle_hough_transform(img, sigma=3.0, num_peaks=20, num_angles=180, angle_pm_90=False):
    """
    Hough Transform 기반의 비대칭도(Skewness) 감지
    :param img: (numpy array)
    :param sigma: (float)
    :param num_peaks: (int)
    :param num_angles: (int)
    :param angle_pm_90: (bool)
    :return: (float)
    """
    imagergb = rgba2rgb(img) if len(img.shape) == 3 and img.shape[2] == 4 else img
    img = rgb2gray(imagergb) if len(imagergb.shape) == 3 else imagergb

    edges = canny(img, sigma=sigma)

    out, theta, distances = hough_line(edges, np.linspace(-np.pi / 2, np.pi / 2, num_angles, endpoint=False))
    hough_line_out = (out, theta, distances)

    _, angles_peaks, _ = hough_line_peaks(out, theta, distances, num_peaks=num_peaks, threshold=0.05 * np.max(out))

    absolute_deviations = [_calculate_deviation(k) for k in angles_peaks]
    average_deviation: np.float64 = np.mean(np.rad2deg(absolute_deviations))
    angles_peaks_degree = [np.rad2deg(x) for x in angles_peaks]

    bin_0_45 = []
    bin_45_90 = []
    bin_0_45n = []
    bin_45_90n = []

    for angle in angles_peaks_degree:
        deviation_sum = int(90 - angle + average_deviation)
        if _compare_sum(deviation_sum):
            bin_45_90.append(angle)
            continue

        deviation_sum = int(angle + average_deviation)
        if _compare_sum(deviation_sum):
            bin_0_45.append(angle)
            continue

        deviation_sum = int(-angle + average_deviation)
        if _compare_sum(deviation_sum):
            bin_0_45n.append(angle)
            continue

        deviation_sum = int(90 + angle + average_deviation)
        if _compare_sum(deviation_sum):
            bin_45_90n.append(angle)

    angles = [bin_0_45, bin_45_90, bin_0_45n, bin_45_90n]
    nb_angles_max = 0
    max_angle_index = -1

    for angle_index, angle in enumerate(angles):
        nb_angles = len(angle)
        if nb_angles > nb_angles_max:
            nb_angles_max = nb_angles
            max_angle_index = angle_index

    if nb_angles_max:
        ans_arr = _get_max_freq_elem(angles[max_angle_index])
        angle = np.mean(ans_arr)
    elif angles_peaks_degree:
        ans_arr = _get_max_freq_elem(angles_peaks_degree)
        angle = np.mean(ans_arr)
    else:
        return None, angles, average_deviation, hough_line_out

    if not angle_pm_90:
        rot_angle = (angle + 45) % 90 - 45
    else:
        rot_angle = (angle + 90) % 180 - 90

    return rot_angle
