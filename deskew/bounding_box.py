import cv2
import numpy as np


def get_angle_bbox(img, angle_max=360, angle_min=0):
    angle = 0

    # 이미지를 GrayScale로 변환하고 전경을 뒤집어 글짜는 흰색, 배경은 검정으로 변환합니다.
    if len(img.shape) < 3:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.bitwise_not(gray)

    # 이미지 임계처리를 통해서 이진화 (배경은 0, 글씨는 255)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    points = []

    for h, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area >= 90:
            for p in cnt:
                points.append(p[0])

    points = np.array(points)

    # `cv2.minAreaRect`함수는 [-90, 0] 범위의 각도값을 반환합니다.
    rect = cv2.minAreaRect(points)
    angle = rect[-1]

    # print(angle)
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = -angle

        if angle < -45:
            angle = -(90 + angle)

    if abs(angle) > angle_max or abs(angle) < angle_min:
        angle = 0

    return angle
