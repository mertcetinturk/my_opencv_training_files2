import cv2
import numpy as np
from collections import deque

cap = cv2.VideoCapture(0)

kernel = np.array((5, 5), np.uint8)

lower_blue = np.array([100, 60, 60], np.uint8)
upper_blue = np.array([140, 255, 255], np.uint8)

blue_points = [deque(maxlen=512)]
green_points = [deque(maxlen=512)]
red_points = [deque(maxlen=512)]
yellow_points = [deque(maxlen=512)]

blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
black_and_white = [(0, 0, 0), (255, 255, 255)]

color_index = 0

paint_window = np.zeros((471, 636, 3)) + 255
paint_window = cv2.rectangle(paint_window, (40, 1), (140, 65), black_and_white[0], 2)
paint_window = cv2.rectangle(paint_window, (160, 1), (255, 65), color[0], -1)
paint_window = cv2.rectangle(paint_window, (275, 1), (370, 65), color[1], -1)
paint_window = cv2.rectangle(paint_window, (390, 1), (485, 65), color[2], -1)
paint_window = cv2.rectangle(paint_window, (505, 1), (600, 65), color[3], -1)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(paint_window, 'CLEAR All', (49, 33), font, 0.5, black_and_white[0], 2, cv2.LINE_AA)
cv2.putText(paint_window, 'BLUE', (185, 33), font, 0.5, black_and_white[1], 2, cv2.LINE_AA)
cv2.putText(paint_window, 'GREEN', (298, 33), font, 0.5, black_and_white[1], 2, cv2.LINE_AA)
cv2.putText(paint_window, 'RED', (420, 33), font, 0.5, black_and_white[1], 2, cv2.LINE_AA)
cv2.putText(paint_window, 'YELLOW', (520, 33), font, 0.5, black_and_white[1], 2, cv2.LINE_AA)
# paint_window ve putText fonksiyonlarında yazdığımız bütün koordinatlar deneme yanılma şekliyle bulunmuştur.

cv2.namedWindow('Paint')

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    frame = cv2.rectangle(frame, (40, 1), (140, 65), black_and_white[0], 2)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), color[0], -1)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), color[1], -1)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), color[2], -1)
    frame = cv2.rectangle(frame, (505, 1), (600, 65), color[3], -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'CLEAR All', (55, 35), font, 0.5, black_and_white[0], 2, cv2.LINE_AA)
    cv2.putText(frame, 'BLUE', (185, 33), font, 0.5, black_and_white[1], 2, cv2.LINE_AA)
    cv2.putText(frame, 'GREEN', (298, 33), font, 0.5, black_and_white[1], 2, cv2.LINE_AA)
    cv2.putText(frame, 'RED', (420, 33), font, 0.5, black_and_white[1], 2, cv2.LINE_AA)
    cv2.putText(frame, 'YELLOW', (520, 33), font, 0.5, black_and_white[1], 2, cv2.LINE_AA)

    if ret is False:
        break

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if len(contours) > 0:
        max_contours = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        ((x, y), radius) = cv2.minEnclosingCircle(max_contours)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(frame, center, radius, (255, 0, 255), 3)

        M = cv2.moments(max_contours)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))  # x ve y değerleri

        if center[1] <= 65:
            if 40 < center[0] < 140:
                blue_points = [deque(maxlen=512)]
                green_points = [deque(maxlen=512)]
                red_points = [deque(maxlen=512)]
                yellow_points = [deque(maxlen=512)]

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                paint_window[67:, :, :] = 255

            elif 160 < center[0] < 255:
                color_index = 0

            elif 275 < center[0] < 370:
                color_index = 1

            elif 390 < center[0] < 485:
                color_index = 2

            elif 505 < center[0] < 600:
                color_index = 3

        else:
            if color_index == 0:
                blue_points[blue_index].appendleft(center)

            elif color_index == 1:
                green_points[green_index].appendleft(center)

            elif color_index == 2:
                red_points[red_index].appendleft(center)

            elif color_index == 3:
                yellow_points[yellow_index].appendleft(center)

    else:
        blue_points.append(deque(maxlen=512))
        blue_index += 1

        green_points.append(deque(maxlen=512))
        green_index += 1

        red_points.append(deque(maxlen=512))
        red_index += 1

        yellow_points.append(deque(maxlen=512))
        yellow_index += 1

    points = [blue_points, green_points, red_points, yellow_points]

    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                # 1 den itibaren dememizin sebebi 0 dediğimizde bir nokta sabit kalıyor.

                if points[i][j][k-1] is None or points[i][j][k] is None:
                    continue

                cv2.line(frame, points[i][j][k-1], points[i][j][k], color[i], 2)
                cv2.line(paint_window, points[i][j][k - 1], points[i][j][k], color[i], 2)

    cv2.imshow('frame', frame)
    cv2.imshow('paint window', paint_window)
    if cv2.waitKey(20) == 27:
        break

cap.release()
cv2.destroyAllWindows()


# Bir ROI alanı belirlenerek çizimlerin belli bir yerin içerisinde yapılmasını sağlayabiliriz.
# Bu sayede de görüntünün altında oluşacak bozulmaları engelleyebiliriz.
