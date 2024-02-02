import cv2
import numpy as np

#1
'''
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.imshow("HSV_image", hsv)
    cv2.imwrite("task_1.png", hsv)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
'''

#2
'''
video = cv2.VideoCapture("http://212.192.144.20:8080/video")


# определение диапазона красного цвета в HSV
lower_red = np.array([50, 150, 50])  # минимальные значения оттенка, насыщенности и яркости
upper_red = np.array([250, 250, 250]) # максимальные значения оттенка, насыщенности и яркости

while True:
    ret, frame = video.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if not ret:
        break
    mask = cv2.inRange(hsv, lower_red, upper_red)

    cv2.imshow("HSV with red", mask)
    if cv2.waitKey(1) & 0xFF == 27:
        break

        video.release()
        cv2.destroyAllWindows()



kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(hsv, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)


cv2.imshow('Opening', opening)
cv2.imshow('Closing', closing)
cv2.waitKey(0)
'''

#4-5

cap = cv2.VideoCapture(0)

# определение диапазона красного цвета в HSV
lower_red = np.array([0, 120, 100])  # минимальные значения оттенка, насыщенности и яркости
upper_red = np.array([60, 255, 255])  # максимальные значения оттенка, насыщенности и яркости
color = (0, 0, 0)
thickness = 4 # толщина

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if not ret:
        break
    mask = cv2.inRange(hsv, lower_red, upper_red)
    kernel = np.ones((5, 5), np.uint8)
    # Операция открытия - 1)эрозия, 2)дилатация.
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Операция закрытия-1)дилатации, 2) операция эрозии.
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    moments = cv2.moments(closing)
    m01 = moments['m01']
    m10 = moments['m10']
    area = moments['m00']

    if area > 10000:
        posX = int(m10 // area)
        posY = int(m01 // area)
        width = height = int(np.sqrt(area))
        # Создаем черную звезду
        star_points = []
        star_radius_outer = width // 20  # радиус внешней окружности
        star_radius_inner = width // 40  # радиус внутренней окружности
        star_angle = np.pi / 2  # начальный угол

        for i in range(10):
            if i % 2 == 0:
                x = posX + int(star_radius_outer * np.cos(star_angle))
                y = posY - int(star_radius_outer * np.sin(star_angle))
            else:
                x = posX + int(star_radius_inner * np.cos(star_angle))
                y = posY - int(star_radius_inner * np.sin(star_angle))
            star_points.append((x, y))
            star_angle += (2 * np.pi) / 10

        star_points = np.array(star_points, np.int32)
        star_points = star_points.reshape((-1, 1, 2))

        # Отображаем черную звезду на кадре
        cv2.fillPoly(frame, [star_points], (0, 0, 0))  # Черный цвет

        # Отображаем черную звезду на кадре
        cv2.fillPoly(frame, [star_points], (0, 0, 0))  # Черный цвет
    cv2.imshow('HSV_frame', mask)
    cv2.imshow('Result_frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


