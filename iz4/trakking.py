import cv2
import numpy as np

# Загрузка видео
video = cv2.VideoCapture("C:/Users/vlad5/PycharmProjects/iz4/1.mp4")
z = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
o = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
status = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter("output.mp4", fourcc, 25, (z, o))

# Чтение первого кадра видео
ret, frame = video.read()

# Выбор области с объектом для отслеживания
bbox = cv2.selectROI(frame, False)

# Определение гистограммы цветов объекта
roi = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

# Определение критериев сравнения гистограммы
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    # Чтение кадра видео
    ret, frame = video.read()

    # Прекращение выполнения, если видео закончилось
    if not ret:
        break

    # Преобразование текущего кадра в HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Вычисление обратного проектирования
    dst = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)

    # Применение среднего сдвига для трекинга
    ret, bbox = cv2.meanShift(dst, bbox, criteria)

    # Отрисовка прямоугольника вокруг объекта
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Отображение кадра
    cv2.imshow('Tracking', frame)
    status = 1
    video_writer.write(frame)
    # Прекращение выполнения при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
video.release()
cv2.destroyAllWindows()