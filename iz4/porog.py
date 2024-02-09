import cv2

# Загрузка видео
video = cv2.VideoCapture("C:/Users/vlad5/PycharmProjects/iz4/1.mp4")
z = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
o = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
status = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter("output.mp4", fourcc, 25, (z, o))

while True:
    # Получение кадра из видео
    ret, frame = video.read()

    if not ret:
        break

    # Преобразование кадра в цветовое пространство HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Определение диапазона цветов объекта (в данном примере: синий цвет)
    lower_blue = (90, 50, 50)
    upper_blue = (130, 255, 255)

    # Применение пороговой обработки для выделения объекта
    mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    # Применение морфологической операции для устранения шума
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Нахождение контуров объекта
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Отрисовка прямоугольника вокруг каждого обнаруженного объекта
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Отображение кадра с обнаруженными объектами
    cv2.imshow('Object Detection', frame)
    status = 1
    video_writer.write(frame)
    # Прерывание по нажатию клавиши 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Освобождение ресурсов
video.release()
cv2.destroyAllWindows()