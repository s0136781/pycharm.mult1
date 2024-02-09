import cv2

# Загрузка готовой модели для обнаружения лиц
face_cascade = cv2.CascadeClassifier("C:/Users/vlad5/PycharmProjects/iz4/haarcascade_frontalface_default.xml")

# Открытие видеофайла
video_capture = cv2.VideoCapture("C:/Users/vlad5/PycharmProjects/iz4/1.mp4")
z = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
o = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
status = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter("output.mp4", fourcc, 25, (z, o))

while True:
    # Чтение кадра из видеопотока
    ret, frame = video_capture.read()

    if not ret:
        break

    # Преобразование кадра в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц на кадре
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Отображение результатов обнаружения на кадре
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Отображение кадра с результатами
    cv2.imshow("Output", frame)
    status = 1
    video_writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
video_capture.release()
cv2.destroyAllWindows()