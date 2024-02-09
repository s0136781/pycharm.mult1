import cv2
import numpy as np

# Загрузка готовой модели для обнаружения объектов
net = cv2.dnn.readNetFromCaffe('C:/Users/vlad5/PycharmProjects/iz4/deploy.prototxt', "C:/Users/vlad5/PycharmProjects/iz4/dnn_model.caffemodel")

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

    # Применение модели для обнаружения объектов на кадре
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Отображение результатов обнаружения на кадре
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Пороговое значение уверенности
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Отображение кадра с результатами
    cv2.imshow("Output", frame)
    status = 1
    video_writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
video_capture.release()
cv2.destroyAllWindows()