import cv2
import os
import time

def track_and_save_objects(video_path, tracker_type, initial_bbox=None):
    # Определите путь для сохранения обработанного видео
    output_path = f"{os.path.splitext(os.path.basename(video_path))[0]}_{tracker_type}.avi"

    # Откройте видеозахват для входного видео
    video_capture = cv2.VideoCapture(video_path)

    # Считайте первый кадр
    success, frame = video_capture.read()

    # Если не предоставлены начальные координаты, выберите объект для отслеживания
    if initial_bbox is None:
        initial_bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)

    # Выберите трекер в зависимости от переданного типа
    if tracker_type == 'CSRT':
        tracker = cv2.TrackerCSRT_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    else:
        print("Invalid tracker type")
        return

    # Инициализация трекера
    tracker.init(frame, initial_bbox)

    # Определите кодек и создайте объект VideoWriter для сохранения обработанного видео
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_path, fourcc, 20.0, (int(video_capture.get(3)), int(video_capture.get(4))))

    # Параметры для оценки качества трекинга
    total_frames = 0
    lost_frames = 0
    total_distance = 0

    start_time = time.time()  # Запомните время начала обработки видео

    # Цикл обработки кадров
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        total_frames += 1

        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Оценка точности отслеживания (пример, среднее расстояние)
            true_center = (initial_bbox[0] + initial_bbox[2] / 2, initial_bbox[1] + initial_bbox[3] / 2)
            tracked_center = (x + w / 2, y + h / 2)
            total_distance += ((true_center[0] - tracked_center[0]) ** 2 + (true_center[1] - tracked_center[1]) ** 2) ** 0.5

        else:
            lost_frames += 1

        # Рассчитать текущий FPS
        current_time = time.time()
        fps = total_frames / (current_time - start_time)

        # Отображение FPS на кадре
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Записать обработанный кадр в выходное видео
        output_video.write(frame)

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    # Оценка частоты потери и точности отслеживания
    tracking_loss_rate = lost_frames / total_frames
    tracking_accuracy = total_distance / total_frames

    print(f"Tracker: {tracker_type}")
    print(f"Tracking Loss Rate: {tracking_loss_rate}")
    print(f"Tracking Accuracy: {tracking_accuracy}")

    # Закрыть видеозахват, записать оставшуюся часть видео и освободить ресурсы
    video_capture.release()
    output_video.release()
    cv2.destroyAllWindows()

# Пример использования для одного видео и трех методов
video_path = '5.mp4'

# Выделите область только для первого метода
initial_bbox = cv2.selectROI("Select Object", cv2.VideoCapture(video_path).read()[1], fromCenter=False, showCrosshair=True)

# Пройдите через каждый метод и выполните трекировку с одним и тем же начальным bbox
for tracker_type in ['CSRT', 'KCF', 'MIL']:
    track_and_save_objects(video_path, tracker_type, initial_bbox)
