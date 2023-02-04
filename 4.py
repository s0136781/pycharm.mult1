import cv2

first_frame = None
video = cv2.VideoCapture(r"C:\Users\vlad5\PycharmProjects\pythonProject4\ЛР4_main_video.mov")
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
status = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('video/out.mov', fourcc,25,(w,h))
while True:
    check, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue
    delta_frame = cv2.absdiff(first_frame, gray)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
    (cnts, _) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contours in cnts:
        if cv2.contourArea(contours) < 1000:
            continue
        status = 1
        video_writer.write(frame)

    cv2.imshow("Gray FRame", gray)
    cv2.imshow("2", delta_frame)
    cv2.imshow("3", thresh_frame)
    cv2.imshow("5", frame)
    key = cv2.waitKey(27)
    if key == ord('q'):
        break
    print(status)
video.release()
cv2.destroyAllWindows