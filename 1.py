import cv2

g1 = cv2.imread(r'C:\Users\vlad5\PycharmProjects\pythonProject1\nec.jpg')
cv2.namedWindow('Display window', cv2.WINDOW_KEEPRATIO)
cv2.imshow('Display window', g1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cap = cv2.VideoCapture(r'C:\Users\vlad5\PycharmProjects\pythonProject1\xm.mp4', cv2.CAP_ANY)
while True:
    ret, frame = cap.read()
    if not(ret):
        break
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF ==27:
        break

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4,480)
while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF ==27:
        break
        cap.release()
        cv2.destroyAllWindows()


video = cv2.VideoCapture(0)
ok, img = video.read()
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter("outputcam.mov", fourcc, 25, (w, h))
while (True):
    ok, img = video.read()
    cv2. imshow('img', img)
    video_writer.write(img)
    if cv2.waitkey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()



