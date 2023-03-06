import cv2

vid_path = r'test_videos\body.mp4'
body_cascade_path = r'haarcascades\haarcascade_fullbody.xml'
window_name = 'Body Detection'

vid = cv2.VideoCapture(vid_path)

while True:
    ret, frame = vid.read()
    frame = cv2.resize(frame, (640, 480))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    body_cascade = cv2.CascadeClassifier(body_cascade_path)
    body = body_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in body:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow(window_name, frame)
    if cv2.waitKey(5) == 27:
        break

vid.release()
cv2.destroyAllWindows()


