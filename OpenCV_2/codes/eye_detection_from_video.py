import cv2

video_path = r'test_videos\eye.mp4'
eye_cascade_path = r'haarcascades\haarcascade_eye.xml'
face_cascade_path = r'haarcascades\haarcascade_frontalface_default.xml'

vid = cv2.VideoCapture(video_path)

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

while True:
    ret, frame = vid.read()
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, minNeighbors=8)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (10, 255, 90), 3)

        roi_frame = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, minNeighbors=8)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_frame, (ex, ey), (ex + ew, ey + eh), (140, 35, 200), 2)

            cv2.imshow('Frame', frame)

    if cv2.waitKey(5) == 27:
        break

vid.release()
cv2.destroyAllWindows()



