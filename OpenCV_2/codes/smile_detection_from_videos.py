import cv2

vid_path = r'test_videos\smile.mp4'
face_cascade_path = r'haarcascades\haarcascade_frontalface_default.xml'
smile_cascade_path = r'haarcascades\haarcascade_smile.xml'
winName = r'Smile Cascade'

vid = cv2.VideoCapture(vid_path)

while True:
    ret, frame = vid.read()
    scale_percent = 0.9  # %90
    width = int(frame.shape[1] * scale_percent)
    height = int(frame.shape[0] * scale_percent)
    dim = (width, height)
    frame = cv2.resize(frame, dim)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (200, 250, 180), 2)

        roi_frame = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]

        smile_cascade = cv2.CascadeClassifier(smile_cascade_path)
        smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.9, minNeighbors=9)

        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_frame, (sx, sy), (sx+sw, sy+sh), (15, 95, 170), 2)

    cv2.imshow(winName, frame)

    if cv2.waitKey(5) == 27:
        break

vid.release()
cv2.destroyAllWindows()
