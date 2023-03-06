import cv2

vid_path = r'test_videos\car.mp4'
cascade_path = r'haarcascades\car.xml'
winName = r'Car Detection'

vid = cv2.VideoCapture(vid_path)

while True:
    ret, frame = vid.read()

    width = int(frame.shape[1] * 90/100)
    height = int(frame.shape[0] * 90/100)
    dim = (width, height)

    frame = cv2.resize(frame, dim)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    car_cascade = cv2.CascadeClassifier(cascade_path)
    cars = car_cascade.detectMultiScale(gray_frame, minNeighbors=5)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.imshow(winName, frame)
    if cv2.waitKey(20) == 27:
        break

vid.release()
cv2.destroyAllWindows()