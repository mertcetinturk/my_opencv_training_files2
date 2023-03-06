import face_recognition
import cv2

path = r'..\videos\face.mp4'
cap = cv2.VideoCapture(path)
color = (0, 0, 255)

while True:
    ret, frame = cap.read()

    face_location = face_recognition.face_locations(frame)

    for index, faceLoc in enumerate(face_location):
        topLeft_y, bottomRight_x, bottomRight_y, topLeft_x = faceLoc
        pt1 = (topLeft_x, topLeft_y)
        pt2 = (bottomRight_x, bottomRight_y)

        cv2.rectangle(frame, pt1, pt2, color, thickness=2)

        cv2.imshow('Test', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
