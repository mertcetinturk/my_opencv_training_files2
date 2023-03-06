import face_recognition
import cv2

image = cv2.imread(r'..\images\Leonardo_Dicaprio_Cannes_2019.jpg')

path = r'..\images\dicaprio.jpg'
dicaprio = face_recognition.load_image_file(path)
dicaprioLoc = face_recognition.face_locations(dicaprio)
dicaprioEncoding = face_recognition.face_encodings(dicaprio)[0]

testImage_path = r'..\images\Leonardo_Dicaprio_Cannes_2019.jpg'
testImage = face_recognition.load_image_file(testImage_path)
faceEncoding = face_recognition.face_encodings(testImage)
faceLoc = face_recognition.face_locations(testImage)

matchedFaces = face_recognition.compare_faces(dicaprioEncoding, faceEncoding)

for index, loc in enumerate(faceLoc):
    topLeft_y, bottomRight_x, bottomRight_y, topLeft_x = loc
    pt1 = (topLeft_x, topLeft_y)
    pt2 = (bottomRight_x, bottomRight_y)
    color = (0, 0, 255)

    if True in matchedFaces:

        cv2.rectangle(image, pt1, pt2, color, thickness=2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, 'Leonardo Dicaprio', pt1, font, 1, color, 2)
        cv2.imshow('Face Detection', image)

    else:
        cv2.rectangle(image, pt1, pt2, color, thickness=2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, 'Unknown Person', pt1, font, 1, color, 2)
        cv2.imshow('Face Detection', image)

cv2.waitKey(0)
cv2.destroyAllWindows()

### NOT SUFFICIENT
