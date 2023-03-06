import cv2

img_path = r'test_images\face.png'
eye_cascade_path = r'haarcascades\haarcascade_eye.xml'
face_cascade_path = r'haarcascades\haarcascade_frontalface_default.xml'

img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

face = face_cascade.detectMultiScale(gray, 1.3, 7)

for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)

    img2 = img[y:y+h, x: x+w]
    gray2 = gray[y:y+h, x: x+w]

    eyes = eye_cascade.detectMultiScale(gray2)

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img2, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()




