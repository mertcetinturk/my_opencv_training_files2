import cv2

img_path = r'test_images\smile.jpg'
face_cascade_path = r'haarcascades\haarcascade_frontalface_default.xml'
smile_cascade_path = r'haarcascades\haarcascade_smile.xml'
winName = r'Smile Cascade'

img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(face_cascade_path)
face = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)

    roi_img = img[y:y+h, x:x+w]
    roi_gray = gray[y:y+h, x:x+w]

    smile_cascade = cv2.CascadeClassifier(smile_cascade_path)
    smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)

    for (sx, sy, sw, sh) in smile:
        cv2.rectangle(roi_img, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

cv2.imshow(winName, img)
cv2.waitKey(0)
cv2.destroyAllWindows()
