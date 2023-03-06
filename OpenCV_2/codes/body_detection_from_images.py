import cv2

img_path = r'test_images\body.jpg'
body_cascade_path = r'haarcascades\haarcascade_fullbody.xml'

img = cv2.imread(img_path)
body_cascade = cv2.CascadeClassifier(body_cascade_path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

body = body_cascade.detectMultiScale(gray)

for (x, y, w, h) in body:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()