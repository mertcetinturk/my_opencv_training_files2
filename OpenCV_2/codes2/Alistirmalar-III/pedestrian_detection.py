import cv2
import imutils

img_path = r'test_images\pedestrian.jpg'

img = cv2.imread(img_path)
img = imutils.resize(img, 500)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  # yaya tespiti yapılan yer

(coordinates, _) = hog.detectMultiScale(img, winStride=(2, 2), padding=(4, 4), scale=1.05)
# print(len(coordinates))  # kaç tane koordinat kümesi yakaladığını görebiliriz.

count = 0
font = cv2.FONT_HERSHEY_SIMPLEX

for (x, y, w, h) in coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
    count += 1

cv2.putText(img, f'People: {str(count)}', (10, 25), font, 1, (255, 0, 0), 2)


cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Aynı işlemler videolar için de yapılabilir.
