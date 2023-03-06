import cv2

img_path = r'test_images\car.jpg'
cascade_path = r'haarcascades\car.xml'

winName = 'Car Detection'

img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

car_cascade = cv2.CascadeClassifier(cascade_path)
cars = car_cascade.detectMultiScale(gray)

for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

cv2.imshow(winName, img)
cv2.waitKey(0)
cv2.destroyAllWindows()

