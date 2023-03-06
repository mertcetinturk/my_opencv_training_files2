import cv2
import numpy as np
import pytesseract
import imutils

img = cv2.imread(r'test_images\licence_plate.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
filtered = cv2.bilateralFilter(gray, 6, 250, 250)
canny = cv2.Canny(filtered, 30, 200)

contours = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(contours)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

screen = None

for c in cnts:
    epsilon = 0.018 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    if len(approx) == 4:
        screen = approx
        break

mask = np.zeros(gray.shape, dtype=np.uint8)

new_img = cv2.drawContours(mask, [screen], 0, (255, 255, 255), -1)

new_img = cv2.bitwise_and(img, img, mask=mask)

(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(botx, boty) = (np.max(x), np.max(y))

cropped = gray[topx:botx+1, topy:boty+1]

text = pytesseract.image_to_string(cropped, lang='eng')
print('Text:', text)

cv2.imshow('Last', cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
