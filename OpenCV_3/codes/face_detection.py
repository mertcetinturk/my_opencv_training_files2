import face_recognition
import cv2

path = r'..\images\leo_and_tobey.jpeg'
image = cv2.imread(path)

face_locations = face_recognition.face_locations(image)
# print(face_locations)
# Koordinatlar sondan başlayıp başa doğru gidiyor
# Yani 4 değerli tupleda son değer x1 ilk değer y1
# İkinci değer x2 ve son olarak da üçüncü değer y2;
# [(175, 736, 283, 629), (140, 569, 247, 462)]
#  (y1    x2   y2   x1)  (y1    x2   y2   x1)

pt1_0 = (629, 175)
pt2_0 = (736, 283)

pt1_1 = (462, 140)
pt2_1 = (569, 247)

color1 = (255, 0, 0)
color2 = (0, 0, 255)

cv2.rectangle(image, pt1_0, pt2_0, color1, 2)
cv2.rectangle(image, pt1_1, pt2_1, color2, 2)

cv2.imshow('Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
