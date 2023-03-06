import cv2
import numpy as np

path = r'test_videos\traffic.avi'
vid = cv2.VideoCapture(path)

backsub = cv2.createBackgroundSubtractorMOG2()
# KNN kullanınca bozulmalar oluşuyor.

c = 0

while True:
    ret, frame = vid.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray image sondaki bozulmaları önledi

        # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # threshold görüntüyü bozdu ve bazı araçların sayılamamasına neden oldu

        fgmask = backsub.apply(gray)

        cv2.line(frame, (50, 0), (50, 300), (0, 255, 0), 2)
        cv2.line(frame, (70, 0), (70, 300), (0, 255, 0), 2)

        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # hata alınmasın diye oluşturulmuş bir try except satırları
        try:
            hierarchy = hierarchy[0]

        except:
            hierarchy = []

        for contour, hier in zip(contours, hierarchy):  # çok fazla değer olduğu için zip içerisine alındı
            (x, y, w, h) = cv2.boundingRect(contour)
            if w > 40 and h > 40:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
                if 70 > x > 50:
                    c += 1
        cv2.putText(frame, 'Car ' + str(c), (90, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Car Counter', frame)
        cv2.imshow('fgmask', fgmask)

    else:
        break

    if cv2.waitKey(20) == 27:
        break

vid.release()
cv2.destroyAllWindows()
