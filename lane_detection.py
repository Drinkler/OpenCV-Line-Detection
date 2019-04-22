import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('countryroad.mp4')

while (cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)


    height = canny.shape[0]
    polygons = np.array([
        [(20, height), (1900, height), (1200, 600), (1120, 600)]
    ])
    mask = np.zeros_like(canny)
    cv2.fillPoly(mask, polygons, 255)
    cropped_image = cv2.bitwise_and(canny, mask)
    

    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    # display lines
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 10)


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()