import cv2
import os

cap = cv2.VideoCapture(0)
# os.system('C:\\Users\\DP\\Desktop\\LED_off.bat')
n=8
while True:
    ret, orig = cap.read()
    copy1 = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(copy1, 169, 255, cv2.THRESH_BINARY)
    cv2.imshow('thres', thresh1)
    cv2.imshow('orig', orig)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        print(n)
        cv2.imwrite(str(n)+'.jpg', orig)
        n+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):#or (move_x=='Ok' and move_y == 'Ok'):
        # cv2.imwrite('4.jpg', orig)
        break
cap.release()
cv2.destroyAllWindows()