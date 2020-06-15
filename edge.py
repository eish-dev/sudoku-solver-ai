import cv2
import numpy as np
import matplotlib.pyplot as plt

#######Reading Image####
image = cv2.imread("data/s_2.jpg")
#cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
#imgr = cv2.resize(image, (960, 540))          

####Converting image to binary form####
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
#thresh = cv2.adaptiveThreshold(blur,255,0,1,19,2)

####Extrating the sudoku puzzle from image####
contours,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

max_area = 0
c = 0

for i in contours:
    area = cv2.contourArea(i)
    if area > 1000:
        if area > max_area:
            max_area = area
            best_cnt = i
            image = cv2.drawContours(image, contours, c, (0, 255, 0))
    c += 1


mask = np.zeros((gray.shape), np.uint8)
cv2.drawContours(mask, [best_cnt], 0, 255, -1)
cv2.drawContours(mask, [best_cnt], 0, 0, 2)

out = np.zeros_like(gray)

out[mask == 255] = gray[mask == 255]

blur = cv2.GaussianBlur(out, (5,5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
contours,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

c = 0
for i in contours:
        area = cv2.contourArea(i)
        if area > 1000/2:
            cv2.drawContours(image, contours, c, (0, 0, 0), 3)
        c+=1

#print(best_cnt)

#approx = cv2.approxPolyDP(best_cnt, 0.1 * cv2.arcLength(best_cnt, True), True) 
#n = approx.ravel()
#x1, y1, x2, y2, x3, y3, x4, y4 = n
#print(y4-y2 , " " , x1-x4)
#h, l = gray.shape
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
x,y,w,h = cv2.boundingRect(best_cnt)
#final_image = gray[y4:y2, x1:x4]
final_image  = gray[y:y+h, x:x+w]
print(final_image.shape)

cv2.imwrite("preprocessed_imgs/s_1_processed.jpg", final_image)

#cell = final_image[311//9:2*311//9 , 311//9:2*311//9]




cv2.imshow('output', final_image)

cv2.waitKey(0)
cv2.destroyAllWindows()