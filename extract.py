import cv2
import numpy as np
import sys
img = cv2.imread("preprocessed_imgs/s_1_processed.jpg", cv2.COLOR_BGR2GRAY)
#print(image)
#cv2.imshow("Output", img)

# Number of rows
nRows = 9
# Number of columns
mCols = 9



# Dimensions of the image
sizeX = img.shape[1]
sizeY = img.shape[0]

print(img.shape)

for i in range(0,nRows):
    for j in range(0, mCols):
        roi = img[i*sizeY//nRows:i*sizeY//nRows + sizeY//nRows ,j*sizeX//mCols:j*sizeX//mCols + sizeX//mCols]
        #cv2.imshow('rois'+str(i)+str(j), roi)
        cv2.imwrite('patches/patch_'+str(i)+str(j)+".jpg", roi)


p1 = cv2.imread("patches/patch_00.jpg")
print(p1.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()