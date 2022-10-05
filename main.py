import cv2
import math
import numpy as np

g1 = cv2.imread(r'C:\Users\vlad5\PycharmProjects\pythonProject\pixel.jpg')
gray=cv2.cvtColor(g1,cv2.COLOR_BGR2GRAY)
newGray=cv2.imread(r'C:\Users\vlad5\PycharmProjects\pythonProject\pixel.jpg')
newGray=cv2.cvtColor(newGray,cv2.COLOR_BGR2GRAY)
h,w=newGray.shape[:2]
cv2.namedWindow('Display window', cv2.WINDOW_KEEPRATIO )
cv2.imshow('Display window', g1)
cv2.waitKey(0)
cv2.destroyAllWindows()

n=3
start=n//2
gauss_matrix=[[0]*n for i in range(n)]
for k in range(n):
    for l in range(n):
        gauss_matrix[k][l]=(1/(2*math.pi))*np.exp(-((k-start)**2+(l-start)**2)/2)
    sum = 0
    for k in range(0,n):
        for l in range(0,n):
          sum += gauss_matrix[k][l]
print(sum)

print(gauss_matrix)
new_sum = 0
for k in range(n):
    for l in range(n):
        gauss_matrix[k][l] = gauss_matrix[k][l] / sum
        new_sum += gauss_matrix[k][l]
print(new_sum)

finishh = h - start
finishw = w - start
for i in range(start,finishh):
    for j in range(start,finishw):
        new_value = 0
        for k in range(n):
            for l in range(n):
                new_value=new_value+gauss_matrix[k][l]*gray[i-start+k][j-start+l]
            newGray[i][j]




