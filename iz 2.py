import cv2
import numpy as np
import math
from numba import njit

@njit
def gaussmatr(n,q):
    return np.array([[(1/(2*math.pi*q*q)*math.exp(-((j-n//2)*(j-n//2)+(i-n//2)*(i-n//2))/2*q*q)) for j in range(n)] for i in range(n)])

@njit
def conv(img,matr,n):
    img2 = img.copy()
    h, w = img2.shape[:2]
    start = n // 2
    finishh = h - start
    finishw = w - start
    for i in range(start, finishh):
        for j in range(start, finishw):

            newpixel = 0
            for k in range(n):
                for l in range(n):
                    newpixel += img[i - start + k][j - start + l] * matr[k][l]

            img2[i][j] = newpixel

    return img2



@njit
def convk(img,matr,n):
    img2 = img.copy()
    h, w = img2.shape[:2]
    start = n // 2
    finishh = h - start
    finishw = w - start
    for i in range(start, finishh):
        for j in range(start, finishw):

            newpixel = 0
            for k in range(n):
                for l in range(n):
                    newpixel += img[i - start + k][j - start + l] * matr[k][l]
            img2[i][j] = newpixel

    return img2

def conv2(img,matr,n):
    img2 = img.copy()
    h, w = img2.shape[:2]
    start = n // 2
    finishh = h - start
    finishw = w - start
    for i in range(start, finishh):
        for j in range(start, finishw):

            newpixel = 0
            for k in range(n):
                for l in range(n):
                    newpixel += img[i - start + k][j - start + l] * matr[k][l]

            img2[i][j] = newpixel
            print(newpixel)
    return img2

@njit
def nonMaximumSupression(img,g_m_x,g_m_y):
    h, w = img.shape[:2]
    gauss_gray = img.copy()
    result = np.zeros((h,w))
    D_matrix, phi_matrix = SobelOper1(img,g_m_x,g_m_y)

    for y in range(1, h-1):
        for x in range(1, w-1):
            if ((phi_matrix[y][x] == 0) | (phi_matrix[y][x] == 4)):
                if ((D_matrix[y][x] > D_matrix[y - 1][x]) & (D_matrix[y][x] > D_matrix[y + 1][x])):
                    gauss_gray[y][x] = D_matrix[y][x]

                else:
                    gauss_gray[y][x] = 0

            if ((phi_matrix[y][x] == 1) | (phi_matrix[y][x] == 5)):
                if ((D_matrix[y][x] > D_matrix[y - 1][x + 1]) & (D_matrix[y][x] > D_matrix[y + 1][x - 1])):
                    gauss_gray[y][x] = D_matrix[y][x]

                else:
                    gauss_gray[y][x] = 0

            if ((phi_matrix[y][x] == 2) | (phi_matrix[y][x] == 6)):
                if ((D_matrix[y][x] > D_matrix[y][x - 1]) & (D_matrix[y][x] > D_matrix[y][x + 1])):
                    gauss_gray[y][x] = D_matrix[y][x]

                else:
                    gauss_gray[y][x] = 0

            if ((phi_matrix[y][x] == 3) | (phi_matrix[y][x] == 7)):
                if ((D_matrix[y][x] > D_matrix[y - 1][x - 1]) & (D_matrix[y][x] > D_matrix[y + 1][x + 1])):
                    gauss_gray[y][x] = D_matrix[y][x]

                else:
                    gauss_gray[y][x] = 0

    return gauss_gray,D_matrix


@njit
def doubleFiltr(matr,lowPr,highPr):
    down = lowPr
    up = highPr
    h = len(matr)
    w = len(matr[0])
    result = np.zeros((h,w))
    for y in range(h):
        for x in range(w):
            if matr[y][x] >= up:
                result[y][x]=255
            elif matr[y][x]<=down:
                result[y][x]=0
            else:
                result[y][x]=127
    return result

@njit
def tang(x, y):
    if (x == 0):
        x = 0.001
    tg = y / x

    a = 0

    if ((x > 0) & (y < 0) & (tg < -2.414)) | ((x < 0) & (y < 0) & (tg > 2.414)):
        a = 0
    elif ((x > 0) & (y < 0) & (tg >= -2.414) & (tg <= -0.414)):
        a = 1
    elif ((x > 0) & (y < 0) & (tg > -0.414)) | ((x > 0) & (y > 0) & (tg < 0.414)):
        a = 2
    elif ((x > 0) & (y > 0) & (tg >= 0.414) & (tg <= 2.414)):
        a = 3
    elif ((x > 0) & (y > 0) & (tg > 2.414)) | ((x < 0) & (y > 0) & (tg < -2.414)):
        a = 4
    elif ((x < 0) & (y > 0) & (tg >= -2.414) & (tg <= -0.414)):
        a = 5
    elif ((x < 0) & (y > 0) & (tg > -0.414)) | ((x < 0) & (y < 0) & (tg < 0.414)):
        a = 6
    elif ((x < 0) & (y < 0) & (tg >= 0.414) & (tg <= 2.414)):
        a = 7

    return a

@njit
def SobelOper1(image_GF,g_m_x,g_m_y):

    h, w = image_GF.shape[:2]
    pad = 1
    size = 3
    Gm = image_GF.copy()
    anglGM = np.zeros((h,w))

    finishh = h - pad
    finishw = w - pad
    gmy = []
    for i in range(pad, finishh):
        for j in range(pad, finishw):
            valueX = 0
            valueY = 0
            for k in range(size):
                for l in range(size):
                    valueX = valueX + g_m_x[k][l] * image_GF[i - pad + k][j - pad + l]
                    valueY = valueY + g_m_y[k][l] * image_GF[i - pad + k][j - pad + l]
            Gm[i][j] = int(math.sqrt(valueX ** 2 + valueY ** 2))

            anglGM[i][j] = tang(valueX,valueY)
            gmy.append(valueY)

    return Gm,anglGM
@njit
def printmaxd(matr):
    for i in matr:
        print(i.min())

def canyTest(img,n,sigm,threshold1,threshold2):
    img2 = cv2.GaussianBlur(img, (n, n), sigm)
    img2 = cv2.Canny(img2,40,230,)
    cv2.putText(img2, f"n = {n}, sigm = {sigm}, threshold1 = {threshold1}, threshold2 = {threshold2}", (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    return img2

def canyTestH(img,n,sigm,threshold1,threshold2,g_m_x,g_m_y,op = "sobel"):
    gaussm = gaussmatr(n, sigm)

    sumgaussmatr = gaussm.sum()
    gaussm = gaussm / sumgaussmatr
    img2 = conv(img, gaussm,n)
    img2, GM = nonMaximumSupression(img2, g_m_x, g_m_y)
    img2 = doubleFiltr(img2, threshold1, threshold2)

    cv2.putText(img2,f"n = {n}, sigm = {sigm}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
    cv2.putText(img2,
                f"threshold1 = {threshold1}, threshold2 = {threshold2}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    cv2.putText(img2,
                 f"operator = {op}",
                 (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    return img2

def resize(img,h,w):

    res = cv2.resize(img,(h,w))
    return res
def main():
    sigm =  0.5
    threshold1 = 30
    threshold2 = 150
    img = cv2.imread(r'img\3.jpg', cv2.IMREAD_GRAYSCALE)
    img = resize(img,400,270)
    cv2.imshow(f"orig", img)
    g_m_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    g_m_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    img2 = canyTestH(img, 11, sigm, threshold1,threshold2, g_m_x, g_m_y)
    cv2.imshow(f"sobel", img2)
    g_m_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
    g_m_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])
    img2 = canyTestH(img, 11, sigm, threshold1, threshold2, g_m_x, g_m_y, "sharr")
    cv2.imshow(f"sharr", img2)
    g_m_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    g_m_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    img3 = canyTestH(img, 11, sigm, threshold1,threshold2, g_m_x, g_m_y,"prewitt")
    cv2.imshow(f"prewitt", img3)
    g_m_x = np.array([[-1, 0], [0, 1]])
    g_m_y = np.array([[1, 0], [0, -1]])
    img4 = canyTestH(img, 11, sigm, threshold1, threshold2, g_m_x, g_m_y, "roberts")
    cv2.imshow(f"roberts", img4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()





