def find_contours():
    img = cv2.imread(r'C:\Users\Arseniy.Zhuk\Documents\example.jpg')
    #cv2.namedWindow('Display window', cv2.WINDOW_AUTOSIZE)
    #cv2.imshow('Display window', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    newGray = cv2.imread(r'C:\Users\Arseniy.Zhuk\Documents\example.jpg')
    newGray = cv2.cvtColor(newGray, cv2.COLOR_BGR2GRAY)
    gauss_gray = cv2.GaussianBlur(gray, (11,11), 0)
    new_gauss_gray = cv2.GaussianBlur(newGray, (11,11), 0)
    h, w = gauss_gray.shape[:2]
    Gx_ker = [[-1,0,1],[-2,0,2],[-1,0,1]]
    Gy_ker = [[-1,-2,-1],[0,0,0],[1,2,1]]
    D_matrix = [[0] * w for i in range(h)]
    phi_matrix = [[0] * w for i in range(h)]
    max_grad = 0
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            gradX = 0
            gradY = 0
            for i in range(3):
                for j in range(3):
                    gradX += gauss_gray[y - 1 + i][x - 1 + j]*Gx_ker[i][j]
                    gradY += gauss_gray[y - 1 + i][x - 1 + j] * Gy_ker[i][j]
            D_matrix[y][x] = math.sqrt(gradX*gradX + gradY*gradY)
            if (D_matrix[y][x] > max_grad):
                max_grad = D_matrix[y][x]
            if (gradX == 0):
                gradX = 0.001
            t_phi = (gradY+0.0) / gradX
            phi = 0
            if ((gradX > 0) & (gradY < 0) & (t_phi < -2.414)) | ((gradX < 0) & (gradY < 0) & (t_phi > 2.414)):
                phi = 0
            elif ((gradX > 0) & (gradY < 0) & (t_phi >= -2.414) & (t_phi <= -0.414) ):
                phi = 1
            elif ((gradX > 0) & (gradY < 0) & (t_phi > -0.414)) | ((gradX > 0) & (gradY > 0) & (t_phi < 0.414)):
                phi = 2
            elif ((gradX > 0) & (gradY > 0) & (t_phi >= 0.414) & (t_phi <= 2.414) ):
                phi = 3
            elif ((gradX > 0) & (gradY > 0) & (t_phi > 2.414)) | ((gradX < 0) & (gradY > 0) & (t_phi < -2.414)):
                phi = 4
            elif ((gradX < 0) & (gradY > 0) & (t_phi >= -2.414) & (t_phi <= -0.414) ):
                phi = 5
            elif ((gradX < 0) & (gradY > 0) & (t_phi > -0.414)) | ((gradX < 0) & (gradY < 0) & (t_phi < 0.414)):
                phi = 6
            elif ((gradX < 0) & (gradY < 0) & (t_phi >= 0.414) & (t_phi <= 2.414) ):
                phi = 7

            phi_matrix[y][x] = phi

    for y in range(h):
        print(D_matrix[y])
    for y in range(h):
        print(phi_matrix[y])

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if ((phi_matrix[y][x] == 0) | (phi_matrix[y][x] == 4)):
                if ((D_matrix[y][x] > D_matrix[y-1][x]) & (D_matrix[y][x] > D_matrix[y+1][x])):
                    gauss_gray[y][x] = 0
                    new_gauss_gray[y][x] = 0
                else:
                    gauss_gray[y][x] = 255
                    new_gauss_gray[y][x] = 255
            if ((phi_matrix[y][x] == 1) | (phi_matrix[y][x] == 5)):
                if ((D_matrix[y][x] > D_matrix[y-1][x+1]) & (D_matrix[y][x] > D_matrix[y+1][x-1])):
                    gauss_gray[y][x] = 0
                    new_gauss_gray[y][x] = 0
                else:
                    gauss_gray[y][x] = 255
                    new_gauss_gray[y][x] = 255
            if ((phi_matrix[y][x] == 2) | (phi_matrix[y][x] == 6)):
                if ((D_matrix[y][x] > D_matrix[y][x-1]) & (D_matrix[y][x] > D_matrix[y][x+1])):
                    gauss_gray[y][x] = 0
                    new_gauss_gray[y][x] = 0
                else:
                    gauss_gray[y][x] = 255
                    new_gauss_gray[y][x] = 255
            if ((phi_matrix[y][x] == 3) | (phi_matrix[y][x] == 7)):
                if ((D_matrix[y][x] > D_matrix[y-1][x-1]) & (D_matrix[y][x] > D_matrix[y+1][x+1])):
                    gauss_gray[y][x] = 0
                    new_gauss_gray[y][x] = 0
                else:
                    gauss_gray[y][x] = 255
                    new_gauss_gray[y][x] = 255
    low_level = max_grad // 25
    high_level = max_grad // 10
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if (D_matrix[y][x] < low_level) & (gauss_gray[y][x] == 0):
                new_gauss_gray[y][x] = 255
            elif (D_matrix[y][x] < high_level) & ((gauss_gray[y][x] == 0)):
                if (D_matrix[y-1][x-1] < high_level) & (D_matrix[y-1][x] < high_level) & (D_matrix[y-1][x+1] < high_level):
                    if (D_matrix[y][x-1] < high_level) & (D_matrix[y][x+1] < high_level):
                        if (D_matrix[y+1][x-1] < high_level) & (D_matrix[y+1][x] < high_level) & (D_matrix[y+1][x+1] < high_level):
                            new_gauss_gray[y][x] = 255

    cv2.namedWindow('Display window', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Display window', new_gauss_gray)

    cv2.waitKey(0)
    cv2.destroyAllWindows()