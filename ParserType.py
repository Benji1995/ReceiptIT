#####################################
#### Receipt Parser: Colruyt Receipts
# Created by: Benjamin Bernaerdts
# Date: 7/25/2020
#####################################

import cv2
import imutils
import os
import numpy as np
import pytesseract
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\Benjamin\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'


# Gen_parser GodClass
class Gen_parser:
    def __init__(self, name, img):
        self.img = img
        self.img_temp = None
        self.img_prog = None
        self.img_Name = name
        self.img_Width = self.img.shape[1]
        self.img_Height = self.img.shape[0]
        self.templatePath = "Templates/"

        self.main()

    def main(self):
        pass


# Colruyt Parser
class Colruyt_parser(Gen_parser):
    def __init__(self, name, img):
        super().__init__(name, img)
        self.store = "Colruyt"

        for t in os.listdir(self.templatePath.split("/")[0]):
            if self.store in t:
                self.img_temp = cv2.imread(self.templatePath + t)

    def main(self):
        img1 = self.img
        img2 = self.img_temp  # cv2.cvtColor(self.img_temp,cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(img1, 120, 255, cv2.THRESH_BINARY_INV)

        kernal = np.ones((5, 5), np.uint8)

        mg = cv2.morphologyEx(mask, cv2.MORPH_BLACKHAT, kernal)
        mth = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernal)

        showImage(mth, "Gradien")
        showImage(mask, "Mask")
        showImage(img1, "Image")

        print(pytesseract.image_to_string(mask))
        # showImage(edges, "edges")
        cv2.waitKey()
        cv2.destroyAllWindows()


def openFilter(img, kernel_size, it_dil, it_er):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv2.dilate(img, kernel, iterations=it_dil)
    img = cv2.erode(img, kernel, iterations=it_er)

    return img


def closeFilter(img, kernel_size, it_dil, it_er):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv2.erode(img, kernel, iterations=it_er)
    img = cv2.dilate(img, kernel, iterations=it_dil)

    return img


def showImage(img, img_name, width=0, height=0):
    if width != 0 and height == 0:
        cv2.imshow(img_name, imutils.resize(img, width=width))
    elif width == 0 and height != 0:
        cv2.imshow(img_name, imutils.resize(img, height=height))
    elif width != 0 and height != 0:
        cv2.imshow(img_name, imutils.resize(img, width=width, height=height))
    else:
        cv2.imshow(img_name, img)


def getLines(edges):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    # for rho, theta in lines[0]:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * (a))
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * (a))

    # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return lines


def getLinesP(edges, theta, threshold, minLineLength, maxLineGap):
    showImage(edges, "Edges")
    lines = cv2.HoughLinesP(edges, 1, theta, threshold, minLineLength, maxLineGap)
    # for x1, y1, x2, y2 in lines[0]:
    # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return lines


def getStoreName(img,kernel,dil_it,erode_it):
    kernel_dim = kernel
    kernel = np.ones((kernel_dim, kernel_dim), np.uint8)
    img = cv2.dilate(img, kernel, iterations=dil_it)
    img = cv2.erode(img, kernel, iterations=erode_it)

    return pytesseract.image_to_string(img)
    ##de kernel dimensie en de iteration settings zijn ideaal voor een colruyt ticket, enkel colruyt eraf te lezen, moet nog getest worden op andere tickets
