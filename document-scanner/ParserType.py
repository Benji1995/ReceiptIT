#####################################
#### Receipt Parser: Colruyt Receipts
# Created by: Benjamin
# Date: 7/25/2020
#####################################
import cv2
import imutils
import os
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'


def showImage(img, width=0, height=0):
    if width != 0 and height == 0:
        cv2.imshow("Image..", imutils.resize(img, width=width))
    elif width == 0 and height != 0:
        cv2.imshow("Image..", imutils.resize(img, height=height))
    elif width != 0 and height != 0:
        cv2.imshow("Image..", imutils.resize(img, width=width, height=height))
    else:
        cv2.imshow("Image..", img)


# Gen_parser GodClass
class Gen_parser:
    def __init__(self, name, img):
        self.img = img
        self.img_temp = None
        self.img_prog = None
        self.img_Name = name.split("/")[1]
        self.img_Width = self.img.shape[1]
        self.img_Height = self.img.shape[0]
        self.templatePath = "Templates/"

    def getDimensions(self):
        print("Image \"{}\" has dimensions: {} x {} pixels".format(self.img_Name, self.img_Width, self.img_Height))


# Colruyt Parser
class Colruyt_parser(Gen_parser):
    def __init__(self, name, img):
        super().__init__(name, img)
        self.store = "Colruyt"

        for t in os.listdir(self.templatePath.split("/")[0]):
            if self.store in t:
                self.img_temp = cv2.imread(self.templatePath + t)
        self.main()

    def main(self):
        self.img_prog = self.img
        img1 = cv2.cvtColor(self.img_prog, cv2.COLOR_BGR2GRAY)
        img2 = self.img_temp  # cv2.cvtColor(self.img_temp,cv2.COLOR_BGR2GRAY)

        test = img1

        edges = cv2.Canny(test, 50, 150, apertureSize=3)
        kernel_size = 2
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        edges = openFilter(edges, 3, 1, 1)
        showImage(edges)
        cv2.waitKey()
        cv2.destroyAllWindows()
        # lines = cv2.HoughLinesP(edges, 0.02, np.pi / 500, 10, minLineLength=40, maxLineGap=2)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))

            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 + 1000 * (a))
            cv2.line(test, (x1, y1), (x2, y2), (0, 255, 0), 2)

            test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)

        print(len(lines))

        titles = ["Edged", "Lines"]
        imges = [cv2.cvtColor(edges, cv2.COLOR_BGR2RGB), cv2.cvtColor(test, cv2.COLOR_BGR2RGB)]

        for i in range(2):
            plt.subplot(1, 2, i + 1), plt.imshow(imges[i])
            plt.title(titles[i])

        plt.show()

    def showProgress(self):
        try:
            showImage(self.img_prog)
        except:
            print("No modification has been done yet.")


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
