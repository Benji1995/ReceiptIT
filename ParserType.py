#####################################
#### Receipt Parser: Colruyt Receipts
# Created by: Benjamin Bernaerdts
# Date: 7/25/2020
#####################################

import os
import pytesseract
import re
import difflib
from JSON_script import *
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils

pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\b.bernaerdts\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'


# pytesseract.pytesseract.tesseract_cmd =


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
        img = self.img
        img2 = self.img_temp

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 50, 200)

        img, lines = getLinesP(img, edge, theta=np.pi/360,threshold=10,minLineLength=10,maxLineGap=20)
        print(len(lines))

        showImage(edge, "Edge")
        showImage(img, "Gamma Adjust")

        cv2.waitKey(0)
        cv2.destroyAllWindows()


class rectangle_parser(Gen_parser):
    def __init__(self, name, img):
        super().__init__(name, img)
        self.store = "Colruyt"

        for t in os.listdir(self.templatePath.split("/")[0]):
            if self.store in t:
                self.img_temp = cv2.imread(self.templatePath + t)

    def main(self):

        image = self.img
        ratio = image.shape[0] / 500.0
        orig = image.copy()
        image = imutils.resize(image, height=500)

        # convert the image to grayscale, blur it, and find edges
        # in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)

        # show the original image and the edge detected image
        # print("STEP 1: Edge Detection")
        # cv2.imshow("Image", image)
        # cv2.imshow("Edged", edged)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # find the contours in the edged image, keeping only the
        # largest ones, and initialize the screen contour
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        # loop over the contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # if our approximated contour has four points, then we
            # can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                break

        # show the contour (outline) of the piece of paper
        # print("STEP 2: Find contours of paper")
        # cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
        # cv2.imshow("Outline", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # apply the four point transform to obtain a top-down
        # view of the original image
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

        # convert the warped image to grayscale, then threshold it
        # to give it that 'black andd white' paper effect
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        T = threshold_local(warped, 11, offset=10, method="gaussian")
        warped = (warped > T).astype("uint8") * 255

        # show the original and scanned images
        # print("STEP 3: Apply perspective transform")
        # cv2.imshow("Original", imutils.resize(orig, height=650))
        cv2.imshow("Scanned", imutils.resize(warped, height=650))
        cv2.waitKey(0)


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


def getLines(img, edges):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return img, lines


def getLinesP(img, edges, rho=1, theta=np.pi / 180, threshold=1, minLineLength=100, maxLineGap=10):
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength, maxLineGap)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img, lines


def getStoreNameFromPic(img, kernel, dil_it, erode_it):
    kernel_dim = kernel
    kernel = np.ones((kernel_dim, kernel_dim), np.uint8)
    img = cv2.dilate(img, kernel, iterations=dil_it)
    img = cv2.erode(img, kernel, iterations=erode_it)

    return pytesseract.image_to_string(img)
    ##de kernel dimensie en de iteration settings zijn ideaal voor een colruyt ticket, enkel colruyt eraf te lezen, moet nog getest worden op andere tickets


def getName(image):
    stores = getStores()[1]
    try:
        store_name = difflib.get_close_matches(getStoreNameFromPic(image, 4, 2, 1), stores)[0]
    except IndexError:
        store_name = getStoreNameFromPic(image, 4, 2, 1)
        if not store_name in stores:
            store_name = re.sub(r'[^\w]', ' ', store_name)
            store_name = " ".join(store_name.split())
            addStore(store_name)

    return store_name


def getLengthOfCoords(x1, x2, y1, y2):
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return length


def convertScale(img, alpha, beta):
    """Add bias and gain to an image with saturation arithmetics. Unlike
    cv2.convertScaleAbs, it does not take an absolute value, which would lead to
    nonsensical results (e.g., a pixel at 44 with alpha = 3 and beta = -210
    becomes 78 with OpenCV, when in fact it should become 0).
    """

    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)


# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=25):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = convertScale(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)
