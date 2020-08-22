# USAGE
# python scan.py --image images/page.jpg

# import the necessary packages
import ParserType
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
from ParserType import *

print('hello')
# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True,	help = "Hier komt het pad naar uw foto")
# args = vars(ap.parse_args())

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
# image = cv2.imread(args["image"])
# image_name = args["image"]

# cp = ParserType.Colruyt_parser(image_name, image)
