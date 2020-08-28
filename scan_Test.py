# USAGE
# python scan.py --image images/page.jpg

# import the necessary packages
import re
import ParserType
from ParserType import *

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread("./images/receipt.jpg")
store_name = getName(image)
cp = ParserType.rectangle_parser(store_name, image)