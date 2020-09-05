# USAGE
# python scan.py --image images/page.jpg

# import the necessary packages
import re
import ParserType
from ParserType import *

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread("./images/Capture.PNG")
store_name = getName(image)
cp = ParserType.Colruyt_parser(store_name, image)