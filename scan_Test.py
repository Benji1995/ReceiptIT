# USAGE
# python scan.py --image images/page.jpg

# import the necessary packages
import difflib
import re
import ParserType
from ParserType import *

#
stores = ["Colruyt","Delhaize","Aldi","Lidl","Albert Heijn","Jumbo","Spar","Carrefour","Carrefour Express"]

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread("./images/receipt.jpg")

try:
    store_name = difflib.get_close_matches(ParserType.getStoreName(image,4,2,1),stores)[0]
except IndexError:
    store_name = ParserType.getStoreName(image,4,2,1)
    if not store_name in stores:
        store_name = re.sub(r'[^\w]', ' ', store_name)
        store_name = " ".join(store_name.split())
        stores.append(store_name)

print(stores)
#cp = ParserType.Colruyt_parser(image_name, image)