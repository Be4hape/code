import numpy as np
from PIL import Image

im = Image.open("1958015.jpg")
nim = im.resize((800, 600))

nim.save("1958015.jpg")