import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


img = mpimg.imread('./trump.png')     

img_convert_ndarray = np.array(img)

# print(img_convert_ndarray)
for line in img_convert_ndarray:
    print(line)