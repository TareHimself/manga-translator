from translator.utils import resize_and_pad
import numpy as np


img = np.zeros((300,180,3),dtype=np.uint8)

print(resize_and_pad(img,target_size=(1000,500)).shape)