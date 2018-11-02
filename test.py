
import skimage.io
import matplotlib.pylab as plt
import cv2
import os
import json
import sys
from chardet import detect
image_path = 'elephant.jpg'


def load_image(self, image_id):
    """Load the specified image and return a [H,W,3] Numpy array.
    """
    # Load image
    image = skimage.io.imread(self.image_info[image_id]['path'])
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    return image


def show_image(image):
    plt.imshow(image)
    plt.show()
       	

if __name__ == '__main__':
    #data_dir = "/home/ai/mbh/data/huaxi/json/train"
    #f = open(os.path.join(data_dir, '4350160_0.json'),'rb+')
    #content = f.read()
    #encoding = detect(content)['encoding']
    #print(encoding)
    #content = content.decode(encoding)
    #contents = content.encode('utf-8')
    #print(detect(contents)['encoding'])    
    image_path = "/home/ai/mbh/data/crop_test/tar/4641434.jpg"
    image = skimage.io.imread(image_path)
    plt.imshow(image)
    plt.show()
    	

