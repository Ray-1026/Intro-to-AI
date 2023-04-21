import os
import cv2

def loadImages(dataPath):
    """
    load all Images in the folder and transfer a list of tuples. The first 
    element is the numpy array of shape (m, n) representing the image. 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    """
    1. Initialize an empty list called dataset.
    2. Use os.listdir() and for loop to make sure all the image can be read.
    3. Use cv2.imread() to read the image, and classify it into 1 for face and 0 for non-face.
    4. Add the image and its classification to a tuple and put the tuple into dataset.
    5. Return the dataset.
    """
    dataset=[]
    num=1
    for i in os.listdir(dataPath):
        root=dataPath+'/'+str(i)+'/'
        for k in os.listdir(root):
          dataset.append((cv2.imread(root+k, cv2.IMREAD_GRAYSCALE), num))
        num-=1
    # End your code (Part 1)
    return dataset
