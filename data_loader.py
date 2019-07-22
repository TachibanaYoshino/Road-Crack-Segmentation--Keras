import numpy as np
import cv2,os
import glob
from itertools import *
import random
from keras.utils import to_categorical


def rotate(image,image1, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH)),cv2.warpAffine(image1, M, (nW, nH))

def flip(x,y):
    if np.random.randint(0, 2) == 0:
        x = cv2.flip(x,1,dst=None)
        y = cv2.flip(y,1,dst=None)

    if np.random.randint(0, 2) == 0:
        x = cv2.flip(x, 0, dst=None)
        y = cv2.flip(y, 0, dst=None)
    return x,y

def data_augment(x, y):

    if np.random.randint(0, 2) == 0:
        x, y = rotate(x, y, random.choice([90,180,270]))

    if np.random.randint(0, 2) == 0:
        x, y = flip(x,y)

    return x, y

def getImageArr(path, path1, width, height, augment=True, imgNorm='divide'):
    img = cv2.imread(path, 1)

    if imgNorm == 'sub_and_divide':
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif imgNorm == 'sub_mean':
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
    elif imgNorm == 'divide':
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img / 255.0


    label = cv2.imread(path1, 0)
    label = cv2.resize(label, (width, height))
    label = label / 255.0

    if augment:
        img, label = data_augment(img, label)

    label = to_categorical(label,num_classes=2)
    assert label.shape[-1]==2

    return img, label


def getSegmentationArr(path, width, height):
    # seg_labels = np.zeros((height, width, 2))

    img = cv2.imread(path, 0)
    img = cv2.resize(img, (width, height))
    label = img / 255.0

    label = to_categorical(label, num_classes=2)
    assert label.shape[-1] == 2

    return label

def imageSegmentationGenerator(images_path, segs_path, batch_size, img_height, img_width, augment, phase='train'):

    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = glob.glob(images_path + '*.jpg')
    images.sort()
    segmentations = glob.glob(segs_path + '*.png')
    segmentations.sort()

    assert len(images) == len(segmentations)
    for im, seg in zip(images, segmentations):
        assert (im.split('/')[-1].split('.')[0] == seg.split('/')[-1].split('.')[0])

    if phase == 'train':
        X = []
        Y = []
        batch = 0
        while True:
            for i in np.random.permutation(np.arange(len(images))):
                im, seg = images[i], segmentations[i]
                x, y = getImageArr(im, seg, img_width, img_height, augment)
                X.append(x)
                Y.append(y)
                batch += 1
                if batch % batch_size == 0:
                    yield np.array(X), np.array(Y)
                    X = []
                    Y = []
                    batch = 0
    else:
        X = []
        Y = []
        batch = 0
        while True:
            for i in np.arange(len(images)):
                im, seg = images[i], segmentations[i]
                x, y = getImageArr(im, seg, img_width, img_height, False)
                X.append(x)
                Y.append(y)
                batch += 1
                if batch % batch_size == 0:
                    yield np.array(X), np.array(Y)
                    X = []
                    Y = []
                    batch = 0

