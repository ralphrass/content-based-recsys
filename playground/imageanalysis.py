# read every transformed image and extract low-level features from it
import os, sys, glob
import numpy as np
from PIL import Image
from skimage.feature import hog, ORB, daisy, BRIEF, match_descriptors
from skimage import data, color, exposure, io
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity, manhattan_distances

def mse(x, y):
    return np.linalg.norm(x - y)

PREMATURE_EXIT = 2
COUNT = 0
similarities = {}

IMG_PATH = '/media/ralph/Extendido/Ralph/Datasets/Yelp7/2016_yelp_dataset_challenge_photos_cropped'

for infile in glob.glob(IMG_PATH + "/*.jpg"):

    print infile

    COUNT += 1
    if COUNT % PREMATURE_EXIT == 0:
        break

    im = io.imread(infile)
    io.imshow(im)
    io.show()

    # HOG
    fd, hog_im = hog(color.rgb2gray(im), orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                     transform_sqrt=True, visualise=True)
    # daisy_extractor, daisy_img = daisy(color.rgb2gray(im), visualize=True)

    io.imshow(hog_im)
    io.show()
    # break
    # find the top five most similar images to this one
    print "Showing top similar images"

    similarities[infile] = []
    similarities_list = []
    feature_list = []
    img_list = []
    subcount = 0

    for otherfile in glob.glob(IMG_PATH + "/*.jpg"):

        subcount += 1
        if subcount % 10 == 0:
            print subcount, "images read"
            break

        if otherfile == infile:
            continue

        im = io.imread(otherfile)
        features = hog(color.rgb2gray(im), orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                       transform_sqrt=True, visualise=False)

        feature_list.append(features)
        img_list.append(im)

    feature_list = np.asarray(feature_list)
    img_list = np.asarray(img_list, dtype=np.uint8)
    distance = np.array([mse(fd.reshape(1, -1), features.reshape(1, -1)) for features in feature_list])
    index = distance.argmin(axis=0)
    print index

    io.imshow(img_list[index])
    io.show()