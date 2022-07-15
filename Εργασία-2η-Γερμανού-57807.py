import cv2
import numpy as np

"""
Εργασία 2η - Όραση Υπολογιστών - Μαρία Αρετή Γερμανού - 57807
"""

# Creating lists of yard-house images and further down using variables too:
filenames = []
img = []
img_grayscale = []

i = 0
for i in range(1, 6, 1):
    filenames.append('images_2/yard-house-0' + str(i) + '.png')
print(filenames)

# Re-assigning the files and reading them colored and grayed:
filenames_1 = 'images_2/yard-house-01.png'
filenames_2 = 'images_2/yard-house-02.png'
filenames_3 = 'images_2/yard-house-03.png'
filenames_4 = 'images_2/yard-house-04.png'
filenames_5 = 'images_2/yard-house-05.png'

# Grayed images:
img_1 = cv2.imread(filenames_1, cv2.IMREAD_GRAYSCALE)
img_2 = cv2.imread(filenames_2, cv2.IMREAD_GRAYSCALE)
img_3 = cv2.imread(filenames_3, cv2.IMREAD_GRAYSCALE)
img_4 = cv2.imread(filenames_4, cv2.IMREAD_GRAYSCALE)
img_5 = cv2.imread(filenames_5, cv2.IMREAD_GRAYSCALE)

img = [img_1, img_2, img_3, img_4, img_5]


# Method of SIFT or SURF Descriptor and detectors of two images. Firstly we detect the key-points of images and then
# we compute descriptors of that key-points, with the detect and compute functions. The return of those functions
# will be a signe directional array. After detecting and computing key-points we use the function matching from below
# to start the crosschecking procedure of every key-point founded in the two images. We cannot use the class
# BFMatcher so we must find another way to find the closest descriptor. For each descriptor in the first set,
# this matcher finds the closest descriptor in the second set by trying each one. This descriptor matcher supports
# masking permissible matches of descriptor sets.

def matching(d1, d2):
    n1 = d1.shape[0]
    n2 = d2.shape[0]
    matches = []

    for i in range(n1):
        fv = d1[i, :]
        diff = d2 - fv
        diff = np.abs(diff)
        distances = np.sum(diff, axis=1)

        i2 = np.argmin(distances)
        minDistance2 = distances[i2]

        # infinitive
        distances[i2] = np.inf

        i3 = np.argmin(distances)
        minDistance3 = distances[i3]

        # "Good" Matching condition
        if minDistance2 / minDistance3 < 0.5:
            matches.append(cv2.DMatch(i, i2, minDistance2))

    return matches


def descriptors_and_crosscheck(img1, img2, methodOfDescriptor):
    if methodOfDescriptor == "SIFT" or methodOfDescriptor == "sift":
        descriptor = cv2.xfeatures2d_SIFT.create()
    elif methodOfDescriptor == "SURF" or methodOfDescriptor == "surf":
        descriptor = cv2.xfeatures2d_SURF.create()
    # Detection of key-points for first image:
    keypoint1 = descriptor.detect(img1)
    # Computation of key-points for first image:
    desc1 = descriptor.compute(img1, keypoint1)

    # Detection of key-points for second image:
    keypoint2 = descriptor.detect(img2)
    # Computation of key-points for second image:
    desc2 = descriptor.compute(img2, keypoint2)

    # Crosschecking calling function matching with descriptors one and two as parameters.
    matches1 = matching(desc1[1], desc2[1])
    matches2 = matching(desc2[1], desc1[1])

    # Loop for creating a list with matches because we can't use the class.
    matches = [m for m in matches1 for n in matches2 if m.distance == n.distance]
    # matches = []
    # for m in matches1:
    #     for n in matches2:
    #         if m.distance == n.distance:
    #             matches.append(m)

    # We create two lists because homography needs two lists with points that the first is the match of the second. We
    # will use a different function for homography below. The code below is identical from lab examples just like
    # matching function. :-)
    img_pt1 = []
    img_pt2 = []
    for match in matches:
        img_pt1.append(keypoint1[match.queryIdx].pt)
        img_pt2.append(keypoint2[match.trainIdx].pt)
    img_pt1 = np.array(img_pt1)
    img_pt2 = np.array(img_pt2)

    # If we run the descriptors and crosscheck function above we will see that the key-points' connection is aligned
    # horizontally:

    dimg = cv2.drawMatches(img1, desc1[0], img2, desc2[0], matches, None)

    # cv2.imwrite("Crosschecking_result_sift.png", dimg)
    #
    # cv2.imwrite("Crosschecking_result_surf.png", dimg)

    # cv2.namedWindow('Crosschecking result:', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('Crosschecking result:', dimg)
    # cv2.waitKey(0)

    return img_pt1, img_pt2


# descriptors_and_crosscheck(img_1, img_2, 'sift')

# descriptors_and_crosscheck(img_1, img_2, 'surf')

# Next we must find homography for stitching the images. The RANSAC algorithm is used to match the key-points. It is
# not enough to find the minimum distance of key-points but to match patterns of neighbor key-points too. Homography
# creates an transformed array because it finds the right rotation of an image to fit with another. Function for
# homography:
def homography(img1, img2, methodOfDescriptor):
    img_1pt, img_2pt = descriptors_and_crosscheck(img1, img2, methodOfDescriptor)
    # Calculate homography based on RANSAC algorithm
    M, mask = cv2.findHomography(img_2pt, img_1pt, cv2.RANSAC)
    return M


def crop(image):
    # Convert the filtered grayscale image to a binary image with threshold
    th_val, img_th = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    # Find contours
    _, contours, _ = cv2.findContours(img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    # Bound the space of the image
    (x, y, w, h) = cv2.boundingRect(cnt)
    # Crop the space
    croppedImage = image[y:y + h, x:x + w]
    return croppedImage


# Before the panorama is built we need to stitch the images one next to another. First we find the translated array
# from homography function and due to method of descriptor and the direction of stitching we create a translated
# matrix and multiply it with the array from the homography function. This will be done with list merging and the
# final list will be the result. I calculate the translation before i stitch the homographed image next to original.
# After the stitching i crop the rotated image because i want it to be aligned next to original.


def stitching(left, right, methodOfDescriptor):
    lim = max(left.shape) * 1.5
    translation = int(lim)

    M = homography(right, left, methodOfDescriptor)
    # Translation matrix
    translationMatrix = np.array([[1, 0, translation], [0, 1, 0], [0, 0, 1]])
    # Matrix Multiplication
    TMpM = np.matmul(translationMatrix, M)
    # Wraping transformation
    result = cv2.warpPerspective(left, TMpM, (translation * 2, translation * 2))
    result[0: right.shape[0], translation:translation + right.shape[1]] = right

    stitchedImage = crop(result)

    return stitchedImage


def panorama(images, methodOfDescriptor):
    # First stitching
    print("First stitching...\n")
    FirstStitching = stitching(images[0], images[1], methodOfDescriptor)

    # Second stitching
    print("Second stitching...\n")
    SecondStitching = stitching(FirstStitching, images[2], methodOfDescriptor)

    # Third stitching
    print("Third stitching..\n")
    ThirdStitching = stitching(SecondStitching, images[3], methodOfDescriptor)

    # Final stitching
    print("Final stitching..\n")
    Final_panorama = stitching(ThirdStitching, images[4], methodOfDescriptor)

    panoramaCropped = Final_panorama[:images[1].shape[0]]
    print("Paronama is completed\n")

    return Final_panorama, panoramaCropped


panorama_SIFT, panorama_SIFT_cropped = panorama(img, "SIFT")
# cv2.namedWindow('Panorama result with sift:', cv2.WINDOW_GUI_EXPANDED)
# cv2.imshow('Panorama result with sift:', panorama_SIFT_cropped)
# cv2.waitKey(0)
#
panorama_SURF, panorama_SURF_cropped = panorama(img, "SURF")
# cv2.namedWindow('Panorama result with surf:', cv2.WINDOW_GUI_EXPANDED)
# cv2.imshow('Panorama result with surf:', panorama_SURF_cropped)
# cv2.waitKey(0)

cv2.imwrite("panorama_SIFT.png", panorama_SIFT)
cv2.imwrite("panorama_SURF.png", panorama_SURF)