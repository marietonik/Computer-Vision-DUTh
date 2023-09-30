import cv2
import numpy as np

"""
Εργασία 1η - Όραση Υπολογιστών - Μαρία Αρετή Γερμανού - 57807
"""

# Creating lists of images and further down using variables too:
filenames_org = []
filenames_ns = []
img = []
shape = []
img_grayscale = []

i = 0

for i in range(1, 6, 1):
    filenames_org.append('images/' + str(i) + '_original.png')
    filenames_ns.append('images/' + str(i) + '_noise.png')
print(filenames_org)

# my list of grayscale images:
for i in range(1, 6, 1):
    temp_org = cv2.imread(filenames_org[i - 1], cv2.IMREAD_GRAYSCALE)
    temp_ns = cv2.imread(filenames_ns[i - 1], cv2.IMREAD_GRAYSCALE)
    img.append(temp_org)
    img.append(temp_ns)

for i in range(1, 11, 1):
    shape.append(img[i - 1].shape)
    print('Original Dimensions: ', shape[i - 1])

# Re-assigning the files:
filenames_org_1 = 'images/1_original.png'
filenames_org_2 = 'images/2_original.png'
filenames_org_3 = 'images/3_original.png'
filenames_org_4 = 'images/4_original.png'
filenames_org_5 = 'images/5_original.png'

filenames_ns_1 = 'images/1_noise.png'
filenames_ns_2 = 'images/2_noise.png'
filenames_ns_3 = 'images/3_noise.png'
filenames_ns_4 = 'images/4_noise.png'
filenames_ns_5 = 'images/5_noise.png'

img_o_1 = cv2.imread(filenames_org_1, cv2.IMREAD_GRAYSCALE)
img_o_2 = cv2.imread(filenames_org_2, cv2.IMREAD_GRAYSCALE)
img_o_3 = cv2.imread(filenames_org_3, cv2.IMREAD_GRAYSCALE)
img_o_4 = cv2.imread(filenames_org_4, cv2.IMREAD_GRAYSCALE)
img_o_5 = cv2.imread(filenames_org_5, cv2.IMREAD_GRAYSCALE)
img_n_1 = cv2.imread(filenames_ns_1, cv2.IMREAD_GRAYSCALE)
img_n_2 = cv2.imread(filenames_ns_2, cv2.IMREAD_GRAYSCALE)
img_n_3 = cv2.imread(filenames_ns_3, cv2.IMREAD_GRAYSCALE)
img_n_4 = cv2.imread(filenames_ns_4, cv2.IMREAD_GRAYSCALE)
img_n_5 = cv2.imread(filenames_ns_5, cv2.IMREAD_GRAYSCALE)

# Assign a colored image for testing later:
img_colored = cv2.imread(filenames_org_1)
# using filter mean or median for de-noising grayscale images with opencv:
median = cv2.medianBlur(img_n_1, 3)
# cv2.namedWindow('median with open cv', cv2.WINDOW_NORMAL)
# cv2.imshow('median with open cv', median)
# cv2.waitKey(0)
cv2.imwrite('org_median_denoised.png', median)


# using filter mean or median for de-noising grayscale images without opencv:
# firstly we use a function that forms a border around an image and applies the respective padding.
# Then we define our window size and its half.
# Constant value border: Applies a padding of a constant value for the whole border and its color is chosen randomly.
def median_filter(img_noise):
    # Check if image is loaded fine
    # if img_noise is None:
    #     print('Error opening image!')
    #     return -1

    # Creating a copy of image:
    img_noise_2 = np.copy(img_noise)

    rows = img_noise.shape[0]
    cols = img_noise.shape[1]

    for k in range(rows - 2):
        for j in range(cols - 2):
            temp = [img_noise[k][j], img_noise[k][j + 1], img_noise[k][j + 2], img_noise[k + 1][j],
                    img_noise[k + 1][j + 1], img_noise[k + 1][j + 2],
                    img_noise[k + 2][j], img_noise[k + 2][j + 1], img_noise[k + 2][j + 2]]
            temp.sort()
            img_noise_2[k + 1][j + 1] = temp[4]
    return img_noise_2


# test result with function median:
# due to high complexity the call of function and the result will be saved and we will use the cv function to move on

# my_median = median_filter(img_n_1)
# cv2.imwrite('my_median_denoised.png', my_median)
# cv2.namedWindow('median without open cv', cv2.WINDOW_NORMAL)
# cv2.imshow('median without open cv', my_median)
# cv2.waitKey(0)

# Taking the resulting gray image from the de-noising above and transforming it to binary image.
# We use Otsu's Binarization that chooses a value and determines it automatically:
flag_otsu = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU

thres1, th1 = cv2.threshold(median, 0, 255, flag_otsu)
thres2, th2 = cv2.threshold(img_o_1, 0, 255, flag_otsu)

# Also i save the resulted images and view the threshold binary image with and without noise
cv2.imwrite('org_bin_otsu.png', th1)
cv2.imwrite('ns_bin_otsu.png', th2)
avg_thres = (thres1 + thres2) // 2
print("Automatically chosen average value from Otsu's Binarization: ", avg_thres)

# img_threshold_ns = cv2.namedWindow("Filtered Binary with Otsu's method: noise", cv2.WINDOW_NORMAL)
# img_threshold_org = cv2.namedWindow("Filtered Binary with Otsu's method: original", cv2.WINDOW_NORMAL)

# cv2.imshow("Filtered Binary with Otsu's method: noise", th1)
# cv2.imshow("Filtered Binary with Otsu's method: original", th2)
# cv2.waitKey(0)

# At this point its time to choose transforms to count words and divide areas of images. A combination of open and
# close transforms is used to connect the areas and words. Dilate and erosion consist the previews transformations.
# Structuring elements are being created to connect the areas by dilation, but this action will change the sizes of
# the areas and results will be wrong. So firstly words are being connected with each other. Also the closing next is
# used to fill vertically gaps of letters.

sqr_1 = np.ones((3, 3), np.uint8)
sqr_2 = np.ones((5, 5), np.uint8)
rect_1 = np.ones((1, 4), np.uint8)

# For text area dilation and closing:
temp2 = cv2.morphologyEx(th1, cv2.MORPH_DILATE, sqr_1, iterations=1)
texts_bound = cv2.morphologyEx(temp2, cv2.MORPH_CLOSE, sqr_2, iterations=8)

# cv2.namedWindow("text_division", cv2.WINDOW_NORMAL)
# cv2.imshow("text_division", texts_bound)
# cv2.waitKey(0)

# For words dilation and closing:
temp = cv2.morphologyEx(th1, cv2.MORPH_DILATE, rect_1, iterations=3)
words_bound = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, rect_1, iterations=1)

# cv2.namedWindow("word_division", cv2.WINDOW_NORMAL)
# cv2.imshow("word_division", words_bound)
# cv2.waitKey(0)

cv2.imwrite('text_division.png', texts_bound)
cv2.imwrite('word_division.png', words_bound)

# The connected components have being formed and the function is used to return the number of the components found.
num, val_pix = cv2.connectedComponents(texts_bound)

# The bounding boxes and measures to be done will be performed according to the number of components found. To design
# the bounding boxes a function is used that stores the coordinates of the box. Through an array of zeros we create a
# mask which later will be helpful to calculate the dimensions of bounding box. An integral image is created in which
# every pixel is sum of its neighbors to the upper left. Also every pixel of looped area is set to 255. Height and
# width is checked with a limit - value, before we move on to the next loop because areas often are consisted by
# higher values than the limit we chose. This check is happening so trash - information will not be considered
# countable region
img_integral = cv2.integral(median, cv2.CV_32F)

for i in range(1, num):
    counter = 0
    text_regions_msk = np.zeros(th1.shape, dtype=np.uint8)
    text_regions_msk[val_pix == i] = 255
    x, y, w, h = cv2.boundingRect(text_regions_msk)

    # If height and width of bounding rectangles are small enough they are consider regions and we increase the counter.
    if w < 12 or h < 12:
        counter += 1
        continue
    num_reg = i - counter
    print("----- Region number", str(num_reg), ": -----")

    # Drawing a rectangle with dimensions from above unfilled with red colored line around it.
    # Also a text number is shown on every region.
    color = (0, 0, 255)
    cv2.rectangle(img_colored, (x, y), (x + w, y + h), color, 4)
    cv2.putText(img_colored, str(num_reg), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)

    # Calculating the bounding box area with the dimensions of width and height.
    box_area = w * h
    print("Bounding Box Area (px): ", box_area)

    # Calculating the number of words horizontally and vertically: from one point to width and from other point to
    # height. With the parameters of the words' bounding box and the words counting area, the number of words is
    # calculated. Also the possible errors made during the collision of the words earlier can be detected by giving
    # one color at each word, this will not show up in this program. Further down the number of words displayed must
    # be decrease by one cause the background has also a label that is counted in the variable.

    num_word, words = cv2.connectedComponents(words_bound[y:y + h, x:x + w])
    # For loop below used only for bounding words and with words_bound whole indexes:
    # for wrd in range(1, num_word):
    #     word_regions_msk = np.zeros(th1.shape, dtype=np.uint8)
    #     word_regions_msk[words == wrd] = 255
    #     x_w, y_w, w_w, h_w = cv2.boundingRect(word_regions_msk)
    #     cv2.rectangle(img_colored, (x_w, y_w), (x_w + w_w, y_w + h_w), (255, 0, 0), 3)

    print("Number of Words: ", num_word)

    # Calculating text area:
    text_area_box = th1[y:y + h, x:x + w]
    text_area = 0
    for y_i in range(1, text_area_box.shape[0]):
        for x_i in range(1, text_area_box.shape[1]):
            if text_area_box[y_i][x_i] == 255:
                text_area += 1

    print("Text Area: ", text_area)
    # Calculating Mean gray-level value in bounding box:
    A = img_integral[y][x]
    D = img_integral[y + h][x + w]
    B = img_integral[y][x + w]
    C = img_integral[y + h][x]
    mglv = (A + D - B - C) / (w * h)
    print("Mean Gray Value of bounding box: ", mglv)

cv2.imwrite('bounding_box_colored.png', img_colored)
# cv2.namedWindow("Bounding box design result colored", cv2.WINDOW_NORMAL)
# cv2.imshow("Bounding box design result colored", img_colored)
# cv2.waitKey(0)

# All the images have been saved in a folder combined with the program
