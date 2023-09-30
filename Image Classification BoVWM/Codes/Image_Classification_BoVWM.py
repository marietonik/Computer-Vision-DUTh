import os
import cv2 as cv
import numpy as np


"""
Εργασία 3η - Όραση Υπολογιστών - Μαρία Αρετή Γερμανού - 57807
"""


# Firstly we create a visual vocabulary based on the model Bug Of Visual Words (BOVW). The creation of the vocabulary
# will become with the use of the K - means algorithm and the use of the images database given "imagedb". The BOVW
# model is grouping the key - points of images given in three steps procedure. On the first step a vocabulary is
# created with the characteristics and key - points that are being detected and computed in the same way as the
# previous homework. Similarly we detect the key-points of images and then we compute descriptors of that key-points,
# with the detect and compute functions. The return of those functions will be a signe directional array. Then we
# cluster them with the use of K - means algorithm. So we extract local features for an image using the SIFT algorithm.

def extract_local_features(image_path, sift):
    # print("Extracting local features...")
    image = cv.imread(image_path)
    keypoints = sift.detect(image)
    descriptors = sift.compute(image, keypoints)
    descriptors = descriptors[1]
    return descriptors


# Next i create a database with features from every image from the training data set. Here we load
# the folder of our images and we create our descriptor. Then we store the descriptors in the train_descriptors array.

def create_feature_database(train_directory, train_folders, sift):
    print("Extracting features...")
    # The number of images used for the training
    number_of_images = 0
    # Initialize an array 128-dimensional, because we use SIFT Descriptor
    train_descriptors = np.zeros((0, 128))
    for folder in train_folders:
        # Create the path of every folder
        folder_path = os.path.join(train_directory, folder)
        # Create a list containing the names of the entries in the directory given by path
        files = os.listdir(folder_path)
        for file in files:
            # Create the path of every image
            path = os.path.join(folder_path, file)
            # Call extract_local_features and take Descriptors
            descriptors = extract_local_features(path, sift)
            # Prevent some bugs, because sometimes it takes some noise like "..."
            if descriptors is None:
                continue
            # Do concatenate the train_descriptors (all ready exists) with new descriptors
            train_descriptors = np.concatenate((train_descriptors, descriptors), axis=0)
            number_of_images += 1
    return train_descriptors, number_of_images


# Create a visual vocabulary based on model Bag Of Visual Words (BoVW) using K-means clustering. We store the
# filename of vocabulary, where k is the number of words that it will contain in the filename variable. Then we use
# the function above to create the database of features and we continue with clustering. At last a vocabulary is
# created as a numpy file.

def create_vocabulary(k, train_directory, train_folders, sift):
    # Check if a directory exists Vocabularies
    if not os.path.isdir("./vocabularies"):
        # Make directory
        os.makedirs("vocabularies")
    filename = "vocabulary_" + str(k) + ".npy"
    print(filename)
    # Call create_feature_database
    train_descriptors, number_of_images = create_feature_database(train_directory, train_folders, sift)
    # Termination criteria
    term_criteria = (cv.TERM_CRITERIA_EPS, int(k / 2), 0.1)
    # K-means clustering  to train visual vocabulary using the bag of visual words approach.
    trainer = cv.BOWKMeansTrainer(k, term_criteria, 1, cv.KMEANS_PP_CENTERS)
    # Gives Descriptors to trainer
    vocabulary_cv = trainer.cluster(train_descriptors.astype(np.float32))
    # Save vocabulary on a numpy file
    np.save("vocabularies/" + filename, vocabulary_cv)
    print("The vocabulary", filename, "has been created and saved successfully")


def train_features(train_directory, train_folders, vocabulary_path_tf, sift):
    # Extract the number of visual words from files of vocabulary_path
    number_of_visual_words = [s for s in vocabulary_path if s.isdigit()]
    number_of_visual_words = ''.join(number_of_visual_words)
    # Filename of BoVW encoded descriptors
    filename = "bovw_encoded_descriptors_" + str(number_of_visual_words)
    # Check if a directory exists
    if not os.path.isdir("./bovw_descs"):
        # Make directory
        os.makedirs("bovw_descs")

        vocabulary_tf = np.load(vocabulary_path_tf)
        number_of_images = 0
        bovw_descriptors = np.zeros((0, vocabulary_tf.shape[0] + 1))
        for folder, class_i in zip(train_folders, range(len(train_folders))):
            # Join the train_directory and folder (names)
            folder_path = os.path.join(train_directory, folder)
            # A list containing the names of the entries in the directory given by path
            files = os.listdir(folder_path)
            for file in files:
                # Join the folder_path and the file (names)
                path = os.path.join(folder_path, file)
                # Call extract_local_features
                descriptor = extract_local_features(path, sift)
                # Call encode_bovw_descriptor
                bovw_descriptor = encode_bovw_descriptor(descriptor, vocabulary_tf)
                # DUMB THING
                bovw_descriptor = np.append(bovw_descriptor, [class_i])
                # Increase the dimension of the existing array by one more dimension
                bovw_descriptor = bovw_descriptor[:, np.newaxis]
                # Reshape the array so as to it can be concatenate
                bovw_descriptor = bovw_descriptor.reshape((bovw_descriptor.shape[1], bovw_descriptor.shape[0]))
                # Do concatenate
                bovw_descriptors = np.concatenate((bovw_descriptors, bovw_descriptor), axis=0)
                number_of_images += 1

    # Save the BoVW encoded descriptors ("Trained")
    np.save("bovw_descs/" + filename, bovw_descriptors)
    print("\nNumber of images : ", number_of_images)
    print("Vocabulary : ", vocabulary_path_tf)
    print("Filename : ", filename, "\n")


# Now the encoding procedure is starting. The creation of a BOVW histogram according to vocabulary is starting. The
# vocabulary is loaded and the descriptors are exported but without using the appropriate function of extraction of
# descriptors. Specifically the distance between every descriptor and vocabulary's vector elements is calculated. At
# start we initialize a vector with zeros with size of vocabulary.shape[0]. Then we calculate the euclidean distance
# (L2 norm) and we are summing all rows (axis = 1). Next we take the position of minimum distance of that keypoint
# from visual words and we increase by one the variable that shows the frequency of the element is shown in the
# image. Next i am encoding each image of the data set using BOVW vocabularies. Also zip function is used to create
# and return an object and wraps zip() usage in a list call. Lastly i create and store histograms in a numpy array file.

def encode_bovw_descriptor(descriptors, voc):
    bovw_descriptor = np.zeros((1, voc.shape[0]))
    for descriptor in range(descriptors.shape[0]):
        distances = np.sum((descriptors[descriptor, :] - voc) ** 2, axis=1)
        index_min = np.argmin(distances)
        bovw_descriptor[0, index_min] += 1
    bovw_descriptor = bovw_descriptor / np.sum(bovw_descriptor)
    return bovw_descriptor


# After the encoding of images with the help of the vocabulary, it is time to sort them into our classes. This step
# will be overcame with two algorithmic methods. The first is the K - Nearest algorithmic method and the second one
# is the Support Vector Machine algorithmic method.


# ______________________________________________________________________________________________________________________
#                              Step one: Call of functions that creating the vocabularies.
# ______________________________________________________________________________________________________________________

print("Creating vocabularies...")
# Create SIFT object
sift_voc = cv.xfeatures2d_SIFT.create()
# Train directory
train_directory_voc = "imagedb/"
# Train folders
train_folders_voc = [dI1 for dI1 in os.listdir(train_directory_voc) if
                     os.path.isdir(os.path.join(train_directory_voc, dI1))]

# ______________________________________________________________________________________________________________________
#             Step two: Call of functions that create histograms according to vocabularies found before.
# ______________________________________________________________________________________________________________________

# Create vocabularies with different number of words
for i in range(50, 201, 100):
    # Call create_vocabulary
    create_vocabulary(i, train_directory_voc, train_folders_voc, sift_voc)

# Vocabulary path
vocabulary_path = "vocabularies/"
# Use all of the vocabularies to create train features
for vocabulary in os.listdir(vocabulary_path):
    # Call train_features
    train_features(train_directory_voc, train_folders_voc, vocabulary_path + vocabulary, sift_voc)

# ______________________________________________________________________________________________________________________
