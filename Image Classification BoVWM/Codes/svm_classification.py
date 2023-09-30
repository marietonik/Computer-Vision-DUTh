# Support-Vector Machines (SVM) classification
# Import libraries
import cv2
import numpy as np
import os
import pandas as pd


# Train an SVM to classify an image based on the type of kernel and epsilon
def train_svm_classifier(training_set, train_class, type_of_kernel, epsilon, svm_path_name):
    """
    training_set: The training set (2D array)
    train_class: The class identifier
    type_of_kernel: The type of kernel that will be used for the SVM
    epsilon: The desired accuracy
    svm_path_name: The path and name of the SVM file that it will be saved

    """

    # Create an empty SVM model
    svm = cv2.ml.SVM_create()
    # Set the type of a SVM formulation
    # The SVM type C_SVC stands for C-Support Vector Classification that means
    # n-class classification (n ≥ 2), allows imperfect separation of classes with penalty multiplier C for outliers
    svm.setType(cv2.ml.SVM_C_SVC)

    # If-statement for setting SVM kernel type
    if type_of_kernel == "RBF":
        # Radial basis function (RBF) kernel
        svm.setKernel(cv2.ml.SVM_RBF)
    elif type_of_kernel == "POLY":
        # Polynomial kernel:
        svm.setKernel(cv2.ml.SVM_POLY)
    elif type_of_kernel == "CHI2":
        # Exponential Chi2 kernel, it is similar to the RBF kernel
        svm.setKernel(cv2.ml.SVM_CHI2)
    elif type_of_kernel == "INTER":
        # Histogram intersection kernel. It is a fast kernel.   [ K(xi,xj)=min(xi,xj) ]
        svm.setKernel(cv2.ml.SVM_INTER)
    elif type_of_kernel == "SIGMOID":
        # Sigmoid kernel
        svm.setKernel(cv2.ml.SVM_SIGMOID)
    elif type_of_kernel == "LINEAR":
        # Linear kernel
        svm.setKernel(cv2.ml.SVM_LINEAR)

    # Termination Criteria
    svm.setTermCriteria((cv2.TERM_CRITERIA_EPS, 100, epsilon))
    # Create an array with labels as much as the images are
    labels = np.array([int((train_class == i)) for i in training_set[:, -1]])
    # Train SVM
    svm.trainAuto(training_set[:, :-1].astype(np.float32), cv2.ml.ROW_SAMPLE, labels)

    # Check if a directory exists (SVMs)
    if not os.path.isdir("./SVMs"):
        # Make directory / mkdir-p
        os.makedirs("SVMs")

    # Save the trained SVM
    svm.save(svm_path_name)

    return 0


# Import functions from feature_extraction.py
from file_3_57807 import *


# One versus All (One-vs-Rest)
# one-vs-All classification
def svm_one_vs_all(svms_path, test_dataset, test_directory, train_parameters, sift):
    """
    svms_path: The path to the directory for SVMs
    test_dataset: A list of the names of classes of the test dataset (is a list of  folders names)
    test_directory: The path of the directory that contains the test dataset
    train_parameters: A tuple with the parameters that will be used to train the SVMs
    sift: SIFT object that will be used by extract_local_features

    """

    # Load vocabulary
    vocabulary = np.load('vocabularies/vocabulary_' + str(train_parameters[0]) + '.npy')
    # Initialize a list named predictions
    predictions = []
    for class_i in range(len(test_dataset)):
        # Define SVM's filename
        svm_filename = 'svm_' + str(class_i) + '_' + str(train_parameters[0]) + 'words_' + '_' \
                       + train_parameters[1] + '_' + str(train_parameters[2])
        if svm_filename in os.listdir(svms_path):
            # Create an empty SVM model
            svm = cv2.ml.SVM_create()
            # Load SVMs
            svm = svm.load(svms_path + svm_filename)
            # Call extract_local_features
            descriptors = extract_local_features(test_directory, sift)
            # Call encode_bovw_descriptor
            bovw_descriptor = encode_bovw_descriptor(descriptors, vocabulary)
            # Prediction
            # cv.ml.STAT_MODEL_RAW_OUTPUT is used for more than one SVMs
            prediction = svm.predict(bovw_descriptor.astype(np.float32), flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
            # Appends an element to the end of the list (list => predictions)
            predictions.append(prediction[1])
        else:
            print("There isn't any file in SVMs with these parameters")

    predicted_class = predictions.index(min(predictions))

    return predicted_class


# Test SVMs using the One vs. All scheme
def test_svms(svms_path, test_dataset, test_directory, train_parameters, sift, result_df):
    """

    :param svms_path: The path to the directory for SVMs
    :param test_dataset: A list of the names of classes of the test dataset (is a list of  folders names)
    :param test_directory: The path of the directory that contains the test dataset
    :param train_parameters: A tuple with the parameters that will be used to train the SVMs
    :param sift:SIFT object that will be used by extract_local_features
    :param result_df: DataFrame of the results for SVM classification

    """

    # Initialize variable for counting the number of images
    number_of_images = 0
    # Initialize variable for counting the number of correctly classified images
    number_of_correct = 0

    for folder, class_i in zip(test_dataset, (range(len(test_dataset)))):
        # Create the path of every class
        folder_path = test_directory + "/" + folder
        # folder_path = os.path.join(test_directory, folder)
        # Create a list containing the names of the entries in the directory given by path
        files = os.listdir(folder_path)
        for file in files:
            # Create the path of every image
            path = folder_path + "/" + file
            # path = os.path.join(folder_path, file)
            # Call svm_one_vs_all
            prediction = svm_one_vs_all(svms_path, test_dataset, path, train_parameters, sift)
            number_of_images += 1
            # Prediction
            if prediction == class_i:
                number_of_correct += 1

            # DataFrame of the results for classification
            result_df = result_df.append(
                pd.Series([path, class_i, prediction, train_parameters[0], train_parameters[1],
                           train_parameters[2]],
                          index=result_df.columns), ignore_index=True)

    # Calculate accuracy
    # The percentage of successful classifications
    accuracy = round(number_of_correct * 100 / number_of_images, 4)

    # Prints
    print("One-Versus-All SVM prediction completed.\n")
    print("Number of vocabulary words =", train_parameters[0])
    print("Type of kernel : ", train_parameters[1])
    print("Epsilon = ", train_parameters[2])
    print("Number of tested pictures: ", number_of_images)
    print("Number of pictures correctly classified: ", number_of_correct)
    print("The success rate is: ", accuracy, "%")
    print("\n\n")

    return result_df


def training_testing(option):
    if option == "Training":
        # Train an SVM for each class, using all of the train features

        # Create SIFT object
        sift = cv2.xfeatures2d_SIFT.create()
        # Train directory
        train_directory = "imagedb"
        # Train dataset
        training_folders = [dI for dI in os.listdir(train_directory) if
                            os.path.isdir(os.path.join(train_directory, dI))]
        # Train features set path
        train_set_path = "train_dbs/"
        # The types of kernel
        # type_of_kernels = ["RBF", "CHI2", "INTER", "SIGMOID"]
        # type_of_kernels = ["RBF", "CHI2"]
        type_of_kernels = ["INTER", "SIGMOID", "LINEAR"]

        # The desired accuracy (epsilon)
        # epsilons = [1.e-08, 1.e-06, 1.e-04]
        # epsilons = [1.e-06, 1.e-04]
        epsilons = [1.e-06]

        for folder, class_i in zip(training_folders, range(len(training_folders))):
            for train_features in os.listdir(train_set_path):
                # The number of visual words that have been used
                number_of_words = int(''.join([s for s in train_features if s.isdigit()]))
                # Create the path of every file (train_features)
                path = os.path.join(train_set_path, train_features)
                training_set_features_path = np.load(path)
                for kernel in type_of_kernels:
                    for epsilon in epsilons:
                        # Set the path and name of the SVM file
                        svms_path = "SVMs/"
                        svm_name = "svm_" + str(class_i) + '_' + str(
                            number_of_words) + "words_" + '_' + kernel + '_' + str(epsilon)
                        svm_path_name = svms_path + svm_name
                        # Call train_svm_classifier
                        train_svm_classifier(training_set_features_path, class_i, kernel, epsilon, svm_path_name)
                        print("SVM with name :", svm_name, "trained.")

    elif option == "Testing":

        # Create SIFT object
        sift = cv2.xfeatures2d_SIFT.create()
        # Test directory
        test_directory = "imagedb_test"
        # Test dataset
        test_folders = [dI for dI in os.listdir(test_directory) if os.path.isdir(os.path.join(test_directory, dI))]

        # DataFrame for SVMs 
        svms_df = pd.DataFrame(columns=['image_path', 'class', 'predicted_class', 'vocabulary_words', 'kernel', 'epsilon'])
        # SVMs path
        svms_path = "SVMs/"

        # The number of visual word TODO: SET THE NUMBER OF WORDS
        # number_of_words = [50, 100, 150, 200, 300, 400, 500, 600, 700]
        number_of_words = [50, 100, 300, 500]

        # The types of kernel
        # type_of_kernels = ["RBF", "POLY", "CHI2", "INTER", "SIGMOID"]
        # type_of_kernels = ["RBF",  "CHI2"]
        type_of_kernels = ["INTER", "SIGMOID", "LINEAR"]
        # The desired accuracy (epsilon)
        # epsilons = [1.e-08, 1.e-06, 1.e-04]
        # epsilons = [1.e-06, 1.e-04]
        epsilons = [1.e-06]

        for word in number_of_words:
            for kernel in type_of_kernels:
                for epsilon in epsilons:
                    svms_df = test_svms(svms_path, test_folders, test_directory, (word, kernel, epsilon), sift, svms_df)

        # Save results to CSV
        svms_df.to_csv("svm.csv")


training_testing("Training")
training_testing("Testing")
