# K-Nearest Neighbors (KNN) Classification
# Import Libraries
import cv2
import numpy as np
import os
import pandas as pd


# Calculate the Euclidean Distance between two vectors
def euclidean_distance0(row1, row2):
    # Euclidean Distance = sqrt(sum i to N (x1_i – x2_i)^2)
    distance = np.sqrt(np.sum((row1 - row2) ** 2))
    return distance


# Second way with loop
def euclidean_distance(row1, row2):
    # Initialize distance
    distance = []
    # Euclidean Distance = sqrt(sum i to N (x1_i – x2_i)^2)
    for x in range(len(row1)):
        distance = np.sqrt(np.sum((row1 - row2) ** 2))
    return distance


# Locate the most similar neighbors
def get_k_neighbors(train_dataset, test_row, number_of_neighbors):
    # Initialize a list named distances
    distances = []
    for train_row in train_dataset:
        # Call euclidean_distance
        distance = euclidean_distance(test_row, train_row[:-1])
        # Appends an element to the end of the list (list => distances)
        distances.append((train_row, distance))
    # Sort the list of tuples by the distance (in descending order)
    distances.sort(key=lambda tup: tup[1])
    # Initialize a list named neighbors
    neighbors = []
    for i in range(number_of_neighbors):
        # Appends an element to the end of the list (list => neighbors)
        neighbors.append(distances[i][0])
    return neighbors


# Make a classification prediction with k-neighbors
def prediction_and_classification(train_dataset, test_row, number_of_neighbors):
    # Call get_k_neighbors
    neighbors = get_k_neighbors(train_dataset, test_row, number_of_neighbors)

    neighbor_classes = [row[-1] for row in neighbors]

    prediction = max(set(neighbor_classes), key=neighbor_classes.count)
    # prediction = The predicted class identifier
    return prediction


# Don't use it on test_knn because can not compare with class, because predictions is list
# KNN Algorithm
def knn_classification(train_dataset, test_row, number_of_neighbors):
    # Initialize a list named predictions
    predictions = []
    for row in test_row:
        # Call prediction_and_classification
        prediction = prediction_and_classification(train_dataset, row, number_of_neighbors)
        # Appends an element to the end of the list (list => predictions)
        predictions.append(prediction)
    return predictions


# Import functions from feature_extraction.py
from file_3_57807 import *


# Test KNN
def test_knn(train_set_path, test_directory, test_folders, number_of_neighbors, sift, result_df):
    #  Split the path name into a pair head and tail.
    head, tail = os.path.split(train_set_path)
    if tail not in os.listdir(head):
        print("This dataset does not exist.")

    # Load the encoded training set
    train_set = np.load(train_set_path)
    number_of_visual_words = int(''.join([s for s in train_set_path if s.isdigit()]))

    # Set the path for vocabularies
    vocabulary_path = "vocabularies/vocabulary_" + str(number_of_visual_words) + ".npy"

    # Load vocabularies
    vocabulary = np.load(vocabulary_path)
    # Initialize
    number_of_images = 0
    number_of_correct = 0
    for folder, class_i in zip(test_folders, range(len(test_folders))):
        folder_path = os.path.join(test_directory, folder)
        # Create a list containing the names of the entries in the directory given by path
        files = os.listdir(folder_path)
        for file in files:
            number_of_images += 1
            # Create the path of every image
            path = os.path.join(folder_path, file)
            # Call extract_local_features
            desc = extract_local_features(path, sift)
            # Call encode_bovw_descriptor
            bovw_desc = encode_bovw_descriptor(desc, vocabulary)
            # Call prediction and classification
            prediction = prediction_and_classification(train_set, bovw_desc, number_of_neighbors)
            if prediction == class_i:
                number_of_correct += 1

            # DataFrame of the results for classification
            result_df = result_df.append(
                pd.Series([path, class_i, prediction, number_of_neighbors, number_of_visual_words],
                          index=result_df.columns), ignore_index=True)
    # Calculate accuracy
    # The percentage of successful classifications
    accuracy = round(number_of_correct * 100 / number_of_images, 4)

    # Prints
    print("K-Nearest-Neighbors prediction completed.\n")
    print("Number of visual words :", number_of_visual_words)
    print("Number of neighbors (K) :", number_of_neighbors)
    print("Number of trained pictures : ", train_set.shape[0])
    print("Number of tested pictures : ", number_of_images)
    print("Number of pictures correctly classified: ", number_of_correct)
    print("The success rate is: ", accuracy, "%")
    print("\n\n")

    return result_df


def run_knn(option):
    if option == "run":
        # Run K-Nearest Neighbor for each class, using all of the train features
        # Create SIFT object
        sift = cv2.xfeatures2d_SIFT.create()
        # Train directory
        test_directory = "imagedb_test"
        # Training dataset
        test_folders = [dI for dI in os.listdir(test_directory) if os.path.isdir(os.path.join(test_directory, dI))]
        # Train set path
        train_set_path = "bovw_descs/"
        knn_data_frame = pd.DataFrame(
            columns=['image_path', 'class', 'predicted_class', 'knn_neighbors', 'vocabulary_words'])
        # Set the k
        number_of_neighbors = [2, 3, 5, 7, 9, 11, 17, 25, 35, 40]
        for train in os.listdir(train_set_path):
            for k in number_of_neighbors:
                knn_data_frame = test_knn(train_set_path + train, test_directory, test_folders, k, sift, knn_data_frame)

        knn_data_frame.to_csv('knn.csv')
    else:
        exit(0)


run_knn("run")
