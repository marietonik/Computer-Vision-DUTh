# Computer-Vision-DUTh
A repository containing all the projects assigned from the "Computer Vision" course in ECE DUTh.
The main tool used is [OpenCV](https://opencv.org/). 

## Text Division and Segmentation Project.
* Median filter for noise removal from images.
* Region Segmentation with dilation and closing techniques.
* Design of bounding box with connected components function.
* Word and region counting as results.

Image Original Data | Resulted Image
:-------------------------:|:-------------------------:
![1_original](https://github.com/marietonik/Computer-Vision-DUTh/assets/53263761/5e91d7d3-11bf-496a-8370-4fef24706e77) | ![bounding_box_colored_words](https://github.com/marietonik/Computer-Vision-DUTh/assets/53263761/13a5fc90-2a3a-4bfa-a42b-2dc722b762d8)

## Panoramic Image Stitching

* Key features detection using Scale Invariant Feature Transform (SIFT) algorithm.
* Key features detection using Speeded up Robust Feature (SURF) algorithm.
* Cross-checking features according to Manhattan distance.
* Homography for image transforms.

Original Photos for the Panorama's creation: |
:-------------------------:|
![original_photos](https://github.com/marietonik/Computer-Vision-DUTh/assets/53263761/d2f420a2-663d-4838-90a6-5ed9a910dfe1) |

Crosschecking Result (SIFT) | Crosschecking Result (SURF)
:-------------------------:|:-------------------------:
![Crosschecking_result_sift](https://github.com/marietonik/Computer-Vision-DUTh/assets/53263761/49a13554-9cee-4163-b9c1-c9928c60824c) | ![Crosschecking_result_surf](https://github.com/marietonik/Computer-Vision-DUTh/assets/53263761/cbe9c946-b631-4a20-8546-3bdbbd76b1b5)

## Image Classification using Bag of Visual Words Model
* Using part of Caltech-256 Dataset.
* Creating Histograms and Descriptors.
* Creating Dictionaries.
* Extraction of local features.
* Training a BoVW model using K-Means.
* Classification with Support Vector Machines (One versus all).
* Classification with K-Nearest-Neighbors.

## Image Classification using Convolutional Neural Networks (CNN)
* Using part of Caltech-256 ![Dataset](https://drive.google.com/file/d/1w1WfTNCuHY-O7z-8exhm-l6XQ4W00xZa/view?usp=drive_link).
* Different Deep Neural Network architectures
* Data Augmentations
* Pre-trained architectural model, VGG.

Contributor: Associate Professor [Lazaros Tsochatzidis](https://github.com/lazatsoc) year 2022.
