## Writeup Template

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/examples.png
[image2]: ./output_images/figure_1.png
[image3]: ./output_images/img0.png
[image4]: ./output_images/img1.png
[image5]: ./output_images/img1.png
[image6]: ./output_images/img1.png
[image7]: ./output_images/img1.png
[video1]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

The code for this project is contained in detection.py.

### Feature Identification

I started by reading in all the `vehicle` and `non-vehicle` images from the KITTI_extracted and Extras subsets. These are approximately the same size, which prevents overlearning on one type of example, and does not have the repeated images of the other sets so associated problems are avoided.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I examined these images and found there were 5966 car images and 5068 non-car images of size:  (64, 64, 3)  and data type: float32.

The next step was to implement feature extraction.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The main feature extraction code for this step is contained in the function get_hog_features at line 87. This function is based on the function presented in lessons and uses skimage.feature.hog with the following parameters:

* colourspace = 'YCrCb'
* orientation bins = 5
* pix_per_cell = 10
* cell_per_block = 2
* image channels to use = "ALL"

These parameters were chosen based on experimentation using the relevant lesson code and my own. The factor which was most effective was the colour space - changing from RGB to YCrCb provided an increased SVC accuracy of almost 10%.

get_hot_features is called by extract_features which flattens the output into a feature vector.

#### 2. Explain how you settled on your final choice of HOG parameters.

Manual tuning and experimentation with test sets yielded SVM classification accuracies approaching 100% with just HOG alone using these parameters, so I felt justified in using them for my feature extraction pipeline.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVC using not only HOG features but also colour histogram features and spacial binning of colour. The parameters and implementation of these features were simpler and easy to optimise manually. The code for extracting these features is found in the functions color_hist and bin_spatial, based on the code from lessons. The SVC training occurs between lines 612-643. I create an array stack of feature vectors, scale these features using the Standard Scaler and define a labels vector corresponding to the feature vector. Next, I shuffle and split the data into a training and test set, create a linear SVC and train it. I also save the the trained classifier and the scaler for re-use.

I also performed grid search on the parameter space to verify that a linear SVC was the best option, which turned out to be the case compared to the rbf kernel.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I did a lot of experimentation with window sizes, where to limit window search to, and window size scaling. I decided on 3 scales of window, with the smallest near the horizon and the largest near the bottom of the image. I found a 75% overlap made the heat map easier to threshold because many windows would stack on a correct classification. The result of such a search are shown below (this image is tuned slightly differently due to different horizon position. I found sky would frequently classify as a car, so horizon position is quite important.

![alt text][image2]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

![alt text][image4] ![alt text][image5] ![alt text][image6]

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I maintained a history of the last three heat maps and summed them, then used a higher threshold on the the result. This meant that detections which persisted over several frames were more likely to be detected and those which did not were more likely to be filtered out. Compare test_video_out and test_video_out2 for examples with and without this improvement.

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My code was very slow because I did not implement HOG subsampling. This could certainly be improved, especially given the high overlap of my window sampling. Sky was frequently detected as a car, so this implementation is sensitive to bounding the window search below the horizon. Further improvements might be found by using a neural network to detect cars rather than an SVC.
