**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car_image]: ./output_images/car1.png
[notcar_image]: ./output_images/notcar1.png
[hog_image]: ./output_images/hog.png
[sliding_image]: ./output_images/sliding.png
[heatmap_image]: ./output_images/heatmap.png
[video1]: ./output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the third and fourth code cells of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][car_image] ![alt text][notcar_image]

I then explored different color spaces. I found out that using YCrCb color space leads to the best results. At first I tried `skimage.hog()` function, but I found that `cv2.HOGDescriptor` runs 10 times faster. According to OpenCV documentation it supports only (8,8) cell size and (16,16) block size. So I only tried different nbins and chose 9 orientations.

I grabbed random images from each of the two classes and displayed them to get a feel for what the hog functions output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][hog_image]

#### 2. Explain how you settled on your final choice of HOG parameters.

With cv2.HOGDescriptor I had only one parameter to choose. I tried different values for nbins and saw how it works on my test set.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in cells 5-7 of the IPython notebook. I tried SVM with different kernels and parameters. For choosing parameters of SVM I used RandomizedSearchCV. I found out that SVM with linear kernel and C==0.0001 performs best for this project. Then I switched to cv2.ml.SVM with the same parameters, because it works 10 times faster.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is contained in cells 8-9 of the IPython notebook. My sliding window starts from y==400. I run it with 3 scales: 1, 1.5, 2. For scales 1 and 1.5 step size is 1 cell, for scale 2 step size is 2 cells. I also added bottom border for each scale: 524, 588, 652. I chose such scales so that the whole cars could be contained in a window on each level.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.

To improve efficiency of my pipeline I computed HOG features and spatially binned color once per scale. `cv2.HOGDescriptor.compute` returns HOG features for all windows of an image with cell step size, so after proper reshaping I could use HOG sub-sampling. This was tricky part of the project, because I couldn't find any documentation on the format. So I had to look at the source code of OpenCV.  

Here are some example images:

![alt text][sliding_image]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this step is contained in cell 10 of the IPython notebook. I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes:

![alt text][heatmap_image]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

On some frames my implementation detect cars in the opposite directions. I couldn't find any restrictions in the rubric for this case. So I decided that more cars detected is actually better. To eliminate this problem I could add one more class in my training set with cars in the opposite direction and may be choose better classifier.

I achieved more than 99% accuracy with SVM classifier. But it's still far from ideal. I could try to add more examples to training set and try to use some neural network architectures.

Using cv2.HOGDescriptor, cv2.ml.SVM and HOG sub-sampling leads to quite good performance. I achieved 4 frames per second on my laptop.