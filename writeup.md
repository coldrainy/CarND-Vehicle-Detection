**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4


### Organization

The organization of the code is that the function like get_hog_feature and draw boxes and so on are in the function.py.And the hog 

feature of the trained images were extracted in trainSVM.py as well as where the SVM classifier was traind.The findcar.py process the 

video and draw the boxes which locate the cars in the video.

### Histogram of Oriented Gradients (HOG)

The code for this step is contained in lines 76 through 87 of the file called function.py.And the function was called in lines 39 through 50 in the

file called trainSVM.py I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![car_and_nocar](/output_images/car_and_nocar.png)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).

I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![hog_feature](/output_images/hogfeature.png)

####  1.final choice of HOG parameters.

Well in fact I only used two sets of parameters.One is provided by the udacity in the course with `orient=9`,`pixels_per_cell=(8, 8)`,

`cells_per_block=(2, 2)`.But the result is not good.I changed the paramters to `orient=15`,`pixels_per_cell=(8, 8)`,and I also set the 

`cells_per_block` adjustable with the start and stop pixel in y direction.

#### 2.classifier trained 

I trained a linear SVM using hog features of all channels combined color features of color histogram and spacial feature.It is in lines 26 through 70 in the trainSVM.py file.

### Sliding Window Search

#### 1. sliding window implement

This part of code was wrote in lines 16 through 82 in findcar.py file.Before deciding the scales and the range of y direction.I observed the test image 

and project video carefully.I find that the car in the middle range almost occupy 64x64 pixels.The further car occupy about half of that.And they are all 

in the range of 400-650 in the y direction.So I make the scales ranges from 1 to 2.And the y ranges from 400 to 650.
![sliding_windows](/output_images/sliding_windows.png)

#### 2. result of the test images
Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![result](/output_images/result.png)
---

### Video Implementation

#### 1. video result
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Filter

I recorded the positions of positive detections in most frame of the video.From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. Besides,I accumulate the heatmaps about 10 frames one time.And then use the threshold again to filter the false positives.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.This part of the code was in the `process_image` function in lines 85 through 137 in the findcar.py file.

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Well,My pipeline still have a lot false positives.And it's a little sensitive to the light and size in the picture.I think the deep learning can help to make it better.
