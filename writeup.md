# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/SignVisualization.PNG "Visualization"
[image2]: ./images/TrainDataset.PNG "Train Data Set"
[image3]: ./images/ValidationDataset.PNG "Validation Data set"
[image4]: ./images/TestDataset.PNG "Test Data Set"
[image5]: ./images/bicyclecrossing.jpg "Byclicle Crossing"
[image6]: ./images/childrencrossing.jpg "Children Crossing"
[image7]: ./images/nopassing.jpg "No passing"
[image8]: ./images/roadwork.jpg "Road work"
[image9]: ./images/straightorright.jpg "Straing or Right"
[image10]: ./images/BileCrossingHistogram.PNG "BileCrossingHistogram.PNG"
[image11]: ./images/ChildrenCrossingHistogram.PNG "ChildrenCrossingHistogram.PNG"
[image12]: ./images/NoPassingHistogram.PNG "NoPassingHistogram.PNG"
[image13]: ./images/RoadWorkHisrogram.PNG "RoadWorkHisrogram.PNG"
[image14]: ./images/StraightOrRightHistogram.PNG "StraightOrRightHistogram.PNG"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! 

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used Python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The image below shows sample images of the sign. 

![Signs Samples][image1]

Additionally, below you can see a charts representing the number of samples per class for each data set (Training, Validation and Test respectively) .


![Training Data Set][image2]


![Validation Data Set][image3]


![Test Data Set][image4]



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided initially not to convert images to  grayscale because I thought I could be loosing information even before trying the performance of the Neural Network. 

I normalized the image using the formula (pixel - 128)/ 128

As some of my training classes had less than 200 samples, I decided to generate additional data to help with the the accuracy during the training process. 

I augmented each class of my training dataset to 1000 samples. To add more data to the the data set, I rotated the image by a random amount (<= 15 degrees) and incurred some random noise.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        							| 
|:---------------------:|:-----------------------------------------------------:| 
| Input         		| 32x32x3 RGB image   									| 
| Convolution 5x5 		| 1x1 stride, valid padding, outputs 28x28x6			|
| RELU					|  														|
| Dropout    			| Externally tunable keep_prob							|
| Max pool      		| Size 2x2, strides 2x2, valid padding, outputs 14x14x6	|
| Convolution 5x5		| Stride 1, valid padding. Outputs 10x10x1				|
| RELU   				|														|
| Dropout				| Externally tunable keep_prob							|
| Max pool				| Size 2x2, strides 2x2, valid padding, outputs 5x5x16	|
| Flatten				| Input 5x5x16, output 400								|
| Fully connected		| Input 400, output 120									|
| RELU          		|														|
| Dropout        		| Externally tunable keep_prob							|
| Fully connected		| Input 120, output 84									|
| RELU          		|														|
| Dropout        		| Externally tunable keep_prob							|
| Fully connected		| Input 84, output 43 (labels)							|
 

To compute the loss function supplied to the optimizer, I took the cross entropy of softmax(logits) with the one-hot-encoded labels of the ground truth data. The loss was defined to be the average of the cross entropy across the batch.

The keep_prob parameter was identical for all dropout layers.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used the following parameters to train my model:
EPOCHS = 40
BATCH_SIZE = 128
rate = 0.001
dropout = .75  # keep_prob for dropout layers
dropout =  1.0 for validation and testing.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 96.6%
* test set accuracy of 94.8%

I began with the LeNet architecture from the lab and modified to accept an input of color depth 3 and yield an output of 43 classes instead of 10.
I was unable to get a validation set accuracy greater than 93%, but test set accuracy was as high as 99.9%ish. 

I tryed implementing a dropout layer after each relu activation with keep_prob a tunable hyperparameter. This helped with the overfitting.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Bicycle crossing][image5] 

![No passing ][image6] 

![Straight or right][image7]

![Road work ][image8] 

![Children crossing][image9]

All five images were not quite square. In resizing and interpolating them down to 32x32 squares, their aspect ratios are skewed. This may  prove a challenge for the network.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bicycle crossing		| Road Work   									| 
| No passing  			| No passing 									|
| Straight or right		| Straight or right								|
| Road work      		| Road Work  					 				|
| Children crossing		| Children crossing    							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. Both, the road work sign and the bicycle crossing have the external red triangle which probably confused the network. Also, the bicycle crossing sign was originally under represented in the initial data set. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 20th cell of the Ipython notebook.

Below are some histogram images representing the certainty of the netowork for each of the 5 image. 
![Bicycle Crossing][image10]
![Children Crossing][image11]
![No Passing][image12]
![Road Work][image13]
![Straing or Right][image14]



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


