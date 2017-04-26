**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/visualized_images.png "Visualization"
[image2]: ./images/train-set-distribution.png "Visualization"
[image3]: ./images/validation-set-distribution.png "Visualization"
[image4]: ./images/test-set-distribution.png "Visualization"
[image5]: ./images/sharpen-grayscale.png "Traffic Sign 2"


[image6]: ./1.png "Traffic Sign 3"
[image7]: ./2.png "Traffic Sign 4"
[image8]: ./3.png "Traffic Sign 5"
[image9]: ./5.png "Grayscaling"
[image10]: ./6.png "Random Noise"
[image11]: ./8.png "Traffic Sign 1"
[image12]: ./9.png "Traffic Sign 1"
[image13]: ./10.png "Traffic Sign 1"
[image14]: ./images/probabilites.png "Traffic Sign 1"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/parametersky/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is (34799, 32, 32, 3)
* The size of the validation set is (4410, 32, 32, 3)
* The size of test set is (12630, 32, 32, 3)
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is composed of 

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because for sign classifier shape is important than color.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image5]

As a last step, I normalized the image data because image data is around 80 and after normalized mean value are betwween (-1,1)



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 3x3	    | 1x1 stride, outputs 10x10x30    									|
| Max pooling	      	| 2x2 stride,  outputs 5x5x30 				|
| Flatten	      	|   outputs 750 				|
| Fully connected		| outputs 300        									|
| RELU					|												|
| Fully connected		| outputs 240        									|
| RELU					|												|
| Fully connected		| outputs 43        									|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used AdamOptimizer with learning rate that is 0.001, batch size 128, epochs 20. mu  0,  sigma 0.07

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

first I just use Lenet network and origin image files to train the model and get about 55% accuracy. I change the output demension in Lenet layer as image sign has 43 classes which is more than 10 classes which Lenet is used to classify. I also try to preprocess data before train model. I tried sharpen image, graysclale image and find that grayscale has magnifficent improvement on accuracy. also I normalize the training data with (data-128)/128.0. at last I tune sigma in LeNet to 0.007 which saves several epochs on finding appropriate weight.

My final model results were:
* training set accuracy of 99%
* validation set accuracy of 94%
* test set accuracy of 92.5%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 I choose LeNet because classifying number and classifying traffic sign are similar. they have the same classify progress. after preprocessing training data, I get a accurracy above 93% and successfully classify images download from internet.

### Test a Model on New Images


Here are eight German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] ![alt text][image12] 
![alt text][image9] ![alt text][image10] ![alt text][image11] ![alt text][image13] 


#### Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

for the images I run test on, there is no specail difficulty here. the images are bright and clear enough except that some of them are rotated by some degrees but this should not cause any difficulty in identify them.
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection     		| Right-of-way at the next intersection   									| 
| Speed limit (30km/h)     			| Speed limit (30km/h) 										|
| Priority road					| Priority road											|
| Keep right	      		| Keep right					 				|
| Turn left ahead			|  Turn left ahead      							|
| General caution			|  General caution      							|
| Road work			|  Road work      							|
| Speed limit (60km/h)			|  Speed limit (60km/h)      							|

The model was able to correctly guess 8 of the 8 traffic signs, which gives an accuracy of 100%. 

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.
at last the softmax probabilities for each of the images are listed blow
![alt text][image14] 

