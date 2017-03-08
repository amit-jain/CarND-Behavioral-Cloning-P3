#**Behavioral Cloning Project** 


[//]: # (Image References)

[image1]: ./images/center.png "Center Image"
[image2]: ./images/center_process.png "Center Processed Image"
[image3]: ./images/left.png "Left Image"
[image4]: ./images/left_process.png "Left Processed Image"
[image5]: ./images/right.png "Right Image"
[image6]: ./images/right_process.png "Right Processed Image"
[image7]: ./images/flipped.png "Flipped Image"
[image8]: ./images/flipped_process.png "Flipped Processed Image"
[image9]: ./images/translated.png "Translated Image"
[image10]: ./images/translated_process.png "Translsated Processed Image"
[image11]: ./images/data.png "Data Distribution"
[image12]: ./images/data_trim.png "Data Distribution After Trim"
[image13]: ./images/data_augmented.png "Data Distribution Augmented"
[image14]: ./images/model.png "Model"


---
###Introduction

The objective of the project is to apply deep learning to teach a car to drive autonomously in a driving simulator. 
The simulator contains tracks on which the car can be driven in either testing or autonomous mode. The test track on 
which the project would be evaluated is a simpler track with a few turns and a lot of almost straight driving. There is 
also a challenge track running through mountains which contains lot of sharp turns and sudden gradient change. 

Essentially while training in the simulator camera images would be collected along with control data like steering 
angle, throttle, brake & speed. For the project we only consider the steering angle and while evaluating the 
throttle and speed etc would be preset. There are 3 camera images collected for the frame simulating left, center & 
right cameras.  

My project includes the following files:
* model.py main class that contains the code to create, train and save the model. The function driving_model returns 
the Keras model used
* driving_data.py class that contains code for loading, preprocessing and retrieving the data
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* data_pre_process.ipynb jupyter notebook for data summarization & cleaning
* writeup_report.md or writeup_report.pdf summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

###2. Data

Data for training the model was captured by driving 2-3 laps of center lane driving and then driving in the reverse 
order to have a more varied data as the track 1 has mostly left turns. I also used 2-3 laps of driving on the 
challenge track to train it recognize very sharp turns so that model can generalize better. I also added one lap of 
recovery data where the car was driver off the road the the recovery to the center lane was recorded.

#### Data munging & augmentation
The distribution of the data was overwhelmingly biased towards zero steering angles as seen below 
![image11]

To have a more balanced data, the zero angle data was trimmed aggressively keeping only 0.15% of it which 
made the distribution a little less biased but still almost 3 times the other data. After trimming the data set had 
16344 examples. At this point I divided the data into 70% training and 30% validation set giving a validation data 
set size of 4904 examples.

![image12]

To have more varied data so that the model could generalize well the training data was augmented in the 
following way:
* Using the left and right camera images along with the center images (The steering angles were corrected by adding 
and subtracting 0.1 to the left and right images respectively). The following images shows some data samples of the 3
 types:

![image1]
![image3]
![image5]

* Flipping the images but only for images with non-zero steering angle
![image7]

* Translating the images randomly by upto 100 pixels
![image9]

The data augmentation techniques increased the size of the training set to a decent 95841 exmaples with a much better 
distribution.
![image13]


#### Data pre-processing
The images were also pre-processed:
* Cropping - The original image size was 160 * 320 RGB images and the to reduce noise and remove irrelevant details 
the image size was cropped to 80 * 280
* Normalization - The images were then normalized to be in the range -1 and 1.
The following are the images after cropping for the above samples
![image2]
![image4]
![image6]
![image8]
![image10]


###Model Architecture and Training Strategy

####1. Initial design and experiments

Initially I tried creating a model over a pre-trained VGG with fine tuning. But the model seemed to overfit the data 
and not learn the features that would make a generalized model. Depending on different approached for the top layer 
it would either drive straight through or drive in circles. 

This approach was abandoned because apart for overfitting the model would take quite a lot of time to train the whole 
model even with using bottleneck features and with or without fine tuning in part because there was a lot of 
experimentation done with the data distribution and augmentation techniques. Intuitively, it seemed that the images 
being trained or has far less information to be learnt than the model was able to represent.
 
####2. Finalized model

The model that was finally used is a variant of the model used by 
[Nvidia model](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

The following changes were done to the model to fit the date and generalize well :
* Replace the first convolution layer kernel from being 5 x 5 to 8 x 8 to try to focus attention on the track.
* Added a BatchNormalization layer after the first 2 convolution layers and before the respective activation layers.
* First and second fully connected layers were changed from 100 and 50 to 256 and 100 respectively.
* Added a dropout with 40% keep probability after all the fully connected layers.

BatchNormalization layers and the Dropout layers primarily worked to regularize the model to not overfit.

The model used an Adam optimizer with a mean squared error as the objective function to determine the performance of 
the model.

The following screen shot gives the model summary used for traininga
![image14]

####3. Future Enhancements

* Early Stopping - The model as trained is not perfect and the model starts to overfit if trained for more epochs. So, 
the training was stopped manually when the validation loss started rising again. This should be automated using early
 stopping technique.
* Data - More data particularly clean data for the challenge track would help. But it seems that 0 angles can be 
trimmed further and keeping only 10% of that data.
* Some more experimentation with the final data on the VGG fine tuned model.
* It would be awesome to have visualization for sample images after intermediate layers to get a deeper understanding
 of what the different layers in the model are learning.
  
### Autonomous Driving Results

#### Track One (Left track in simulator)
[![Track One](https://img.youtube.com/vi/garNZVPOXJA/0.jpg)](https://www.youtube.com/watch?v=garNZVPOXJA)

#### Challenge Track (Right Track in simulator)
[![Track Challenge](https://img.youtube.com/vi/tOu-yae-sdU/0.jpg)](https://www.youtube.com/watch?v=tOu-yae-sdU)