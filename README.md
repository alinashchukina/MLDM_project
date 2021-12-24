# MLDM_project

## Petfinder task

Our team participates in the [Petfinder competition](https://www.kaggle.com/c/petfinder-pawpularity-score/overview/description). The task is to determine how cute is the photo of an animal on a scale from 1 to 100. This mark will help people to make nicer photos of stray cats and dogs, which will increase the chance of adoption of an animal.

## Data and metrics

Each observation includes one image - a photo of an animal, some metadata of this photo - such as flag, that a photo is focused or blurred, flag, that both eyes of an animal can be seen in a photo, e.t.c., and finally the target - Pawpularity Score - number from 1 to 100, where 100 represents very cute and lovely photo of an animal an 1 represents not nice photo at all. There are 9912 images in train dataset and 8 generated images in test dataset.

The metrics is RMSE.

## Our experiments

### Main model

For our main model we build a convolutional neural network on images, which predicts target divided by 100. This is a regression model, which directly predicts target. This model achieves RMSE=21, which is comparable with the current best score on leaderboard.

For training and validating the model we use only the train non-generated data, which we divide 90% train and 10% on validation. We preprocess all data resizing images to the size 512 x 512 and normalizing them with ImageNet mean and std. We divide target by 100 getting them in the interval (0, 1] to eliminate problems with large weights and instability in training. We take pretrained GoogLeNet, where we replace last linear layer on linear layer with 1 output. We minimize MSE loss with Adam. We also decrease learning rate each 60 batches by 10 times, batch size is 24. The model is trained on GPU.

### Classification

We proposed a hypothesis that one-hundred-point scale is too large and difficult for people to estimate photo and there is no visible difference between for example 75 and 76. So we made an experiment turning the task into classification task. We splitted one-hundred-point scale into 10 classes: 1st class for images which have target value between 1 and 10, 2nd - between 11 and 20, ..., 10th - between 91 and 100. We trained model to predict classes, and then we transformed classes back to pawpularity score: we assigned to each class the mean of its boundaries. The 1st class got pawpularity 5, 2nd - 15, ..., 10th - 95. If the model classifies images perfectly into these 10 classes, then the RMSE will not be larger then 5. Unfortunatelly, we got a bit worse result, about 25. But still, we plan to refine and finalise this experiment, because it showed some potential - the score is still quite close to regression result.

### Other experiments

We made some other experiments, such as throwing out images with more than 90 points, because the center of scores distribution was near 20-40 points and images with 100 points seemed as outliers. Though, it didn't give us a breakthrough in score. We also tried to take logarithm instead of dividing by 100 in our regression solution, but it also didn't give any significant increase in quality. We tried to add metadata to last layer of the network but it also didn't work.

We tried other models, such as Resnet-18 and -34, Mobilenet and some others, but it seems that they are not deep and complicated enough to capture such difficult characteristic as pawpularity. We also experimented with optimizer and learning rate sheduler.

### Plans to do

We plan to refine main regression model and classification model by adding early stopping, adding metadata in more complex ways (e.g. with some linear layers or preprocessed in some way), tuning parameters more and providing new hypothesis on the process of photo estimation.

## Our team

Alina Shchukina, https://github.com/alinashchukina

Maria Filippova, https://github.com/mfilippova
