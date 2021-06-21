# Final Report
**MDS 2021 Capstone Project with the Gerontology Diabetes Research Lab (GDRL)**

By: Ela Bandari, Lara Habashy, Peter Yang and Javairia Raza

## Executive Summary
Subclinical lipohypertrophy is traditionally evaluated based on visual inspection or palpation. Recent work has shown that lipohypertrophy may be detected by ultrasound imaging (Kapeluto et al., 2018). However, the criteria used to classify lipohypertrophy using ultrasound imaging is only familiar to and implemented by a small group of physicians (Madden, 2021). In an effort to improve the accessibility and efficiency of this method of detection, we have developed a supervised machine learning model to detect lipohypertrophy in ultrasound images. 

To develop our machine learning model, we tested a variety of image augmentation techniques and ultimately decided on augmenting our dataset by adding random flipping, contrast and brightness. We then fit a variety of pre-existing model architectures trained on thousands of images to our augmented dataset. We optimized the parameters by which the model learned and then carefully examined the different models’ performance and fine tuned them to our dataset before selecting the best performing model. Our final model accurately classified 76% of unseen test data. We have made our model accessible to potential users via a [web interface](https://share.streamlit.io/xudongyang2/lipo_deployment/demo_v2.py). Our work has demonstrated the potential that supervised machine learning holds in accurately detecting lipohypertrophy; however, the scarcity of the ultrasound images has been a contributor to the model’s shortcomings. We have outlined a set of recommendations for improving this project; the most notable of which is re-fitting the model using a larger dataset.


## Introduction
Subclinical lipohypertrophy is a common complication for diabetic patients who inject insulin. It is defined as the growth of fat cells and fibrous tissue in the deepest layer of the skin following repeated insulin injections in the same area. It is critical that insulin is not injected into areas of lipohypertrophy as it reduces the effectiveness of the insulin. Fortunately, research by Kapeluto et al. (2018)  has found ultrasound imaging techniques are more accurate in finding these masses than a physical examination of the body by a healthcare professional. But, currently, the criteria to classify lipohypertrophy using ultrasound imaging is only implemented by a small group of physicians. To expand the usability of this criteria to a larger set of healthcare professionals, the capstone partner is interested in seeing if we can leverage supervised machine learning techniques to accurately classify the presence of lipohypertrophy given an ultrasound image.

```{figure} image/example.png
---
height: 500px
name: example
---
Some examples of images found in our dataset. Top row is negative (no lipohypertrophy present) and bottom row is positive (lipohypertrophy present) images where the yellow annotations indicate the exact area of the mass. 
```

Our current dataset, as shown in Figure 2 (below), includes 218 negative images (no lipohypertrophy present) and 135 positive images (lipohypertrophy present) which is typically considered to be a very small dataset for a deep learning model.Thus, our specific data science objectives for this project include:

1) Use the provided dataset to develop and evaluate the efficacy of an image classification model.

    1.1 Data Augmentations
  
    1.2 Transfer Learning
  
    1.3 Optimization of Learning
  
    1.4 Object Detection
  
2) Deploy the model for a non-technical audience.



```{figure} image/counts_df.png
---
height: 100px
name: counts_df
---
The total number of positive and negative examples in our dataset.
```

 
## Literature Review

Throughout the project, the team has consulted literature to gain insight and direction on current practices relevant to our problem and our dataset to guide and validate our decision-making. Please see Appendix 1 for our literature review.

## Data Science Methods 

3.1 Data Preparation

Given our small dataset, image augmentation techniques are used to expand the size of the training set and ideally make the model more generalizable. We explored the pytorch and the albumentations library for augmentation as they proved to improve accuracy (Al-Dhabyani et al., 2019). We also attempted to leverage a machine learning tool called autoalbument from the latter library that finds the best transformations for you. However, following several runs, the accuracy did not improve from baseline. We suspect it may be that the transformations were too complex for the model to learn anything significant. Using simpler transformations to augment the data was complemented by results found in our literature review(Esteva et al., 2017;Loey et al., 2020). A variety of classic transformations were tested and the model’s performance on these augmented datasets were documented here. The transformations that led to the best performance were adding random vertical and horizontal flipping with a probability of 50% along with random brightness and contrast.

```{figure} image/transformed_image.png
---
height: 200px
name: transformation
---
Image transformation.
```

3.2 Model Development 

The augmented data is then used to train a CNN model using transfer learning. There are many popular architectures for transfer learning models that have proven successful as discussed in our literature review. As such, we investigated the following structures: VGG16, ResNet, DenseNet and Inception. Each of the models were incorporated with our small dataset, trained in separate experiments utilizing techniques to optimize the parameters of the model to maximize its ability to learn. To compare the performance of the different architectures, the team considered the accuracy and recall scores of the models when tested on our test set. 


```{figure} image/model_acc_bar_chart.png
---
height: 150px
name: acc_chart
---
The results of accuracy.
```

```{figure} image/model_recall_bar_chart.png
---
height: 150px
name: recall_chart
---
The results of recall.
```


The model that utilizes the densenet architecture appears to be the best performing model for our dataset. Furthermore, we conducted a manual analysis of tricky negative and positive images to test model performance using an interactive interface. This application allowed us to compare how confident the models were when misclassifying cases and showcased how well DenseNet was doing. 

In an effort to reduce the high confidence levels for misclassifications, a problem known as overfitting, we considered adding dropout layers in our CNN structure. In this case, overfitting occurs as a result of having trained our models on such a small training set. Another way to reduce overfitting is to average the predictions from all four models, called an ensemble. However, an ensemble would not be feasible as it requires lots of resources such as enough CPU to train the models. Although the dropout layers did manage to reduce overfitting in the models, the overall reduced accuracy was not significant enough to implement those layers in our optimal model. To see this exploration, click here.

Furthermore, as our capstone partner was far more interested in reducing false negatives, areas where insulin should not be injected, we investigated various techniques to optimize recall. Positive weight (pos_weight) is one argument of the loss function used in our CNN model which, if greater than 1, prioritizes the positive examples more such that there is a heavier penalization (loss) on labelling the positive lipohypertrophy examples incorrectly. However, this method was not implemented in our final model as it proved to be unstable. After conducting a few experiments, we found high variance in the results. 
Finally, the choice of our optimal model as DenseNet was further motivated by its size and compatibility with our data product, further discussed in the data product section.

```{figure} image/size_df.png
---
height: 250px
name: size_df
---
Size comparison.
```

Our next objective was to implement object detection into our pipeline, giving us the ability to identify the exact location of lipohypertrophy on unseen test images. To implement object detection using a popular framework called YOLO, first, using the annotated ultrasound images, the team created bounding boxes around the location of the lipohypertrophy masses in the positive training images. Next, using the YOLOv5 framework, the Yolov5m model was trained for approximately 200 epochs with an image size of 300 and a batch size of 8. The team also experimented with different training parameters to find the optimal ones and found the former specifics optimal.

```{figure} image/object_detect_example.png
---
height: 500px
name: obj_det
---
Object Detection.
```

## Data Product and Results 

The first deliverable of the data product is the source code including an environmental file (.yml), command-line enabled python scripts to perform train-validation-test split on image folders, modeling, evaluation, and deployment, and a Makefile that automates the entire process. The team expects our partner to use the source code to refit the lipohypertrophy classification model as more ultrasound images become available. The python scripts are built with optional arguments for model hyperparameters, which makes this a flexible tool to work with. The Makefile renders the process automatic and makes the model refitting effortless. However, the source code requires a certain degree of python and command line interface knowledge.

The second deliverable is a web application for our partner to interact with the lipohypertrophy classification and object detection models. The app is deployed on Streamlit share and Heroku, free-service platforms. The team expects our partner to use this app to upload one or multiple cropped ultrasound images, select one of them with the select box, and get an instance prediction of lipohypertrophy presence and prediction confidence. Finally, if it is a positive prediction, a red box will appear on the image to indicate the proposed lipohypertrophy area. This web app has good accessibility as our partner can interact with it without IT clearance issues, thus making it easy to work with. However, the final models are hosted on Streamlite share after the team put in a special request to get more resources. After the Streamlit granted additional resources, it proved to be much faster than Heroku. Future (larger) models could be hosted on cloud based servers since they are more flexible and secure. The team has also considered creating a desktop app for Lipohypertrophy classification. A desktop app allows our partner to interact with the model without the need of internet access, and has low maintenance cost. However, downloading it from the internet requires internal IT clearance and this method was deemed less preferable by our partner. 

```{figure} image/demo_gif_v2.gif
---
height: 300px
name: web_app
---
Web app.
```

## Conclusions and recommendations

In this project, we aimed to investigate whether supervised machine learning techniques could be leveraged to detect the presence of lipohypertrophy in ultrasound images. Our trained models have demonstrated that this is indeed a possibility as they are accurately predicting the presence of lipohypertrophy on 76% of previously unseen data. Furthermore, our team has developed two data products. The data products are: 

1) Well-documented source code and an automated machine learning pipeline
2) An interactive web application

The source code will allow future researchers and developers to inspect and integrate all aspects of the model. The Makefile included with the project makes it seamless to update the model with an expanded dataset. The web application allows healthcare providers to easily interact with our machine learning model to discover which sites are safe for insulin injection and which sites should be avoided. 

Although our project has demonstrated that machine learning can be used to detect lipohypertrophy, there are some key limitations that should be addressed before it is used in the clinical setting. These key limitations are as follows:

1. Scarcity of data - We had merely 353 images to develop our model which is a small dataset in the realm of deep learning. The scarcity of our dataset caused variability in different runs of the experiments conducted. We have also noticed that our model’s performance is sensitive to data splitting (i.e. which images are in the training, validation and test folders). 

2. Limited resources - Limited computing power has been a recurrent challenge in this project and has made it difficult to experiment with tuning all the parameters of our model concurrently. 

3. Lack of Oversight - We believe that there should be an auditing process involved to ensure that our machine learning model does not propagate any biases that could cause harm to specific patient populations. It would also be useful to have other independent radiologists label the dataset in order to draw comparisons between physicians’ and algorithms’ performance.

To address the limitations outlined above, our team has made the following recommendations: 

1.  Addition of more labelled images – In line with the size of the dataset in similar studies (Xiao et al., 2018; Cheng & Malhi, 2017), we recommend increasing the dataset to at least 1000 images.

2.  Increasing computational resources - we recommend obtaining funding for additional computing resources for both the development and deployment of future models.

3. Add additional functionality such as native cropping.


## References
[insert here]

