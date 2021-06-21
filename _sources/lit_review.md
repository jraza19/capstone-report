## Literature Review 

Prediction of masses in ultrasound images using machine learning techniques has been an ongoing effort in clinical practice for the past few decades. To assist physicians in diagnosing disease, many scholars have implemented techniques such as regression, decision trees, Naive Bayesian classifiers, and neural networks on patients' ultrasound imaging data. (ref) Further, many studies involving ultrasound images have preprocessed the images to extract features. This STUDY shows that CNNs using ultrasound images perform better than radiomic models at predicting breast cancer tumours. Another recent study shows success in classifying liver masses into 1 out 5 categories with 84% accuracy using a CNN model. (ref)

Furthermore, recent research has delved into various complex image augmentation approaches such as using GANs to generate images(ref), enlarging the dataset used for training which naturally improves the performance. The study also found that traditional transformations managed to improve model performance. Many other studies such as Esteva et al., 2017;Loey et al 2020 confirmed that minimal transformations such as flipping the images, achieved higher prediction accuracy in their application. 

We also found that transfer learning architectures are crucial for yielding reliable results as any given patient's dataset is likely too small to construct a CNN architecture. This study built a CNN using DenseNet models for the prediction of breast cancer masses and achieved a test accuracy of 91%. Zhang, H., Han, L., Chen, K. also studying the diagnosis of breast cancer tumours in ultrasound images found that when compared with other architectures such as InceptionV3, VGG16, ResNet50, and VGG19, InceptionV3 performs the best. Prediction competitions on Kaggle have proven successful with architectures such as VGG16 and Resnet, predicting masses in ultrasound images with 70-80% accuracy. [add ref-Ela]

For the object detection piece, we also considered RCNNs but after some research we saw YOLO performs well in Kaggle competitions and is generally faster than RCNNs so we went with the YOLO model (reference).



