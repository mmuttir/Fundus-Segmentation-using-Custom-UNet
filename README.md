# Fundus Image Segmentation using Deep Learning

Fundus segmentation is the task of identifying and separating the optic disc and/or the blood vessels present in fundus images. Fundus images are images of the retina, which can be captured using specialized cameras. The optic disc is the point where the blood vessels enter and exit the retina, and it appears as a bright circular shape in the center of the fundus image. Accurate segmentation of the optic disc and blood vessels is essential for diagnosis and monitoring of various retinal diseases such as diabetic retinopathy, glaucoma, and age-related macular degeneration.

Fundus segmentation has significant importance in the field of ophthalmology as it can aid in the diagnosis and management of various retinal diseases. Early detection of such diseases can lead to better prognosis and treatment outcomes. Fundus segmentation can also be used for tracking disease progression and evaluating treatment efficacy. Automated segmentation techniques can save time and effort for clinicians and reduce the subjectivity associated with manual segmentation. Therefore, the development of accurate and efficient fundus segmentation algorithms using deep learning has become an active area of research in recent years.

## DRIVE Dataset:

The ISIC-2018 dataset is used for conducting this project. The dataset contains the 2596 pairs of images and masks. All of these images are of different shapes and contains a variety of skin lesions.

## Dataset Preparation

In the first step, I loaded and preprocessed the DRIVE dataset and split the data into training and validation sets and preprocessed to a common size.

## Model Architecture

I created a custom UNet model with four encoder and decoder blocks. Compared to the standard UNet architecture, I made the following modifications:

1. I used 4 encoder and decoder blocks instead of the usual 3 blocks to capture more complex features and increase the model's capacity.
2. I added dropout layers after each pooling operation to prevent overfitting and improve generalization.
3. I used 5x5 convolution filters instead of the usual 3x3 filters to capture larger spatial context and improve the model's ability to distinguish between similar features.
4. I used padding='same' in all convolution layers to maintain the spatial dimensions of the input and output feature maps, which simplifies the upsampling process in the decoder blocks.

These modifications were made to adapt the UNet architecture to the specific requirements of fundus image segmentation and improve the model's performance on this task.


## Model Training

I compiled and trained the custom UNet model using the training and validation data prepared earlier. Data augmentation was used to improve the model's generalization. The best model was saved for further evaluation.

## Model Evaluation

The best saved model was loaded and made predictions on the validation set. I also calculated various evaluation metrics such as IoU, Dice Coefficient, Precision, and Recall to assess the model's performance. Additionally, I applied a simple post-processing step using morphological operations to refine the predicted segmentation masks.

## Hyperparameter Tuning
I performed hyperparameter tuning using GridSearchCV from scikit-learn library. I explored the effect of different hyperparameters such as the number of filters, dropout rate, and learning rate on the model's performance.

## Results:

The below images shows the:

- Input image
- Ground truth
- Predicted mask

(https://github.com/mmuttir/Fundus-Segmentation-using-Custom-UNet/blob/main/01_test_0.png)

## Conclusion

This project demonstrated the use of deep learning for fundus image segmentation. I implemented a custom UNet model and trained it on the DRIVE dataset, achieving promising results on the validation set. Additionally, I explored various evaluation metrics and performed hyperparameter tuning to optimize the model's performance. This project can serve as a starting point for further research in the field of medical image analysis.

