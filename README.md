# Fundus Image Segmentation using Deep Learning

This project implements a custom UNet model to perform segmentation on fundus images. The model is trained using the DRIVE dataset and evaluated using various evaluation metrics.

## Dataset Preparation

In the first step, I loaded and preprocessed the DRIVE dataset using the code in Chunk 1 of the Jupyter notebook. The dataset was split into training and validation sets and preprocessed to a common size.

## Model Architecture

I created a custom UNet model with four encoder and decoder blocks. The model was designed to take in an input image and output a corresponding binary mask indicating the pixels that belong to the optic disc.

## Model Training

I compiled and trained the custom UNet model using the training and validation data prepared earlier. Data augmentation was used to improve the model's generalization. The best model was saved for further evaluation.

## Model Evaluation

The best saved model was loaded and made predictions on the validation set. I also calculated various evaluation metrics such as IoU, Dice Coefficient, Precision, and Recall to assess the model's performance. Additionally, we applied a simple post-processing step using morphological operations to refine the predicted segmentation masks.

## Hyperparameter Tuning
I performed hyperparameter tuning using GridSearchCV from scikit-learn library. We explored the effect of different hyperparameters such as the number of filters, dropout rate, and learning rate on the model's performance.

## Conclusion

This project demonstrated the use of deep learning for fundus image segmentation. I implemented a custom UNet model and trained it on the DRIVE dataset, achieving promising results on the validation set. Additionally, I explored various evaluation metrics and performed hyperparameter tuning to optimize the model's performance. This project can serve as a starting point for further research in the field of medical image analysis.

