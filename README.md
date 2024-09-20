# Facial Keypoint Detection with CNN

This project implements a **Facial Keypoint Detection** system using a Convolutional Neural Network (CNN). The network takes in grayscale images of faces and predicts 68 keypoints (x, y) for each face, identifying important facial features such as the eyes, nose, and mouth.

## Project Overview

The project includes:
- A custom CNN architecture designed to detect keypoints from face images.
- Dataset transformation with techniques like rescaling, cropping, normalization, and conversion to tensors.
- Training with a regression-based loss function (MSE) to minimize the Euclidean distance between predicted and actual keypoints.
- Visualization of predicted keypoints and feature maps learned by the CNN.
- **Complete face detection pipeline** using a Haar Cascade classifier for face detection and applying the trained model for facial keypoint detection.

## Model Architecture

The CNN architecture consists of the following layers:
- **Convolutional Layers**: 5 convolutional layers with varying channel sizes and kernel sizes.
- **Pooling Layers**: Max-pooling layers after each convolutional block to reduce spatial dimensions.
- **Dropout Layer**: A dropout layer to prevent overfitting during training.
- **Fully Connected Layer**: A final fully connected layer with 136 outputs representing 68 keypoints.

## Dataset Transformation

To prepare the images for training, the following transformations are applied:
- **Rescaling**: Rescale images to 240x240 pixels.
- **Random Cropping**: Crop the rescaled images to 224x224 pixels.
- **Normalization**: Convert pixel values to the range [0, 1] and normalize keypoints to the range [-1, 1].
- **Tensor Conversion**: Convert images and keypoints into PyTorch tensors.

## Face and Keypoint Detection Pipeline

Once the model is trained, it can be used to detect facial keypoints in any image containing faces. The steps in the pipeline include:

1. **Face Detection**: 
   - Faces are detected using a **Haar Cascade Classifier** (from OpenCV).
   - A bounding box is drawn around each detected face.

2. **Preprocessing**:
   - Each detected face is converted to **grayscale**.
   - The grayscale face is **normalized** to have pixel values in the range [0, 1].
   - The face is **rescaled** to the size expected by the CNN (224x224).
   - The processed image is reshaped into a tensor to match the input format for the model.

3. **Keypoint Prediction**:
   - The preprocessed face is fed into the trained CNN model to predict the **facial keypoints**.
   - The predicted keypoints are then **un-normalized** to be visualized on the original image.

4. **Visualization**:
   - The predicted keypoints are displayed on the image along with the detected face using matplotlib.

## Training Process

- **Loss Function**: Mean Squared Error (MSE) is used to measure the Euclidean distance between predicted and true keypoints.
- **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **Batch Size**: 10 images per batch.
- **Epochs**: Initial training with 5 epochs to observe performance.

## Results

After training, the model is tested on unseen data, and the predicted keypoints are compared to the actual keypoints. The results show how well the model generalizes to new images.

### Visualization of Keypoints
- **Ground Truth Keypoints**: Displayed in green.
- **Predicted Keypoints**: Displayed in magenta.

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- NumPy
- Matplotlib

## Future Work

- Improve accuracy by experimenting with deeper architectures and fine-tuning hyperparameters.
- Implement data augmentation techniques for more robust training.
- Deploy the model using a web interface for real-time facial keypoint detection.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

