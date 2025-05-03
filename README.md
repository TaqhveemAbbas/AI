# Brain Tumor Detection Using Convolutional Neural Networks with Enhanced Visualization for Improved Diagnostic Accuracy 
Brain Tumor Classification Using CNN
This project focuses on classifying brain tumor types using Convolutional Neural Networks (CNNs) with TensorFlow and Keras. Based on MRI scan images, it identifies four categories of brain tumors: glioma, meningioma, pituitary, and no tumor.

📌 Overview
Model Type: Convolutional Neural Network (CNN)

Categories: Glioma, Meningioma, Pituitary, No Tumor

Tools Used: TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn

Accuracy Achieved: ~95.7%

Dataset: Pre-labeled brain MRI images split into training and testing folders.

🔧 Features
Image preprocessing using ImageDataGenerator with extensive data augmentation.

Model architecture with four convolutional layers and dropout for regularization.

Evaluation using confusion matrix and sample prediction visualization.

Training and validation accuracy/loss visualization.

🧠 Model Architecture
text
Copy
Edit
Input Layer: 150x150x3 RGB images
↓
Conv2D (32 filters, 3x3) + ReLU
↓
MaxPooling2D (2x2)
↓
Conv2D (64 filters, 3x3) + ReLU
↓
MaxPooling2D (2x2)
↓
Conv2D (128 filters, 3x3) + ReLU
↓
MaxPooling2D (2x2)
↓
Conv2D (128 filters, 3x3) + ReLU
↓
MaxPooling2D (2x2)
↓
Flatten
↓
Dense (512) + ReLU
↓
Dropout (0.5)
↓
Output: Dense (4 categories) + Softmax
🚀 How It Works
Data Preprocessing:

Training images are normalized and augmented with rotations, flips, shifts, and zooms.

Testing images are normalized for evaluation.

Model Training:

Trained for 50 epochs with adam optimizer and categorical_crossentropy loss.

Evaluation:

Achieved ~95.7% accuracy.

Confusion matrix generated to analyze class-wise performance.

Visualization:

Displays accuracy curves, confusion matrix, and example predictions.

📊 Results
Final Test Accuracy: 95.78%

Loss: 0.134

Performance Indicators:

Confusion matrix reveals strong class-wise prediction capability.

Visual sample predictions align well with true labels.

🛠️ Technologies Used
Language: Python

Libraries: TensorFlow, Keras, Matplotlib, Seaborn, NumPy, Pandas

IDE: VSCode / Jupyter Notebook

📈 Future Enhancements
Introduce transfer learning (e.g., MobileNet, ResNet).

Integrate Grad-CAM for tumor region localization.

Expand dataset and add AR-based visualization (future scope).
