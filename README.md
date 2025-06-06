# Capstone-Project


🩺 Project Title:
Design & Development of a Non-Invasive Method for Early Detection of Skin Cancer using ResNet-50

🔍 Project Objective:
The main goal was to develop an AI-based, non-invasive system using a deep learning model (ResNet-50) that can accurately classify skin lesions as either benign or malignant using dermatoscopic images.

🧠 Why This Project Matters:
Skin cancer is one of the most common and dangerous types of cancer.

Early detection can save lives, but many people don’t have access to dermatologists.

This model helps automate skin cancer detection using image analysis, especially helpful in rural or under-resourced areas.

🏗️ Project Workflow (Step-by-Step)
1. Dataset Collection
Used the HAM10000 dataset (a standard medical image dataset).

Contains over 10,000 labeled images of skin lesions (both benign and malignant).

Training: 70%, Validation: 15%, Testing: 15%.

2. Data Preprocessing
Removed low-quality and duplicate images.

All images were resized to 224x224 pixels (required by ResNet-50).

Normalized image values and encoded labels.

Applied data augmentation to increase diversity and reduce overfitting.

3. Model Used: ResNet-50
ResNet-50 is a deep convolutional neural network with 50 layers.

Used transfer learning: started with a pre-trained model and fine-tuned it for skin cancer images.

Added custom layers for final classification.

4. Model Training
Used TensorFlow or PyTorch frameworks.

Applied early stopping and learning rate tuning to avoid overfitting.

Used K-Fold Cross Validation to make sure the model performs well on all parts of the dataset.

5. Model Evaluation
Evaluated performance using:

✅ Accuracy (up to 90.4%)

✅ Precision & Recall

✅ F1-score

✅ Confusion matrix

Achieved 87%+ accuracy after 20 epochs of training.

6. Results & Visualization
Confusion matrix showed good detection of cancerous lesions.

Graphs showed:

Increasing accuracy

Decreasing loss

Used sample images to compare predicted vs actual labels.

🧪 Conclusion
The model is effective at detecting skin cancer non-invasively.

ResNet-50 was able to capture important features and classify with high accuracy.

The system could assist doctors, speed up diagnosis, and help in early treatment.

🔮 Future Scope
Train on larger and more diverse datasets (more skin tones and lesion types).

Integrate with mobile apps for real-time screening.

Use explainable AI to make the model's predictions more transparent to doctors.

⚙️ Technologies & Tools Used
Python, TensorFlow/Keras

OpenCV (for image preprocessing)

Scikit-learn (for evaluation)

Google Colab / Jupyter Notebook

🗂️ Key Points to Mention in Interview
Solved a real healthcare problem using deep learning.

Gained experience in data preprocessing, transfer learning, CNN architecture, and model evaluation.

Learned to handle medical datasets and apply deep learning in a real-world scenario.
