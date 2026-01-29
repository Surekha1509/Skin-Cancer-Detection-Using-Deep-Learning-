# Skin-Cancer-Detection-Using-Deep-Learning-
This project focuses on skin cancer detection using deep learning, aiming to support early identification of malignant skin lesions through image classification. The system analyzes dermoscopic images and classifies them as Benign or Malignant, highlighting potentially high-risk cases. 


🧬 *Skin Cancer Detection Using Deep Learning*

📌 *Project Overview*

Skin cancer is one of the most common cancers worldwide, and early detection plays a critical role in successful treatment.
This project explores the use of deep learning image classification models to assist in identifying potentially high-risk skin lesions.

We implemented and compared multiple neural network architectures and selected the best-performing model based on accuracy, loss, and recall, with a strong emphasis on minimizing false negatives.

🔍 *What the Application Does*

Analyzes dermoscopic skin lesion images

Classifies images as Benign or Malignant

Provides prediction confidence scores

Focuses on high recall to reduce missed malignant cases

Offers a simple and interactive web interface for image upload and prediction

prediction

🧠 *Models Used*

The following deep learning architectures were implemented and compared:

Custom CNN (Baseline)

VGG16 (Transfer Learning)

EfficientNet-B0 (Transfer Learning) ✅ Final Model

EfficientNet-B0 was selected as the final model due to:

Strong performance on both small and large datasets

Lower validation loss

Better balance between accuracy and computational efficiency

efficiency

📊 *Dataset*

Source: Public dermoscopic skin lesion dataset

Initially explored a large dataset (~12,000 images) (https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset)

Due to limited GPU resources and long training times, a balanced subset of 2,000 images was selected

Final Dataset Size: 2,000 images

Class Distribution: Balanced (Benign / Malignant)

Reason for Subset Selection:

Limited GPU resources

Long training times on the full dataset

Faster experimentation while maintaining strong performance

🛠️ *Techniques Applied*

Image resizing and normalization

Data augmentation (rotation, flipping, zooming)

Transfer learning with frozen base models

Early stopping to prevent overfitting

Threshold tuning to improve recall

recall

🚀 *Deployment*

The final model is deployed as a Hugging Face Space, allowing real-time image classification through a web interface.

🔗 *Live Demo*:

📁 *Project Structure*
├── models/
│   ├── cnn_model.keras
│   ├── vgg_model.keras
│   └── effnet_model.keras
├── app.py               # Web application
├── training_notebook.ipynb
├── requirements.txt
└── README.md



🧪 *Evaluation Metrics*
Accuracy

Precision

Recall (prioritized metric)

Loss

Confusion Matrix

Classification Report

*Conclusion*

Our project shows that deep learning can effectively assist in skin cancer detection.
EfficientNetB0 achieved the best performance for our task, especially in terms of recall.

⚠️ This project is for educational and research purposes only and is not a medical diagnosis.


🧑‍💻 *Author*

Developed as a final bootcamp project focusing on applied deep learning and responsible AI use.
