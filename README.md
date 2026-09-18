# Group 6 Capstone Project

A machine learning capstone project consisting of two applications that demonstrate **Supervised Learning** and **Unsupervised Learning** concepts.

## 📌 Projects

### 1. 📧 Email Spam Detection

**Type:** Supervised Machine Learning

A machine learning application that classifies emails as:

* **Spam**
* **Not Spam (Ham)**

The model is trained using labelled email data and uses text preprocessing and feature extraction to make predictions on new emails.

**Key concepts:**

* Labelled datasets
* Text preprocessing
* Feature extraction
* Train/Test Split
* Classification
* Model Evaluation
* Model Saving and Reuse
* Prediction through an application interface

---

### 2. 👥 Customer Persona Segmenter

**Type:** Unsupervised Machine Learning

A customer segmentation application that groups customers into different personas based on characteristics such as:

* Annual Income
* Spending Score

The project uses **K-Means Clustering** to identify customer groups without predefined labels.

**Key concepts:**

* Unlabelled data
* Data preprocessing
* Feature scaling
* K-Means Clustering
* Cluster analysis
* Customer persona mapping
* Data visualization

---

## 🏗️ Repository Structure

```text
Group-6-Capstone-Project/
│
├── README.md
│
├── supervised/
│   └── email-spam-detection/
│       ├── README.md
│       ├── requirements.txt
│       │
│       ├── backend/
│       │   ├── train_model.py
│       │   ├── app.py
│       │   └── spam_emails.csv
│       │
│       └── frontend/
│           ├── index.html
│           ├── style.css
│           └── script.js
│
└── unsupervised/
    └── customer-persona-segmenter/
        ├── README.md
        ├── requirements.txt
        ├── dataset_unsupervised.py
        ├── train_kmeans.py
        ├── main_unsupervised.py
        └── app_unsupervised.py
```

> Model files and generated datasets may be created automatically during the training/setup process and are not necessarily created manually.

## 🛠️ Technologies Used

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn

### Backend

* FastAPI
* Uvicorn

### Frontend / Visualization

* HTML
* CSS
* JavaScript
* Streamlit
* Matplotlib
* Seaborn

### Development Tools

* Git
* GitHub
* Visual Studio Code

## 🔄 Machine Learning Workflow

### Supervised Learning

```text
Labelled Dataset
       ↓
Data Preprocessing
       ↓
Feature Extraction
       ↓
Train/Test Split
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Save Model
       ↓
New Email
       ↓
Prediction
       ↓
Spam / Not Spam
```

### Unsupervised Learning

```text
Customer Dataset
       ↓
Data Preprocessing
       ↓
Feature Scaling
       ↓
K-Means Clustering
       ↓
Customer Groups
       ↓
Persona Mapping
       ↓
Visualization / Prediction
```

## 🎯 Objectives

* Apply machine learning concepts to practical problems.
* Understand the difference between supervised and unsupervised learning.
* Build and evaluate machine learning models.
* Integrate trained models with application backends.
* Provide simple user interfaces for interacting with the models.
* Demonstrate an end-to-end machine learning workflow.

## 👥 Team

**Group 6 — Capstone Project**

This repository contains the collaborative work of the Group 6 team.

## 📂 Project Documentation

Detailed setup instructions, implementation details, and usage instructions are available in the individual project README files:

* `supervised/email-spam-detection/README.md`
* `unsupervised/customer-persona-segmenter/README.md`

---

## 📌 Note

This repository is developed as part of the **Capstone Project** to demonstrate practical implementation of machine learning concepts using Python and related technologies.
