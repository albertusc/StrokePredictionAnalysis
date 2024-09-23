# Brain Stroke Prediction

## Overview
This project predicts the likelihood of a brain stroke using machine learning models. The dataset includes health and demographic features.

## Dataset
- Features: gender, age, hypertension, heart disease, marital status, work type, residence type, average glucose level, BMI, smoking status, and stroke occurrence.

Link to the dataset: https://www.kaggle.com/datasets/jillanisofttech/brain-stroke-dataset

## Project Structure

### Libraries
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```

### Data Preprocessing
- Load dataset
- Exploratory data analysis
- Encode categorical variables
- Handle imbalanced data
- Remove outliers

### Visualization
- Distribution of stroke classes
- Kernel Density Estimation of age by stroke

### Models
- Decision Tree
- Random Forest

## Conclusion
This project demonstrates the use of machine learning models to predict brain strokes. The models were trained and evaluated, showing their performance in terms of accuracy, classification report, and confusion matrix.
