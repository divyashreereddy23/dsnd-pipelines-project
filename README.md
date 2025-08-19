# Customer Review Recommendation Pipeline

A machine learning pipeline project that predicts whether customers recommend products based on their reviews. This project implements a complete end-to-end data science workflow including data exploration, preprocessing, model training, hyperparameter tuning, and evaluation.

### Project Overview
This project builds a robust machine learning pipeline to classify customer reviews into recommendations (1) or non-recommendations (0). The dataset contains mixed data types including numerical features, categorical features, and text data (review titles and review text), requiring sophisticated preprocessing techniques.

### Key Features

* Handles mixed data types (numerical, categorical, and text)
* Implements comprehensive preprocessing pipelines
* Compares multiple classification algorithms
* Performs hyperparameter optimization
* Achieves >90% ROC-AUC score
* Production-ready model deployment

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* Python 3.8 or higher
* pip package manager
* Jupyter Notebook or JupyterLab
* At least 4GB of RAM for model training

### Dependencies

```
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
matplotlib==3.7.0
seaborn==0.12.0
joblib==1.3.0
jupyter==1.0.0
```

### Installation

1. Clone the repository to your local machine:

```
git clone https://github.com/divyashreereddy23/dsnd-pipelines-project.git
cd dsnd-pipelines-project
```

2. Create a virtual environment (recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:

```
pip install -r requirements.txt
```

4. Verify the installation:

```
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"
```

5. Launch Jupyter Notebook:

```
jupyter notebook
```

## Project Structure

dsnd-pipelines-project/
│
├── data/
│   └── reviews.csv                 # Customer reviews dataset
│
├── starter.ipynb                   # Main project notebook
├── best_model_*.pkl                # Saved trained model (generated)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── LICENSE.txt                     # License file

## Dataset

The dataset contains 18,442 customer reviews with the following features:

### Features

* Clothing ID (**Integer**): *Specific piece being reviewed*
* Age (**Integer**): *Reviewer's age*
* Title (**String**): *Review title*
* Review Text (**String**): *Main review content*
* Positive Feedback Count (**Integer**): *Number of positive feedback from other customers*
* Division Name (**Categorical**): *Product high-level division*
* Department Name (**Categorical**): *Product department*
* Class Name (**Categorical**): *Product class*

### Target Variable

Recommended IND (Binary): 1 = Recommended, 0 = Not Recommended

### Data Characteristics

* No missing values in numerical features
* Some missing values in text fields
* Class imbalance: ~82% positive (recommended) class

## Implementation Details

1. Data Exploration

- Statistical analysis of all features
- Distribution visualizations
- Class imbalance assessment
- Text length analysis
- Correlation analysis

2. Preprocessing Pipeline

The project implements sophisticated preprocessing for each data type:

* Numerical Features

- Median imputation for missing values
- StandardScaler normalization

* Categorical Features

- Missing value imputation with 'missing' category
- One-hot encoding with unknown category handling

* Text Features

- TF-IDF vectorization
- Separate processing for Title (100 features) and Review Text (400 features)
- Bi-gram inclusion
- English stop words removal

3. Models Implemented

Three classification algorithms are compared:

* Logistic Regression (with class weight balancing)
* Random Forest Classifier (100 estimators)
* Gradient Boosting Classifier (100 estimators)

4. Model Evaluation

Comprehensive evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* 5-Fold Cross-Validation

5. Hyperparameter Tuning

GridSearchCV optimization for the best performing model with extensive parameter grids.

## Running the Project

### Step 1: Data Exploration

Run the data exploration cells to understand the dataset:

#### Load and explore data
```
df = pd.read_csv('data/reviews.csv')
df.info()
```

### Step 2: Build Pipeline

Execute the pipeline building section:

#### Define feature groups and create preprocessing pipeline
```
numerical_features = ['Age', 'Positive Feedback Count', 'Clothing ID']
categorical_features = ['Division Name', 'Department Name', 'Class Name']
text_features = ['Title', 'Review Text']
```

### Step 3: Train Models

Train all three models and compare performance:
#### Train models
```
for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
```

### Step 4: Fine-Tune

Optimize the best model with hyperparameter tuning:
#### Hyperparameter tuning
```
grid_search = GridSearchCV(best_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

### Step 5: Save Model

The final trained model is saved for deployment:
```
joblib.dump(best_model, 'best_model_tuned.pkl')
```

## Model Performance

### Final Results (Tuned Model)

- Accuracy: ~88%
- Precision: ~89%
- Recall: ~97%
- F1-Score: ~93%
- ROC-AUC: ~93%

### Key Insights

* Text features are the most important predictors
* Review length correlates with recommendation likelihood
* Model handles class imbalance effectively
* Strong generalization with minimal overfitting

## Testing

### Unit Tests
Run the test suite to verify pipeline components:

#### Test preprocessing pipeline
```
assert preprocessor.fit_transform(X_train).shape[1] > 500
print("✓ Preprocessing creates 500+ features")
```

#### Test model predictions
```
assert len(y_pred) == len(y_test)
print("✓ Predictions match test set size")
```

#### Test probability outputs
```
assert all(0 <= p <= 1 for p in y_pred_proba)
print("✓ Probabilities in valid range")
```

### Cross-Validation

The project includes 5-fold cross-validation to ensure model stability:
```
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(f"Mean CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
```

## Deployment

To use the trained model for predictions:
```
import joblib

# Load the saved model
model = joblib.load('best_model_modelname_tuned.pkl')   # Change 'modelname' with whatever model has been selected 

# Prepare new data (must have same features)
new_review = {
    'Clothing ID': 1234,
    'Age': 35,
    'Title': 'Great product!',
    'Review Text': 'I love this item, fits perfectly...',
    'Positive Feedback Count': 5,
    'Division Name': 'General',
    'Department Name': 'Tops',
    'Class Name': 'Blouses'
}

# Make prediction
prediction = model.predict([new_review])
probability = model.predict_proba([new_review])
```

## Project Instructions

This section contain all the deliverables for this project.

## Built With

* [scikit-learn](https://scikit-learn.org/) - Machine learning library
* [pandas](https://pandas.pydata.org/) - Data manipulation and analysis
* [NumPy](https://numpy.org/) - Numerical computing
* [Matplotlib](https://matplotlib.org/) - Plotting library
* [Seaborn](https://seaborn.pydata.org/) - Statistical data visualization
* [Jupyter](https://jupyter.org/) - Interactive development environment

## Authors

* **Name** - Divyashree Reddy
* **Email** - divyashreereddy23@gmail.com

## License

[License](LICENSE.txt)

## Acknowledgments

- Dataset provided by the Data Science Nanodegree program
- Inspired by real-world e-commerce recommendation systems
- Thanks to the scikit-learn community for excellent documentation

## Contact
For questions or feedback about this project, please open an issue on GitHub or contact via the repository.
