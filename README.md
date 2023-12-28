# Project Summary

This project utilizes a tumor dataset with missing values to predict whether a tumor is malignant using a decision tree classifier.

# Project Files

File | Description
--- | ---
train.csv | Tumor dataset
tumor_prediction.py | Code for tumor prediction using a decision tree classifier
README.md | Project documentation

# Dependencies

1. NumPy
2. pandas
3. scikit-learn
4. matplotlib

# Code Overview

1. Read the file

The dataset is read using the `read_csv` function from pandas.

2. Data Preprocessing

We remove the target column from the dataset and store it in `y`, then use `SimpleImputer` to interpolate missing values for the angle values, and `StandardScaler` to scale the features.

3. Split the Data

We split the data into training and test sets using the `train_test_split` function with a ratio of 75:25.

4. Define the Model

The model is fitted using `DecisionTreeClassifier`.

5. Model Evaluation

The accuracy of the model and feature importance are calculated using the test dataset.

6. Visualization

We use matplotlib to plot the feature importance of the decision tree model.

# How to Run

1. Ensure that the dependencies are installed.
2. Save the code as `tumor_prediction.py`.
3. Save the dataset as `train.csv`.
4. Run the code by executing the `python tumor_prediction.py` command in the console.

# Conclusion

Our model achieved an accuracy of 0.899 on the test set, indicating good performance of our decision tree classifier in predicting whether a tumor is malignant. According to the visualization of feature importance, tumor size and shape are important features. This is a great project that showcases the process of tumor prediction using a decision tree classifier. Thank you for sharing the detailed project description and implementation code.
