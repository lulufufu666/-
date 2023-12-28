import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load data
filename = "train.csv"
datas = pd.read_csv(filename)
header = datas.columns.tolist()
header = header[1:len(header)-1:1]

X = datas.drop('target', axis=1)
y = datas['target']

# data pre-processing
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# split the dataset into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=46)

# build the decision tree model
tree = DecisionTreeClassifier(max_depth=40, random_state=0)
tree.fit(X_train, y_train)

# print out the model evaluation metrics
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on testing set: {:.3f}".format(roc_auc_score(y_test, tree.predict(X_test))))
print("Classification report:\n", classification_report(y_test, tree.predict(X_test),
                                                        target_names=["Non-5G", "5G"]))

# visualize the feature importance
def plot_feature_importance(model):
    n_features = len(header)
    plt.figure(figsize=(10, 6))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), header)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance for Classification")
    plt.show()

plot_feature_importance(tree)
