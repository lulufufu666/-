import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

filename = "D:\\train.csv"
datas = pd.read_csv(filename)
header = datas.columns.tolist()
header = header[1:len(header)-1:1]

X = datas.drop('target', axis=1)
y = datas['target']

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=46)

tree = DecisionTreeClassifier(max_depth=40, random_state=0)
# 4
# 后将max_depth参数去掉可以全局看到哪些特征的重要性更高

tree.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on testing set: {:.3f}".format(roc_auc_score(y_test, tree.predict(X_test))))

print("Accuracy on testing set: {:.3f}".format(tree.score(X_test, y_test)))
print("Feature importance:\n{}".format(tree.feature_importances_))
print("classification report:\n", classification_report(y_test, tree.predict(X_test),
                                                        target_names=["非5g", "5g"]))


def plot_feature_importance_cancer(model):
    n_features = len(header)
    print(n_features)
    plt.figure(figsize=(17, 14))
    plt.barh(list(range(n_features+1)), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), header)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.show()


plot_feature_importance_cancer(tree)





