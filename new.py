import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

data_set = pd.read_csv("meteorite-landings.csv")

data_set.drop("name", axis=1, inplace=True)
data_set.drop("id", axis=1, inplace=True)
data_set.drop("nametype", axis=1, inplace=True)
data_set.drop("recclass", axis=1, inplace=True)

y = data_set["fall"]
data_set.drop("fall" , axis=1 , inplace=True)

x = data_set

numeric_columns = data_set.select_dtypes(include=[np.number]).columns

categorical_columns = data_set.select_dtypes(include=[object]).columns
label_encoder = LabelEncoder()
for column in categorical_columns:
    data_set[column] = label_encoder.fit_transform(data_set[column])
    
imputer = SimpleImputer()
data_set[numeric_columns] = imputer.fit_transform(data_set[numeric_columns])


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=46)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train, y_train)

print('knn eğitim verisi doğruluk değeri: ', knn_model.score(x_train, y_train))
print('knn test verisi doğruluk değeri: ', knn_model.score(x_test, y_test))

tree_model = DecisionTreeClassifier(max_depth=2)
tree_model.fit(x_train, y_train)

print('tree eğitim verisi doğruluk değeri: ', tree_model.score(x_train, y_train))
print('tree test verisi doğruluk değeri: ', tree_model.score(x_test, y_test))

naive_model = GaussianNB()
naive_model.fit(x_train, y_train)

print('naive eğitim verisi doğruluk değeri: ', naive_model.score(x_train, y_train))
print('naive test verisi doğruluk değeri: ', naive_model.score(x_test, y_test))

tahmin_knn_model = knn_model.predict(x_test)
tahmin_tree_model = tree_model.predict(x_test)
tahmin_naive_model = naive_model.predict(x_test)

cf_matris_knn_model = confusion_matrix(y_test, tahmin_knn_model)
cf_matris_tree_model = confusion_matrix(y_test, tahmin_tree_model)
cf_matris_naive_model = confusion_matrix(y_test, tahmin_naive_model)

def confusion_matrix_visualization(matrix, name):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(x=j, y=1, s=matrix[i, j], va='center', ha='center', size = 'xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title(name, fontsize=18)
    plt.show()
    
cf_matrisler = [cf_matris_knn_model, cf_matris_tree_model]
    
cf_matrisler_name = [["Knn Algoritması", tahmin_knn_model], ["Tree Algoritması", tahmin_tree_model], ["Naive Bayes Algoritması", tahmin_naive_model]]
    
for i, c in zip(cf_matrisler, cf_matrisler_name):
    TP = i[1, 1]
    TN = i[0, 0]
    FP = i[0, 1]
    FN = i[1, 0]
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = (TP) / (TP + FN)
    specifity = (TN) / (TN + FP)
    
    print('')
    print('')
    print(c[0], ' accuracy değeri: ', accuracy)
    print(c[0], ' sensitivity değeri: ', sensitivity)
    print(c[0], ' specifity değeri: ', specifity)
    print("\n\n\n")
    print(c[0], "{0}".format(classification_report(y_test, c[1],)))
    print("\n\n\n")
    print("\n\n\n")
    print(confusion_matrix_visualization(i, c[0]))
    print("\n\n\n")
    print("----------------------------------------------------------")
    
    print('')
    print('')
    