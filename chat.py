import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükle
data = pd.read_csv("meteorite-landings.csv")

# Gerekli sütunları seç
X = data[['name', 'nametype', 'recclass', 'mass', 'fall', 'year', 'reclat', 'reclong']]
y = data['fall']

# Etiketleri kodla
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Metinsel değerleri dönüştür
X_encoded = X.apply(label_encoder.fit_transform)

# Eksik değerleri doldur
imputer = SimpleImputer(strategy='most_frequent')
X_encoded = imputer.fit_transform(X_encoded)

# Veri setini eğitim ve test olarak böle
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# K-En Yakın Komşu (KNN) sınıflandırıcısı
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_preds = knn_model.predict(X_test)
knn_cm = confusion_matrix(y_test, knn_preds)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_model.predict_proba(X_test)[:, 1])

# Rastgele Orman sınıflandırıcısı
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_cm = confusion_matrix(y_test, rf_preds)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])

# Naive Bayes sınıflandırıcısı
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)
nb_cm = confusion_matrix(y_test, nb_preds)
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_model.predict_proba(X_test)[:, 1])

# Karar Ağaçları sınıflandırıcısı
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
dt_cm = confusion_matrix(y_test, dt_preds)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_model.predict_proba(X_test)[:, 1])

# Destek Vektör Makineleri (SVM) sınıflandırıcısı
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)
svm_cm = confusion_matrix(y_test, svm_preds)
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_model.predict_proba(X_test)[:, 1])

# Karmaşıklık Matrisi ve ROC Eğrisi Grafikleri
plt.figure(figsize=(6, 6))

plt.subplot(1, 1, 1)
sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('KNN Confusion Matrix')

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))

plt.subplot(1, 1, 1)
plt.plot(knn_fpr, knn_tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN ROC Curve')

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))

plt.subplot(1, 1, 1)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))

plt.subplot(1, 1, 1)
plt.plot(rf_fpr, rf_tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))

plt.subplot(1, 1, 1)
sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Naive Bayes Confusion Matrix')

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))

plt.subplot(1, 1, 1)
plt.plot(nb_fpr, nb_tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Bayes ROC Curve')

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))

plt.subplot(1, 1, 1)
sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Decision Trees Confusion Matrix')

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))

plt.subplot(1, 1, 1)
plt.plot(dt_fpr, dt_tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision Trees ROC Curve')

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))

plt.subplot(1, 1, 1)
sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Confusion Matrix')

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))

plt.subplot(1, 1, 1)
plt.plot(svm_fpr, svm_tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC Curve')

plt.tight_layout()
plt.show()
