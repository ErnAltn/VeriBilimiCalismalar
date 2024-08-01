import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Veri setini yükle
data_set = pd.read_csv("meteorite-landings.csv")

# Gereksiz sütunları kaldır
data_set.drop(["name", "id", "nametype", "recclass", "GeoLocation"], axis=1, inplace=True)

# Hedef değişkeni etiketle
label_encoder = LabelEncoder()
data_set["fall"] = label_encoder.fit_transform(data_set["fall"])

# Eksik değerleri doldur
imputer = SimpleImputer()
numeric_columns = data_set.select_dtypes(include=[np.number]).columns
data_set[numeric_columns] = imputer.fit_transform(data_set[numeric_columns])

# Bağımsız değişkenleri ve hedef değişkeni ayır
X = data_set.drop("fall", axis=1)
y = data_set["fall"]

# Veri setini eğitim ve test olarak böle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verileri ölçeklendir
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# K-En Yakın Komşu (KNN) sınıflandırıcısı
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_y_pred = knn_model.predict(X_test)
knn_y_pred_proba = knn_model.predict_proba(X_test)[:, 1]
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_y_pred_proba)
knn_auc = roc_auc_score(y_test, knn_y_pred_proba)

knn_cm = confusion_matrix(y_test, knn_y_pred)
knn_cr = classification_report(y_test, knn_y_pred)

print("K-En Yakın Komşu (KNN) Sınıflandırıcısı:")
print("Karmaşıklık Matrisi:")
print(knn_cm)
print("Sınıflandırma Raporu:")
print(knn_cr)

# Rastgele Orman sınıflandırıcısı
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)
rf_y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_y_pred_proba)
rf_auc = roc_auc_score(y_test, rf_y_pred_proba)

rf_cm = confusion_matrix(y_test, rf_y_pred)
rf_cr = classification_report(y_test, rf_y_pred)

print("Rastgele Orman Sınıflandırıcısı:")
print("Karmaşıklık Matrisi:")
print(rf_cm)
print("Sınıflandırma Raporu:")
print(rf_cr)


# Naive Bayes sınıflandırıcısı
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_y_pred = nb_model.predict(X_test)
nb_y_pred_proba = nb_model.predict_proba(X_test)[:, 1]
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_y_pred_proba)
nb_auc = roc_auc_score(y_test, nb_y_pred_proba)

nb_cm = confusion_matrix(y_test, nb_y_pred)
nb_cr = classification_report(y_test, nb_y_pred)

print("Naive Bayes Sınıflandırıcısı:")
print("Karmaşıklık Matrisi:")
print(nb_cm)
print("Sınıflandırma Raporu:")
print(nb_cr)


# Karar Ağaçları sınıflandırıcısı
dt_model = DecisionTreeClassifier(max_depth=2)
dt_model.fit(X_train, y_train)
dt_y_pred = dt_model.predict(X_test)
dt_y_pred_proba = dt_model.predict_proba(X_test)[:, 1]
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_y_pred_proba)
dt_auc = roc_auc_score(y_test, dt_y_pred_proba)

dt_cm = confusion_matrix(y_test, dt_y_pred)
dt_cr = classification_report(y_test, dt_y_pred)

print("Karar Ağaçları Sınıflandırıcısı:")
print("Karmaşıklık Matrisi:")
print(dt_cm)
print("Sınıflandırma Raporu:")
print(dt_cr)

# ROC eğrilerini çizdir
plt.plot(knn_fpr, knn_tpr, label=f"KNN (AUC = {knn_auc:.2f})")
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.2f})")
plt.plot(nb_fpr, nb_tpr, label=f"Naive Bayes (AUC = {nb_auc:.2f})")
plt.plot(dt_fpr, dt_tpr, label=f"Decision Trees (AUC = {dt_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrileri')
plt.legend(loc='lower right')
plt.show()
