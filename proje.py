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
from sklearn.metrics import r2_score


# Veri setini yükle
data = pd.read_csv("meteorite-landings.csv")

# Gerekli sütunları seç
X = data[['recclass','mass', 'fall', 'reclat', 'reclong']]
y = data['fall']

# Etiketleri kodla
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Metinsel değerleri dönüştür
X_encoded = X.apply(label_encoder.fit_transform)

# Eksik değerleri doldur
imputer = SimpleImputer(strategy='most_frequent')
X_encoded = imputer.fit_transform(X_encoded)

# Veri setini eğitim ve test olarak bölme
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.1, train_size=0.9, random_state=46)
# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, train_size=0.8, random_state=46)
# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.25, train_size=0.75, random_state=46)

# K-En Yakın Komşu (KNN) sınıflandırıcısı
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_preds = knn_model.predict(X_test)
knn_cm = confusion_matrix(y_test, knn_preds)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_model.predict_proba(X_test)[:, 1])
knn_auc = auc(knn_fpr, knn_tpr)
# K-En Yakın Komşu (KNN) sınıflandırıcısı
knn_accuracy = knn_cm.diagonal().sum() / knn_cm.sum()
print("KNN Doğruluk: {:.3f}".format(knn_accuracy))

# Rastgele Orman sınıflandırıcısı
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_cm = confusion_matrix(y_test, rf_preds)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
rf_auc = auc(rf_fpr, rf_tpr)

# Rastgele Orman sınıflandırıcısının başarı oranını hesapla
rf_accuracy = rf_cm.diagonal().sum() / rf_cm.sum()
print("Rastgele Orman Başarı Oranı: {:.3f}".format(rf_accuracy))

# Naive Bayes sınıflandırıcısı
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)
nb_cm = confusion_matrix(y_test, nb_preds)
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_model.predict_proba(X_test)[:, 1])
nb_auc = auc(nb_fpr, nb_tpr)

# Naive Bayes sınıflandırıcısının başarı oranını hesapla
nb_accuracy = nb_cm.diagonal().sum() / nb_cm.sum()
print("Naive Bayes Başarı Oranı: {:.3f}".format(nb_accuracy))

# Karar Ağaçları sınıflandırıcısı
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
dt_cm = confusion_matrix(y_test, dt_preds)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_model.predict_proba(X_test)[:, 1])
dt_auc = auc(dt_fpr, dt_tpr)

# Karar Ağaçları sınıflandırıcısının başarı oranını hesapla
dt_accuracy = dt_cm.diagonal().sum() / dt_cm.sum()
print("Karar Ağaçları Başarı Oranı: {:.3f}".format(dt_accuracy))

# Destek Vektör Makinesi (SVM) sınıflandırıcısı
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)
svm_cm = confusion_matrix(y_test, svm_preds)
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_model.predict_proba(X_test)[:, 1])
svm_auc = auc(svm_fpr, svm_tpr)

# Destek Vektör Makinesi (SVM) sınıflandırıcısının başarı oranını hesapla
svm_accuracy = svm_cm.diagonal().sum() / svm_cm.sum()
print("SVM Başarı Oranı: {:.3f}".format(svm_accuracy))

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

# Performans metriklerini bir satır sütun grafiği olarak göster
plt.figure(figsize=(6, 6))

plt.subplot(1, 2, 1)
ax1 = sns.barplot(x=['KNN', 'Random Forest', 'Naive Bayes', 'Decision Trees', 'SVM'],
                  y=[knn_auc, rf_auc, nb_auc, dt_auc, svm_auc])
plt.xlabel('Algoritma')
plt.ylabel('AUC')
plt.title('Algoritmaların AUC Performansı')

# Sayısal değerleri grafiklere ekle
for i, v in enumerate([knn_auc, rf_auc, nb_auc, dt_auc, svm_auc]):
    ax1.text(i, v, f"{v:.3f}", horizontalalignment='center', verticalalignment='bottom', fontsize=8)

plt.subplot(1, 2, 2)
ax2 = sns.barplot(x=['KNN', 'Random Forest', 'Naive Bayes', 'Decision Trees', 'SVM'],
                  y=[knn_cm.diagonal().sum() / knn_cm.sum(), rf_cm.diagonal().sum() / rf_cm.sum(),
                     nb_cm.diagonal().sum() / nb_cm.sum(), dt_cm.diagonal().sum() / dt_cm.sum(),
                     svm_cm.diagonal().sum() / svm_cm.sum()])
plt.xlabel('Algoritma')
plt.ylabel('Doğruluk')
plt.title('Algoritmaların Doğruluk Performansı')

# Sayısal değerleri grafiklere ekle
for i, v in enumerate([knn_cm.diagonal().sum() / knn_cm.sum(), rf_cm.diagonal().sum() / rf_cm.sum(),
                       nb_cm.diagonal().sum() / nb_cm.sum(), dt_cm.diagonal().sum() / dt_cm.sum(),
                       svm_cm.diagonal().sum() / svm_cm.sum()]):
    ax2.text(i, v, f"{v:.3f}", horizontalalignment='center', verticalalignment='bottom', fontsize=8)

plt.tight_layout()
plt.show()

print(data["nametype"].value_counts())