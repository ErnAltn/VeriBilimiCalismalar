import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Veri setini okuma
data = pd.read_csv('meteorite-landings.csv')  # CSV dosyası olarak varsayalım, formatı değişebilir

# Veri setini matris formatına dönüştürme
matrix = data.values

# Matrisin boyutunu kontrol etme
print(matrix.shape)

# Veri setini normalize etme
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)

# Veri setini normalize etme
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)

# Veri setini eğitim ve test verilerine bölme
X_train, X_test, y_train, y_test = train_test_split(normalized_data, labels, test_size=0.2, random_state=42)

# KNN modelini oluşturma
knn = KNeighborsClassifier(n_neighbors=5)

# Modeli eğitme
knn.fit(X_train, y_train)

# Test verileri üzerinde tahmin yapma
y_pred = knn.predict(X_test)

# Tahmin başarısını değerlendirme
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Tahminlerin olasılıklarını hesaplama
y_prob = knn.predict_proba(X_test)
positive_prob = y_prob[:, 1]  # Sadece pozitif sınıfın olasılıklarını al

# ROC eğrisini oluşturma
fpr, tpr, thresholds = roc_curve(y_test, positive_prob)

# ROC eğrisini çizdirme
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# ROC AUC skorunu hesaplama
roc_auc = roc_auc_score(y_test, positive_prob)
print("ROC AUC Score:", roc_auc)