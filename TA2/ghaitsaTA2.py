# Import library yang diperlukan
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset dari file CSV
df = pd.read_csv('ai4i2020.csv')

# Pisahkan fitur (X) dan label (y)
# Drop kolom-kolom yang tidak digunakan sebagai fitur
X = df.drop(labels=['UDI', 'Product ID', 'Type', 'Air temperature [K]', 'Process temperature [K]', 
                    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure'], axis=1)
y = df['Machine failure']  # Kolom 'Machine failure' sebagai target (label)

# Memisahkan dataset menjadi data latih (X_train, y_train) dan data uji (X_test, y_test)
# dengan rasio 80:20 menggunakan train_test_split. random_state=42 digunakan untuk hasil yang dapat direproduksi.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melakukan penskalaan fitur menggunakan StandardScaler
# agar setiap fitur memiliki mean 0 dan varians 1 untuk meningkatkan performa model.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Skala data yang dilatih
X_test = scaler.transform(X_test)        # Skala data yang diuji

# Memilih model klasifikasi LogisticRegression dengan parameter random_state=42 untuk hasil yang konsisten saat dilatih ulang.
model = LogisticRegression(random_state=42)

# Latih model dengan data latih
model.fit(X_train, y_train)

# Menggunakan model yang telah dilatih untuk memprediksi label dari data uji (X_test).
y_pred = model.predict(X_test)

# Evaluasi model dengan cross-validation
# Estimasi yang Lebih Akurat tentang Performa Model: Cross-validation memungkinkan kita untuk
# mendapatkan estimasi yang lebih akurat tentang seberapa baik model akan berperforma pada data yang
# belum pernah dilihat sebelumnya (data yang diuji). Dengan melakukan pengujian pada beberapa subset dari data latih,
# kita dapat menghindari bias yang mungkin muncul dari satu pembagian tertentu dari data latih ke data yang diuji.
cv_scores = cross_val_score(model, X_train, y_train, cv=5)  # Cross-validation dengan 5 fold
print(f'Akurasi Cross-Validation: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})')

# Evaluasi model dengan metrik akurasi
# Metode evaluasi akurasi (accuracy_score) digunakan karena dalam kasus klasifikasi biner seperti ini,
# akurasi memberikan gambaran tentang seberapa baik model memprediksi dengan benar kelas-kelas yang ada
# (yaitu, apakah prediksi benar atau salah). Namun, perlu diingat bahwa akurasi tidak selalu merupakan metrik yang sempurna,
# terutama jika kelas target tidak seimbang atau jika kesalahan prediksi dari satu kelas lebih penting daripada yang lain.
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi: {accuracy:.2f}')

report = classification_report(y_test, y_pred)
print(f'Report:\n{report}')