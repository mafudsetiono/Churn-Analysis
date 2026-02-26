#📊 Churn Prediction & Retention Optimization Dashboard

Proyek ini merupakan end-to-end implementasi machine learning untuk memprediksi customer churn dan mengoptimalkan strategi retensi berbasis dampak finansial.

Aplikasi dibangun menggunakan Python, Scikit-learn, dan Streamlit, serta dilengkapi dengan simulasi ROI untuk membantu pengambilan keputusan bisnis.

#🎯 Business Problem

Customer churn menyebabkan kehilangan pendapatan berulang dan meningkatkan biaya akuisisi pelanggan baru.
Tanpa sistem prediktif, perusahaan hanya bereaksi setelah pelanggan berhenti, bukan sebelum.

Tujuan proyek ini adalah:

Memprediksi pelanggan yang berisiko churn

Meminimalkan false negative (pelanggan berisiko yang terlewat)

Mengoptimalkan strategi retensi berdasarkan dampak finansial

Menghitung estimasi ROI program retensi

#📂 Dataset Overview

Total pelanggan: 667

Target variable: Churn (Yes/No)

Fitur utama:

International Plan

Voice Mail Plan

Customer Service Calls

Total Day Charge

Total Evening Charge

#⚙️ Machine Learning Process
1️⃣ Data Preprocessing

Encoding variabel kategorikal (Yes/No → 0/1)

Feature scaling menggunakan StandardScaler

Train-test split (80:20)

2️⃣ Model Comparison

Beberapa model dibandingkan:

Logistic Regression (baseline)

Ridge Logistic Regression

Random Forest

3️⃣ Hyperparameter Tuning

Random Forest dituning menggunakan GridSearchCV dengan fokus pada optimasi recall churn.

#🏆 Final Model

Model akhir: Random Forest (Tuned)

Performance pada test set:

Accuracy: 90%

Recall (Churn): 84%

Precision (Churn): 59%

ROC-AUC: ~0.90

Model ini dipilih karena memberikan keseimbangan optimal antara deteksi churn dan efisiensi biaya intervensi.

#💰 Financial Impact Simulation

Dashboard menghitung:

Expected Loss

Expected Saved

Net Benefit

Retention Priority

ROI Program Retensi

Contoh hasil:

Avg churn probability: 23%

Total expected loss: $7,673

Priority 1 customers: 61

Estimated ROI: 2.47x

Artinya:
Setiap $1 yang dikeluarkan untuk intervensi menghasilkan potensi return sebesar $2.47.

#🖥️ Streamlit App Features

✔ Single Customer Prediction
✔ Batch Prediction
✔ Financial Impact Simulator
✔ Retention Priority Ranking
✔ Executive Dashboard
✔ ROI Analysis
✔ Profile Page

#🚀 How to Run

Clone repository:

git clone https://github.com/username/repository-name.git
cd repository-name

Install dependencies:

pip install -r requirements.txt

Run Streamlit app:

streamlit run app.py
#📈 Key Insight

International Plan dan Customer Service Calls memiliki pengaruh signifikan terhadap churn.

Model tuning meningkatkan recall dan menurunkan false positive.

Strategi retensi menjadi lebih terarah dan efisien secara finansial.

#🧠 Business Value

Proyek ini tidak hanya membangun model prediksi, tetapi juga:

Menghubungkan model ke dampak finansial nyata

Memberikan prioritas retensi berbasis ROI

Mendukung data-driven decision making

#👤 Author

Mafud Satrio Setiono
Data Analyst | Machine Learning | Business Insight

Terbuka untuk peluang Data Analyst / Data Scientist 🚀
