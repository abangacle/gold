import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Membaca data harga emas
data_emas = pd.read_csv('gold.csv')
data_emas['Date'] = pd.to_datetime(data_emas['Date'])

# Menambahkan fitur berdasarkan tanggal
data_emas['Year'] = data_emas['Date'].dt.year
data_emas['Month'] = data_emas['Date'].dt.month
data_emas['Day'] = data_emas['Date'].dt.day

# Membuat fitur dan target
X = data_emas[['Year', 'Month', 'Day']]
y = data_emas['Close']

# Membuat model Regresi Random Forest
model2023 = RandomForestRegressor(n_estimators=100, max_depth=5000, max_leaf_nodes=800, random_state=42)
model2023.fit(X, y)

# Membuat dataframe untuk setiap tanggal pada tahun 2023
dates_2023 = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
prediction_data_2023 = pd.DataFrame({
    'Year': dates_2023.year,
    'Month': dates_2023.month,
    'Day': dates_2023.day
})

# Melakukan prediksi harga emas untuk tahun 2023
prediction_array_2023 = prediction_data_2023.to_numpy()
predicted_values_2023 = model2023.predict(prediction_array_2023)

model2024 = RandomForestRegressor(n_estimators=500, max_depth=10, max_leaf_nodes=2000, random_state=42)
model2024.fit(X, y)

# Membuat dataframe untuk setiap tanggal pada tahun 2024
dates_2024 = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
prediction_data_2024 = pd.DataFrame({
    'Year': dates_2024.year,
    'Month': dates_2024.month,
    'Day': dates_2024.day
})

# Melakukan prediksi harga emas untuk tahun 2024
prediction_array_2024 = prediction_data_2024.to_numpy()
predicted_values_2024 = model2024.predict(prediction_array_2024)

# Menyiapkan aplikasi Streamlit
st.title('Aplikasi Prediksi Harga Emas')
st.sidebar.header('Pilih Data')
option = st.sidebar.selectbox('', ('Data Historis', 'Prediksi 2023', 'Prediksi 2024', 'Data Historis dan Hasil Prediksi'))

# Menambahkan penjelasan di aplikasi
st.write('Aplikasi ini bertujuan untuk memvisualisasikan data historis harga emas dan melakukan prediksi harga emas untuk tahun 2023 dan 2024 menggunakan metode Regresi Random Forest. Pengguna dapat memilih opsi yang ingin dilihat dari sidebar aplikasi.')

# Menampilkan grafik harga emas
if option == 'Data Historis':
    st.subheader('Grafik Harga Emas Sebelumnya')
    st.write('Grafik ini menunjukkan perubahan harga emas dari waktu ke waktu berdasarkan data historis yang disediakan.')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data_emas['Date'].values, data_emas['Close'].values, color='black')
    ax.set_title('Data Historis Harga Emas')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga Emas (USD)')
    st.pyplot(fig)

# Menampilkan grafik prediksi harga emas untuk tahun 2023
if option == 'Prediksi 2023':
    st.subheader('Grafik Prediksi Harga Emas Tahun 2023')
    st.write('Grafik ini memperlihatkan prediksi harga emas untuk tahun 2023 berdasarkan model Regresi Random Forest yang dilatih dengan data historis.')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dates_2023, predicted_values_2023, color='red')
    ax.set_title('Prediksi Harga Emas Tahun 2023')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga Emas (USD)')
    st.pyplot(fig)


# Menampilkan grafik prediksi harga emas untuk tahun 2024
if option == 'Prediksi 2024':
    st.subheader('Grafik Prediksi Harga Emas Tahun 2024')
    st.write('Grafik ini memperlihatkan prediksi harga emas untuk tahun 2024 berdasarkan model Regresi Random Forest yang dilatih dengan data historis.')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dates_2024, predicted_values_2024, color='blue')
    ax.set_title('Prediksi Harga Emas Tahun 2024')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga Emas (USD)')
    st.pyplot(fig)

# Menampilkan grafik harga emas dan prediksi untuk tahun 2023
if option == 'Data Historis dan Hasil Prediksi':
    st.subheader('Grafik Harga Emas dan Prediksi Tahun 2023-2024')
    st.write('Grafik ini membandingkan data historis harga emas dengan prediksi harga emas untuk tahun 2023 dan 2024.')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data_emas['Date'].values, data_emas['Close'].values, color='black', label='Data Historis Harga Emas')
    ax.plot(dates_2023, predicted_values_2023, color='red', label='Prediksi 2023')
    
    # Menambahkan prediksi untuk tahun 2024
    ax.plot(dates_2024, predicted_values_2024, color='blue', label='Prediksi 2024')
    
    ax.set_title('Data Historis dan Prediksi Harga Emas Tahun 2023-2024')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga Emas (USD)')
    ax.legend()
    st.pyplot(fig)

