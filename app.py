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
X = data_emas[['Year', 'Month', 'Day']].values
y = data_emas['Close'].values

# Membuat model Regresi Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Membuat dataframe untuk setiap tanggal pada tahun 2023
dates_2023 = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
prediction_data_2023 = pd.DataFrame({
    'Year': dates_2023.year,
    'Month': dates_2023.month,
    'Day': dates_2023.day
})

# Melakukan prediksi harga emas untuk tahun 2023
prediction_array_2023 = prediction_data_2023[['Year', 'Month', 'Day']].values
predicted_values_2023 = model.predict(prediction_array_2023)

# Menyiapkan aplikasi Streamlit
st.title('Aplikasi Prediksi Harga Emas')
st.sidebar.header('Pilih Data')
option = st.sidebar.selectbox('', ('Data Historis', 'Prediksi 2023', 'Data Historis dan Hasil Prediksi 2023'))

# Menambahkan penjelasan di aplikasi
st.write('Aplikasi ini bertujuan untuk memvisualisasikan data historis harga emas dan melakukan prediksi harga emas untuk tahun 2023 menggunakan metode Regresi Random Forest. Pengguna dapat memilih opsi yang ingin dilihat dari sidebar aplikasi.')

# Menampilkan grafik harga emas
if option == 'Data Historis':
    st.subheader('Grafik Harga Emas Sebelumnya')
    st.write('Grafik ini menunjukkan perubahan harga emas dari waktu ke waktu berdasarkan data historis yang disediakan.')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data_emas['Date'], data_emas['Close'], color='black')
    ax.set_title('Data Historis Harga Emas')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga Emas (USD)')
    st.pyplot(fig)

if option == 'Prediksi 2023':
    st.subheader('Grafik Prediksi Harga Emas Tahun 2023')
    st.write('Grafik ini memperlihatkan prediksi harga emas untuk tahun 2023 berdasarkan model Regresi Random Forest yang dilatih dengan data historis.')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dates_2023, predicted_values_2023, color='red')
    ax.set_title('Prediksi Harga Emas Tahun 2023')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga Emas (USD)')
    st.pyplot(fig)

if option == 'Data Historis dan Hasil Prediksi 2023':
    st.subheader('Grafik Harga Emas dan Prediksi Tahun 2023')
    st.write('Grafik ini membandingkan data historis harga emas dengan prediksi harga emas untuk tahun 2023.')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data_emas['Date'], data_emas['Close'], color='black', label='Data Historis Harga Emas')
    ax.plot(dates_2023, predicted_values_2023, color='red', label='Prediksi 2023')
    ax.set_title('Data Historis dan Prediksi Harga Emas Tahun 2023')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga Emas (USD)')
    ax.legend()
    st.pyplot(fig)
