import pickle
import pandas as pd
import streamlit as st

# โหลดโมเดลและตัวแปลงที่บันทึกไว้
with open('model_penguin_66130701919.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# สร้าง UI สำหรับรับข้อมูลจากผู้ใช้
st.title("Penguin Species Prediction")

st.header("Enter Penguin Data")

# รับข้อมูลจากผู้ใช้
island = st.selectbox('Island', ['Torgersen', 'Biscoe', 'Dream'])
sex = st.selectbox('Sex', ['MALE', 'FEMALE'])
culmen_length_mm = st.number_input('Culmen Length (mm)', min_value=0.0, step=0.1)
culmen_depth_mm = st.number_input('Culmen Depth (mm)', min_value=0.0, step=0.1)
flipper_length_mm = st.number_input('Flipper Length (mm)', min_value=0.0, step=0.1)
body_mass_g = st.number_input('Body Mass (g)', min_value=0, step=1)

# เมื่อผู้ใช้กรอกข้อมูลและกดปุ่มทำนาย
if st.button('Predict'):
    # เตรียมข้อมูลใหม่
    x_new = pd.DataFrame({
        'island': [island],
        'culmen_length_mm': [culmen_length_mm],
        'culmen_depth_mm': [culmen_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g],
        'sex': [sex]
    })
    x_new['sex'] = x_new['sex'].str.upper()
    # แปลงค่า island และ sex โดยใช้ตัวแปลงที่บันทึกไว้
    x_new['island'] = island_encoder.transform(x_new['island'])
    x_new['sex'] = sex_encoder.transform(x_new['sex'])

    # ทำนายพันธุ์เพนกวิน
    y_pred = model.predict(x_new)

    # แปลงผลลัพธ์จากตัวเลขเป็นชื่อพันธุ์
    predicted_species = species_encoder.inverse_transform(y_pred)

    # แสดงผลลัพธ์
    st.write(f"Predicted Penguin Species: {predicted_species[0]}")
