import streamlit as st
import numpy as np
import joblib


model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Tüdőrák kockázat becslő", layout="centered")
st.title("🫁 Tüdőrák kockázat becslő alkalmazás")
st.write("Add meg az alábbi adatokat, és megtippeljük a tüdőrák kockázatát.")

st.header("🧍 Alapadatok")
age = st.number_input("Kor", min_value=10, max_value=120, step=1, value=50)
gender = st.selectbox("Mi a nemed?", ["Férfi", "Nő"])

st.header("🚶‍♂️ Életmód és környezeti tényezők")
smoking = st.selectbox("Szoktál dohányozni?", ["Nem", "Igen"])
alcohol = st.selectbox("Fogyasztasz rendszeresen alkoholt?", ["Nem", "Igen"])
pollution = st.selectbox("Élsz vagy dolgozol erősen szennyezett levegőjű környezetben?", ["Nem", "Igen"])
smoking_family = st.selectbox("Dohányzott valaki a közvetlen családodból?", ["Nem", "Igen"])

st.header("🏥 Egészségi állapot és tünetek")
finger_discoloration = st.selectbox("Előfordult már, hogy elszíneződtek az ujjaid?", ["Nem", "Igen"])
mental_stress = st.selectbox("Tapasztalsz gyakran mentális stresszt?", ["Nem", "Igen"])
illness = st.selectbox("Van valamilyen hosszú távú betegséged?", ["Nem", "Igen"])
immune_weakness = st.selectbox("Gyakran legyengül az immunrendszered?", ["Nem", "Igen"])
breathing = st.selectbox("Szoktál légzési nehézségeket tapasztalni?", ["Nem", "Igen"])
throat = st.selectbox("Szoktál torokfájást vagy torokdiszkomfortot érezni?", ["Nem", "Igen"])
chest = st.selectbox("Előfordul nálad mellkasi szorítás vagy nyomásérzés?", ["Nem", "Igen"])
family_history = st.selectbox("Volt már tüdőbetegség a családodban?", ["Nem", "Igen"])
stress_immune = st.selectbox("Volt már, hogy a stressz hatással volt az immunrendszeredre?", ["Nem", "Igen"])


def binary(x): return 1 if x == "Igen" or x == "Férfi" else 0

user_input = np.array([[
    age,
    binary(gender),
    binary(smoking),
    binary(finger_discoloration),
    binary(mental_stress),
    binary(pollution),
    binary(illness),
    binary(immune_weakness),
    binary(breathing),
    binary(alcohol),
    binary(throat),
    binary(chest),
    binary(family_history),
    binary(smoking_family),
    binary(stress_immune)
]])

if st.button("Kockázat becslése"):
    prediction = model.predict(user_input)[0]
    probability = model.predict_proba(user_input)[0][1]

    if prediction == 1:
        st.error("⚠️ A modell szerint fennáll a tüdőrák kockázata.")
    else:
        st.success("✅ A modell szerint NINCS tüdőrák kockázat.")

    st.write(f"**Valószínűség:** {probability:.2%}")