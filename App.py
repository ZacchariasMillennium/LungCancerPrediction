import streamlit as st
import numpy as np
import joblib


model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Tüdőrák kockázat becslő", layout="centered")
st.title("🫁 Tüdőrák kockázat becslő alkalmazás")
st.write("Add meg az alábbi adatokat, és megtippeljük a tüdőrák kockázatát.")

age = st.number_input("Kor", min_value=10, max_value=120, step=1, value=50)
gender = st.selectbox("Nem", ["Férfi", "Nő"])
smoking = st.selectbox("Dohányzik?", ["Nem", "Igen"])
finger_discoloration = st.selectbox("Ujj elszíneződése?", ["Nem", "Igen"])
mental_stress = st.selectbox("Mentális stressz?", ["Nem", "Igen"])
pollution = st.selectbox("Légszennyezettségnek kitett?", ["Nem", "Igen"])
illness = st.selectbox("Hosszú távú betegség?", ["Nem", "Igen"])
immune_weakness = st.selectbox("Legyengült immunrendszer?", ["Nem", "Igen"])
breathing = st.selectbox("Légzési nehézség?", ["Nem", "Igen"])
alcohol = st.selectbox("Alkoholfogyasztás?", ["Nem", "Igen"])
throat = st.selectbox("Torokfájás vagy diszkomfort?", ["Nem", "Igen"])
chest = st.selectbox("Mellkasi szorítás?", ["Nem", "Igen"])
family_history = st.selectbox("Családban volt hasonló betegség?", ["Nem", "Igen"])
smoking_family = st.selectbox("Családtag dohányzott?", ["Nem", "Igen"])
stress_immune = st.selectbox("Stressz/immun probléma?", ["Nem", "Igen"])

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