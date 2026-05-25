import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Chargement du modèle et des features
model = joblib.load('model_hospitalisation.pkl')
with open('features.json', 'r') as f:
    features = json.load(f)

st.set_page_config(page_title="Prédiction Hospitalisation", page_icon="🏥", layout="centered")

st.title("🏥 Prédiction d'Hospitalisation aux Urgences")
st.markdown("Remplis les informations du patient pour prédire s'il sera hospitalisé.")

st.divider()

# --- Formulaire ---
col1, col2 = st.columns(2)

with col1:
    departement = st.selectbox("Département", ['Pédiatrie', 'Traumatologie', 'Médecine Générale', 'Chirurgie', 'Gynécologie'])
    categorie = st.selectbox("Catégorie de triage", ['C1', 'C2', 'C3', 'C4', 'C5'])
    genre = st.selectbox("Genre", ['m', 'f'])
    assurance = st.selectbox("Assurance", ['A', 'B', 'C', 'D', 'Without'])

with col2:
    cause = st.selectbox("Cause consultation", ['Maladie', 'Accident', 'Autre'])
    diagnostic = st.text_input("Code Diagnostic CIM-10", value="R509")
    age_annee = st.number_input("Âge (années)", min_value=0, max_value=120, value=25)
    age_mois = st.number_input("Âge (mois)", min_value=0, max_value=11, value=0)
    heure = st.slider("Heure d'admission", 0, 23, 12)

st.divider()

# --- Prédiction ---
if st.button("🔍 Prédire", use_container_width=True):
    age_total = age_annee * 12 + age_mois

    input_data = pd.DataFrame([{
        'DEPARTEMENT': departement,
        'CATEGORIE': categorie,
        'GENRE': genre,
        'ASSURANCE': assurance,
        'CAUSE CONSULTATION': cause,
        'DIAGNOSTIC CODE CM-10': diagnostic,
        'AGE_total': age_total,
        'Heure Admission': heure
    }])

    proba = model.predict_proba(input_data)[0][1]
    seuil = 0.3

    st.subheader("Résultat :")

    if proba >= seuil:
        st.error(f"⚠️ Hospitalisation probable — Score : {proba:.1%}")
    else:
        st.success(f"✅ Hospitalisation peu probable — Score : {proba:.1%}")

    st.progress(float(proba), text=f"Probabilité d'hospitalisation : {proba:.1%}")

st.divider()
st.caption("Projet IA — Prédiction hospitalisation aux urgences | Par Solange ILINGA")