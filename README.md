# 🏥 Prédiction d'Hospitalisation aux Urgences

> Modèle de machine learning pour prédire si un patient admis aux urgences sera hospitalisé, à partir de ses informations d'admission.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6.1-orange?logo=scikit-learn) ![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red?logo=streamlit) ![Status](https://img.shields.io/badge/Status-Portfolio-green)

---

## 📌 Objectif

Prédire si un patient sera **hospitalisé ou non** à partir des informations disponibles lors de son admission aux urgences, en utilisant deux approches de classification supervisée.

---

## 📊 Dataset

| Caractéristique | Valeur |
|---|---|
| Source | Registre d'admissions aux urgences |
| Période | 2018 – 2020 |
| Nombre de patients | 32 024 |
| Colonnes | 15 |
| Taux d'hospitalisation | ~4% (classe déséquilibrée) |

**Variables utilisées :**
- Démographiques : âge, genre, assurance, catégorie de triage
- Cliniques : diagnostic CIM-10, cause de consultation, département
- Temporelles : heure d'admission

---

## ⚙️ Pipeline ML

```
Données brutes
    └── Nettoyage (dropna, normalisation, filtrage âges aberrants)
        └── Feature Engineering (AGE_total, extraction heure)
            └── Préprocessing (OneHotEncoder + StandardScaler)
                └── Modèles (Random Forest / Logistic Regression)
                    └── Évaluation (Classification Report, ROC-AUC, Courbe ROC)
```

---

## 📈 Résultats

| Modèle | Accuracy | Recall (hospitalisés) | F1-score | ROC-AUC |
|---|---|---|---|---|
| Random Forest (seuil=0.5) | 0.965 | 0.206 | 0.316 | 0.852 |
| Random Forest (seuil=0.2) | 0.947 | 0.480 | 0.419 | 0.852 |
| **Logistic Regression** | **0.853** | **0.778** | **0.294** | **0.903** |

> ✅ **Meilleur modèle retenu : Logistic Regression** — ROC-AUC de 0.90 et Recall de 78% sur la classe hospitalisée, ce qui est prioritaire dans un contexte médical.

---

## 🔍 Features les plus importantes (Random Forest)

1. `AGE_total` — âge en mois (feature la plus déterminante)
2. `Heure Admission` — l'heure d'arrivée influence fortement le risque
3. `CATEGORIE` — le niveau de triage (C3, C4)
4. `DIAGNOSTIC CODE CIM-10` — notamment K35 (appendicite), A09, J00
5. `ASSURANCE`, `GENRE`, `DEPARTEMENT` — impact secondaire

---

## 🚀 Application Streamlit

L'application permet de saisir les informations d'un patient et d'obtenir une prédiction en temps réel.

**👉 [Accéder à l'application](https://hospitalisation-app-mwwmugbxfzwlhjdciybgez.streamlit.app/)**

**Fonctionnalités :**
- Formulaire de saisie des informations patient
- Prédiction instantanée avec score de probabilité
- Indicateur visuel du niveau de risque

---

## 🗂️ Structure du projet

```
hospitalisation-app/
├── app.py                        # Application Streamlit
├── model_hospitalisation.pkl     # Modèle entraîné (Logistic Regression)
├── features.json                 # Liste des features utilisées
├── requirements.txt              # Dépendances Python
└── README.md                     # Ce fichier
```

---



---

## ⚠️ Limites du modèle

- **Déséquilibre de classes** (96% / 4%) : peu d'exemples d'hospitalisés pour l'entraînement
- **Precision faible** (0.18) : beaucoup de fausses alertes sur la classe hospitalisée
- **Pas de signes vitaux** : tension, fréquence cardiaque, saturation absents du dataset
- **Pas de validation croisée** : résultats basés sur un seul split train/test

> Ce projet est une **preuve de concept** à visée académique. Il ne constitue pas un outil médical certifié.

---

## 👩‍💻 Auteurs

- DIESSONGO Johanne
- FEUDJEU II La Grâce
- ILINGA Solange

---

## 📚 Technologies utilisées

`Python` `Pandas` `NumPy` `Scikit-learn` `Matplotlib` `Seaborn` `Streamlit` `Joblib`
