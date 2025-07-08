# ✅ Version REORGANISÉE pour garder **exactement** ta structure d'origine
# ✅ Plus légère, plus rapide, sans duplication, sans print inutile

import streamlit as st
from io import BytesIO
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle
from graphviz import Digraph

# --- CONFIG ---
st.set_page_config(page_title="Accidents Routiers France", layout="wide")

# --- UTILS ---
def detect_mixed_types(df):
    return [col for col in df.columns if len(set(df[col].apply(type))) > 1]

def convert_mixed_types(df, columns):
    for col in columns:
        df[col] = df[col].astype(str)

# --- HOME & ABOUT ---
def display_home():
    st.title("🚗 Accidents Routiers en France")
    st.image("https://www.efurgences.net/images/sampledata/accidents/circulation00.jpg", width=200)
    st.markdown("## Contexte")
    st.write("""
Chaque année en France, des milliers d'accidents corporels sont enregistrés.
Données : fichier BAAC (Bulletin d’Analyse des Accidents Corporels).
Visualisation, prédiction, et exploration interactive.
    """)

def display_about():
    st.header("📄 À propos")
    st.write("Données issues de [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/)")

# --- PIPELINE ---
def display_pipeline():
    diagram = Digraph()
    diagram.attr(rankdir='LR')
    diagram.node("A", "Collecte & Exploration")
    diagram.node("B", "Préparation")
    diagram.node("C", "Analyse Exploratoire")
    diagram.node("D1", "Modèle Bimodal")
    diagram.node("D2", "Modèle Multimodal")
    diagram.edge("A", "B")
    diagram.edge("B", "C")
    diagram.edge("C", "D1")
    diagram.edge("C", "D2")
    st.graphviz_chart(diagram)

# --- VISUALISATIONS ---
def display_corr_map(df):
    corr = df.select_dtypes(include=['number']).corr()
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    all_deps = [f"{i:02d}" for i in range(1, 96)] + ["2A", "2B"]
    df_map = df.groupby("dep").size().reset_index(name="nb_accidents")
    df_map['dep'] = df_map['dep'].astype(str).str.zfill(2)
    df_full = pd.DataFrame({'dep': all_deps}).merge(df_map, how='left', on='dep').fillna(0)

    fig_map = px.choropleth_mapbox(
        df_full, geojson="https://france-geojson.gregoiredavid.fr/repo/departements.geojson",
        locations="dep", color="nb_accidents", featureidkey="properties.code",
        mapbox_style="carto-positron", zoom=4.5, center={"lat": 46.6, "lon": 2.5}, opacity=0.7)
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

# --- PREDICTION ---
def display_prediction():
    st.subheader("Prédiction Bimodale")
    with open("models/streamlit_bin_xgboost.pkl", "rb") as f:
        model = pickle.load(f)
    df = pd.read_csv("data/stream_value_df.csv", sep=';')
    convert_mixed_types(df, detect_mixed_types(df))
    X = df.drop('grav', axis=1)
    y = df['grav']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train, X_test, _, _ = train_test_split(X, y, stratify=y, random_state=42)
    encoder.fit(X_train)

    selections = {}
    for col in X.columns:
        options = X[col].unique().tolist()
        selections[col] = st.selectbox(col, options)

    new_data = pd.DataFrame([selections])
    X_new = encoder.transform(new_data)

    if st.button("Prédire"):
        pred = model.predict(X_new)[0]
        proba = model.predict_proba(X_new)[0]
        st.success(f"Résultat: {'Indemne' if pred==0 else 'Non Indemne'}")
        st.write({f"Classe {i}": f"{round(p*100,2)}%" for i, p in enumerate(proba)})

# --- MAIN MENU ---
def main():
    menu = st.sidebar.selectbox("Menu", [
        "🏠 Accueil", "📄 À propos", "📈 Pipeline", "🔍 Visualisations", "🤖 Prédiction"
    ])

    if menu == "🏠 Accueil":
        display_home()
    elif menu == "📄 À propos":
        display_about()
    elif menu == "📈 Pipeline":
        display_pipeline()
    elif menu == "🔍 Visualisations":
        df = pd.read_csv("data/accidents_dep.csv")
        display_corr_map(df)
    elif menu == "🤖 Prédiction":
        display_prediction()

if __name__ == "__main__":
    main()
