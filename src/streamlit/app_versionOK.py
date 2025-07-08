import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
import joblib
import os
from io import BytesIO
import streamlit as st
from graphviz import Digraph
# -------------------------------------------------
# Configuration de la page Streamlit
# -------------------------------------------------
st.set_page_config(page_title="Accidents Routiers France", layout="wide")
# -------------------------------------------------
# Fonction pour afficher la page d'accueil
# -------------------------------------------------
def display_home():
    st.title("🚗 Les Accidents Routiers en France")
    st.image("https://www.efurgences.net/images/sampledata/accidents/circulation00.jpg", width=200)

    st.markdown("### 🎯 Objectifs de l'application")
    st.markdown("""
- **Visualiser des statistiques globales**
- Filtrer dynamiquement les données
- **Explorer des graphiques par variable**
- Analyser les **corrélations** et visualiser la **carte géographique**
- **Prédire la gravité** d'un accident
- **Comparer plusieurs modèles** de Machine Learning
    """)

    st.markdown("### 🧭 Structure de l'application")
    st.markdown("""
- **Accueil** : Présentation générale
- **À propos** : Informations sur le projet
    """)

    st.markdown("### 🗂️ Présentation des données et notre approche")
    st.markdown("""
- **Les données** : Présentation des données utilisées
- **Notre approche** : Méthodologie et traitement des données
    """)

    st.markdown("### 🔍 Axes d'analyse")
    st.markdown("""
- **Analyse & Prédiction Bimodale** : Avec ou Sans Gravité
- **Analyse & Prédiction Multi-modale** : Indemne, blessé léger, hospitalisé, tué
    """)

    st.markdown("### 🧩 Fonctionnalités par axe")
    st.markdown("""
1. **Analyse interactive** : Graphiques dynamiques
2. **Filtre dynamique** : Critères personnalisables
3. **Visualisations** : Corrélations, carte choroplèthe
4. **Prédiction** : Gravité via Machine Learning
5. **Comparaison de modèles** : Évaluation des performances
    """)

    st.markdown("> 🚦 *Cette application vous aidera à mieux comprendre les accidents routiers en France et à contribuer à la sécurité routière.*")
# -------------------------------------------------
# Fonction pour afficher la section "Notre approche"
# -------------------------------------------------
def display_approach():
    st.header("🧩 Notre approche")

    st.markdown("## 📈 Pipeline global de la démarche")
    with st.expander("📊 Voir le pipeline du projet"):
        diagram = Digraph()
        diagram.attr(rankdir='LR', size='10,5')

        diagram.node("A", "📥 Collecte des données")
        diagram.node("B", "🧹 Préparation\n- Nettoyage\n- Fusion\n- Feature Engineering")
        diagram.node("C", "📊 Analyse exploratoire")
        diagram.node("D", "🤖 Modélisation ML\n(RandomForest, GB, LogReg)")
        diagram.node("E", "🧠 Évaluation")
        diagram.node("F", "🚀 Déploiement Streamlit")

        diagram.edges(["AB", "BC", "CD", "DE", "EF"])
        st.graphviz_chart(diagram)

    st.markdown("## 🧬 Schéma logique des jointures de données")
    with st.expander("🔗 Voir le schéma de fusion (clé `Num_Acc`)"):
        diagram = Digraph()
        diagram.attr(rankdir='TB', size='6,6')

        diagram.node("C", "📄 CARACTERISTIQUES\n(1 ligne par accident)", shape='box')
        diagram.node("L", "📄 LIEUX\n(1 ligne par accident)", shape='box')
        diagram.node("V", "📄 VEHICULES\n(plusieurs par accident)", shape='box')
        diagram.node("U", "📄 USAGERS\n(plusieurs par accident)", shape='box')
        diagram.node("F", "🧩 DataFrame final\n(1 ligne = 1 usager)", shape='ellipse', style='filled', fillcolor='lightblue')

        diagram.edge("C", "F", label="Num_Acc")
        diagram.edge("L", "F", label="Num_Acc")
        diagram.edge("V", "F", label="Num_Acc + num_veh")
        diagram.edge("U", "F", label="Num_Acc + num_veh")

        st.graphviz_chart(diagram)


# -------------------------------------------------
# Fonction pour afficher la section "À propos"
# -------------------------------------------------
def display_about():
    st.header("📄 À propos du projet")

    st.markdown("## 🛣️ Contexte")
    st.markdown("""
Chaque année en France, des milliers d'accidents corporels sont enregistrés.  
Ces données sont collectées par les forces de l’ordre sur les lieux des accidents et centralisées dans le **fichier BAAC** (*Bulletin d’Analyse des Accidents Corporels*).

Ce projet vise à exploiter ces données pour mieux comprendre les facteurs de risque liés aux accidents et développer des outils de visualisation et de prédiction.
    """)

    st.markdown("## 🎯 Objectifs")
    st.markdown("""
- Identifier les variables influençant la **gravité d’un accident**
- Proposer des **visualisations dynamiques**
- Utiliser le **Machine Learning** pour la prédiction
- Rendre l’information accessible à tous via une **application Streamlit**
    """)

    st.markdown("## 📁 Données utilisées")
    with st.expander("🔎 Voir les sources de données utilisées"):
        st.markdown("""
Les données sont issues du site [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/) pour les années **2019 à 2023**.

Elles sont réparties en 4 ensembles :
- **Caractéristiques** : Informations générales sur l'accident
- **Lieux** : Localisation précise
- **Usagers** : Données individuelles des personnes impliquées
- **Véhicules** : Types et caractéristiques des véhicules
        """)

    st.markdown("## ⏳ Ligne du temps du projet")
    with st.expander("📅 Déroulement du projet étape par étape"):
        st.markdown("""
**🔍 Phase 1 – Exploration**  
- Collecte et analyse initiale des fichiers BAAC  
- Prise en main des données complexes et disparates

**🧹 Phase 2 – Préparation**  
- Nettoyage et fusion des données  
- Transformation pour modélisation centrée sur les usagers

**📊 Phase 3 – Analyse exploratoire**  
- Statistiques descriptives  
- Visualisations interactives (Plotly, Seaborn)

**🤖 Phase 4 – Modélisation**  
- Machine Learning : RandomForest, LogisticRegression, GradientBoosting  
- Évaluation : Accuracy, F1-score, AUC

**🧠 Phase 5 – Application Streamlit**  
- Interface intuitive avec filtres, graphiques, prédiction et comparaison
        """)

    st.markdown("## 👥 Équipe projet")
    st.markdown("""
Projet réalisé dans le cadre de la formation **Data Scientist** de [DataScientest](https://datascientest.com), en partenariat avec **Orange**.

**Membres de l'équipe :**
- 👩‍💻 Carine **LOMBARDI**
- 👩‍💻 Salomé **LILIENFELD**
- 👨‍💻 Nicolas **SCHLEWITZ**
- 👨‍💻 Youssef **FOUDIL**
    """)

    st.markdown("## 🙏 Remerciements")
    st.markdown("""
Nous remercions :
- **Orange** pour la confiance et le soutien
- **Manon & Kalome**, nos mentors DataScientest pour leur disponibilité et leur expertise
- Tous ceux qui ont contribué à faire de ce projet une réussite
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.image("https://logo-marque.com/wp-content/uploads/2021/09/Orange-S.A.-Logo-650x366.png", width=100)
    with col2:
        st.image("https://datascientest.com/wp-content/uploads/2022/03/logo-2021.png", width=120)
# -------------------------------------------------
# Fonction pour afficher les données
# -------------------------------------------------
def display_donnees_Description():
    #st.title("📊 Les données utilisées")
    st.markdown("## 📊 Structure des données analysées")
    st.markdown("""
    Notre jeu de données recense **543 446 accidents** de la route survenus en France entre **2019 et 2023**.

    Avant modélisation, le dataset peut contenir **de 43 à 386 variables**, selon qu’il ait été **dichotomisé ou non**.

    Les variables sont réparties en **5 grandes familles** :
    - 🕒 Variables temporelles
    - 🗺️ Variables géographiques
    - 🌦️ Variables environnementales
    - 🚧 Variables liées à l'accident
    - 👤 Variables liées à l’usager

    Nous décrivons ici brièvement ces familles afin de mieux comprendre la préparation à la modélisation.
    """)

    st.markdown("### 🕒 1.1.1 Variables temporelles")
    st.markdown("""
    - **année** : globalement homogène, sauf 2020 (effet Covid-19)  
    - **mois** : ~8% d'accidents par mois  
    - **jour** : répartition uniforme (~14% par jour)  
    - **heure** : majorité des accidents en journée  
    - **lum** : 2/3 des accidents ont lieu en pleine lumière
    """)

    st.markdown("### 🗺️ 1.1.2 Variables géographiques")
    st.markdown("""
    - **situ** : 87% des accidents ont lieu sur chaussée  
    - **nbv_cat** : 2/3 sur routes à 2 voies  
    - **dep** : les départements les plus peuplés ont le plus d’accidents  
    - **agg** : 2/3 des accidents en agglomération  
    - **catr** : presque la moitié dans les métropoles  
    - **int** : 2/3 hors intersection  
    - **circ** : 2/3 sur voies bidirectionnelles (surtout départementales)  
    - **vosp** : 88% "sans objet" (pas de voie réservée)  
    - **prof** : 80% des routes sont plates  
    - **plan** : 80% rectilignes  
    - **infra** : 83% "non renseigné"
    """)

    st.markdown("### 🌦️ 1.1.3 Conditions environnementales")
    st.markdown("""
    - **atm** : 80% des cas, météo normale  
    - **surf** : 80% des cas, route en bon état
    """)

    st.markdown("### 🚧 1.1.4 Conditions de l'accident")
    st.markdown("""
    - **col** : type de collision assez homogène  
    - **senc** : sens de circulation utile pour le contexte  
    - **obs** : 85% sans objet (peu d'obstacles)  
    - **obsm** : 2/3 véhicule, 15% piéton  
    - **choc** : >50% en choc avant  
    - **manv** : 50% sans changement de direction  
    - **motor** : 82% des véhicules sont à essence  
    - **trajet** : 1/3 concerne le loisir  
    - **catv_cat** : 2/3 des véhicules sont des voitures  
    - **vma_cat** : moitié des accidents à 40-50 km/h  
    - **nbveh_cat** : 50% avec 1 véhicule, 30% avec 2
    """)

    st.markdown("### 👤 1.1.5 Caractéristiques des usagers")
    st.markdown("""
    - **sexe** : plus de 2/3 sont des hommes  
    - **age_cat** : les 25-34 ans sont les plus représentés  
    - **place** : 3/4 sont des conducteurs  
    - **catu** : 3/4 également des conducteurs  
    - **secu1** : 60% utilisent une ceinture  
    - **secu2**/**secu3** : souvent absents  
    - **locp** : localisation du piéton à 90% "sans objet"  
    - **etap** : état du piéton à 90% "sans objet"  
    - **nbacc_cat** : la moitié des cas impliquent 2 usagers
    """)

    def convert_df_to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Données filtrées')
        return output.getvalue()

    st.markdown("### 📥 Exploration du jeu de données pour l'année 2023")

    st.markdown("""
    Cette section vous permet d'explorer les jeux de données issus des fichiers BAAC :  
    **CARACTÉRISTIQUES**, **LIEUX**, **VÉHICULES** et **USAGERS**.  
    Vous pouvez appliquer des filtres, visualiser des statistiques, explorer les distributions pour les données de l'année 2023.
    """)
    st.markdown("---")

    def show_data_section(title, filepath, description):
        with st.expander(title):
            st.markdown(f"**Contenu** : {description}")
            df = pd.read_csv(filepath)

            # 🔍 Filtres dynamiques
            st.markdown("### 🔧 Filtres")
            filterable_cols = df.select_dtypes(include=["object", "int", "category"]).columns.tolist()
            filter_cols = st.multiselect("Colonnes à filtrer :", filterable_cols, key=f"filters_{title}")

            for col in filter_cols:
                options = df[col].dropna().unique()
                selected = st.multiselect(f"{col} :", sorted(options), default=options[:5], key=f"vals_{col}_{title}")
                df = df[df[col].isin(selected)]

            st.markdown(f"✅ **{len(df)} lignes sélectionnées**")

            # 👁️ Affichage & analyse
            st.markdown("### 🧾 Aperçu des données")
            st.dataframe(df.head())

            st.markdown("### 📊 Statistiques descriptives")
            st.dataframe(df.describe())

            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            cat_cols = df.select_dtypes(include='object').columns.tolist()

            st.markdown("### 📈 Visualisation")
            viz_type = st.radio("Type :", ["Numérique", "Catégorielle"], horizontal=True, key=f"viz_{title}")

            if viz_type == "Numérique" and numeric_cols:
                col = st.selectbox("Choisir une variable numérique :", numeric_cols, key=f"num_{title}")
                fig = px.histogram(df, x=col, nbins=30, title=f"Distribution de {col}")
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Catégorielle" and cat_cols:
                col = st.selectbox("Choisir une variable catégorielle :", cat_cols, key=f"cat_{title}")
                chart_type = st.radio("Format :", ["Barres", "Camembert"], horizontal=True, key=f"chart_{title}")
                top_values = df[col].value_counts().nlargest(10).reset_index()
                top_values.columns = [col, "Nombre"]

                if chart_type == "Barres":
                    fig = px.bar(top_values, x=col, y="Nombre", title=f"Top 10 catégories de {col}")
                else:
                    fig = px.pie(top_values, names=col, values="Nombre", title=f"Répartition de {col}")
                st.plotly_chart(fig, use_container_width=True)

            elif not numeric_cols and not cat_cols:
                st.info("Aucune variable visualisable.")

    # 🔄 Appel pour chaque dataset
    show_data_section("🧾 Données CARACTÉRISTIQUES - Année 2023)", 
                      "data/accidents_caracteristiques_2023.csv",
                      "Date, heure, météo, luminosité, type de route, intersection...")
    
    show_data_section("📍 Données LIEUX - Année 2023", 
                      "data/accidents_lieux_2023.csv",
                      "Département, commune, type de voie, zone urbaine...")

    show_data_section("🚗 Données VÉHICULES - Année 2023", 
                      "data/accidents_vehicules_2023.csv",
                      "Type, motorisation, âge, manoeuvre...")

    show_data_section("🧍 Données USAGERS - Année 2023", 
                      "data/accidents_usagers_2023.csv",
                      "Sexe, âge, gravité, place dans le véhicule, type d'usager...")
def display_donnees_Caracteristiques():
    st.title("🧾 Données CARACTÉRISTIQUES Année 2023")
    st.markdown("**Contenu** : Date, heure, météo, luminosité, type de route, intersection...")

    df = pd.read_csv("data/accidents_caracteristiques_2023.csv")

    # Réinitialisation
    reset = st.checkbox("🔄 Réinitialiser les filtres", key="reset_car")

    if not reset:
        st.markdown("### 🔧 Filtres")
        filterable_cols = df.select_dtypes(include=["object", "int", "category"]).columns.tolist()
        selected_filters = st.multiselect("Colonnes à filtrer :", filterable_cols, key="filtres_car")

        for col in selected_filters:
            options = df[col].dropna().unique()
            selected_values = st.multiselect(f"{col} :", sorted(options), default=options[:5], key=f"valeurs_{col}_car")
            df = df[df[col].isin(selected_values)]

    st.markdown(f"✅ **{len(df)} lignes après filtrage**")

    # Nombre de lignes à afficher
    max_rows = min(500, len(df))  # Limite de sécurité
    n_rows = st.slider("Nombre de lignes à afficher :", min_value=1, max_value=max_rows, value=5, step=1, key="nrows_car")

    st.download_button(
        "💾 Télécharger les données filtrées (.csv)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="caracteristiques_filtrees.csv",
        mime="text/csv"
    )

    # Affichage du DataFrame
    st.markdown(f"### 🧾 Aperçu des {n_rows} premières lignes")
    st.dataframe(df.head(n_rows))

    st.markdown("### 📊 Statistiques descriptives")
    st.dataframe(df.describe())

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    viz_type = st.radio("Type de variable :", ["Numérique", "Catégorielle"], horizontal=True, key="viz_car")

    if viz_type == "Numérique" and numeric_cols:
        col = st.selectbox("Variable numérique :", numeric_cols, key="num_car")
        st.plotly_chart(px.histogram(df, x=col, nbins=30), use_container_width=True)

    elif viz_type == "Catégorielle" and cat_cols:
        col = st.selectbox("Variable catégorielle :", cat_cols, key="cat_car")
        top = df[col].value_counts().nlargest(10).reset_index()
        top.columns = [col, "Nombre"]
        chart = st.radio("Type :", ["Barres", "Camembert"], horizontal=True, key="chart_car")
        fig = px.bar(top, x=col, y="Nombre") if chart == "Barres" else px.pie(top, names=col, values="Nombre")
        st.plotly_chart(fig, use_container_width=True)
def display_donnees_Lieux():
    st.title("📍 Données LIEUX Année 2023")
    st.markdown("**Contenu** : Département, commune, type de voie, zone urbaine...")

    df = pd.read_csv("data/accidents_lieux_2023.csv")

    reset = st.checkbox("🔄 Réinitialiser les filtres", key="reset_lieux")

    if not reset:
        st.markdown("### 🔧 Filtres")
        filterable_cols = df.select_dtypes(include=["object", "int", "category"]).columns.tolist()
        selected_filters = st.multiselect("Colonnes à filtrer :", filterable_cols, key="filtres_lieux")

        for col in selected_filters:
            options = df[col].dropna().unique()
            selected_values = st.multiselect(f"{col} :", sorted(options), default=options[:5], key=f"valeurs_{col}_lieux")
            df = df[df[col].isin(selected_values)]

    st.markdown(f"✅ **{len(df)} lignes après filtrage**")
    max_rows = min(500, len(df))
    n_rows = st.slider("Nombre de lignes à afficher :", 1, max_rows, 5, key="nrows_lieux")

    st.download_button(
        "💾 Télécharger les données filtrées (.csv)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="lieux_filtrees.csv",
        mime="text/csv"
    )

    st.markdown(f"### 🧾 Aperçu des {n_rows} premières lignes")
    st.dataframe(df.head(n_rows))
    st.markdown("### 📊 Statistiques descriptives")
    st.dataframe(df.describe())

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    viz_type = st.radio("Type de variable :", ["Numérique", "Catégorielle"], horizontal=True, key="viz_lieux")

    if viz_type == "Numérique" and numeric_cols:
        col = st.selectbox("Variable numérique :", numeric_cols, key="num_lieux")
        st.plotly_chart(px.histogram(df, x=col, nbins=30), use_container_width=True)

    elif viz_type == "Catégorielle" and cat_cols:
        col = st.selectbox("Variable catégorielle :", cat_cols, key="cat_lieux")
        top = df[col].value_counts().nlargest(10).reset_index()
        top.columns = [col, "Nombre"]
        chart = st.radio("Type :", ["Barres", "Camembert"], horizontal=True, key="chart_lieux")
        fig = px.bar(top, x=col, y="Nombre") if chart == "Barres" else px.pie(top, names=col, values="Nombre")
        st.plotly_chart(fig, use_container_width=True)
def display_donnees_Vehicules():
    st.title("🚗 Données VÉHICULES Année 2023")
    st.markdown("**Contenu** : Type, motorisation, âge du véhicule, manoeuvre...")

    df = pd.read_csv("data/accidents_vehicules_2023.csv")

    reset = st.checkbox("🔄 Réinitialiser les filtres", key="reset_veh")

    if not reset:
        st.markdown("### 🔧 Filtres")
        filterable_cols = df.select_dtypes(include=["object", "int", "category"]).columns.tolist()
        selected_filters = st.multiselect("Colonnes à filtrer :", filterable_cols, key="filtres_veh")

        for col in selected_filters:
            options = df[col].dropna().unique()
            selected_values = st.multiselect(f"{col} :", sorted(options), default=options[:5], key=f"valeurs_{col}_veh")
            df = df[df[col].isin(selected_values)]

    st.markdown(f"✅ **{len(df)} lignes après filtrage**")
    max_rows = min(500, len(df))
    n_rows = st.slider("Nombre de lignes à afficher :", 1, max_rows, 5, key="nrows_veh")

    st.download_button(
        "💾 Télécharger les données filtrées (.csv)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="vehicules_filtrees.csv",
        mime="text/csv"
    )

    st.markdown(f"### 🧾 Aperçu des {n_rows} premières lignes")
    st.dataframe(df.head(n_rows))
    st.markdown("### 📊 Statistiques descriptives")
    st.dataframe(df.describe())

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    viz_type = st.radio("Type de variable :", ["Numérique", "Catégorielle"], horizontal=True, key="viz_veh")

    if viz_type == "Numérique" and numeric_cols:
        col = st.selectbox("Variable numérique :", numeric_cols, key="num_veh")
        st.plotly_chart(px.histogram(df, x=col, nbins=30), use_container_width=True)

    elif viz_type == "Catégorielle" and cat_cols:
        col = st.selectbox("Variable catégorielle :", cat_cols, key="cat_veh")
        top = df[col].value_counts().nlargest(10).reset_index()
        top.columns = [col, "Nombre"]
        chart = st.radio("Type :", ["Barres", "Camembert"], horizontal=True, key="chart_veh")
        fig = px.bar(top, x=col, y="Nombre") if chart == "Barres" else px.pie(top, names=col, values="Nombre")
        st.plotly_chart(fig, use_container_width=True)
def display_donnees_Usagers():
    st.title("🧍 Données USAGERS Année 2023")
    st.markdown("**Contenu** : Sexe, âge, gravité, type d'usager, place dans le véhicule...")

    df = pd.read_csv("data/accidents_usagers_2023.csv")

    reset = st.checkbox("🔄 Réinitialiser les filtres", key="reset_usa")

    if not reset:
        st.markdown("### 🔧 Filtres")
        filterable_cols = df.select_dtypes(include=["object", "int", "category"]).columns.tolist()
        selected_filters = st.multiselect("Colonnes à filtrer :", filterable_cols, key="filtres_usa")

        for col in selected_filters:
            options = df[col].dropna().unique()
            selected_values = st.multiselect(f"{col} :", sorted(options), default=options[:5], key=f"valeurs_{col}_usa")
            df = df[df[col].isin(selected_values)]

    st.markdown(f"✅ **{len(df)} lignes après filtrage**")
    max_rows = min(500, len(df))
    n_rows = st.slider("Nombre de lignes à afficher :", 1, max_rows, 5, key="nrows_usa")

    st.download_button(
        "💾 Télécharger les données filtrées (.csv)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="usagers_filtrees.csv",
        mime="text/csv"
    )

    st.markdown(f"### 🧾 Aperçu des {n_rows} premières lignes")
    st.dataframe(df.head(n_rows))
    st.markdown("### 📊 Statistiques descriptives")
    st.dataframe(df.describe())

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    viz_type = st.radio("Type de variable :", ["Numérique", "Catégorielle"], horizontal=True, key="viz_usa")

    if viz_type == "Numérique" and numeric_cols:
        col = st.selectbox("Variable numérique :", numeric_cols, key="num_usa")
        st.plotly_chart(px.histogram(df, x=col, nbins=30), use_container_width=True)

    elif viz_type == "Catégorielle" and cat_cols:
        col = st.selectbox("Variable catégorielle :", cat_cols, key="cat_usa")
        top = df[col].value_counts().nlargest(10).reset_index()
        top.columns = [col, "Nombre"]
        chart = st.radio("Type :", ["Barres", "Camembert"], horizontal=True, key="chart_usa")
        fig = px.bar(top, x=col, y="Nombre") if chart == "Barres" else px.pie(top, names=col, values="Nombre")
        st.plotly_chart(fig, use_container_width=True)
# -------------------------------------------------
# Fonction pour afficher l'analyse interactive Bimodale
# ------------------------------------------------
def display_analysis_bi():
    st.title("📊 Analyse interactive Bi-Modal")
    st.markdown("### 🎯 Objectif : Répartition des accidents **Avec vs Sans Gravité**")

    # Chargement des données
    df_graph = pd.read_csv("data/accidents_graphiques.csv")

    # Sélection de la variable
    variable = st.selectbox(
        "🔽 Choisissez une variable à analyser :",
        df_graph.columns.drop("gravite")
    )

    # Histogramme interactif
    fig_hist = px.histogram(
        df_graph,
        x=variable,
        color="gravite",
        barmode="group",
        title=f"Répartition des accidents selon la variable : {variable}",
        labels={"gravite": "Gravité"}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Statistiques descriptives
    with st.expander("📈 Voir les statistiques descriptives globales et par gravité"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Moyenne globale")
            st.write(df_graph[variable].describe())

        with col2:
            st.markdown("#### Moyenne par classe de gravité")
            st.write(df_graph.groupby("gravite")[variable].describe())

    # Camembert optionnel
    with st.expander("🧁 Afficher un camembert de distribution des modalités (si applicable)"):
        if df_graph[variable].nunique() < 20:
            pie_data = df_graph[variable].value_counts().reset_index()
            pie_data.columns = [variable, "count"]
            fig_pie = px.pie(pie_data, names=variable, values="count", title=f"Distribution des valeurs de {variable}")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("🔎 Trop de modalités pour une visualisation en camembert.")
# -------------------------------------------------
# Fonction pour afficher l'analyse interactive Multi-modal
# -------------------------------------------------
def display_analysis_multi():
    st.title("📊 Analyse interactive Multi-Modale")
    st.markdown("### 🎯 Objectif : Étude des accidents selon les **4 niveaux de gravité**")

    df_graph = pd.read_csv("data/accidents_graphiques_multi.csv")

    # Sélection de variable
    variable = st.selectbox(
        "🔽 Choisissez une variable à analyser :",
        df_graph.columns.drop("gravite")
    )

    # Histogramme groupé
    fig_hist = px.histogram(
        df_graph,
        x=variable,
        color="gravite",
        barmode="group",
        title=f"Répartition des accidents selon la variable : {variable}",
        labels={"gravite": "Niveau de gravité"}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Statistiques descriptives
    with st.expander("📈 Voir les statistiques descriptives globales et par niveau de gravité"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Statistiques globales")
            st.write(df_graph[variable].describe())
        with col2:
            st.markdown("#### Statistiques par niveau de gravité")
            st.write(df_graph.groupby("gravite")[variable].describe())

    # Camembert optionnel
    with st.expander("🧁 Camembert des modalités de la variable sélectionnée (si applicable)"):
        if df_graph[variable].nunique() < 20:
            pie_data = df_graph[variable].value_counts().reset_index()
            pie_data.columns = [variable, "count"]
            fig_pie = px.pie(pie_data, names=variable, values="count", title=f"Répartition des modalités : {variable}")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("🔍 Trop de modalités pour un graphique en camembert.")
# -------------------------------------------------
# Fonction pour afficher le filtre dynamique Multi
# -------------------------------------------------
def display_dynamic_filter_multi():
    st.subheader("Filtre dynamique")
    st.write("Utilisez ce filtre pour affiner vos résultats.")
    st.header("🔎 Filtrage dynamique des données")
    df_filtre = pd.read_csv("data/accidents_geolocalises_multi.csv")
    with st.expander("🎛️ Filtres avancés"):
        filtre_dep = st.multiselect("Départements", df_filtre["dep"].unique(), default=df_filtre["dep"].unique())
        filtre_nuit = st.radio("Accident de nuit ?", options=["Tous", 1, 0])
        filtre_weekend = st.radio("Accident le week-end ?", options=["Tous", 1, 0])
    
    df_filtre = df_filtre[df_filtre["dep"].isin(filtre_dep)]
    if filtre_nuit != "Tous":
        df_filtre = df_filtre[df_filtre["nuit"] == filtre_nuit]
    if filtre_weekend != "Tous":
        df_filtre = df_filtre[df_filtre["weekend"] == filtre_weekend]
    
    st.dataframe(df_filtre)
    st.markdown(f"🧮 **{len(df_filtre)}** accidents correspondent à vos critères.")
# -------------------------------------------------
# Fonction pour afficher le filtre dynamique Bimodal
# -------------------------------------------------
def display_dynamic_filter():
    st.subheader("Filtre dynamique")
    st.write("Utilisez ce filtre pour affiner vos résultats.")
    st.header("🔎 Filtrage dynamique des données")
    df_filtre = pd.read_csv("data/accidents_geolocalises.csv")
    with st.expander("🎛️ Filtres avancés"):
        filtre_dep = st.multiselect("Départements", df_filtre["dep"].unique(), default=df_filtre["dep"].unique())
        filtre_nuit = st.radio("Accident de nuit ?", options=["Tous", 1, 0])
        filtre_weekend = st.radio("Accident le week-end ?", options=["Tous", 1, 0])
    
    df_filtre = df_filtre[df_filtre["dep"].isin(filtre_dep)]
    if filtre_nuit != "Tous":
        df_filtre = df_filtre[df_filtre["nuit"] == filtre_nuit]
    if filtre_weekend != "Tous":
        df_filtre = df_filtre[df_filtre["weekend"] == filtre_weekend]
    
    st.dataframe(df_filtre)
    st.markdown(f"🧮 **{len(df_filtre)}** accidents correspondent à vos critères.")
# -------------------------------------------------
# Fonction pour afficher les visualisations
# -------------------------------------------------
def display_visualizations():
    st.subheader("Visualisations")
    st.write("Découvrez nos visualisations de données.")
    st.header("📊 Analyse de Corrélation et Carte Choroplèthe")
    df_viz = pd.read_csv("data/accidents_dep.csv")

    st.subheader("📈 Matrice de corrélation")
    corr = df_viz.select_dtypes(include=['number']).corr()
    fig_corr, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig_corr)

    st.subheader("🌍 Carte des accidents par département")
    fig_map = px.choropleth(
        df_viz,
        geojson="https://france-geojson.gregoiredavid.fr/repo/departements.geojson",
        locations="dep",
        color="nb_accidents",
        featureidkey="properties.code",
        color_continuous_scale="OrRd",
        projection="mercator"
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig_map, use_container_width=True)
# -------------------------------------------------
# Fonction pour afficher les visualisations multi
# -------------------------------------------------
def display_visualizations_multi():
    st.subheader("Visualisations")
    st.write("Découvrez nos visualisations de données.")
    st.header("📊 Analyse de Corrélation et Carte Choroplèthe")
    df_viz = pd.read_csv("data/accidents_dep_multi.csv")

    st.subheader("📈 Matrice de corrélation")
    corr = df_viz.select_dtypes(include=['number']).corr()
    fig_corr, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig_corr)

    st.subheader("🌍 Carte des accidents par département")
    fig_map = px.choropleth(
        df_viz,
        geojson="https://france-geojson.gregoiredavid.fr/repo/departements.geojson",
        locations="dep",
        color="nb_accidents",
        featureidkey="properties.code",
        color_continuous_scale="OrRd",
        projection="mercator"
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig_map, use_container_width=True)
# -------------------------------------------------
# Fonction pour la prédiction
# -------------------------------------------------
def display_prediction():
    st.subheader("Prédiction")
    st.write("Faites des prédictions basées sur les données.")
    st.header("🧠 Prédire la gravité d'un accident")
    st.markdown("Entrez les paramètres d'un accident pour prédire s'il est probable que la personne soit indemne ou blessée/tuée.")

    model_path = "data/model_rf.pkl"
    if not os.path.exists(model_path):
        st.warning("Modèle non trouvé. Création d'un modèle de test...")
        data = pd.read_csv("data/accidents_modele.csv")
        X = data[["dep", "atm", "surf", "lum", "int", "catr", "vma", "heure", "weekend", "nuit"]]
        y = data["gravite"]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, model_path)
        st.success("Modèle entraîné avec succès !")

    model = joblib.load(model_path)

    col1, col2, col3 = st.columns(3)
    with col1:
        dep = st.selectbox("Département", list(range(1, 96)))
        atm = st.selectbox("Conditions atmosphériques", list(range(1, 9)))
        surf = st.selectbox("État de la route", list(range(1, 9)))
    with col2:
        lum = st.selectbox("Luminosité", list(range(1, 6)))
        inter = st.selectbox("Type d'intersection", list(range(1, 7)))
        catr = st.selectbox("Catégorie de route", list(range(1, 10)))
    with col3:
        vma = st.slider("Vitesse maximale autorisée", 30, 130, 50, step=10)
        heure = st.slider("Heure de l'accident", 0, 23, 12)
        nuit = st.radio("Accident la nuit", [0, 1])
        weekend = st.radio("Week-end", [0, 1])

    input_df = pd.DataFrame([[dep, atm, surf, lum, inter, catr, vma, heure, weekend, nuit]],
                            columns=["dep", "atm", "surf", "lum", "int", "catr", "vma", "heure", "weekend", "nuit"])

    if st.button("Prédire"):
        prediction = model.predict(input_df)[0]
        st.success(f"🧾 Résultat : {'Indemne' if prediction == 0 else 'Blessé ou Tué'}")
        st.write("---")
        st.write("**Données d'entrée utilisées :**")
        st.dataframe(input_df)
# -------------------------------------------------
# Fonction pour la prédiction multi
# -------------------------------------------------
def display_prediction_multi():
    st.subheader("Prédiction")
    st.write("Faites des prédictions basées sur les données.")
    st.header("🧠 Prédire la gravité d'un accident")
    st.markdown("Entrez les paramètres d'un accident pour prédire s'il est probable que la personne soit indemne, bléssée légée, bléssée grave ou tuée.")

    model_path = "data/model_rf_multi.pkl"
    if not os.path.exists(model_path):
        st.warning("Modèle non trouvé. Création d'un modèle de test...")
        data = pd.read_csv("data/accidents_modele_multi.csv")
        X = data[["dep", "atm", "surf", "lum", "int", "catr", "vma", "heure", "weekend", "nuit"]]
        y = data["gravite"]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, model_path)
        st.success("Modèle entraîné avec succès !")

    model = joblib.load(model_path)

    col1, col2, col3 = st.columns(3)
    with col1:
        dep = st.selectbox("Département", list(range(1, 96)))
        atm = st.selectbox("Conditions atmosphériques", list(range(1, 9)))
        surf = st.selectbox("État de la route", list(range(1, 9)))
    with col2:
        lum = st.selectbox("Luminosité", list(range(1, 6)))
        inter = st.selectbox("Type d'intersection", list(range(1, 7)))
        catr = st.selectbox("Catégorie de route", list(range(1, 10)))
    with col3:
        vma = st.slider("Vitesse maximale autorisée", 30, 130, 50, step=10)
        heure = st.slider("Heure de l'accident", 0, 23, 12)
        nuit = st.radio("Accident la nuit", [0, 1])
        weekend = st.radio("Week-end", [0, 1])

    input_df = pd.DataFrame([[dep, atm, surf, lum, inter, catr, vma, heure, weekend, nuit]],
                            columns=["dep", "atm", "surf", "lum", "int", "catr", "vma", "heure", "weekend", "nuit"])

    if st.button("Prédire"):
        prediction = model.predict(input_df)[0]
        st.success(f"🧾 Résultat : {'Indemne' if prediction == 0 else 'Blessé ou Tué'}")
        st.write("---")
        st.write("**Données d'entrée utilisées :**")
        st.dataframe(input_df)
# -------------------------------------------------
# Fonction pour la comparaison
# -------------------------------------------------
def display_model_comparison():
    st.title("🤖 Comparaison de modèles de Machine Learning")
    st.markdown("Comparez les performances de plusieurs modèles sur la prédiction de la gravité des accidents.")

    # Chargement des données
    data = pd.read_csv("data/accidents_modele.csv")
    X = data[["dep", "atm", "surf", "lum", "int", "catr", "vma", "heure", "weekend", "nuit"]]
    y = data["gravite"]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=500)
    }

    is_binary = len(y.unique()) == 2
    average_type = "binary" if is_binary else "weighted"
    multi_class_auc = "ovr" if not is_binary else "raise"

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=average_type)
        try:
            auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class=multi_class_auc, average=average_type)
        except:
            auc = None

        results.append({
            "Modèle": name,
            "Accuracy": round(acc, 3),
            "F1-score": round(f1, 3),
            "AUC": round(auc, 3) if auc else "N/A"
        })

    df_results = pd.DataFrame(results)

    with st.expander("📋 Résultats détaillés des modèles"):
        st.dataframe(df_results)

        # Générer fichier CSV
        buffer = BytesIO()
        df_results.to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="📥 Télécharger les résultats (CSV)",
            data=buffer,
            file_name="comparaison_modeles.csv",
            mime="text/csv"
        )

    with st.expander("📊 Visualisation comparative"):
        try:
            fig = px.bar(df_results.melt(id_vars=["Modèle"], value_vars=["Accuracy", "F1-score", "AUC"]),
                         x="Modèle", y="value", color="variable", barmode="group",
                         title="Comparaison des métriques par modèle")
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("❗ Courbes non disponibles (AUC manquant ou modèle sans predict_proba).")
# -------------------------------------------------
# Fonction pour la comparaison multi
# -------------------------------------------------
def display_model_comparison_multi():
    st.title("🤖 Comparaison de modèles de Machine Learning")
    st.markdown("Comparez les performances de plusieurs modèles sur la prédiction de la gravité des accidents.")

    # Chargement des données
    data = pd.read_csv("data/accidents_modele.csv")
    X = data[["dep", "atm", "surf", "lum", "int", "catr", "vma", "heure", "weekend", "nuit"]]
    y = data["gravite"]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=500)
    }

    is_binary = len(y.unique()) == 2
    average_type = "binary" if is_binary else "weighted"
    multi_class_auc = "ovr" if not is_binary else "raise"

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=average_type)
        try:
            auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class=multi_class_auc, average=average_type)
        except:
            auc = None

        results.append({
            "Modèle": name,
            "Accuracy": round(acc, 3),
            "F1-score": round(f1, 3),
            "AUC": round(auc, 3) if auc else "N/A"
        })

    df_results = pd.DataFrame(results)

    with st.expander("📋 Résultats détaillés des modèles"):
        st.dataframe(df_results)

        # Générer fichier CSV
        buffer = BytesIO()
        df_results.to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="📥 Télécharger les résultats (CSV)",
            data=buffer,
            file_name="comparaison_modeles.csv",
            mime="text/csv"
        )

    with st.expander("📊 Visualisation comparative"):
        try:
            fig = px.bar(df_results.melt(id_vars=["Modèle"], value_vars=["Accuracy", "F1-score", "AUC"]),
                         x="Modèle", y="value", color="variable", barmode="group",
                         title="Comparaison des métriques par modèle")
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("❗ Courbes non disponibles (AUC manquant ou modèle sans predict_proba).")
# -------------------------------------------------
# Fonction pour MENU principal
# -------------------------------------------------
def main():
    menu = st.sidebar.selectbox(
        "Sélectionnez une option :",
        [
            "🚗 Accidents Routiers en France",
            "📝 Les données",
            "🧩 Notre approche",
            "📈 Analyse & Prédiction Bimodal",
            "📊 Analyse & Prédiction Multi-modal"
        ]
    )

    if menu == "🚗 Accidents Routiers en France":
        sub_menu = st.sidebar.radio(
            "Choisissez une option :",
            ["🏠 Accueil", "📄 À propos"]
        )
        if sub_menu == "🏠 Accueil":
            display_home()
        elif sub_menu == "📄 À propos":
            display_about()

    elif menu == "📝 Les données":
        st.title("📊 Les données des accidents routiers")
        sub_menu = st.sidebar.radio(
            "📂 Accès rapide aux données :",
            [
                "Présentation globale",
                "Données CARACTÉRISTIQUES",
                "Données LIEUX",
                "Données VÉHICULES",
                "Données USAGERS"
            ]
        )

        if sub_menu == "Présentation globale":
            display_donnees_Description()
        elif sub_menu == "Données CARACTÉRISTIQUES":
            display_donnees_Caracteristiques()
        elif sub_menu == "Données LIEUX":
            display_donnees_Lieux()
        elif sub_menu == "Données VÉHICULES":
            display_donnees_Vehicules()
        elif sub_menu == "Données USAGERS":
            display_donnees_Usagers()

    elif menu == "🧩 Notre approche":
        display_approach()

    elif menu == "📈 Analyse & Prédiction Bimodal":
        sub_menu = st.sidebar.radio(
            "Choisissez une option :",
            ["📊 Analyse interactive", "🎛️ Filtre dynamique", "🔮 Visualisations", "🧠 Prédiction", "🤖 Comparaison de modèles"]
        )
        if sub_menu == "📊 Analyse interactive":
            display_analysis_bi()
        elif sub_menu == "🎛️ Filtre dynamique":
            display_dynamic_filter()
        elif sub_menu == "🔮 Visualisations":
            display_visualizations()
        elif sub_menu == "🧠 Prédiction":
            display_prediction()
        elif sub_menu == "🤖 Comparaison de modèles":
            display_model_comparison()

    elif menu == "📊 Analyse & Prédiction Multi-modal":
        sub_menu = st.sidebar.radio(
            "Choisissez une option :",
            ["📊 Analyse interactive", "🎛️ Filtre dynamique", "🔮 Visualisations", "🧠 Prédiction", "🤖 Comparaison de modèles"]
        )
        if sub_menu == "📊 Analyse interactive":
            display_analysis_multi()
        elif sub_menu == "🎛️ Filtre dynamique":
            display_dynamic_filter_multi()
        elif sub_menu == "🔮 Visualisations":
            display_visualizations_multi()
        elif sub_menu == "🧠 Prédiction":
            display_prediction_multi()
        elif sub_menu == "🤖 Comparaison de modèles":
            display_model_comparison_multi()
if __name__ == "__main__":
    main()