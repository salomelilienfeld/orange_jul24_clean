import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel, f_regression, mutual_info_regression, RFE, RFECV
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import pickle
import os
from io import BytesIO
import streamlit as st
from graphviz import Digraph
from PIL import Image
from pathlib import Path
import gdown
import joblib
import imblearn

st.write(f"✅ scikit-learn version: {sklearn.__version__}")
st.write(f"✅ imbalanced-learn version: {imblearn.__version__}")

# # ========================
# # 📥 Téléchargement Google Drive
# # ========================
# def telecharger_depuis_drive(file_id, local_path):
#     os.makedirs(Path(local_path).parent, exist_ok=True)
#     if not os.path.exists(local_path):
#         url = f"https://drive.google.com/uc?id={file_id}"
#         gdown.download(url, local_path, quiet=False)

# with st.spinner("📥 Chargement initial des fichiers (une seule fois) ..."):
#     telecharger_depuis_drive("13s7wsKJNuQp6nI_i4J3tufyX4qP_ZRYW", "data/accidents_graphiques_multi.csv")
#     telecharger_depuis_drive("1REgG5T14B3IgeD0Q6uAIu0oGlCY7RdMv", "data/accidents_graphiques.csv")
#     telecharger_depuis_drive("1cc8w4b1vhwHJGLGIc4bghVcrMdR1z48x", "models/streamlit_bin_randomforest_none_param_grid_rf.pkl")
#     telecharger_depuis_drive("1zPyVAmngnUdFVuT5-BNVU2dJ8cLLxkJU", "models/streamlit_randomforest_multi_none_param_grid_rf.pkl")
#     telecharger_depuis_drive("1znCeKIaroHNLY5F_qTwolPAzNcTXzU8n", "models/streamlit_randomforest_multi_oversampling_param_grid_rf.pkl")
# st.success("✅ Fichiers téléchargés avec succès")



# -------------------------------------------------
# Configuration de la page Streamlit
# -------------------------------------------------
st.set_page_config(page_title="Prédiction Accidents Routiers", layout="wide")
# -------------------------------------------------
# Fonction pour afficher la page d'accueil
# -------------------------------------------------

@st.cache_data
def load_csv(csv_path, sep=','):
    df = pd.read_csv(csv_path, sep=sep)
    mixed_columns = detect_mixed_types(df)    
    df = convert_mixed_types(df, mixed_columns)  
    return df

# @st.cache_resource
# def load_model(model_path):
#     return joblib.load(model_path)

@st.cache_resource
def load_model(model_path):
    import joblib
    try:
        return joblib.load(model_path)
    except ModuleNotFoundError as e:
        st.error(f"🧨 Module manquant pour charger le modèle : **{e.name}**")
        raise

def display_home():
    st.markdown("#### 🛣️ Contexte")

    st.markdown("""
    Chaque année, des milliers d’**accidents corporels** sont recensés sur les routes françaises.  
    Ces événements sont soigneusement enregistrés par les forces de l’ordre dans le fichier **BAAC**  
    (*Bulletin d’Analyse des Accidents Corporels*).

    👉 Ce projet vise à **analyser ces données** pour mieux comprendre les **circonstances** et **tendances** liées aux accidents,  
    et à développer des outils interactifs pour **visualiser**, **explorer** et **prédire** leur gravité.
    """)
    st.markdown("---")
    st.markdown("#### 🎯 Objectifs de l'application")
    st.markdown("""
    Cette application vous permet de :
    
    - 📊 **Visualiser** des statistiques globales sur les accidents  
    - 🧭 **Filtrer dynamiquement** les données selon différents critères  
    - 📈 **Explorer des graphiques** interactifs pour chaque variable  
    - 🧩 Analyser les **corrélations** entre variables  
    - 🗺️ Afficher une **carte géographique** des accidents  
    - 🧠 **Prédire la gravité** d’un accident à partir de ses caractéristiques  
    - 🤖 **Comparer plusieurs modèles** de Machine Learning
    """)

    st.markdown("---")
    st.info("💡 *Naviguez via le menu latéral pour explorer les différentes fonctionnalités.*")

def display_approach():
    st.markdown("#### 📈 Pipeline global de la démarche")
    with st.expander("📊 Voir le pipeline du projet", expanded=True):
        diagram = Digraph()
        diagram.attr(rankdir='LR', size='14,5')

        # Nœuds avec styles et couleurs pour différencier les étapes
        diagram.node("A", "📥 Collecte & Exploration\n- Collecte\n- Exploration", shape='box', style='filled', fillcolor='lightyellow')
        diagram.node("B", "🧹 Préparation\n- Fusion \n- Nettoyage\n- Création Dictionnaire\n- Feature Engineering\n- Encodage & Normalisation", shape='box', style='filled', fillcolor='lightblue')
        diagram.node("C", "📊 Analyse exploratoire\n- Statistiques\n- Visualisations\n- Corrélations\n- Cartes & Graphiques\n- Khiops", shape='box', style='filled', fillcolor='lightgreen')
        diagram.node("D_bimodal", "🤖 Modélisation ML\nBinaire \n", shape='box', style='filled', fillcolor='lightpink')
        diagram.node("D_multimodal", "🤖 Modélisation ML\nMulticlasse", shape='box', style='filled', fillcolor='lightpink')
        diagram.node("E_bimodal", "🧠 Évaluation Binaire", shape='box', style='filled', fillcolor='lightcoral')
        diagram.node("E_multimodal", "🧠 Évaluation Multiclasse", shape='box', style='filled', fillcolor='lightcoral')

        # Arêtes avec labels pour clarifier le flux
        diagram.edge("A", "B")
        diagram.edge("B", "C")
        diagram.edge("C", "D_bimodal", label='Classification\nIndemne vs Non Indemne')
        diagram.edge("C", "D_multimodal", label='Classification\nIndemne / Blessé léger / Blessé grave / Tué')        
        diagram.edge("D_bimodal", "E_bimodal")
        diagram.edge("D_multimodal", "E_multimodal")

        st.graphviz_chart(diagram)

    st.markdown("""Ce projet suit un pipeline structuré en plusieurs étapes clés pour analyser les accidents routiers en France :
Voici le pipeline global de notre démarche projet, qui suit une architecture classique en data science, depuis l’ingestion des données jusqu’à l’évaluation de modèles prédictifs
- **📥 Exploration** : Collecte et exploration des données
- **🧹 Préparation** : Fusion des données, Nettoyage, création d'un dictionnaire, feature engineering, encodage et normalisation
- **📊 Analyse exploratoire** : Statistiques descriptives, visualisations dynamiques, corrélations, carte choroplèthe, graphiques interactifs, utilisation de Khiops
- **🤖 Modélisation ML** : Deux approches de modélisation, bimodale et multimodale
- **🧠 Évaluation des performances** : Analyse des résultats des modèles
""")

def CollecteDonnees():
    col1, col2 = st.columns([1, 2])  # [largeur image, largeur texte]
    image_path = "src/streamlit/data/CollecteDesDonnees.png"
    with col1:
        try:
            image = Image.open(image_path)
            st.image(image, caption="Schéma du processus de collecte", width=300)
        except FileNotFoundError:
            st.error(f"Image non trouvée : {image_path}")

    with col2:
        st.markdown("#### 📥 Collecte des données & Exploration des Données")
        st.markdown("""
        L'objectif de cette étape consiste à collecter et explorer les données à notre disposition.
        """)
        #st.markdown("#### 📥 Collecte des données")
        with st.expander("📥 Collecte des données"):
            st.markdown("""
            Les données sont issues du site [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/),
            nous avons sélectionnés les données des années **2019 à 2023** pour notre projet.

            Notre jeux de données recense **543 446 usagers accidentés** de la route survenus en France entre **2019 et 2023**.
                                
            Elles sont réparties en 4 ensembles :
            - **Caractéristiques** : Informations générales sur l'accident
            - **Lieux** : Localisation précise
            - **Usagers** : Données individuelles des personnes impliquées
            - **Véhicules** : Types et caractéristiques des véhicules
            """)

        #st.markdown("#### 📊 Exploration des données")
        with st.expander("📊 Exploration des données"):
            st.markdown("""
            La première étape de ce projet a consisté à explorer les données pour en comprendre la structure et les caractéristiques.
            ⚠️ Nous avons récupérés ces données brutes, non corrigées des erreurs de saisie. 
            L'exploration a permis d’identifier :
            - les variables qualitatives et quantitatives  
            - les valeurs nulles ou manquantes  
            - les valeurs aberrantes 
            - les doublons dans les lieux des accidents, etc...
            """)           
        #st.markdown("#### 🧮 Conclusion ")
        with st.expander("🧮 Conclusion "):
            st.markdown("""
            - **Données hétérogènes :**
                - Multiples sources (usagers, véhicules, lieux, caractéristiques de l’accident)
                - Structure relationnelle (plusieurs usagers par accident, plusieurs véhicules, etc.)
            - **Qualité des données :**
                - Valeurs manquantes ou mal codées (zéros, points, cellules vides)
                - Données aberrantes (ex. : âge = 9999, vitesse = 0 sur autoroute)
                - Doublons ou incohérences dans les identifiants (Num_Acc, num_veh...)
            - **Déséquilibre des classes :**
                - Très peu de cas "tués" vs beaucoup de "indemnes" ➝ problème pour la classification
                - Peut fausser les métriques standards (accuracy élevée sans performance réelle)
                """)

def PrepDonnees():
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("src/streamlit/data/PreparationDonnees.png", width=250)

    with col2:
        st.markdown("#### 🧹 Préparation des données")
        st.markdown("""
        L'objectif de cette première étape consiste à créer un DataFrame propre et optimisé, prêt pour la modélisation.
        """)

        with st.expander("🔗 Fusion des données", expanded=False):
            st.markdown("""
            Ce DataFrame a été conçu pour représenter 1 accident par usager impliqué, 
            facilitant ainsi une approche de modélisation centrée sur la prédiction de la gravité de l'accident pour chaque usager. 
            """)
            diagram = Digraph()
            diagram.attr(rankdir='TB', size='7,6')

            diagram.node("C", "📄 CARACTERISTIQUES\n(1 ligne par accident)", shape='box', style='filled', fillcolor='lightyellow')
            diagram.node("L", "📄 LIEUX\n(1 ligne par accident)", shape='box', style='filled', fillcolor='lightcyan')
            diagram.node("V", "📄 VEHICULES\n(plusieurs par accident)", shape='box', style='filled', fillcolor='lightpink')
            diagram.node("U", "📄 USAGERS\n(plusieurs par accident)", shape='box', style='filled', fillcolor='lightgreen')
            diagram.node("F", "🧩 DataFrame final\n(1 ligne = 1 usager)", shape='ellipse', style='filled', fillcolor='lightblue')

            diagram.edge("C", "F", label="Num_Acc")
            diagram.edge("L", "F", label="Num_Acc")
            diagram.edge("V", "F", label="Num_Acc + num_veh")
            diagram.edge("U", "F", label="Num_Acc + num_veh")

            st.graphviz_chart(diagram)
        
        with st.expander("🧽 Préprocessing & Feature Engineering", expanded=False):
            st.markdown("#### 🧽 Preprocessing")
            st.markdown("""
            - Suppression des doublons et des valeurs aberrantes (âge sur 4 chiffres, etc.)
            pour garantir la qualité et la cohérence des données.
            - Corrections des espaces dans les champs d'identifiants
            pour Assurer la cohérence des identifiants et éviter des erreurs lors des jointures ou analyses.
            - Gestion des valeurs manquantes
            pour éviter qu’elles n’altèrent la fiabilité du modèle.
            - Correction des erreurs de saisie (âge ou vitesse aberrantes, etc.)
            pour améliorer la fiabilité des données.
            """)
            st.markdown("#### 🧽 Feature engineering ")
            st.markdown("""
            - Changement de type de certaines variables
            pour rendre les variables cohérentes et exploitables par les algorithmes.
            - Création de nouvelles variables pertinentes (nb_user_acc_cat, nb_user_veh_cat, etc.)
            pour améliorer la modélisation.
            - Discrétisation de certaines colonnes (âge, etc.)
            pour simplifier la structure des données, améliorer l’interprétabilité et la performance.
            - Encodage des variables catégorielles
            pour convertir les variables catégoriques en formats numériques exploitables par les modèles.
            - Normalisation des données
            pour mettre à l’échelle toutes les variables pour certains modèles sensibles à l’échelle.
            - Ré-encodage de la variable cible en respectant l'ordre des catégories de gravité
            pour garantir que les modèles comprennent la hiérarchie des catégories de gravité.
            - Suppression des variables non pertinentes
            pour réduire la complexité du modèle, éviter le surapprentissage et améliorer la performance.
            """)
            

        with st.expander("📘 Création d'un dictionnaire des données", expanded=False):
            st.markdown("#### 👓 Pour faciliter la compréhension et la lecture des données")
            st.markdown(
            """Un dictionnaire a été élaboré, listant toutes les variables avec leurs descriptions et les valeurs possibles.
            Ce dictionnaire est en grande partie basé sur la description des bases de données annuelles sur le site du gouvernement : 
            [https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/](https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/)
            """
            )
            st.markdown("#### 📚 Dictionnaire des données")
            st.markdown("""
            | Variable   | Description |
            |------------|-------------|
            |jour|Jour de l'accident|
            |mois|Mois de l'accident|
            |annee|Annee de l'accident|
            |heure|Heure de l'accident|
            |hrmn|Heure et minutes de l'accident|
            |lum|Conditions d’éclairage dans lesquelles l'accident s'est produit|
            |dep|Département : Code INSEE|
            |com|Commune : Le numéro de commune est un code donné par l‘INSEE|
            |agg|Localisé en agglomération ou non|
            |int|Intersection|
            |atm|Conditions atmosphériques|
            |col|Type de collision|
            |lat|Latitude|
            |lon|Longitude|
            |catr|Catégorie de route|
            |circ|Régime de circulation|
            |nbv|Nombre total de voies de circulation|
            |vosp|Signale l’existence d’une voie réservée, indépendamment du fait que l’accident ait lieu ou non sur cette voie|
            |prof|Profil en long décrit la déclivité de la route à l'endroit de l'accident|
            |plan|Tracé en plan|
            |surf|Etat de la surface|
            |infra|Aménagement - Infrastructure|
            |situ|Situation de l’accident|
            |vma|Vitesse maximale autorisée sur le lieu et au moment de l’accident|
            |senc|Sens de circulation|
            |obs|Obstacle fixe heurté|
            |obsm|Obstacle mobile heurté|
            |choc|Point de choc initial|
            |manv|Manoeuvre principale avant l’accident|
            |motor|Type de motorisation du véhicule|
            |place|Permet de situer la place occupée dans le véhicule par l'usager au moment de l'accident|
            |catu|Catégorie d'usager|
            |grav|Gravité de blessure de l'usager|
            |sexe|Sexe de l'usager|
            |an_nais|Année de naissance de l'usager|
            |trajet|Motif du déplacement au moment de l’accident|
            |secu1|Présence et utilisation de l’équipement de sécurité|
            |secu2|Présence et utilisation de l’équipement de sécurité|
            |secu3|Présence et utilisation de l’équipement de sécurité|
            |locp|Localisation du piéton|
            |etatp|Cette variable permet de préciser si le piéton accidenté était seul ou non|
            |catv_cat *|Catégorie du véhicule revue|
            |heure_cat *|Créneau Horaire de l'accident|
            |age_cat *|Catégorie d'âge de l'usager|
            |nbv_cat *|Catégorie de nombre de voies|
            |vma_cat *|Catégorie de vitesse du véhicule|
            |nbacc_cat *|Nombre de victimes impliquées dans l'accident|
            |nbveh_cat *|Nombre de victimes dans le véhicule|
            |accident_type *|Type d'accident|

            * : Variables ajoutées pour catégoriser ou simplifier certaines données.
                    """)
        
        with st.expander("#### 🧮 Conclusion ", expanded=False):
            st.markdown("#### Avant de passer à l'analyse exploratoire")
            st.markdown("""
            Cette étape a structuré les données pour la modélisation, tout en conservant leur valeur explicative.
            """)

def AnalyseDonnees():
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("src/streamlit/data/AnalyseDesDonnees.png", width=250, caption="Étapes d'analyse des données")

    with col2:
        st.markdown("#### 📊 Analyse des données")
        st.markdown("""
L'analyse descriptive des données a été réalisée en plusieurs étapes :

- **Statistiques descriptives** : Moyennes, médianes, écarts-types  
- **Visualisations dynamiques** : Graphiques interactifs avec Plotly et Seaborn
- **Corrélations** : Analyse des relations entre les variables  
- **Carte choroplèthe** : Visualisation géographique des accidents par département  
- **Utilisation de Khiops** : Utilisation de l'outil Khiops (OpenSource Orange) pour la classification des données non traitées afin de mieux comprendre les relations entre les variables.

Une section dédiée permet d'explorer les données de manière interactive, avec des filtres dynamiques et des graphiques adaptés aux variables sélectionnées dans le menu "Exploration des données".  
        """)

        with st.expander("#### 🧮 Conclusion ", expanded=False):
            st.markdown("""
Cette étape nous a permis de mieux comprendre les données, d'identifier les variables pertinentes et de préparer le terrain pour la modélisation.
- 📈 Visualisation des corrélations entre la gravité (grav) et d'autres variables
- 🧪 Utilisation de tests χ² pour valider statistiquement la dépendance entre grav et d'autres variables.
- ✅ Cette étape a permis de sélectionner les variables les plus corrélées à la gravité et de mieux comprendre les facteurs de risque.
""")

def ModelPredictDonnees():
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("src/streamlit/data/Modelisation.png", width=300, caption="Étapes de Modélisation")

    with col2:
        st.markdown("#### 🤖 Modélisation Machine Learning" )
        st.markdown("""
        Notre étude vise à modéliser la gravité des accidents de la route en France, en ciblant quatre catégories
        d'usagers : indemnes, blessés légers, hospitalisés et tués. 
        Étant donné la faible proportion des usagers tués et de mauvais résultat en multi-class 
        nous avons testé une approche par classification binaire. 
                    
        Celle-ci oppose la catégorie des usagers indemnes aux trois autres catégories regroupées, 
        afin d'appliquer des modèles de classification binaire sur des données équilibrées. 
        (Gravité corporel ou Indemne).
                        
        2 Approches de modélisation ont été testées :
        - **Binaire**    : Modélisation de la gravité des accidents de la route en France en ciblant 2 catégories : Indemne ou Non Indemne.
        - **Multiclasse** : Modélisation de la gravité des accidents de la route en France en ciblant 4 catégories: Indemnes, blessés légers, hospitalisés et tués.
       """)
        with st.expander("#### 🧮 Conclusion ", expanded=False):
            st.markdown("""
        Cette étape a permis d’évaluer différentes stratégies de modélisation :
        - **Binaire** : Approche plus simple, mais moins informative    
        - **Multiclasse** : Plus complexe, mais plus représentative de la réalité des accidents
        
        Et plus particulièrement :
        - De **comparer plusieurs modèles de classification** (RandomForest, GradientBoosting, CatBoost, LogisticRegression)
        - **D'évaluer les performances** de chaque modèle sur les données de test
        - **D'identifier le meilleur modèle** pour notre Use Case 
        """)            

def EvalModeles():
    st.markdown("#### 🧠 Évaluation des performances"
                    )
    st.markdown("""
Les performances des modèles ont été évaluées sur les données de test :
- **RandomForest** : Meilleur compromis entre précision et robustesse
- **GradientBoosting** : Très performant mais plus sensible aux sur-ajustements
- **CatBoost** : Au même niveau que GradientBoosting
- **LogisticRegression** : Simple et efficace pour les problèmes linéaires
- **Comparaison des modèles** : Visualisation des métriques de performance
    """)

def display_conclusion():
    st.markdown("#### 🎉 Conclusion"
                    )
    st.markdown("""
Ce projet a permis de :
- Comprendre les facteurs de risque liés aux accidents routiers
- Développer des outils de visualisation et de prédiction
- Rendre l'information accessible à tous via une application Streamlit
    """)



# -------------------------------------------------
# Fonction pour afficher la section "À propos"
# -------------------------------------------------
def display_about():
    st.markdown("#### 👥 Équipe projet")
    st.markdown("""
Projet réalisé dans le cadre de la formation **Data Scientist** de [DataScientest](https://datascientest.com), en partenariat avec **Orange**.

**Membres de l'équipe :**
- 👩‍💻 Salomé **LILIENFELD**
- 👨‍💻 Nicolas **SCHLEWITZ**
- 👨‍💻 Youssef **FOUDIL**
- 👩‍💻 Carine **LOMBARDI**
    """)
    st.markdown("---")
    st.markdown("#### 🙏 Remerciements")
    st.markdown("""
Nous remercions :
- **Orange** pour la confiance et le soutien
- **Manon & Kalome**, nos mentors DataScientest pour leur disponibilité et leur expertise
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.image("https://logo-marque.com/wp-content/uploads/2021/09/Orange-S.A.-Logo-650x366.png", width=100)
    with col2:
        st.image("https://datascientest.com/wp-content/uploads/2022/03/logo-2021.png", width=366)



# -------------------------------------------------
# Fonction pour afficher les données
# -------------------------------------------------
def display_donnees_Description():
    #st.title("📊 Les données utilisées")
    st.markdown("#### 📊 Structure des données analysées")
    st.markdown("""
    Notre jeu de données fournie les données des accidents corporels de la circulation par année et est répartie en 4 fichiers distincts :
    - 📝 La rubrique CARACTERISTIQUES qui décrit les circonstances générales de l’accident (Date, heure, météo, luminosité, type de route, intersection...)
    - 📍 La rubrique LIEUX qui décrit le lieu principal de l’accident même si celui-ci s’est déroulé à une intersection (Département, commune, type de voie, zone urbaine...)
    - 🚗 La rubrique VEHICULES impliqués (Type, motorisation, âge du véhicule, manoeuvre...)
    - 🚶‍♂️ La rubrique USAGERS impliqués (Sexe, âge, gravité, type d'usager, place dans le véhicule...)
    
    Pour chaque accident, il peut y avoir plusieurs usagers et plusieurs véhicules.
    Chaque accident est identifié par un numéro unique (Num_Acc) et peut impliquer plusieurs véhicules et usagers.
    Les données sont structurées de manière relationnelle, où chaque rubrique est liée par le numéro d'identifiant de l'accident ("Num_Acc").
    Quand un accident comporte plusieurs véhicules, il faut aussi pouvoir relier chaque véhicule à ses occupants. Ce lien est fait par la variable id_vehicule.
                 
    Au final, ce sont 20 fichiers .csv (4. Types * 5 années) qui ont été fusionnés  pour obtenir un jeu de données unique, centré sur les usagers impliqués dans les accidents.
    Nous avons donc un jeu de données avec 1 ligne = 1 usager impliqué dans un accident.
            """)        
    
    st.markdown("#### 📊 Aperçu des données")
    st.markdown("""
                

    Notre jeux de données recense **543 446 usagers accidentés** de la route survenus en France entre **2019 et 2023**.

    Les variables sont réparties en **5 grandes familles** :
    - 🕒 Variables temporelles
    - 🗺️ Variables géographiques
    - 🌦️ Variables environnementales
    - 🚧 Variables liées à l'accident
    - 👤 Variables liées à l’usager

    Nous décrivons ici brièvement ces familles afin de mieux comprendre la préparation à la modélisation.
    """)

    st.markdown("#### 🕒 1.1.1 Variables temporelles")
    st.markdown("""
    - **année** : globalement homogène, sauf 2020 (effet Covid-19)  
    - **mois** : ~8% d'accidents par mois  
    - **jour** : répartition uniforme (~14% par jour)  
    - **heure** : majorité des accidents en journée  
    - **lum** : 2/3 des accidents ont lieu en pleine lumière
    """)

    st.markdown("#### 🗺️ 1.1.2 Variables géographiques")
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

    st.markdown("#### 🌦️ 1.1.3 Conditions environnementales")
    st.markdown("""
    - **atm** : 80% des cas, météo normale  
    - **surf** : 80% des cas, route en bon état
    """)

    st.markdown("#### 🚧 1.1.4 Conditions de l'accident")
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

    st.markdown("#### 👤 1.1.5 Caractéristiques des usagers")
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

    # st.markdown("#### 📥 Exploration du jeu de données pour l'année 2023")

    # st.markdown("""
    # Cette section vous permet d'explorer les jeux de données issus des fichiers BAAC :  
    # **CARACTÉRISTIQUES**, **LIEUX**, **VÉHICULES** et **USAGERS**.  
    # Vous pouvez appliquer des filtres, visualiser des statistiques, explorer les distributions pour les données de l'année 2023.
    # """)
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

    # # 🔄 Appel pour chaque dataset
    # show_data_section("🧾 Données CARACTÉRISTIQUES - Année 2023)", 
    #                   "data/accidents_caracteristiques_2023.csv",
    #                   "Date, heure, météo, luminosité, type de route, intersection...")
    
    # show_data_section("📍 Données LIEUX - Année 2023", 
    #                   "data/accidents_lieux_2023.csv",
    #                   "Département, commune, type de voie, zone urbaine...")

    # show_data_section("🚗 Données VÉHICULES - Année 2023", 
    #                   "data/accidents_vehicules_2023.csv",
    #                   "Type, motorisation, âge, manoeuvre...")

    # show_data_section("🧍 Données USAGERS - Année 2023", 
    #                   "data/accidents_usagers_2023.csv",
    #                   "Sexe, âge, gravité, place dans le véhicule, type d'usager...")

def display_donnees_Observations():
    st.markdown("## 📝 Observations sur le jeu de données")
    #st.title("💡 Observations sur le jeu de données : ")
    st.markdown("""
    - une absence de la colonne ‘id_usager’ dans les fichiers ‘Usagers’ des années 2019 et 2020,
    - des doublons d’identifiants d’accidents dans le fichier ‘Lieux’,
    - un changement de nom d’identifiant du numéro de l’accident dans le fichier ‘Caractéristiques’ pour l’année 2022,
    - une présence de types mixtes dans les colonnes ['voie', 'v2', 'nbv', 'lartpc'] des fichiers ‘Lieux’,
    - ainsi que dans la colonne ['addr'] des fichiers ‘Caracteristiques’,
    - enfin, une présence d’espaces dans les identifiants ‘id_vehicules’ et ‘id_usagers’ dans les fichiers ‘Vehicules’ et ‘Usagers’.")
    """)
    st.markdown("### 🛠️ Solutions apportées : ")
    st.markdown("""
    - Pour l’absence de la colonne ‘id_usager’ dans les fichiers ‘Usagers’ des années 2019 et 2020, nous avons créé une nouvelle colonne ‘id_usager’ en incrémentant un identifiant unique pour chaque usager.
    - Pour les doublons d’identifiants d’accidents dans le fichier ‘Lieux’, nous avons supprimé les doublons en gardant la première occurrence.
    - Pour le changement de nom d’identifiant du numéro de l’accident dans le fichier ‘Caractéristiques’ pour l’année 2022, nous avons renommé la colonne ‘num_acc’ en ‘Num_Acc’ pour uniformiser avec les autres années.
    - Pour les types mixtes dans les colonnes ['voie', 'v2', 'nbv', 'lartpc'] des fichiers ‘Lieux’, nous avons converti ces colonnes en chaînes de caractères pour éviter les erreurs de type.
    - Pour la présence d’espaces dans les identifiants ‘id_vehicules’ et ‘id_usagers’ dans les fichiers ‘Vehicules’ et ‘Usagers’, nous avons supprimé les espaces superflus.
    - Enfin, pour la colonne ['addr'] des fichiers ‘Caracteristiques’, nous avons converti les valeurs en chaînes de caractères et supprimé les espaces superflus.
    """)

                
def display_donnees_Caracteristiques():
    st.markdown("## 🧾 Données CARACTÉRISTIQUES Année 2023")
    #st.title("🧾 Données CARACTÉRISTIQUES Année 2023")
    st.markdown("**Contenu** : Date, heure, météo, luminosité, type de route, intersection...")

    df = pd.read_csv("src/streamlit/data/accidents_caracteristiques_2023.csv")

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
    st.markdown("## 📍 Données LIEUX Année 2023")
    #st.title("📍 Données LIEUX Année 2023")
    st.markdown("**Contenu** : Département, commune, type de voie, zone urbaine...")

    df = pd.read_csv("src/streamlit/data/accidents_lieux_2023.csv")

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
    st.markdown("## 🚗 Données VÉHICULES Année 2023")
    #st.title("🚗 Données VÉHICULES Année 2023")
    st.markdown("**Contenu** : Type, motorisation, âge du véhicule, manoeuvre...")

    df = pd.read_csv("src/streamlit/data/accidents_vehicules_2023.csv")

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
    st.markdown("## 🧍 Données USAGERS Année 2023")
    #st.title("🧍 Données USAGERS Année 2023")
    st.markdown("**Contenu** : Sexe, âge, gravité, type d'usager, place dans le véhicule...")

    df = pd.read_csv("src/streamlit/data/accidents_usagers_2023.csv")

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
# Fonction pour afficher l'analyse interactive Binaire
# ------------------------------------------------
def display_analysis_bi():
    #st.title("📈 Analyse interactive Binaire")
    st.markdown("### 🎯 Objectif : Répartition des accidents **Avec vs Sans Gravité**")

    # Chargement des données
    # AMAJ
    df_graph = pd.read_csv("src/streamlit/data/accidents_graphiques.csv")

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
# Fonction pour afficher l'analyse interactive Multiclasse
# -------------------------------------------------
def display_analysis_multi():
    #st.title("📊 Analyse Multiclasse")
    st.markdown("### 🎯 Objectif : Étude des accidents selon les **4 niveaux de gravité**")
    # AMAJ
    df_graph = pd.read_csv("src/streamlit/data/accidents_graphiques_multi.csv")

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
    #st.title("📊 Analyse Multiclasse")    
    st.subheader("🔎Filtre dynamique")
    st.write("Utilisez ce filtre pour affiner vos résultats.")
    #st.header("🔎 Filtrage dynamique des données")
    # AMAJ
    df_filtre = pd.read_csv("src/streamlit/data/accidents_graphiques_multi.csv")
    
    # Définir les options de filtres à partir des définitions
    dep_options = df_filtre["dep"].unique()
    agg_options = ["Tous", "Hors agglomération", "En agglomération"]
    catv_options = df_filtre["catv_cat_s"].unique()
    secu_options = df_filtre["secu1"].unique()
    sexe_options = ["Tous", "Masculin", "Féminin"]
    age_options = ["Tous"] + sorted(df_filtre["age_cat"].unique().tolist())

    with st.expander("🎛️ Filtres avancés"):
        filtre_dep = st.multiselect("Départements", dep_options, default=dep_options)
        filtre_agg = st.radio("Accident en agglomération ?", options=agg_options)
        filtre_type_vehicule = st.multiselect("Type de véhicule", catv_options, default=catv_options)
        filtre_secu = st.multiselect("Elément de sécurité", secu_options, default=secu_options)
        filtre_sexe = st.radio("Sexe de l'usager ?", options=sexe_options, key="filtre_sexe")
        filtre_age_cat = st.selectbox("Catégorie d'âge de l'usager", options=age_options, index=0, key="filtre_age_cat")
    
    # Appliquer les filtres
    df_filtre = df_filtre[df_filtre["dep"].isin(filtre_dep)]
    if filtre_agg != "Tous":
        df_filtre = df_filtre[df_filtre["agg"] == filtre_agg]
    if filtre_type_vehicule:
        df_filtre = df_filtre[df_filtre["catv_cat_s"].isin(filtre_type_vehicule)]
    if filtre_secu:
        df_filtre = df_filtre[df_filtre["secu1"].isin(filtre_secu)]
    if filtre_sexe != "Tous":
        df_filtre = df_filtre[df_filtre["sexe"] == filtre_sexe]
    if filtre_age_cat != "Tous":
        df_filtre = df_filtre[df_filtre["age_cat"] == filtre_age_cat]
    
    # Résumé et téléchargement
    st.markdown(f"✅ **{len(df_filtre)} lignes après filtrage**")
    max_rows = min(500, len(df_filtre))
    n_rows = st.slider("Nombre de lignes à afficher :", 1, max_rows, 5, key="nrows_filtre")
    st.download_button(
        "💾 Télécharger les données filtrées (.csv)",
        df_filtre.to_csv(index=False).encode("utf-8"),
        file_name="bimodal_filtrees.csv",
        mime="text/csv"
    )

    st.dataframe(df_filtre.head(n_rows))
    st.markdown(f"🧮 **{len(df_filtre)}** accidents correspondent à vos critères.")
# -------------------------------------------------
# Fonction pour afficher le filtre dynamique Binaire
# -------------------------------------------------
def display_dynamic_filter():
    st.subheader("🔎Filtre dynamique")
    st.write("Utilisez ce filtre pour affiner vos résultats.")
    #st.header("🔎 Filtrage dynamique des données")
    df_filtre = pd.read_csv("src/streamlit/data/accidents_graphiques.csv")
    
    # Définir les options de filtres à partir des définitions
    dep_options = df_filtre["dep"].unique()
    agg_options = ["Tous", "Hors agglomération", "En agglomération"]
    catv_options = df_filtre["catv_cat_s"].unique()
    secu_options = df_filtre["secu1"].unique()
    sexe_options = ["Tous", "Masculin", "Féminin"]
    age_options = ["Tous"] + sorted(df_filtre["age_cat"].unique().tolist())

    with st.expander("🎛️ Filtres avancés"):
        filtre_dep = st.multiselect("Départements", dep_options, default=dep_options)
        filtre_agg = st.radio("Accident en agglomération ?", options=agg_options)
        filtre_type_vehicule = st.multiselect("Type de véhicule", catv_options, default=catv_options)
        filtre_secu = st.multiselect("Elément de sécurité", secu_options, default=secu_options)
        filtre_sexe = st.radio("Sexe de l'usager ?", options=sexe_options, key="filtre_sexe")
        filtre_age_cat = st.selectbox("Catégorie d'âge de l'usager", options=age_options, index=0, key="filtre_age_cat")
    
    # Appliquer les filtres
    df_filtre = df_filtre[df_filtre["dep"].isin(filtre_dep)]
    if filtre_agg != "Tous":
        df_filtre = df_filtre[df_filtre["agg"] == filtre_agg]
    if filtre_type_vehicule:
        df_filtre = df_filtre[df_filtre["catv_cat_s"].isin(filtre_type_vehicule)]
    if filtre_secu:
        df_filtre = df_filtre[df_filtre["secu1"].isin(filtre_secu)]
    if filtre_sexe != "Tous":
        df_filtre = df_filtre[df_filtre["sexe"] == filtre_sexe]
    if filtre_age_cat != "Tous":
        df_filtre = df_filtre[df_filtre["age_cat"] == filtre_age_cat]
    
    # Résumé et téléchargement
    st.markdown(f"✅ **{len(df_filtre)} lignes après filtrage**")
    max_rows = min(500, len(df_filtre))
    n_rows = st.slider("Nombre de lignes à afficher :", 1, max_rows, 5, key="nrows_filtre")
    st.download_button(
        "💾 Télécharger les données filtrées (.csv)",
        df_filtre.to_csv(index=False).encode("utf-8"),
        file_name="bimodal_filtrees.csv",
        mime="text/csv"
    )

    st.dataframe(df_filtre.head(n_rows))
    st.markdown(f"🧮 **{len(df_filtre)}** accidents correspondent à vos critères.")
# -------------------------------------------------
# Fonction pour afficher les visualisations
# -------------------------------------------------
def display_visualizations():
    st.subheader("Visualisations")
    st.write("Découvrez nos visualisations de données.")
    st.header("📊 Analyse de Corrélation et Carte Choroplèthe")
    df_viz = pd.read_csv("src/streamlit/data/accidents_dep.csv")

    st.subheader("📈 Matrice de corrélation")
    corr = df_viz.select_dtypes(include=['number']).corr()
    fig_corr, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig_corr)

    # st.plotly_chart(fig_map, use_container_width=True)
    import plotly.express as px

    # ✅ 1) Liste complète de tous les départements, au format str
    all_deps = [f"{i:02d}" for i in range(1, 96)] + ["2A", "2B"]

    # ✅ 2) Groupby et forcer str + zfill
    df_map = (
        df_viz.groupby("dep")
        .agg(nb_accidents=("dep", "count"))
        .reset_index()
    )
    df_map["dep"] = df_map["dep"].astype(str).str.zfill(2)

    # ✅ 3) Merge sur liste complète pour remplir les manquants
    df_all_deps = pd.DataFrame({"dep": all_deps})
    df_map = df_all_deps.merge(df_map, on="dep", how="left").fillna(0)
    df_map["nb_accidents"] = df_map["nb_accidents"].astype(int)

    # ✅ 4) Carte Mapbox
    st.subheader("🌍 Carte des accidents par département (version Mapbox)")

    fig_map = px.choropleth_mapbox(
        df_map,
        geojson="https://france-geojson.gregoiredavid.fr/repo/departements.geojson",
        locations="dep",
        color="nb_accidents",
        featureidkey="properties.code",
        color_continuous_scale="OrRd",
        mapbox_style="carto-positron",   # Fond clair élégant (tu peux tester "carto-darkmatter" pour sombre)
        center={"lat": 46.6, "lon": 2.5},  # Centre de la France
        zoom=4.5,
        opacity=0.7,
    )

    fig_map.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )

    st.plotly_chart(fig_map, use_container_width=True)



# -------------------------------------------------
# Fonction pour afficher les visualisations multi
# -------------------------------------------------
def display_visualizations_multi():
    st.subheader("Visualisations")
    st.write("Découvrez nos visualisations de données.")
    st.header("📊 Analyse de Corrélation et Carte Choroplèthe")
    df_viz = pd.read_csv("src/streamlit/data/accidents_dep_multi.csv")

    st.subheader("📈 Matrice de corrélation")
    corr = df_viz.select_dtypes(include=['number']).corr()
    fig_corr, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig_corr)

    import plotly.express as px

    # ✅ 1) Liste complète des départements
    all_deps = [f"{i:02d}" for i in range(1, 96)] + ["2A", "2B"]

    # ✅ 2) Agréger et formatter
    df_map = (
        df_viz.groupby("dep")
        .agg(nb_accidents=("dep", "count"))
        .reset_index()
    )
    df_map["dep"] = df_map["dep"].astype(str).str.zfill(2)

    # ✅ 3) Merge sur la liste complète
    df_all_deps = pd.DataFrame({"dep": all_deps})
    df_map = df_all_deps.merge(df_map, on="dep", how="left").fillna(0)
    df_map["nb_accidents"] = df_map["nb_accidents"].astype(int)

    # ✅ 4) Carte Mapbox multimodal
    st.subheader("🌍 Carte des accidents par département (multimodal)")

    fig_map = px.choropleth_mapbox(
        df_map,
        geojson="https://france-geojson.gregoiredavid.fr/repo/departements.geojson",
        locations="dep",
        color="nb_accidents",
        featureidkey="properties.code",
        color_continuous_scale="OrRd",
        mapbox_style="carto-positron",  # ou "carto-darkmatter" pour thème sombre
        center={"lat": 46.6, "lon": 2.5},
        zoom=4.5,
        opacity=0.7,
    )

    fig_map.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )

    st.plotly_chart(fig_map, use_container_width=True)

# Fonction pour detecter les types de données mixtes
def detect_mixed_types(df):
    mixed_type_columns = []
    for col in df.columns:
        unique_types = set(df[col].apply(type))
        if len(unique_types) > 1:
            mixed_type_columns.append(col)
    return mixed_type_columns

# Fonction pour convertir types de données mixtes en str
def convert_mixed_types(df, mixed_type_columns):
    for col in mixed_type_columns:
        df[col] = df[col].astype('str')
    return df

# Fonction pour charger et préparer les données, mise en cache
def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path, sep=';')
    mixed_columns = detect_mixed_types(df)    
    convert_mixed_types(df, mixed_columns)    
    X = df.drop('grav', axis=1)
    y = df['grav']
    X['annee'] = X['annee'].astype('category')
    X['dep']=X['dep'].astype('category')
    X['dep'] = X['dep'].astype('str')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X, y, X_train, X_test, y_train, y_test

# -------------------------------------------------
# Fonction pour la prédiction
# -------------------------------------------------
def display_prediction():
    st.subheader("🧠 Prédire la gravité d'un accident avec le modèle XGBoost Année 2023")
    st.markdown("Entrez les paramètres d'un accident pour prédire s'il est probable que la personne soit indemne ou blessée/tuée.")

    model_path = "models/streamlit_bin_xgboost_none_param_grid_light.joblib"
    loaded_pickle_model = load_model(model_path)

    X, y, X_train, X_test, y_train, y_test = load_and_prepare_data("data/stream_value_df.csv")

    # Initialiser le OneHotEncoder avec handle_unknown='ignore' pour gérer les modalités inconnues dans le test
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop=None)  # drop='first' pour éviter la multicolinéarité, mais dans les modèles Arbres ou Forets, ça peut être mis à None

    # Encoder uniquement sur l'entraînement
    encoder.fit(X_train)

    columns = X.columns
    col_content = {}

    for col in columns:
        filename = f"data/{col}_possibles.csv"
        df = pd.read_csv(filename)
        col_content[col] = df[col].dropna().tolist()

    jours_possibles = col_content['jour']
    mois_possibles = col_content['mois']
    lum_possibles = col_content['lum']
    agg_possibles = col_content['agg']
    int_possibles = col_content['int']
    atm_possibles = col_content['atm']
    col_possibles = col_content['col']
    catr_possibles = col_content['catr']
    circ_possibles = col_content['circ']
    vosp_possibles = col_content['vosp']
    prof_possibles = col_content['prof']
    plan_possibles = col_content['plan']
    surf_possibles = col_content['surf']
    infra_possibles = col_content['infra']
    situ_possibles = col_content['situ']
    senc_possibles = col_content['senc']
    obs_possibles = col_content['obs']
    obsm_possibles = col_content['obsm']
    choc_possibles = col_content['choc']
    manv_possibles = col_content['manv']
    motor_possibles = col_content['motor']
    place_possibles = col_content['place']
    catu_possibles = col_content['catu']
    sexe_possibles = col_content['sexe']
    trajet_possibles = col_content['trajet']
    secu1_possibles = col_content['secu1']
    secu2_possibles = col_content['secu2']
    secu3_possibles = col_content['secu3']
    locp_possibles = col_content['locp']
    etatp_possibles = col_content['etatp']
    annees_possibles = col_content['annee']
    heure_cat_possibles = col_content['heure_cat']
    age_cat_possibles = col_content['age_cat']
    catv_cat_s_possibles = col_content['catv_cat_s']
    nbv_cat_possibles = col_content['nbv_cat']
    vma_cat_possibles = col_content['vma_cat']
    accident_type_possibles = col_content['accident_type']
    dep_possibles = col_content['dep']

    col1, col2, col3 = st.columns(3)
    with col1:
        annee_selection = st.selectbox('Choisissez une annee', annees_possibles)
        mois_selection = st.selectbox('Choisissez un mois', mois_possibles)
        jour_selection = st.selectbox('Choisissez un jour', jours_possibles)
        heure_cat_selection = st.selectbox('Choisissez une heure', heure_cat_possibles)
        dep_selection = st.selectbox('Choisissez un département', dep_possibles)
        lum_selection = st.selectbox('Choisissez une luminosité', lum_possibles)
        agg_selection = st.selectbox('Agglomération', agg_possibles)
        int_selection = st.selectbox('Intersection', int_possibles)
        atm_selection = st.selectbox('Conditions atmosphériques', atm_possibles)
        catr_selection = st.selectbox('Catégorie de route', catr_possibles)
        circ_selection = st.selectbox('Régime de circulation', circ_possibles)
        vosp_selection = st.selectbox('Voie réservée', vosp_possibles)
        prof_selection = st.selectbox('Profil de la route', prof_possibles)
        plan_selection = st.selectbox('Tracé en plan', plan_possibles)
        surf_selection = st.selectbox('Etat de la surface', surf_possibles)
        infra_selection = st.selectbox('Aménagement - Infrastructure', infra_possibles)
        nbv_cat_selection = st.selectbox('Nombre de voies', nbv_cat_possibles)

    with col2:
        col_selection = st.selectbox('Type de collision', col_possibles)
        situ_selection = st.selectbox('Situation de l’accident', situ_possibles)
        senc_selection = st.selectbox('Sens de circulation', senc_possibles)
        obs_selection = st.selectbox('Obstacle fixe heurté', obs_possibles)
        obsm_selection = st.selectbox('Obstacle mobile heurté', obsm_possibles)
        choc_selection = st.selectbox('Point de choc initial', choc_possibles)
        manv_selection = st.selectbox('Manoeuvre principale avant l’accident', manv_possibles)
        motor_selection = st.selectbox('Type de motorisation du véhicule', motor_possibles)
        place_selection = st.selectbox('Place occupée dans le véhicule', place_possibles)
        catv_cat_s_selection = st.selectbox('Catégorie de véhicule', catv_cat_s_possibles)
        vma_cat_selection = st.selectbox('Catégorie de vitesse', vma_cat_possibles)
        accident_type_selection = st.selectbox("Type d'accident", accident_type_possibles)

    with col3:
        catu_selection = st.selectbox("Catégorie d'usager", catu_possibles)
        sexe_selection = st.selectbox("Sexe de l'usager", sexe_possibles)
        trajet_selection = st.selectbox('Motif du déplacement', trajet_possibles)
        secu1_selection = st.selectbox('Equipement de sécurité 1', secu1_possibles)
        secu2_selection = st.selectbox('Equipement de sécurité 2', secu2_possibles)
        secu3_selection = st.selectbox('Equipement de sécurité 3', secu3_possibles)
        locp_selection = st.selectbox('Localisation du piéton', locp_possibles)
        etatp_selection = st.selectbox('Etat du piéton', etatp_possibles)
        age_cat_selection = st.selectbox("Catégorie d'age", age_cat_possibles)

    new_data = pd.DataFrame({
        'jour': [jour_selection],
        'mois': [mois_selection],
        'lum': [lum_selection],
        'dep': [dep_selection],
        'agg': [agg_selection],
        'int': [int_selection],
        'atm': [atm_selection],
        'col': [col_selection],
        'catr': [catr_selection],
        'circ': [circ_selection],
        'vosp': [vosp_selection],
        'prof': [prof_selection],
        'plan': [plan_selection],
        'surf': [surf_selection],
        'infra': [infra_selection],
        'situ': [situ_selection],
        'senc': [senc_selection],
        'obs': [obs_selection],
        'obsm': [obsm_selection],
        'choc': [choc_selection],
        'manv': [manv_selection],
        'motor': [motor_selection],
        'place': [place_selection],
        'catu': [catu_selection],
        'sexe': [sexe_selection],
        'trajet': [trajet_selection],
        'secu1': [secu1_selection],
        'secu2': [secu2_selection],
        'secu3': [secu3_selection],
        'locp': [locp_selection],
        'etatp': [etatp_selection],
        'annee': [annee_selection],
        'heure_cat': [heure_cat_selection],
        'age_cat': [age_cat_selection],
        'catv_cat_s': [catv_cat_s_selection],
        'nbv_cat': [nbv_cat_selection],
        'vma_cat': [vma_cat_selection],
        'accident_type': [accident_type_selection],
    })

    X_new_encoded = encoder.transform(new_data)

    # 🧠 Mapping des classes
    gravite_labels = {
        0: "Indemne",
        1: "Non Indemne"
    }

    if st.button("Prédire"):
        prediction = loaded_pickle_model.predict(X_new_encoded)[0]
        proba = loaded_pickle_model.predict_proba(X_new_encoded)[0]

        # Résultat clair
        st.success(f"🧾 Résultat : **{gravite_labels[prediction]}**")

        # Probabilités détaillées
        st.markdown("### 🔢 Probabilités par classe")
        probas_df = pd.DataFrame({
            "Gravité": [gravite_labels[i] for i in loaded_pickle_model.classes_],
            "Probabilité": [round(p * 100, 2) for p in proba]
        })
        st.dataframe(probas_df)

# -------------------------------------------------
# Fonction pour la prédiction multi
# -------------------------------------------------
def display_prediction_multi():
    #st.title("📈 Analyse Binaire")
    st.subheader("🧠 Prédire la gravité d'un accident avec CatBoost SMOTE Année 2023")
    #st.write("Faites des prédictions basées sur les données.")
    #st.header("🧠 Prédire la gravité d'un accident")
    st.markdown("Entrez les paramètres d'un accident pour prédire s'il est probable que la personne soit indemne ou blessée/tuée.")
    # AMAJ

    model_path = "models/streamlit_catboost_multi_smote_param_grid_catboost_light.pkl"

    loaded_pickle_model = load_model(model_path)

    X, y, X_train, X_test, y_train, y_test = load_and_prepare_data("data/stream_value_df.csv")

# -------------------------------------------------
# Encodage
# -------------------------------------------------

    # Initialiser le OneHotEncoder avec handle_unknown='ignore' pour gérer les modalités inconnues dans le test
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop=None)  # drop='first' pour éviter la multicolinéarité, mais dans les modèles Arbres ou Forets, ça peut être mis à None

    # Encoder uniquement sur l'entraînement
    encoder.fit(X_train)

    columns = X.columns
    col_content = {}

    for col in columns:
        filename = f"data/{col}_possibles.csv"
        df = pd.read_csv(filename)
        col_content[col] = df[col].dropna().tolist()

    jours_possibles = col_content['jour']
    mois_possibles = col_content['mois']
    lum_possibles = col_content['lum']
    agg_possibles = col_content['agg']
    int_possibles = col_content['int']
    atm_possibles = col_content['atm']
    col_possibles = col_content['col']
    catr_possibles = col_content['catr']
    circ_possibles = col_content['circ']
    vosp_possibles = col_content['vosp']
    prof_possibles = col_content['prof']
    plan_possibles = col_content['plan']
    surf_possibles = col_content['surf']
    infra_possibles = col_content['infra']
    situ_possibles = col_content['situ']
    senc_possibles = col_content['senc']
    obs_possibles = col_content['obs']
    obsm_possibles = col_content['obsm']
    choc_possibles = col_content['choc']
    manv_possibles = col_content['manv']
    motor_possibles = col_content['motor']
    place_possibles = col_content['place']
    catu_possibles = col_content['catu']
    sexe_possibles = col_content['sexe']
    trajet_possibles = col_content['trajet']
    secu1_possibles = col_content['secu1']
    secu2_possibles = col_content['secu2']
    secu3_possibles = col_content['secu3']
    locp_possibles = col_content['locp']
    etatp_possibles = col_content['etatp']
    annees_possibles = col_content['annee']
    heure_cat_possibles = col_content['heure_cat']
    age_cat_possibles = col_content['age_cat']
    catv_cat_s_possibles = col_content['catv_cat_s']
    nbv_cat_possibles = col_content['nbv_cat']
    vma_cat_possibles = col_content['vma_cat']
    accident_type_possibles = col_content['accident_type']
    dep_possibles = col_content['dep']



    col1, col2, col3 = st.columns(3)
    with col1:
        annee_selection = st.selectbox('Choisissez une annee', annees_possibles)
        mois_selection = st.selectbox('Choisissez un mois', mois_possibles)
        jour_selection = st.selectbox('Choisissez un jour', jours_possibles)
        heure_cat_selection = st.selectbox('Choisissez une heure', heure_cat_possibles)
        dep_selection = st.selectbox('Choisissez un département', dep_possibles)
        lum_selection = st.selectbox('Choisissez une luminosité', lum_possibles)
        agg_selection = st.selectbox('Agglomération', agg_possibles)
        int_selection = st.selectbox('Intersection', int_possibles)
        atm_selection = st.selectbox('Conditions atmosphériques', atm_possibles)
        catr_selection = st.selectbox('Catégorie de route', catr_possibles)
        circ_selection = st.selectbox('Régime de circulation', circ_possibles)
        vosp_selection = st.selectbox('Voie réservée', vosp_possibles)
        prof_selection = st.selectbox('Profil de la route', prof_possibles)
        plan_selection = st.selectbox('Tracé en plan', plan_possibles)
        surf_selection = st.selectbox('Etat de la surface', surf_possibles)
        infra_selection = st.selectbox('Aménagement - Infrastructure', infra_possibles)
        nbv_cat_selection = st.selectbox('Nombre de voies', nbv_cat_possibles)

    with col2:
        col_selection = st.selectbox('Type de collision', col_possibles)
        situ_selection = st.selectbox('Situation de l’accident', situ_possibles)
        senc_selection = st.selectbox('Sens de circulation', senc_possibles)
        obs_selection = st.selectbox('Obstacle fixe heurté', obs_possibles)
        obsm_selection = st.selectbox('Obstacle mobile heurté', obsm_possibles)
        choc_selection = st.selectbox('Point de choc initial', choc_possibles)
        manv_selection = st.selectbox('Manoeuvre principale avant l’accident', manv_possibles)
        motor_selection = st.selectbox('Type de motorisation du véhicule', motor_possibles)
        place_selection = st.selectbox('Place occupée dans le véhicule', place_possibles)
        catv_cat_s_selection = st.selectbox('Catégorie de véhicule', catv_cat_s_possibles)
        vma_cat_selection = st.selectbox('Catégorie de vitesse', vma_cat_possibles)
        accident_type_selection = st.selectbox("Type d'accident", accident_type_possibles)

    with col3:
        catu_selection = st.selectbox("Catégorie d'usager", catu_possibles)
        sexe_selection = st.selectbox("Sexe de l'usager", sexe_possibles)
        trajet_selection = st.selectbox('Motif du déplacement', trajet_possibles)
        secu1_selection = st.selectbox('Equipement de sécurité 1', secu1_possibles)
        secu2_selection = st.selectbox('Equipement de sécurité 2', secu2_possibles)
        secu3_selection = st.selectbox('Equipement de sécurité 3', secu3_possibles)
        locp_selection = st.selectbox('Localisation du piéton', locp_possibles)
        etatp_selection = st.selectbox('Etat du piéton', etatp_possibles)
        age_cat_selection = st.selectbox("Catégorie d'age", age_cat_possibles)

    new_data = pd.DataFrame({
        'jour': [jour_selection],
        'mois': [mois_selection],
        'lum': [lum_selection],
        'dep': [dep_selection],
        'agg': [agg_selection],
        'int': [int_selection],
        'atm': [atm_selection],
        'col': [col_selection],
        'catr': [catr_selection],
        'circ': [circ_selection],
        'vosp': [vosp_selection],
        'prof': [prof_selection],
        'plan': [plan_selection],
        'surf': [surf_selection],
        'infra': [infra_selection],
        'situ': [situ_selection],
        'senc': [senc_selection],
        'obs': [obs_selection],
        'obsm': [obsm_selection],
        'choc': [choc_selection],
        'manv': [manv_selection],
        'motor': [motor_selection],
        'place': [place_selection],
        'catu': [catu_selection],
        'sexe': [sexe_selection],
        'trajet': [trajet_selection],
        'secu1': [secu1_selection],
        'secu2': [secu2_selection],
        'secu3': [secu3_selection],
        'locp': [locp_selection],
        'etatp': [etatp_selection],
        'annee': [annee_selection],
        'heure_cat': [heure_cat_selection],
        'age_cat': [age_cat_selection],
        'catv_cat_s': [catv_cat_s_selection],
        'nbv_cat': [nbv_cat_selection],
        'vma_cat': [vma_cat_selection],
        'accident_type': [accident_type_selection],
    })

    X_new_encoded = encoder.transform(new_data)

    # 🧠 Mapping des classes
    gravite_labels = {
        0: "Indemne",
        1: "Blessé léger",
        2: "Blessé grave",
        3: "Tué"
    }

    if st.button("Prédire"):
        prediction = int(loaded_pickle_model.predict(X_new_encoded)[0])
        proba = loaded_pickle_model.predict_proba(X_new_encoded)[0]

        # Résultat clair
        st.success(f"🧾 Résultat : **{gravite_labels[prediction]}**")

        # Probabilités détaillées
        st.markdown("### 🔢 Probabilités par classe")
        probas_df = pd.DataFrame({
            "Gravité": [gravite_labels[i] for i in loaded_pickle_model.classes_],
            "Probabilité": [round(p * 100, 2) for p in proba]
        })
        st.dataframe(probas_df)

# -------------------------------------------------
# Fonction pour la comparaison
# -------------------------------------------------

import pickle
import pandas as pd
import streamlit as st
import plotly.express as px

def display_model_comparison():
    st.header("🤖 Comparaison de modèles de Machine Learning (Binaire) Année 2023")

    # ✅ Charge les 4 .pkl 'info'
    info_files = [
        "models/streamlit_rf_info.pkl",
        "models/streamlit_bin_xgboost_none_param_grid_light_info.pkl",
        "models/streamlit_lr_info.pkl",
        "models/streamlit_catboost_info.pkl"
    ]

    infos = []
    for fpath in info_files:
        with open(fpath, 'rb') as f:
            infos.append(pickle.load(f))

    # ✅ Assemble proprement en DataFrame
    df_comparaison = pd.DataFrame(infos)
    # Renommer pour cohérence
    df_comparaison.rename(columns={'best_score_cv':'best_f1_cv'}, inplace=True)
    # Supprimer colonnes inutiles si présentes
    for col in ['param_grid', 'n_features_in']:
        if col in df_comparaison.columns:
            df_comparaison.drop(col, axis=1, inplace=True)

    st.dataframe(df_comparaison)

    # ✅ Téléchargement CSV
    csv = df_comparaison.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Télécharger le CSV", csv, file_name="comparaison_modeles.csv", mime="text/csv")

    # ✅ Visualisation Plotly complète (AUC + Précision + F1)
    try:
        fig = px.bar(
            df_comparaison.melt(
                id_vars=["model_name"],
                value_vars=[
                    "auc_train", "auc_test",
                    "precision_train", "precision_test",
                    "f1_train", "f1_test",
                    "best_f1_cv"
                ]
            ),
            x="model_name",
            y="value",
            color="variable",
            barmode="group",
            title="Comparaison des métriques : AUC, Précision & F1"
        )
        fig.update_yaxes(range=[0.5,1])
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"❗ Erreur de graphique : {e}")

import pickle
import pandas as pd
import streamlit as st
import plotly.express as px

def display_model_comparison_multi():
    st.header("🤖 Comparaison de modèles de Machine Learning (Multiclasse) Année 2023")

    # ✅ Charge les 4 .pkl 'info'
    info_files = [
        "models/streamlit_xgboost_multi_none_param_grid_light_info.pkl",
        "models/streamlit_xgboost_multi_undersampling_param_grid_light_info.pkl",
        "models/streamlit_xgboost_multi_smote_param_grid_light_info.pkl",
        "models/streamlit_xgboost_multi_oversampling_param_grid_light_info.pkl",

        "models/streamlit_catboost_multi_none_param_grid_catboost_light_info.pkl",
        "models/streamlit_catboost_multi_undersampling_param_grid_catboost_light_info.pkl",
        "models/streamlit_catboost_multi_smote_param_grid_catboost_light_info.pkl",
        "models/streamlit_catboost_multi_oversampling_param_grid_catboost_light_info.pkl",
        
        "models/streamlit_randomforest_multi_none_param_grid_rf_info.pkl",
        "models/streamlit_randomforest_multi_undersampling_param_grid_rf_info.pkl",
        "models/streamlit_randomforest_multi_smote_param_grid_rf_info.pkl",
        "models/streamlit_randomforest_multi_oversampling_param_grid_rf_info.pkl"  
    ]

    infos = []
    for fpath in info_files:
        with open(fpath, 'rb') as f:
            infos.append(pickle.load(f))

    # ✅ Assemble proprement en DataFrame
    df_comparaison = pd.DataFrame(infos)
    # Renommer pour cohérence
    df_comparaison.rename(columns={'best_score_cv':'best_f1_cv'}, inplace=True)
    # Supprimer colonnes inutiles si présentes
    for col in ['param_grid', 'n_features_in']:
        if col in df_comparaison.columns:
            df_comparaison.drop(col, axis=1, inplace=True)
    df_comparaison['f1_train']=df_comparaison['f1_train'].round(2)
    df_comparaison['f1_test']=df_comparaison['f1_test'].round(2)
    df_comparaison['auc_train']=df_comparaison['auc_train'].round(2)
    df_comparaison['auc_test']=df_comparaison['auc_test'].round(2)
    df_comparaison['precision_train']=df_comparaison['precision_train'].round(2)
    df_comparaison['precision_test']=df_comparaison['precision_test'].round(2)
    df_comparaison.sort_values(by='f1_test',ascending=False,inplace=True)

    st.dataframe(df_comparaison)

    # ✅ Téléchargement CSV
    csv = df_comparaison.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Télécharger le CSV", csv, file_name="comparaison_modeles.csv", mime="text/csv")

    # ✅ Visualisation Plotly complète (AUC + Précision + F1)

    df_comparaison["model_sampling"] = df_comparaison["model_name"] + " - " + df_comparaison["sampling"]

    try:
        fig = px.bar(
            df_comparaison.melt(
                id_vars=["model_sampling"],
                value_vars=[
                    "auc_train", "auc_test",
                    "precision_train", "precision_test",
                    "f1_train", "f1_test",
                    "best_f1_cv"
                ]
            ),
            x="model_sampling",
            y="value",
            color="variable",
            barmode="group",
            title="Comparaison des métriques : AUC, Précision & F1"
        )
        fig.update_yaxes(range=[0.5,1])
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"❗ Erreur de graphique : {e}")


    # try:
    #     fig = px.bar(
    #         df_comparaison.melt(
    #             id_vars=["model_name"],
    #             value_vars=[
    #                 "auc_train", "auc_test",
    #                 "precision_train", "precision_test",
    #                 "f1_train", "f1_test",
    #                 "best_f1_cv"
    #             ]
    #         ),
    #         x="model_name",
    #         y="value",
    #         color="variable",
    #         barmode="group",
    #         title="Comparaison des métriques : AUC, Précision & F1"
    #     )
    #     st.plotly_chart(fig, use_container_width=True)
    # except Exception as e:
    #     st.warning(f"❗ Erreur de graphique : {e}")

def add_custom_css():
    st.markdown("""
        <style>
        .stButton > button {
            background-color: #e8f0fe;
            color: black;
            border-radius: 8px;
            border: none;
            margin: 5px 0px;
            padding: 10px 16px;
            width: 100%;
            font-weight: bold;
        }
        .stButton > button:hover {
            background-color: #c9ddff;
            color: black;
        }

        /* En-tête centré */
        .title-style {
            text-align: center;
            color: #003366;
        }

        .block-container {
            padding-top: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    add_custom_css()
    # Init
    if "active_menu" not in st.session_state:
        st.session_state.active_menu = "🏠 Accueil & À propos"

    # En-tête
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("src/streamlit/data/logo.png", width=100)
    with col2:
        st.markdown("<h2 class='title-style'>Prédiction de la gravité des Accidents routiers en France</h2>", unsafe_allow_html=True)
    st.markdown("---")

    with st.sidebar:
        st.header("📌 Menu de navigation")

        # CSS
        st.markdown("""
            <style>
                section[data-testid="stSidebar"] button {
                    width: 100% !important;
                    padding: 0.5rem 1rem;
                    margin: 0.2rem 0;
                    border-radius: 8px;
                    font-size: 16px;
                    text-align: left !important;
                    display: block !important;
                    background-color: #f0f2f6;
                    color: black;
                }
                section[data-testid="stSidebar"] button:focus:not(:active) {
                    background-color: #2E86C1 !important;
                    color: white !important;
                    font-weight: bold;
                }
            </style>
        """, unsafe_allow_html=True)

        # Boutons de navigation
        menu_items = {
            "🏠 Accueil & À propos": "🏠 Accueil & À propos",
            "🧩 Notre approche": "🧩 Notre approche",
            "📝 Exploration des Données": "📝 Exploration des Données",
            "📈 Analyse & Prédiction Binaire": "📈 Analyse & Prédiction Binaire",
            "📊 Analyse & Prédiction Multiclasse": "📊 Analyse & Prédiction Multiclasse",
            "🏁 Conclusion": "🏁 Conclusion"
        }

        for label, item in menu_items.items():
            if st.button(label, key=f"btn_{item}"):
                st.session_state.active_menu = item

    # Navigation vers actif
    menu = st.session_state.active_menu

    if menu == "🏠 Accueil & À propos":
        onglet_accueil, onglet_apropos = st.tabs(["🏠 Accueil", "📄 À propos"])
        with onglet_accueil:
            display_home()
        with onglet_apropos:
            display_about()

    elif menu == "🧩 Notre approche":
        st.markdown("### 🧩 Notre approche globale")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🌐 Approche globale",
            "📥 Collecte et exploration des données",
            "🧹 Préparation des données",
            "📊 Analyse exploratoire",
            "🤖 Modélisation"
        ])
        with tab1:
            display_approach()
        with tab2:
            CollecteDonnees()
        with tab3:
            PrepDonnees()
        with tab4:
            AnalyseDonnees()
        with tab5:
            ModelPredictDonnees()

    elif menu == "📝 Exploration des Données":
        st.markdown("### 📝 Exploration des données (Fichier BAC)")
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🏠 Présentation globale",
            "📝 Observations et corrections",
            "🧾 Données CARACTÉRISTIQUES",
            "📍 Données LIEUX",
            "🚗 Données VÉHICULES",
            "🚶‍♂️ Données USAGERS"
        ])
        with tab1:
            display_donnees_Description()
        with tab2:
            display_donnees_Observations()
        with tab3:
            display_donnees_Caracteristiques()
        with tab4:
            display_donnees_Lieux()
        with tab5:
            display_donnees_Vehicules()
        with tab6:
            display_donnees_Usagers()

    elif menu == "📈 Analyse & Prédiction Binaire":
        st.markdown("### 📈 Analyse interactive Binaire")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Analyse interactive",
            "🎛️ Filtre dynamique",
            "🔮 Visualisations",
            "🧠 Prédiction",
            "🤖 Comparaison de modèles"
        ])
        with tab1:
            display_analysis_bi()
        with tab2:
            display_dynamic_filter()
        with tab3:
            display_visualizations()
        with tab4:
            display_prediction()
        with tab5:
            display_model_comparison()

    elif menu == "📊 Analyse & Prédiction Multiclasse":
        st.markdown("### 📊 Analyse & Prédiction Multiclasse")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Analyse interactive",
            "🎛️ Filtre dynamique",
            "🔮 Visualisations",
            "🧠 Prédiction",
            "🤖 Comparaison de modèles"
        ])
        with tab1:
            display_analysis_multi()
        with tab2:
            display_dynamic_filter_multi()
        with tab3:
            display_visualizations_multi()
        with tab4:
            display_prediction_multi()
        with tab5:
            display_model_comparison_multi()

    elif menu == "🏁 Conclusion":

        image_path = "data/importance.png"


        st.title("🏁 Conclusion")
        st.write("Notre projet a démontré que la prédiction de la gravité des accidents routiers est possible à partir de données publiques, mais reste un défi complexe.")
        st.markdown("<h4 style='color: #2E86C1;'>✅ Résultats obtenus</h4>", unsafe_allow_html=True)
        st.write("- Le meilleur modèle 2023 pour notre Use Case Binaire est XGBoost.")
        st.write("- Le meilleur modèle 2023 pour notre Use Case Multiclasse est CatBoost avec SMOTE.")
        st.write("- Les résultats de nos prédictions sont satisfaisants, même sur la classe la plus déséquilibrée, entre autres grâce au feature engineering mis en place pour déduire le type d’accident.")
        st.markdown("<h4 style='color: #2E86C1;'>🧠 Critiques et perspectives</h4>", unsafe_allow_html=True)
        st.write("- La quarantaine de variables du jeu de données sont finalement assez limitées.")
        st.write("- Un effort significatif a été porté sur le feature engineering afin de maximiser la valeur de ces variables.")
        st.write("- La performance des modèles a pu être améliorée, malgré des temps d’exécution conséquents.")
        st.write("- Cependant, cela n’a pas permis d’intégrer des variables exogènes (comme les données météo) ni d’approfondir l’explicabilité des modèles (par exemple avec SHAP).")
        st.write("- Nous avons par contre indentifié les variables les plus importantes.")
        image = Image.open(image_path)
        st.image(image, caption="importance des variables", width=600)
        st.write("- Le déploiement via Streamlit est resté en environnement local, sans passage sur un Cloud.")
        st.markdown("<h4 style='color: #2E86C1;'>🚀 Apports du projet</h4>", unsafe_allow_html=True)
        st.markdown("Ce projet nous a permis :")
        st.write("- D’explorer un jeu de données riche et complexe.")
        st.write("- De créer des modèles robustes de classification binaire et multiclasse.")
        st.write("- De proposer un outil valorisable avec Streamlit.")

if __name__ == "__main__":
    main()