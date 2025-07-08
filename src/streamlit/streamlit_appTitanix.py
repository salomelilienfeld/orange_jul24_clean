import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Chargement des données
df = pd.read_csv("./data/train2.csv")

# Configuration de la mise en page
st.set_page_config(page_title="Classification Titanic", layout="wide")

# Titre et contexte avec couleurs
st.markdown("<h1 style='text-align: center; color: #4B0082;'>Projet de Classification Binaire : Gravité des Passagers du Titanic</h1>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #555555;'>
    Le Titanic a coulé lors de son voyage inaugural en 1912, entraînant la mort de plus de 1500 passagers. 
    L'objectif de ce projet est de prédire la survie des passagers en fonction de leurs caractéristiques. 
    Nous allons explorer les données, visualiser les résultats et construire un modèle de classification binaire.
    </div>
""", unsafe_allow_html=True)

# Sidebar avec logos et boutons
st.sidebar.image("logo.png", width=150)  
st.sidebar.header("Navigation")
pages = ["📜 Présentation du Projet", "🏠 Exploration", "📊 Data Visualization", "📈 Modélisation", "🙏 Remerciements"]
page = st.sidebar.radio("Aller vers", pages)

# Présentation du projet
if page == pages[0]:
    st.subheader("Présentation du Projet")
    st.markdown("""
        <div style='color: #333333;'>
        Ce projet vise à analyser les données des passagers du Titanic afin de prédire leur survie. 
        En utilisant des techniques de machine learning, nous cherchons à comprendre quels facteurs ont influencé 
        la survie des passagers lors de ce tragique événement historique.
        
        ### Objectifs
        - Explorer les données des passagers du Titanic.
        - Visualiser les relations entre différentes caractéristiques et la survie.
        - Construire et évaluer des modèles de classification pour prédire la survie.
        
        ### Importance
        Comprendre les facteurs de survie peut non seulement aider à mieux appréhender cet événement historique, 
        mais aussi fournir des enseignements sur la prise de décision en situations de crise.
        </div>
    """, unsafe_allow_html=True)

# Exploration des données
if page == pages[1]:
    st.subheader("Exploration des Données")
    st.write("Affichage des 10 premières lignes du jeu de données :")
    st.dataframe(df.head(10))
    st.write("Forme du DataFrame :", df.shape)
    st.write("Description des données :")
    st.dataframe(df.describe())
    
    if st.checkbox("Afficher les valeurs manquantes"):
        st.write(df.isna().sum())

    # Sous-pages pour l'exploration
    st.markdown("### Sous-pages")
    if st.button("Afficher les statistiques avancées"):
        st.subheader("Statistiques Avancées")
        st.write(df.describe(include='all'))

    if st.button("Afficher la distribution des âges"):
        st.subheader("Distribution des Âges")
        fig, ax = plt.subplots()
        sns.histplot(df['Age'], bins=30, kde=True, ax=ax, color='#FF6347')  # Couleur tomate
        ax.set_title("Distribution de l'Âge des Passagers", color='#4B0082')  # Couleur indigo
        st.pyplot(fig)

# Visualisation des données
if page == pages[2]:
    st.subheader("Visualisation des Données")
    col1, col2 = st.columns(2)
    
    with col1:
        fig1, ax1 = plt.subplots()
        sns.countplot(x='Survived', data=df, ax=ax1, palette='pastel')
        ax1.set_title("Survie des Passagers", color='#4B0082')
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        sns.countplot(x='Sex', data=df, ax=ax2, palette='pastel')
        ax2.set_title("Répartition du Genre des Passagers", color='#4B0082')
        st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    sns.countplot(x='Pclass', data=df, ax=ax3, palette='pastel')
    ax3.set_title("Répartition des Classes des Passagers", color='#4B0082')
    st.pyplot(fig3)

    # Sous-pages pour la visualisation
    st.markdown("### Sous-pages")
    if st.button("Afficher la distribution des âges"):
        st.subheader("Distribution des Âges")
        fig4, ax4 = plt.subplots()
        sns.histplot(df['Age'], bins=30, kde=True, ax=ax4, color='#FF6347')
        ax4.set_title("Distribution de l'Âge des Passagers", color='#4B0082')
        st.pyplot(fig4)

# Modélisation
if page == pages[3]:
    st.subheader("Modélisation")
    
    # Sélecteur pour les sous-pages de modélisation
    model_pages = ["Modélisation Bimodale", "Modélisation Multimodale"]
    model_page = st.selectbox("Choisissez un type de modélisation", model_pages)

    if model_page == "Modélisation Bimodale":
        st.write("Nous allons construire un modèle de classification binaire pour prédire la survie des passagers.")
        
        # Préparation des données
        df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
        y = df['Survived']
        X_cat = df[['Pclass', 'Sex', 'Embarked']]
        X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]
        
        # Remplissage des valeurs manquantes
        X_cat.fillna(X_cat.mode().iloc[0], inplace=True)
        X_num.fillna(X_num.median(), inplace=True)
        
        X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
        X = pd.concat([X_cat_scaled, X_num], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
        scaler = StandardScaler()
        X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
        X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

        # Choix du modèle
        choix = ['Random Forest', 'SVC', 'Logistic Regression']
        option = st.selectbox('Choix du modèle', choix)
        
        def prediction(classifier):
            if classifier == 'Random Forest':
                clf = RandomForestClassifier()
            elif classifier == 'SVC':
                clf = SVC()
            elif classifier == 'Logistic Regression':
                clf = LogisticRegression()
            clf.fit(X_train, y_train)
            return clf

        clf = prediction(option)
        display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
        
        if display == 'Accuracy':
            st.write("Précision du modèle :", clf.score(X_test, y_test))
        elif display == 'Confusion matrix':
            st.dataframe(confusion_matrix(y_test, clf.predict(X_test)))

    elif model_page == "Modélisation Multimodale":
        st.write("Nous allons construire un modèle de classification multimodale pour prédire la survie des passagers.")
        
        # Préparation des données (similaire à la bimodale mais avec plus de caractéristiques)
        df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
        y = df['Survived']
        X_cat = df[['Pclass', 'Sex', 'Embarked']]
        X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]
        
        # Remplissage des valeurs manquantes
        X_cat.fillna(X_cat.mode().iloc[0], inplace=True)
        X_num.fillna(X_num.median(), inplace=True)
        
        X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
        X = pd.concat([X_cat_scaled, X_num], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
        scaler = StandardScaler()
        X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
        X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

        # Choix du modèle
        choix = ['Random Forest', 'SVC', 'Logistic Regression']
        option = st.selectbox('Choix du modèle', choix)
        
        def prediction(classifier):
            if classifier == 'Random Forest':
                clf = RandomForestClassifier()
            elif classifier == 'SVC':
                clf = SVC()
            elif classifier == 'Logistic Regression':
                clf = LogisticRegression()
            clf.fit(X_train, y_train)
            return clf

        clf = prediction(option)
        display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
        
        if display == 'Accuracy':
            st.write("Précision du modèle :", clf.score(X_test, y_test))
        elif display == 'Confusion matrix':
            st.dataframe(confusion_matrix(y_test, clf.predict(X_test)))

# Remerciements
if page == pages[4]:
    st.subheader("Remerciements")
    st.write("Merci d'avoir consulté ce projet !")

# Footer
st.markdown("---")
st.markdown("Développé dans le cadre du projet Fil rouge du programme Data Science | © 2025")