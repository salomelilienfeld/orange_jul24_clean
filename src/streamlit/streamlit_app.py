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

#---------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import logging
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Définition des variables globales
def_variables_initiales = {
    'Num_Acc':"Numéro d'identifiant de l’accident.",
    'jour':"Jour de l'accident",
    'mois':"Mois de l'accident",
    'an':"Année de l'accident",
    'hrmn':"Heure et minutes de l'accident",
    'lum':"Conditions d’éclairage dans lesquelles l'accident s'est produit",
    'dep':"Département : Code INSEE",
    'com':"Commune : Le numéro de commune est un code donné par l‘INSEE",
    'agg':"Localisé en agglomération ou non",
    'int':"Intersection",
    'atm':"Conditions atmosphériques",
    'col':"Type de collision",
    'adr':"Adresse postale",
    'lat':"Latitude",
    'lon':"Longitude",
    'catr':"Catégorie de route",
    'voie':"Numéro de la route",
    'v1':"Indice numérique du numéro de route",
    'v2':"Lettre indice alphanumérique de la route",
    'circ':"Régime de circulation",
    'nbv':"Nombre total de voies de circulation",
    'vosp':"Signale l’existence d’une voie réservée, indépendamment du fait que l’accident ait lieu ou non sur cette voie",
    'prof':"Profil en long décrit la déclivité de la route à l'endroit de l'accident",
    'pr':"Numéro du PR de rattachement (numéro de la borne amont)",
    'pr1':"Distance en mètres au PR (par rapport à la borne amont)",
    'plan':"Tracé en plan",
    'lartpc':"Largeur du terre-plein central (TPC) s'il existe (en m)",
    'larrout':"Largeur de la chaussée affectée à la circulation des véhicules ne sont pas compris les bandes d'arrêt d'urgence, les TPC et les places de stationnement (en m)",
    'surf':"Etat de la surface",
    'infra':"Aménagement - Infrastructure",
    'situ':"Situation de l’accident",
    'vma':"Vitesse maximale autorisée sur le lieu et au moment de l’accident",
    'num_veh':"Identifiant du véhicule repris pour chacun des usagers occupant ce véhicule (y compris les piétons qui sont rattachés aux véhicules qui les ont heurtés)",
    'senc':"Sens de circulation",
    'catv':"Catégorie du véhicule",
    'obs':"Obstacle fixe heurté",
    'obsm':"Obstacle mobile heurté",
    'choc':"Point de choc initial",
    'manv':"Manoeuvre principale avant l’accident",
    'motor':"Type de motorisation du véhicule",
    'occutc':"Nombre d’occupants dans le transport en commun",
    'place':"Permet de situer la place occupée dans le véhicule par l'usager au moment de l'accident",
    'catu':"Catégorie d'usager",
    'grav':"Gravité de blessure de l'usager",
    'sexe':"Sexe de l'usager",
    'an_nais':"Année de naissance de l'usager",
    'trajet':"Motif du déplacement au moment de l’accident",
    'secu1':"Présence et utilisation de l’équipement de sécurité",
    'secu2':"Présence et utilisation de l’équipement de sécurité",
    'secu3':"Présence et utilisation de l’équipement de sécurité",
    'locp':"Localisation du piéton",
    'actp':"Action du piéton",
    'etatp':"Cette variable permet de préciser si le piéton accidenté était seul ou non",
    'catv_cat':"Catégorie du véhicule revue",
    'heure_cat':"Créneau Horaire de l'accident",
    'age_cat':"Catégorie d'age de l'usager",
    'nbv_cat':"Catégorie de nombre de voies",
    'vma_cat':"Catégorie de vitesse du véhicule"
}


def_variable = {
    'jour':"Jour de l'accident",
    'mois':"Mois de l'accident",
    'annee':"Annee de l'accident",
    'heure':"Heure de l'accident",
    'hrmn':"Heure et minutes de l'accident",
    'lum':"Conditions d’éclairage dans lesquelles l'accident s'est produit",
    'dep':"Département : Code INSEE",
    'com':"Commune : Le numéro de commune est un code donné par l‘INSEE",
    'agg':"Localisé en agglomération ou non",
    'int':"Intersection",
    'atm':"Conditions atmosphériques",
    'col':"Type de collision",
    'lat':"Latitude",
    'lon':"Longitude",
    'catr':"Catégorie de route",
    'circ':"Régime de circulation",
    'nbv':"Nombre total de voies de circulation",
    'vosp':"Signale l’existence d’une voie réservée, indépendamment du fait que l’accident ait lieu ou non sur cette voie",
    'prof':"Profil en long décrit la déclivité de la route à l'endroit de l'accident",
    'plan':"Tracé en plan",
    'surf':"Etat de la surface",
    'infra':"Aménagement - Infrastructure",
    'situ':"Situation de l’accident",
    'vma':"Vitesse maximale autorisée sur le lieu et au moment de l’accident",
    'senc':"Sens de circulation",
    'obs':"Obstacle fixe heurté",
    'obsm':"Obstacle mobile heurté",
    'choc':"Point de choc initial",
    'manv':"Manoeuvre principale avant l’accident",
    'motor':"Type de motorisation du véhicule",
    'place':"Permet de situer la place occupée dans le véhicule par l'usager au moment de l'accident",
    'catu':"Catégorie d'usager",
    'grav':"Gravité de blessure de l'usager",
    'sexe':"Sexe de l'usager",
    'an_nais':"Année de naissance de l'usager",
    'trajet':"Motif du déplacement au moment de l’accident",
    'secu1':"Présence et utilisation de l’équipement de sécurité",
    'secu2':"Présence et utilisation de l’équipement de sécurité",
    'secu3':"Présence et utilisation de l’équipement de sécurité",
    'locp':"Localisation du piéton",
    'etatp':"Cette variable permet de préciser si le piéton accidenté était seul ou non",
    'catv_cat':"Catégorie du véhicule revue",
    'heure_cat':"Créneau Horaire de l'accident",
    'age_cat':"Catégorie d'age de l'usager",
    'nbv_cat':"Catégorie de nombre de voies",
    'vma_cat':"Catégorie de vitesse du véhicule",
<<<<<<< HEAD
    'nbacc_cat':"Nombre d'usagers impliqués dans l'accident",
    'nbveh_cat':"Nombre de véhicules impliqués dans l'accident"
=======
    'nbacc_cat':"Nombre de victimes impliquées dans l'accident",
    'nbveh_cat':"Nombre de victimes dans le véhicule"
>>>>>>> 327dd5e738f2650d66be6f04ed71e9b9935294ec
}
# Valeurs des variables
val_variables_initiales = {
    'lum':{1:"Plein jour",
           2:"Crépuscule ou aube",
           3:"Nuit sans éclairage public",
           4:"Nuit avec éclairage public non allumé",
           5:"Nuit avec éclairage public allumé"},
    'agg':{1:"Hors agglomération",
           2:"En agglomération"},
    'int':{1:"Hors intersection",
           2:"Intersection en X",
           3:"Intersection en T",
           4:"Intersection en Y",
           5:"Intersection à plus de 4 branches",
           6:"Giratoire",
           7:"Place",
           8:"Passage à niveau",
           9:"Autre intersection"},
    'atm':{-1:"Non renseigné",
            1:"Normale",
            2:"Pluie légère",
            3:"Pluie forte",
            4:"Neige - grêle",
            5:"Brouillard - fumée",
            6:"Vent fort - tempête",
            7:"Temps éblouissant",
            8:"Temps couvert",
            9:"Autre"},
    'col':{-1:"Non renseigné",
            1:"Deux véhicules - frontale",
            2:"Deux véhicules – par l’arrière",
            3:"Deux véhicules – par le coté",
            4:"Trois véhicules et plus – en chaîne",
            5:"Trois véhicules et plus - collisions multiples",
            6:"Autre collision",
            7:"Sans collision"},
    'catr':{1:"Autoroute",
            2:"Route nationale",
            3:"Route Départementale",
            4:"Voie Communales",
            5:"Hors réseau public",
            6:"Parc de stationnement ouvert à la circulation publique",
            7:"Routes de métropole urbaine",
            9:"autre"},
    'circ':{-1:"Non renseigné",
            1:"A sens unique",
            2:"Bidirectionnelle",
            3:"A chaussées séparées",
            4:"Avec voies d’affectation variable"},
    'vosp':{-1:"Non renseigné",
            0:"Sans objet",
            1:"Piste cyclable",
            2:"Bande cyclable",
            3:"Voie réservée"},
    'prof':{-1:"Non renseigné",
            1:"Plat",
            2:"Pente",
            3:"Sommet de côte",
            4:"Bas de côte"},
    'plan':{-1:"Non renseigné",
            1:"Partie rectiligne",
            2:"En courbe à gauche",
            3:"En courbe à droite",
            4:"En S"},
    'surf':{-1:"Non renseigné",
            1:"Normale",
            2:"Mouillée",
            3:"Flaques",
            4:"Inondée",
            5:"Enneigée",
            6:"Boue",
            7:"Verglacée",
            8:"Corps gras – huile",
            9:"Autre"},
    'infra':{-1:"Non renseigné",
            0:"Aucun",
            1:"Souterrain - tunnel",
            2:"Pont - autopont",
            3:"Bretelle d’échangeur ou de raccordement",
            4:"Voie ferrée",
            5:"Carrefour aménagé",
            6:"Zone piétonne",
            7:"Zone de péage",
            8:"Chantier",
            9:"Autres"},
    'situ':{-1:"Non renseigné",
            0:"Aucun",
            1:"Sur chaussée",
            2:"Sur bande d’arrêt d’urgence",
            3:"Sur accotement",
            4:"Sur trottoir",
            5:"Sur piste cyclable",
            6:"Sur autre voie spéciale",
            8:"Autres"},
    'senc':{-1:"Non renseigné",
            0:"Inconnu",
            1:"PK ou PR ou numéro d’adresse postale croissant",
            2:"PK ou PR ou numéro d’adresse postale décroissant",
            3:"Absence de repère"},
    'catv':{0:"Indéterminable",
            1:"Bicyclette",
            2:"Cyclomoteur <50cm3",
            3:"Voiturette (Quadricycle à moteur carrossé) (anciennement voiturette ou tricycle à moteur)",
            4:"Référence inutilisée depuis 2006 (scooter immatriculé)",
            5:"Référence inutilisée depuis 2006 (motocyclette)",
            6:"Référence inutilisée depuis 2006 (side-car)",
            7:"VL seul",
            8:"Référence inutilisée depuis 2006 (VL + caravane)",
            9:"Référence inutilisée depuis 2006 (VL + remorque)",
            10:"VU seul 1,5T <= PTAC <= 3,5T avec ou sans remorque (anciennement VU seul 1,5T <= PTAC <= 3,5T)",
            11:"Référence inutilisée depuis 2006 (VU (10) + caravane)",
            12:"Référence inutilisée depuis 2006 (VU (10) + remorque)",
            13:"PL seul 3,5T <PTCA <= 7,5T",
            14:"PL seul > 7,5T",
            15:"PL > 3,5T + remorque",
            16:"Tracteur routier seul",
            17:"Tracteur routier + semi-remorque",
            18:"Référence inutilisée depuis 2006 (transport en commun)",
            19:"Référence inutilisée depuis 2006 (tramway)",
            20:"Engin spécial",
            21:"Tracteur agricole",
            30:"Scooter < 50 cm3",
            31:"Motocyclette > 50 cm3 et <= 125 cm3",
            32:"Scooter > 50 cm3 et <= 125 cm3",
            33:"Motocyclette > 125 cm3",
            34:"Scooter > 125 cm3",
            35:"Quad léger <= 50 cm3 (Quadricycle à moteur non carrossé)",
            36:"Quad lourd > 50 cm3 (Quadricycle à moteur non carrossé)",
            37:"Autobus",
            38:"Autocar",
            39:"Train",
            40:"Tramway",
            41:"3RM <= 50 cm3",
            42:"3RM > 50 cm3 <= 125 cm3",
            43:"3RM > 125 cm3",
            50:"EDP à moteur",
            60:"EDP sans moteur",
            80:"VAE",
            99:"Autre véhicule"},
    'obs':{-1:"Non renseigné",
            0:"Sans objet",
            1:"Véhicule en stationnement",
            2:"Arbre",
            3:"Glissière métallique",
            4:"Glissière béton",
            5:"Autre glissière",
            6:"Bâtiment, mur, pile de pont",
            7:"Support de signalisation verticale ou poste d’appel d’urgence",
            8:"Poteau",
            9:"Mobilier urbain",
            10:"Parapet",
            11:"Ilot, refuge, borne haute",
            12:"Bordure de trottoir",
            13:"Fossé, talus, paroi rocheuse",
            14:"Autre obstacle fixe sur chaussée",
            15:"Autre obstacle fixe sur trottoir ou accotement",
            16:"Sortie de chaussée sans obstacle",
            17:"Buse – tête d’aqueduc"},
    'obsm':{-1:"Non renseigné",
            0:"Aucun",
            1:"Piéton",
            2:"Véhicule",
            4:"Véhicule sur rail",
            5:"Animal domestique",
            6:"Animal sauvage",
            9:"Autre"},
    'choc':{-1:"Non renseigné",
            0:"Aucun",
            1:"Avant",
            2:"Avant droit",
            3:"Avant gauche",
            4:"Arrière",
            5:"Arrière droit",
            6:"Arrière gauche",
            7:"Côté droit",
            8:"Côté gauche",
            9:"Chocs multiples (tonneaux)"},
    'manv':{-1:"Non renseigné",
            0:"Inconnue",
            1:"Sans changement de direction",
            2:"Même sens, même file",
            3:"Entre 2 files",
            4:"En marche arrière",
            5:"A contresens",
            6:"En franchissant le terre-plein central",
            7:"Dans le couloir bus, dans le même sens",
            8:"Dans le couloir bus, dans le sens inverse",
            9:"En s’insérant",
            10:"En faisant demi-tour sur la chaussée",
            11:"Changeant de file à gauche",
            12:"Changeant de file à droite",
            13:"Déporté à gauche",
            14:"Déporté à droite",
            15:"Tournant à gauche",
            16:"Tournant à droite",
            17:"Dépassant à gauche",
            18:"Dépassant à droite",
            19:"Traversant la chaussée",
            20:"Manoeuvre de stationnement",
            21:"Manoeuvre d’évitement",
            22:"Ouverture de porte",
            23:"Arrêté (hors stationnement)",
            24:"En stationnement (avec occupants)",
            25:"Circulant sur trottoir",
            26:"Autres manoeuvres"},
    'motor':{-1:"Non renseigné",
            0:"Inconnu",
            1:"Hydrocarbures",
            2:"Hybride électrique",
            3:"Electrique",
            4:"Hydrogène",
            5:"Humaine",
            6:"Autre"},
    'place':{1:"Conducteur",
            2:"Passager principal",
            3:"Passager arrière droit",
            4:"Passager arrière gauche",
            5:"Passager arrière centre",
            6:"Passager avant centre",
            7:"Passager gauche",
            8:"Passager centre",
            9:"Passager droit"},
    'catu':{1:"Conducteur",
            2:"Passager",
            3:"Piéton"},
    'grav':{1:"Indemne",
            2:"Tué",
            3:"Blessé hospitalisé",
            4:"Blessé léger"},
    'sexe':{1:"Masculin",
            2:"Féminin"},
    'trajet':{-1:"Non renseigné",
               0:"Non renseigné",
               1:"Domicile – travail",
               2:"Domicile – école",
               3:"Courses – achats",
               4:"Utilisation professionnelle",
               5:"Promenade – loisirs",
               9:"Autre"},
    'secu1':{-1:"Non renseigné",
            0:"Aucun équipement",
            1:"Ceinture",
            2:"Casque",
            3:"Dispositif enfants",
            4:"Gilet réfléchissant",
            5:"Airbag (2RM/3RM)",
            6:"Gants (2RM/3RM)",
            7:"Gants + Airbag (2RM/3RM)",
            8:"Non déterminable",
            9:"Autre"},
    'secu2':{-1:"Non renseigné",
            0:"Aucun équipement",
            1:"Ceinture",
            2:"Casque",
            3:"Dispositif enfants",
            4:"Gilet réfléchissant",
            5:"Airbag (2RM/3RM)",
            6:"Gants (2RM/3RM)",
            7:"Gants + Airbag (2RM/3RM)",
            8:"Non déterminable",
            9:"Autre"},
    'secu3':{-1:"Non renseigné",
            0:"Aucun équipement",
            1:"Ceinture",
            2:"Casque",
            3:"Dispositif enfants",
            4:"Gilet réfléchissant",
            5:"Airbag (2RM/3RM)",
            6:"Gants (2RM/3RM)",
            7:"Gants + Airbag (2RM/3RM)",
            8:"Non déterminable",
            9:"Autre"},
    'locp':{-1:"Non renseigné",
            0:"Sans objet",
            1:"Sur chaussée à + 50 m du passage piéton",
            2:"Sur chaussée à – 50 m du passage piéton",
            3:"Sur passage piéton sans signalisation lumineuse",
            4:"Sur passage piéton avec signalisation lumineuse",
            5:"Sur trottoir",
            6:"Sur accotement",
            7:"Sur refuge ou BAU",
            8:"Sur contre allée",
            9:"Inconnue"},
    'etatp':{-1:"Non renseigné",
              1:"Seul",
              2:"Accompagné",
              3:"En groupe"},
<<<<<<< HEAD
=======
    'actp':{-1:"Non renseigné",
              0:"Se déplaçant",
              1:"Se déplaçant sens véhicule heurtant",
              2:"Se déplaçant sens inverse du véhicule"},
>>>>>>> 327dd5e738f2650d66be6f04ed71e9b9935294ec
    'catv_cat':{1:"Vélo",
                2:"Vélo électrique",
                3:"Trotinette ou Skate",
                4:"Trotinette ou Skate électrique",
                5:"2 Roues 50 cm3",
                6:"2 Roues 125 cm3",
                7:"2 Roues > 125 cm3",
                8:"Tracteur",
                9:"Utilitaire",
                10:"Poid Lourd",
                11:"Véhicule léger",
                12:"Bus et Car",
                13:"Train et Tram",
                14:"Autres"},
    'heure_cat':{1:"00H-06H",
                2:"06H-08H",
                3:"08H-10H",
                4:"10H-12H",
                5:"12H-14H",
                6:"14H-16H",
                7:"16H-18H",
                8:"18H-20H",
                9:"20H-22H",
                10:"22H-24H"},
    'age_cat':{1:"0-13",
                2:"14-17",
                3:"18-24",
                4:"25-34",
                5:"35-44",
                6:"45-54",
                7:"55-64",
                8:"65-74",
                9:"75+"},
    'nbv_cat':{0:"Non renseigné",
               1:"1 voie",
               2:"2 voies",
               3:"3 voies",
               4:"4 voies",
               5:"5 voies",
               6:"6 voies",
               7:"+6 voies"},
    'vma_cat':{0:"Non renseigné",
               1:"0-10kmh",
               2:"10-20kmh",
               3:"20-30kmh",
               4:"30-40kmh",
               5:"40-50kmh",
               6:"50-60kmh",
               7:"60-70kmh",
               8:"70-80kmh",
               9:"80-90kmh",
               10:"90-100kmh",
               11:"100-110kmh",
               12:"110-120kmh",
               13:"120-130kmh",
               14:"+130kmh"},
    'jour':{0:"Lundi",
            1:"Mardi",
            2:"Mercredi",
            3:"Jeudi",
            4:"Vendredi",
            5:"Samedi",
            6:"Dimanche"},
    'mois':{1:"Janvier",
            2:"Février",
            3:"Mars",
            4:"Avril",
            5:"Mai",
            6:"Juin",
            7:"Juillet",
            8:"Août",
            9:"Septembre",
            10:"Octobre",
            11:"Novembre",
            12:"Décembre"}
}

val_variable = {
    'lum':{1:"Plein jour",
           2:"Crépuscule ou aube",
           3:"Nuit sans éclairage public",
           4:"Nuit avec éclairage public non allumé",
           5:"Nuit avec éclairage public allumé"},
    'agg':{1:"Hors agglomération",
           2:"En agglomération"},
    'int':{1:"Hors intersection",
           2:"Intersection en X",
           3:"Intersection en T",
           4:"Intersection en Y",
           5:"Intersection à plus de 4 branches",
           6:"Giratoire",
           7:"Place",
           8:"Passage à niveau",
           9:"Autre intersection"},
    'atm':{-1:"Non renseigné",
            1:"Normale",
            2:"Pluie légère",
            3:"Pluie forte",
            4:"Neige - grêle",
            5:"Brouillard - fumée",
            6:"Vent fort - tempête",
            7:"Temps éblouissant",
            8:"Temps couvert",
            9:"Autre"},
    'col':{-1:"Non renseigné",
            1:"Deux véhicules - frontale",
            2:"Deux véhicules – par l’arrière",
            3:"Deux véhicules – par le coté",
            4:"Trois véhicules et plus – en chaîne",
            5:"Trois véhicules et plus - collisions multiples",
            6:"Autre collision",
            7:"Sans collision"},
    'catr':{1:"Autoroute",
            2:"Route nationale",
            3:"Route Départementale",
            4:"Voie Communales",
            5:"Hors réseau public",
            6:"Parc de stationnement ouvert à la circulation publique",
            7:"Routes de métropole urbaine",
            9:"autre"},
    'circ':{-1:"Non renseigné",
            1:"A sens unique",
            2:"Bidirectionnelle",
            3:"A chaussées séparées",
            4:"Avec voies d’affectation variable"},
    'vosp':{-1:"Non renseigné",
            0:"Sans objet",
            1:"Piste cyclable",
            2:"Bande cyclable",
            3:"Voie réservée"},
    'prof':{-1:"Non renseigné",
            1:"Plat",
            2:"Pente",
            3:"Sommet de côte",
            4:"Bas de côte"},
    'plan':{-1:"Non renseigné",
            1:"Partie rectiligne",
            2:"En courbe à gauche",
            3:"En courbe à droite",
            4:"En S"},
    'surf':{-1:"Non renseigné",
            1:"Normale",
            2:"Mouillée",
            3:"Flaques",
            4:"Inondée",
            5:"Enneigée",
            6:"Boue",
            7:"Verglacée",
            8:"Corps gras – huile",
            9:"Autre"},
    'infra':{-1:"Non renseigné",
            0:"Aucun",
            1:"Souterrain - tunnel",
            2:"Pont - autopont",
            3:"Bretelle d’échangeur ou de raccordement",
            4:"Voie ferrée",
            5:"Carrefour aménagé",
            6:"Zone piétonne",
            7:"Zone de péage",
            8:"Chantier",
            9:"Autres"},
    'situ':{-1:"Non renseigné",
            0:"Aucun",
            1:"Sur chaussée",
            2:"Sur bande d’arrêt d’urgence",
            3:"Sur accotement",
            4:"Sur trottoir",
            5:"Sur piste cyclable",
            6:"Sur autre voie spéciale",
            8:"Autres"},
    'senc':{-1:"Non renseigné",
            0:"Inconnu",
            1:"PK ou PR ou numéro d’adresse postale croissant",
            2:"PK ou PR ou numéro d’adresse postale décroissant",
            3:"Absence de repère"},
    'obs':{-1:"Non renseigné",
            0:"Sans objet",
            1:"Véhicule en stationnement",
            2:"Arbre",
            3:"Glissière métallique",
            4:"Glissière béton",
            5:"Autre glissière",
            6:"Bâtiment, mur, pile de pont",
            7:"Support de signalisation verticale ou poste d’appel d’urgence",
            8:"Poteau",
            9:"Mobilier urbain",
            10:"Parapet",
            11:"Ilot, refuge, borne haute",
            12:"Bordure de trottoir",
            13:"Fossé, talus, paroi rocheuse",
            14:"Autre obstacle fixe sur chaussée",
            15:"Autre obstacle fixe sur trottoir ou accotement",
            16:"Sortie de chaussée sans obstacle",
            17:"Buse – tête d’aqueduc"},
    'obsm':{-1:"Non renseigné",
            0:"Aucun",
            1:"Piéton",
            2:"Véhicule",
            4:"Véhicule sur rail",
            5:"Animal domestique",
            6:"Animal sauvage",
            9:"Autre"},
    'choc':{-1:"Non renseigné",
            0:"Aucun",
            1:"Avant",
            2:"Avant droit",
            3:"Avant gauche",
            4:"Arrière",
            5:"Arrière droit",
            6:"Arrière gauche",
            7:"Côté droit",
            8:"Côté gauche",
            9:"Chocs multiples (tonneaux)"},
    'manv':{-1:"Non renseigné",
            0:"Inconnue",
            1:"Sans changement de direction",
            2:"Même sens, même file",
            3:"Entre 2 files",
            4:"En marche arrière",
            5:"A contresens",
            6:"En franchissant le terre-plein central",
            7:"Dans le couloir bus, dans le même sens",
            8:"Dans le couloir bus, dans le sens inverse",
            9:"En s’insérant",
            10:"En faisant demi-tour sur la chaussée",
            11:"Changeant de file à gauche",
            12:"Changeant de file à droite",
            13:"Déporté à gauche",
            14:"Déporté à droite",
            15:"Tournant à gauche",
            16:"Tournant à droite",
            17:"Dépassant à gauche",
            18:"Dépassant à droite",
            19:"Traversant la chaussée",
            20:"Manoeuvre de stationnement",
            21:"Manoeuvre d’évitement",
            22:"Ouverture de porte",
            23:"Arrêté (hors stationnement)",
            24:"En stationnement (avec occupants)",
            25:"Circulant sur trottoir",
            26:"Autres manoeuvres"},
    'motor':{-1:"Non renseigné",
            0:"Inconnu",
            1:"Hydrocarbures",
            2:"Hybride électrique",
            3:"Electrique",
            4:"Hydrogène",
            5:"Humaine",
            6:"Autre"},
    'place':{1:"Conducteur",
            2:"Passager principal",
            3:"Passager arrière droit",
            4:"Passager arrière gauche",
            5:"Passager arrière centre",
            6:"Passager avant centre",
            7:"Passager gauche",
            8:"Passager centre",
            9:"Passager droit"},
    'catu':{1:"Conducteur",
            2:"Passager",
            3:"Piéton"},
    'grav':{1:"Indemne",
<<<<<<< HEAD
            2:"Tué",
            3:"Blessé hospitalisé",
            4:"Blessé léger"},
=======
            2:"Blessé léger",
            3:"Blessé hospitalisé",
            4:"Tué"},
>>>>>>> 327dd5e738f2650d66be6f04ed71e9b9935294ec
    'sexe':{1:"Masculin",
            2:"Féminin"},
    'trajet':{-1:"Non renseigné",
               0:"Non renseigné",
               1:"Domicile – travail",
               2:"Domicile – école",
               3:"Courses – achats",
               4:"Utilisation professionnelle",
               5:"Promenade – loisirs",
               9:"Autre"},
    'secu1':{-1:"Non renseigné",
            0:"Aucun équipement",
            1:"Ceinture",
            2:"Casque",
            3:"Dispositif enfants",
            4:"Gilet réfléchissant",
            5:"Airbag (2RM/3RM)",
            6:"Gants (2RM/3RM)",
            7:"Gants + Airbag (2RM/3RM)",
            8:"Non déterminable",
            9:"Autre"},
    'secu2':{-1:"Non renseigné",
            0:"Aucun équipement",
            1:"Ceinture",
            2:"Casque",
            3:"Dispositif enfants",
            4:"Gilet réfléchissant",
            5:"Airbag (2RM/3RM)",
            6:"Gants (2RM/3RM)",
            7:"Gants + Airbag (2RM/3RM)",
            8:"Non déterminable",
            9:"Autre"},
    'secu3':{-1:"Non renseigné",
            0:"Aucun équipement",
            1:"Ceinture",
            2:"Casque",
            3:"Dispositif enfants",
            4:"Gilet réfléchissant",
            5:"Airbag (2RM/3RM)",
            6:"Gants (2RM/3RM)",
            7:"Gants + Airbag (2RM/3RM)",
            8:"Non déterminable",
            9:"Autre"},
    'locp':{-1:"Non renseigné",
            0:"Sans objet",
            1:"Sur chaussée à + 50 m du passage piéton",
            2:"Sur chaussée à – 50 m du passage piéton",
            3:"Sur passage piéton sans signalisation lumineuse",
            4:"Sur passage piéton avec signalisation lumineuse",
            5:"Sur trottoir",
            6:"Sur accotement",
            7:"Sur refuge ou BAU",
            8:"Sur contre allée",
            9:"Inconnue"},
    'etatp':{-1:"Non renseigné",
              1:"Seul",
              2:"Accompagné",
              3:"En groupe"},
<<<<<<< HEAD
    'catv_cat':{1:"Vélo",
                2:"Vélo électrique",
                3:"Trotinette ou Skate",
                4:"Trotinette ou Skate électrique",
                5:"2 Roues 50 cm3",
                6:"2 Roues 125 cm3",
                7:"2 Roues > 125 cm3",
                8:"Tracteur",
                9:"Utilitaire",
                10:"Poid Lourd",
                11:"Véhicule léger",
                12:"Bus et Car",
                13:"Train et Tram",
                14:"Autres"},
=======
    'catv_cat_s':{0:"Pieton",
                1:"EDP",
                2:"Cycliste",
                3:"Motocycliste",
                4:"Véhicule léger",
                5:"Véhicule lourd",
                6:"Train",
                7:"Autres"},
>>>>>>> 327dd5e738f2650d66be6f04ed71e9b9935294ec
    'heure_cat':{1:"00H-06H",
                2:"06H-08H",
                3:"08H-10H",
                4:"10H-12H",
                5:"12H-14H",
                6:"14H-16H",
                7:"16H-18H",
                8:"18H-20H",
                9:"20H-22H",
                10:"22H-24H"},
    'age_cat':{1:"0-13",
                2:"14-17",
                3:"18-24",
                4:"25-34",
                5:"35-44",
                6:"45-54",
                7:"55-64",
                8:"65-74",
                9:"75+"},
    'nbv_cat':{0:"Non renseigné",
               1:"1 voie",
               2:"2 voies",
               3:"3 voies",
               4:"4 voies",
               5:"5 voies",
               6:"6 voies",
               7:"+6 voies"},
    'vma_cat':{0:"Non renseigné",
               1:"0-10kmh",
               2:"10-20kmh",
               3:"20-30kmh",
               4:"30-40kmh",
               5:"40-50kmh",
               6:"50-60kmh",
               7:"60-70kmh",
               8:"70-80kmh",
               9:"80-90kmh",
               10:"90-100kmh",
               11:"100-110kmh",
               12:"110-120kmh",
               13:"120-130kmh",
               14:"+130kmh"},
    'jour':{0:"Lundi",
            1:"Mardi",
            2:"Mercredi",
            3:"Jeudi",
            4:"Vendredi",
            5:"Samedi",
            6:"Dimanche"},
    'mois':{1:"Janvier",
            2:"Février",
            3:"Mars",
            4:"Avril",
            5:"Mai",
            6:"Juin",
            7:"Juillet",
            8:"Août",
            9:"Septembre",
            10:"Octobre",
            11:"Novembre",
            12:"Décembre"}
}


# Fonction pour detecter les types de données mixtes
def detect_mixed_types(df):
    mixed_type_columns = []
    for col in df.columns:
        unique_types = set(df[col].apply(type))
        if len(unique_types) > 1:
            logging.debug(f"colonne {col} type de la colonne {unique_types}")
            mixed_type_columns.append(col)
    return mixed_type_columns

# Fonction pour convertir types de données mixtes en str
def convert_mixed_types(df, mixed_type_columns):
    for col in mixed_type_columns:
        df[col] = df[col].astype('str') 

# Fonction pour réparer tous les types de données mixtes
def fix_all_mixed_types(list_df):
    for df in list_df:
        mixed_columns = detect_mixed_types(df)
        logging.info(f"Colonnes avec types de données mixtes avant fix : {mixed_columns}")
        convert_mixed_types(df, mixed_columns)
        mixed_columns = detect_mixed_types(df)
        logging.info(f"Colonnes avec types de données mixtes après fix : {mixed_columns}")


# Configurer le logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("--- Début du Notebook ---")
logging.info("--- Chargement des données")
value_df = pd.read_csv('../../data/value_df.csv', sep=";")
features_df = pd.read_csv('../../data/features_df.csv', sep=";")

# Suppression des types mixtes
list_df = [value_df, features_df]
logging.info("Detection des types de données mixtes dans le df fusionné")
fix_all_mixed_types(list_df)

# Construction des features
features_list = list(features_df.columns)
logging.info(f"La liste des features est: {features_list}")

# Construction de la target
target = value_df['grav']
logging.info("La variable cible target est: 'grav'")

# fusion des dataframes
df = pd.concat([features_df,target], axis = 1)

# suppression dep_971 ,  dep_972, dep_973, dep_974, dep_975, dep_976, dep_977, dep_978, dep_986, dep_987, dep_988
#df = df.drop(['dep_971','dep_972','dep_973','dep_974','dep_975','dep_976','dep_977','dep_978','dep_986','dep_987','dep_988'], axis=1)
#df.head()

# Séparation du dataframe en variables explicatives et variable cible
features = df.drop('grav', axis=1)
#target = df['grav']
target = df['grav'].replace({ 'Tué': 1, 'Blessé hospitalisé': 1, 'Blessé léger': 1, 'Indemne': 0})

# Séparation Train / Test
#X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 66)

# Prepare features and target
# Sélectionner 10% du jeu de données
df_sample = df.sample(frac=0.1, random_state=42)  # 10% des données
features = df_sample.drop(columns=["grav"])
target = df_sample["grav"]
target = target.replace({'Tué': 1, 'Blessé hospitalisé': 1, 'Blessé léger': 1, 'Indemne': 0})

if features.empty or target.empty or len(features) <= 1:
    raise ValueError("Insufficient data for processing.")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=66)

# Convert y_train to integer if it's not already
y_train = y_train.astype(int)



#-------------------------------------------------------------------------------------------------

# Configuration de la mise en page
st.set_page_config(page_title="Prédiction de la Gravité des Accidents Routiers", layout="wide")

# Titre et contexte
st.markdown("<h1 style='text-align: center; color: #4B0082;'>Prédiction de la Gravité des Accidents Routiers en France</h1>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #555555;'>
    L'objectif de ce projet est de prédire la gravité des accidents routiers en France. 
    Ce rapport présentera en détail la première étape de ce projet de Data Science, qui consiste à étudier et à appliquer des méthodes de nettoyage du jeu de données. 
    Une fois le jeu de données assaini, la prochaine étape consistera à extraire les caractéristiques pertinentes pour estimer la gravité des accidents. 
    Nous procéderons ensuite à la création d'un modèle prédictif, suivi de son entraînement, afin de comparer ses performances avec des données historiques. 
    Enfin, sur la base des résultats obtenus, nous développerons un système de scoring des zones à risque, prenant en compte des facteurs tels que les conditions météorologiques et l'emplacement géographique.
    </div>
""", unsafe_allow_html=True)

# Sidebar avec logos et boutons
st.sidebar.image("logo.png", width=150)  
st.sidebar.header("Navigation")
pages = ["📜 Présentation du Projet", "🏠 Exploration des Données", "📊 Visualisation", "📈 Modélisation", "🙏 Remerciements"]
page = st.sidebar.radio("Aller vers", pages)

# Présentation du projet
if page == pages[0]:
    st.subheader("1. Présentation du Projet")
    st.markdown("""
        <div style='color: #333333;'>
        L'objectif de ce projet est de prédire la gravité des accidents routiers en France. 
        Pour chaque accident corporel survenu sur une voie ouverte à la circulation publique, des informations sont collectées par les forces de l'ordre.
        
        ### 1.1 L’Observatoire National Interministériel de la Sécurité Routière
        Les bases de données extraites du fichier BAAC répertorient tous les accidents corporels de la circulation survenus au cours d'une année donnée en France métropolitaine.
        
        ### 1.2 Le Bulletin d’Analyse des Accidents Corporels de la Circulation (BAAC)
        Le Bulletin d'Analyse des Accidents de la Circulation est une fiche statistique remplie par les Forces de l'Ordre à la suite d'un accident corporel.
        
        ### 1.3 Les Spécifications des données
        La base Etalab fournit les données des accidents corporels de la circulation par année et est répartie en 4 fichiers distincts.
        </div>
    """, unsafe_allow_html=True)

# Exploration des données
if page == pages[1]:
    st.subheader("3. Exploration des Données")
    st.write("Affichage des 10 premières lignes du jeu de données :")
    #df = pd.read_csv("./data/train.csv") 
    st.dataframe(df.head(10))
    st.write("Forme du DataFrame :", df.shape)
    st.write("Description des données :")
    st.dataframe(df.describe())
    
    if st.checkbox("Afficher les valeurs manquantes"):
        st.write(df.isna().sum())

    # Analyse des valeurs manquantes
    st.subheader("Analyse des Valeurs Manquantes")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])

# Visualisation des données
if page == pages[2]:
    st.subheader("4. Visualisation des Données")
    # Exemple de visualisation : Répartition par gravité
    fig, ax = plt.subplots()
    sns.countplot(x='grav', data=df, ax=ax, palette='pastel')
    ax.set_title("Répartition de la Gravité des Accidents", color='#4B0082')
    st.pyplot(fig)

    # Autres visualisations peuvent être ajoutées ici
    # Exemple : Répartition par département
    if st.checkbox("Afficher la répartition par département"):
        fig2, ax2 = plt.subplots()
        sns.countplot(y='dep', data=df, ax=ax2, palette='pastel')
        ax2.set_title("Répartition des Accidents par Département", color='#4B0082')
        st.pyplot(fig2)

# Modélisation
if page == pages[3]:
    st.subheader("5. Modélisation")
    
    # Préparation des données
    st.write("Nous allons construire un modèle de classification pour prédire la gravité des accidents.")
    
    # Exemple de préparation des données
    y = df['grav']
    X = df.drop(columns=['grav', 'id_usager'])  # Supprimez les colonnes non pertinentes

    # Traitement des valeurs manquantes
    X.fillna(method='ffill', inplace=True)

    # Séparation des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

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
    st.subheader("🙏 Remerciements")
    st.write("Merci d'avoir consulté ce projet !")

# Footer
st.markdown("---")
st.markdown("Développé dans le cadre du projet de Data Science | © 2025")