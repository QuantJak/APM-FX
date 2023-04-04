import pickle
from pathlib import Path
#from optimisation import *

import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_option_menu import option_menu as om
from datetime import datetime, timedelta
import yfinance as yf
import streamlit.components.v1 as components
from scipy import optimize
from scipy.stats import norm
import scipy.optimize as sci_opt
import scipy.optimize as minimize

import numpy as np
from scipy.optimize import fsolve
import scipy.optimize as sci_opt
from scipy.stats import norm
from scipy import optimize

import numpy as np
import pandas as pd
import hydralit_components as hc
import folium
from streamlit_folium import st_folium
import streamlit_authenticator as stauth
import streamlit.components.v1 as html
from PIL import Image
from scipy.optimize import Bounds

import io
import matplotlib.pyplot as plt
import scipy.stats as si
import sympy as sy
from sympy.stats import Normal, cdf
from sympy import init_printing

init_printing()

import sched, time
import xlwings as xw

import plotly.express as px
from scipy import interpolate
from datetime import datetime, date

from scipy.interpolate import interp1d

st.set_page_config(
    page_title="Oaklins Atlas Capital",
    page_icon=Image.open("LOGO.jpeg"),
    layout="wide",

    initial_sidebar_state="expanded",
)

col1, col2, col3 = st.columns(3)
with col1:
    image = Image.open('LOGO.jpeg')
    st.image(image, width=150, output_format='center')


# fonction du navigateur bar
def navBar():
    menu_data = [
        {'id': "info", 'icon': "book", 'label': "S'informer"},
        {'id': 'spot', 'icon': "person lines fill", 'label': "Spot"},
        {'id': 'taux', 'icon': "person lines fill", 'label': "Taux d'intérêt"},
        {'id': 'vol', 'icon': "person lines fill", 'label': "Nappe de Volatilité"},
        {'id': 'Produits', 'icon': "far fa-chart-bar", 'label': " Produits de Couverture"},  # no tooltip message
        {'id': 'contact', 'icon': "person lines fill", 'label': "Contacter nous"},

        # { 'label': " "},{ 'label': " "},{ 'label': " "},{'label': " "},
        # {'label': " "},{'label': " "},{ 'label': " "},{ 'label': " "},{ 'label': " "},{ 'label': " "},{'label': " "},{'label': " "},
        # {'label': " "},{'label': " "},{'label': " "},{'label': " "},{'label': " "},{'label': " "},{'label': " "},{'label': " "},

    ]

    over_theme = {'txc_inactive': '#FFFFFF'}
    menu_id = hc.nav_bar(
        menu_definition=menu_data,
        override_theme=over_theme,
        home_name='Acceuil',

        login_name='Déconnexion',
        hide_streamlit_markers=False,  # will show the st hamburger as well as the navbar now!
        sticky_nav=True,  # at the top or not
        sticky_mode='pinned',  # jumpy or not-jumpy, but sticky or pinned
    )
    return menu_id


# fonction de black and sholes

@st.cache
def BS_CALL(s, k, T1, rd, rf, sigma1):
    d1 = (np.log(s / k) + ((rd - rf) + sigma1 ** 2 / 2) * T1) / (sigma1 * np.sqrt(T1)) if k != 0 else 0
    d2 = d1 - sigma1 * np.sqrt(T1)
    Call = s * norm.cdf(d1, 0, 1) * np.exp(-rf * T1) - k * np.exp(-rd * T1) * norm.cdf(d2, 0, 1)
    return Call
@st.cache
def BS_PUT(s, k, T1, rd, rf, sigma1):
    d1 = (np.log(s / k) + ((rd - rf) + sigma1 ** 2 / 2) * T1) / (sigma1 * np.sqrt(T1)) if k != 0 else 0
    d2 = d1 - sigma1 * np.sqrt(T1)
    Put = k * np.exp(-rd * T1) * norm.cdf(-d2, 0, 1) - s * norm.cdf(-d1, 0, 1) * np.exp(-rf * T1)
    return Put

@st.cache
# fonction de fk
def cours_terme(s, T, rd, rf):
    F_k = s * np.exp((rd - rf) * T)
    return F_k

@st.cache
def find_strike1(s, rd, rf, T, sigma1, p1, p2):
    def objectif_func(k1):
        return (N * p2 * (k1 - s * np.exp((rd - rf) * T)) * np.exp(-(rd - rf) * T)) + N * p1 * BS_PUT(s, k1, T1, rd, rf,
                                                                                                      sigma1)

    sol = optimize.root_scalar(objectif_func, bracket=[1, 1e8], method='brentq')
    return sol.root

@st.cache
def find_strike2(s, rd, rf, T, sigma1, p1, p2):
    def objectif_func(k1):
        return (-N * p2 * (k1 - s * np.exp((rd - rf) * T)) * np.exp(-(rd - rf) * T)) + N * p1 * BS_CALL(s, k1, T1, rd,
                                                                                                        rf, sigma1)

    sol = optimize.root_scalar(objectif_func, bracket=[1, 1e8], method='brentq')
    return sol.root

@st.cache
def find_strike(s, rd, rf, T1, sigma1):
    def objectif_func(k1):
        return (k1 - s * np.exp((rd - rf) * T1)) * np.exp(-(rd - rf) * T1) + BS_PUT(s, k1, T1, rd, rf, sigma1)

    sol = optimize.root_scalar(objectif_func, bracket=[1, 1e8], method='brentq')
    return sol.root


col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    ##--- USER AUTHENTICATION ---
    names = ["Oumaima", " Omar"]
    usernames = ["oumaima", "omar"]

    # load hashed passwords
    file_path = Path(__file__).parent / "hashed_pw.pkl"
    with file_path.open("rb") as file:
        hashed_passwords = pickle.load(file)

    authenticator = stauth.Authenticate(names, usernames, hashed_passwords, "sales_dashboard", "abcdef",
                                        cookie_expiry_days=0)

    name, authentication_status, username = authenticator.login("Login", "main")

    if authentication_status == False:
        st.error("Identifiant/Mot de Passe est incorrecte")

    if authentication_status == None:
        st.warning("Saisir votre Identifiant et Mot de Passe")

if authentication_status:

    menu_id = navBar()

    if menu_id == 'Acceuil':

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        st.markdown(
            """
            <style>
            .reportview-container {
                background:st.image(img, width=200, output_format='center')
            }
           .sidebar .sidebar-content {
                background: st.image(img, width=300, output_format='center')
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "<h1 style='text-align: center;color: #194489;font-size:70px'>Stratégies de couverture du risque de change </h1>",
            unsafe_allow_html=True)

        st.markdown(
            "<h2 style='text-align: center; color: #6cb44c;'> Protégez votre entreprise des variations de change </h2>",
            unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col2:
            img = Image.open("finance.jpg")

            st.image(img, width=800, use_column_width=False)

        col1, col2, col3 = st.columns(3)
        with col2:

            st.markdown(
                "<h1 style='text-align: center; color: #6cb44c;font-size:40px'>Qui sommes nous ? </h1>",
                unsafe_allow_html=True)

            st.write("""Affiliée à Oaklins, la banque d’affaires midmarket la plus expérimentée au monde, Oaklins Atlas Capital met à la disposition de ses clients une capacité d’exécution mondiale et l’expérience collective exceptionnelle de 800 professionnels dans 45 pays.
        Avec une offre complète de services financiers et un vaste réseau relationnel local et mondial, Oaklins Atlas Capital apporte des solutions globales, intégrées et personnalisées à chacun de ses clients.""")
        st.header('')
        col1, col2, col3 = st.columns(3)
        with col2:

            st.markdown(
                "<h1 style='text-align: center; color: #6cb44c;font-size:40px'>Nos services </h1>",
                unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col2:
            img = Image.open("services1.jpg")

            st.image(img, width=900, use_column_width=False)
        st.header('')
        st.header('')
        col1, col2, col3 = st.columns(3)
        st.header('')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.header("")
            st.markdown(
                "<h1 style='text-align: left; color: #6cb44c;font-size:40px'>Problématique </h1>",
                unsafe_allow_html=True)
        with col1:

            st.write(
                "Considerons une entreprise résidant au Maroc et qui importe ou exporte des biens de l'étranger et paye ou sera payé par ses fournisseurs en devise étrangère (USD,EUR) L'entreprise veut se protèger contre la hausse ou la baisse du cours de change pour sécuriser ses marges.")
            # st.write( "- Proposition de produits sur mesure afin d’optimiser les frais de couverture (FX / commodities) des clients")

        with col3:

            img = Image.open("2.jpg")

            st.image(img, width=500, use_column_width=False, output_format='right')

        col1, col2 = st.columns(2)
        st.header("")
        st.header("")
        st.header("")

        with col1:

            img = Image.open("change-à-terme1-750x476.jpg")

            st.image(img, width=500, use_column_width=False)
        with col2:

            st.header("")
            st.markdown(
                "<h1 style='text-align: left; color: #6cb44c;font-size:40px'>Nos Solutions</h2>",
                unsafe_allow_html=True)
            st.write(
                """ Pour vous aider à trouver la ou les stratégies les mieux adaptées à votre situation, nous avons conçu un outil simple et dynamique qui vous propose un panier de stratégies de couverture  et qui vous permettra de  sélectionner lesquelles  qui vous semblent les plus pertinentes par rapport à vos objectifs. """)
            st.write("""- Création de produits structurés sur mesure  pour les besoins de couverture des clients""")
            st.write(
                """- Proposition de produits sur mesure afin d’optimiser les frais de couverture (FX / commodities) des clients""")
            # st.write("""Les options de change restent néanmoins des instruments de couverture totalement personnalisables en fonction de critères tels que votre appétence au risque, votre niveau de trésorerie disponible, votre cours budget ou encore votre sentiment par rapport à l’évolution des cours. """)




    elif menu_id == "info":

        col1, col2 = st.columns(2)
        with col1:

            sens = st.radio(
                " LE SENS DE L'OPERATION"
                , ('IMPORT', 'EXPORT'))
        with col2:
            if sens == 'IMPORT':

                liste = st.selectbox('LISTE DES PRODUITS DE COUVERTURE', (
                    'FORWARD(IMPORT)', 'CALL', 'CALL PARTICIPATIF', 'TUNNEL SYMETRIQUE(IMPORT)',
                    'TUNNEL ASYMETRIQUE (1/m)(IMPORT)'))
                st.header("         ")
            else:
                liste = st.selectbox('LISTE DES PRODUITS DE COUVERTURE', (
                    'FORWARD(EXPORT)', 'PUT', 'PUT PARTICIPATIF', 'TUNNEL SYMETRIQUE(EXPORT)',
                    'TUNNEL ASYMETRIQUE (1/m)(EXPORT)'))

        if liste == 'FORWARD(IMPORT)':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> Définition </h2>",
                    unsafe_allow_html=True)

                st.markdown(
                    "Un contrat à terme est un instrument de couverture qui permet à son détenteur de fixer aujourd'hui un cours d'achat de devise pour un dénouement futur")

            with col2:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> Avantage  </h2>",
                    unsafe_allow_html=True)
                st.markdown("- Garantie d'un cours minimum de cession ")
                st.markdown("- Couverture Totale ")
                st.markdown("- Pas de prime à payer  ")

            with col3:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> PDF  </h2>",
                    unsafe_allow_html=True)
                import os
                import base64


                def get_binary_file_downloader_html(bin_file, file_label='File'):
                    with open(bin_file, 'rb') as f:
                        data = f.read()
                    bin_str = base64.b64encode(data).decode()
                    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Télécharger   {file_label}</a>'
                    return href


                st.markdown(get_binary_file_downloader_html('f-i.pdf', 'fichier PDF'), unsafe_allow_html=True)

            st.markdown(
                "<h2 style='text-align: left ;color:#6cb44c;font-size:35px;'> Exemple  </h2>",
                unsafe_allow_html=True)
            st.markdown(
                "Une société qui import une marchandise de l'étranger (Europe) à un nominal qui vaut 1M d'euros pour le 19 juillet 2022. La banque lui propose un Forward avec un strike K=10.50 ")
            st.markdown("- Si au 19  juillet 2022 l'euro vaut 10.40 . L'importateur achète 1M d'euros au prix K= 10.50")
            st.markdown("- Si au 19  juillet 2022 l'euro vaut 10.60 . L'importateur achète 1M d'euros au prix K=10.50")

        if liste == 'FORWARD(EXPORT)':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> Définition </h2>",
                    unsafe_allow_html=True)

                st.markdown(
                    "Le contrat à terme (Forward)  est un instrument de couverture qui permet à son détenteur de fixer aujourd'hui un cour de cession de la devise pour un dénouement futur")

            with col2:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> Avantage  </h2>",
                    unsafe_allow_html=True)
                st.markdown("- Garantie d'un cours minimum d'de cession ")
                st.markdown("- Couverture Totale ")
                st.markdown("- Pas de prime à payer  ")

            with col3:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> PDF  </h2>",
                    unsafe_allow_html=True)
                import os
                import base64


                @st.cache
                def get_binary_file_downloader_html(bin_file, file_label='File'):
                    with open(bin_file, 'rb') as f:
                        data = f.read()
                    bin_str = base64.b64encode(data).decode()
                    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Télécharger   {file_label}</a>'
                    return href


                st.markdown(get_binary_file_downloader_html('f-e.pdf', 'fichier PDF'), unsafe_allow_html=True)

            st.markdown(
                "<h2 style='text-align: left ;color:#6cb44c;font-size:35px;'> Exemple  </h2>",
                unsafe_allow_html=True)
            st.markdown(
                "Une société qui exporte une mrchandise à l'étranger (Europe) à un nominal qui vaut 1M d'euros pour le 19 juillet 2022. La banque lui propose un Forward avec un strike K=10.50 ")
            st.markdown("- Si au 19  juillet 2022 l'euro vaut 10.40 . L'exportateur cède  1M d'euros au prix K= 10.50")
            st.markdown("- Si au 19  juillet 2022 l'euro vaut 10.60 . L'exportateur cède 1M d'euros au prix K=10.50")

        if liste == 'CALL':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> Définition </h2>",
                    unsafe_allow_html=True)

                st.markdown(
                    "Un call vanille de strike k et de nominal N est un instrument de couverture qui donne à son détenteur le droit et non l'obligation d'acheter un montant de devise à un cours détérminé à l'avance en contrepartie du paiement d'une prime")

            with col2:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> Avantage  </h2>",
                    unsafe_allow_html=True)
                st.markdown("- Garantie d'un cours maximum d'achat ")
                st.markdown("- Couverture Totale ")
                st.markdown("- Possibilité de profiter pleiement de la baisse  ")

            with col3:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> PDF  </h2>",
                    unsafe_allow_html=True)
                import os
                import base64


                @st.cache
                def get_binary_file_downloader_html(bin_file, file_label='File'):
                    with open(bin_file, 'rb') as f:
                        data = f.read()
                    bin_str = base64.b64encode(data).decode()
                    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Télécharger   {file_label}</a>'
                    return href


                st.markdown(get_binary_file_downloader_html('c-i.pdf', 'fichier PDF'), unsafe_allow_html=True)
            st.markdown(
                "<h2 style='text-align: left ;color:#6cb44c;font-size:35px;'> Exemple  </h2>",
                unsafe_allow_html=True)
            st.markdown(
                "Une société qui importe une marchandise de l'étranger (Europe) à un nominal qui vaut 1M d'euros pour le 19 juillet 2022. La banque lui propose un Call avec un strike K=10.50 ")
            st.markdown("- Si au 19  juillet 2022 l'euro vaut 10.40 . L'importateur achète 1M d'euros au prix S= 10.4")
            st.markdown("- Si au 19  juillet 2022 l'euro vaut 10.60 . L'importateur achète 1M d'euros au prix K=10.50")

        if liste == 'PUT':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> Définition </h2>",
                    unsafe_allow_html=True)

                st.markdown(
                    "Un Put classique ou vanille de strike k et de nominal N est un instrument de couverture qui donne a son détenteur le droit et non l'obligation de céder un montant de devise fixe (Nominal) à un cours détérminé à l'avance (prix d'exercice k) en contrepartie du paiement d'une prime")

            with col2:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> Avantage  </h2>",
                    unsafe_allow_html=True)
                st.markdown("- Garantie d'un cours maximum de vente ")
                st.markdown("- Couverture Totale ")
                st.markdown("- Possibilité de profiter pleiement de la hausse  ")

            with col3:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> PDF  </h2>",
                    unsafe_allow_html=True)
                import os
                import base64


                @st.cache
                def get_binary_file_downloader_html(bin_file, file_label='File'):
                    with open(bin_file, 'rb') as f:
                        data = f.read()
                    bin_str = base64.b64encode(data).decode()
                    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Télécharger   {file_label}</a>'
                    return href


                st.markdown(get_binary_file_downloader_html('p-e.pdf', 'fichier PDF'), unsafe_allow_html=True)

            st.markdown(
                "<h2 style='text-align: left ;color:#6cb44c;font-size:35px;'> Exemple  </h2>",
                unsafe_allow_html=True)
            st.markdown(
                "Une société qui exporte une marchandise à l'étranger (Europe) à un nominal qui vaut 1M d'euros pour le 19 juillet 2022. La banque lui propose un Put avec un strike K=10.50 ")
            st.markdown("- Si au 19  juillet 2022 l'euro vaut 10.4 .  L'exportateur vent 1M d'euros au prix K= 10.5")
            st.markdown(
                "- Si au 19  juillet 2022 l'euro vaut 10.64 . L'exportateur vent 1M d'euros au prix prix S=10.64")

        if liste == 'CALL PARTICIPATIF':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> Définition </h2>",
                    unsafe_allow_html=True)

                st.markdown(
                    "Un Call participatif 50% de strike K et de nominal N est une stratégie de couverture qui donne à son détenteur le droit et non l’obligation d’acheter un montant de devise égal à 50% du nominal à un cours déterminé à l’avance K. L’autre  moitié sera achetée à K quelque soit le niveau du spot à l’échéance.")

            with col2:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> Avantage  </h2>",
                    unsafe_allow_html=True)
                st.markdown("- Garantie d'un cours maximum d'achat K ")
                st.markdown("- Couverture Totale ")
                st.markdown("- Possibilité de profiter de la baisse(50% uniquement) ")
                st.markdown("- Pas de Prime à payer ")

            with col3:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> PDF  </h2>",
                    unsafe_allow_html=True)
                import os
                import base64


                @st.cache
                def get_binary_file_downloader_html(bin_file, file_label='File'):
                    with open(bin_file, 'rb') as f:
                        data = f.read()
                    bin_str = base64.b64encode(data).decode()
                    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Télécharger   {file_label}</a>'
                    return href


                st.markdown(get_binary_file_downloader_html('c-p-i.pdf', 'fichier PDF'), unsafe_allow_html=True)

            st.markdown(
                "<h2 style='text-align: left ;color:#6cb44c;font-size:35px;'> Exemple  </h2>",
                unsafe_allow_html=True)
            st.markdown(
                "Une société qui import la  marchandise de l'étranger (Europe) à un nominal qui vaut 1M d'euros pour le 19 juillet 2022. La banque lui propose un Call Participatif(50%) avec un strike K=10.50 ")
            st.markdown(
                "- Si au 19  juillet 2022 l'euro vaut S=10.60 .  L'importateur achète 1M d'euros au prix K= 10.5")
            st.markdown("- Si au 19  juillet 2022 l'euro vaut S=10.40 ")
            st.markdown("  L'importateur achète 500 000 d'euros au prix  S=10.60")
            st.markdown("  L'importateur achète 500 000 d'euros au prix pix K = 10.50")

        if liste == 'PUT PARTICIPATIF':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> Définition </h2>",
                    unsafe_allow_html=True)

                st.markdown(
                    "Un Put Participative 50% de strike k et de nominal N est un instrument de couverture qui donne a son détenteur le droit et non l'obligation de céder un montant de devise égale à 50% du nominal à un cours détérminé a l'avance K.L'autre moitié sera achetée à K quelque soit le niveau du spot à l'échéance.")

            with col2:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> Avantage  </h2>",
                    unsafe_allow_html=True)
                st.markdown("- Garantie d'un cours minimun de cession K ")
                st.markdown("- Couverture Totale ")
                st.markdown("- Possibilité de profiter de la hausse(50% uniquement) ")
                st.markdown("-  Pas de Prime à payer ")

            with col3:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> PDF  </h2>",
                    unsafe_allow_html=True)
                import os
                import base64


                @st.cache
                def get_binary_file_downloader_html(bin_file, file_label='File'):
                    with open(bin_file, 'rb') as f:
                        data = f.read()
                    bin_str = base64.b64encode(data).decode()
                    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Télécharger   {file_label}</a>'
                    return href


                st.markdown(get_binary_file_downloader_html('p-p-e.pdf', 'fichier PDF'), unsafe_allow_html=True)

            st.markdown(
                "<h2 style='text-align: left ;color:#6cb44c;font-size:35px;'> Exemple  </h2>",
                unsafe_allow_html=True)
            st.markdown(
                "Une société qui exporte une marchandise à l'étranger (Europe) à un nominal qui vaut 1M d'euros pour le 19 juillet 2022. La banque lui propose un Put Participatif(50%) avec un strike K=10.50 ")
            st.markdown("- Si au 19  juillet 2022 l'euro vaut S=10.40 .  L'exportateur vent 1M d'euros au prix K= 10.5")
            st.markdown("- Si au 19  juillet 2022 l'euro vaut S=10.60 ")
            st.markdown("  L'exportateur vent 500 000 d'euros au prix  S=10.60")
            st.markdown("  L'exportateur vent 500 000 d'euros au prix pix K = 10.50")

        if liste == 'TUNNEL SYMETRIQUE(IMPORT)':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> Définition </h2>",
                    unsafe_allow_html=True)

                st.markdown(
                    "Tunnel (K1,K2) est une stratégie de couverture qui fixe un couloir délimité par une borne inférieure K1 (cours minimum d'achat) et une borne supérieure K2 (cours maximum d'chat)")

            with col2:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> Avantage  </h2>",
                    unsafe_allow_html=True)
                st.markdown("- Garantie d'un cours maximum d'achat K2. ")
                st.markdown("- Couverture Totale ")
                st.markdown("- Possibilité de profiter de la baisse")
                st.markdown("- Pas de prime à payer")

            with col3:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> PDF  </h2>",
                    unsafe_allow_html=True)
                import os
                import base64


                @st.cache
                def get_binary_file_downloader_html(bin_file, file_label='File'):
                    with open(bin_file, 'rb') as f:
                        data = f.read()
                    bin_str = base64.b64encode(data).decode()
                    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Télécharger   {file_label}</a>'
                    return href


                st.markdown(get_binary_file_downloader_html('t-i.pdf', 'fichier PDF'), unsafe_allow_html=True)

            st.markdown(
                "<h2 style='text-align: left ;color:#6cb44c;font-size:35px;'> Exemple  </h2>",
                unsafe_allow_html=True)
            st.markdown(
                "Une société qui importe une marchandise de l'étranger (Europe) à un nominal qui vaut 1M d'euros pour le 19 juillet 2022. La banque lui propose un Tunnel symétrique avec un strike K1=10.50 et k2 =10.60")
            st.markdown(
                "- Si au 19  juillet 2022 l'euro vaut S=10.40 .  L'importateur achète 1M d'euros au prix K1= 10.5")
            st.markdown(
                "- Si au 19  juillet 2022 l'euro vaut S=10.55 . L'importateur achète 1M d'euros au prix S= 10.55")
            st.markdown(
                "- Si au 19  juillet 2022 l'euro vaut S=10.65 . L'importateur achète 1M d'euros au prix K2= 10.60")

        if liste == 'TUNNEL SYMETRIQUE(EXPORT)':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> Définition </h2>",
                    unsafe_allow_html=True)

                st.markdown(
                    "Tunnel (K1,K2) est une stratégie de couverture qui fixe un couloir délimité par une borne inférieure K1 (cours minimum de cession) et une borne supérieure K2 (cours maximum de cession) ")

            with col2:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> Avantage  </h2>",
                    unsafe_allow_html=True)
                st.markdown("- Garantie d'un cours maximum de cession K1 ")
                st.markdown("- Couverture Totale ")
                st.markdown("- Possibilité de profiter de la hausse ")
                st.markdown("-  Pas de prime à payer")

            with col3:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> PDF  </h2>",
                    unsafe_allow_html=True)
                import os
                import base64


                @st.cache
                def get_binary_file_downloader_html(bin_file, file_label='File'):
                    with open(bin_file, 'rb') as f:
                        data = f.read()
                    bin_str = base64.b64encode(data).decode()
                    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Télécharger   {file_label}</a>'
                    return href


                st.markdown(get_binary_file_downloader_html('t-e.pdf', 'fichier PDF'), unsafe_allow_html=True)

            st.markdown(
                "<h2 style='text-align: left ;color:#6cb44c;font-size:35px;'> Exemple  </h2>",
                unsafe_allow_html=True)
            st.markdown(
                "Une société qui exporte la marchandise à l'étranger (Europe) à un nominal qui vaut 1M d'euros pour le 19 juillet 2022. La banque lui propose un Tunnel symétrique avec un strike K1=10.50 et k2 =10.60")
            st.markdown("- Si au 19  juillet 2022 l'euro vaut S=10.40 . L'exportateur cède 1M d'euros au prix K1= 10.5")
            st.markdown("- Si au 19  juillet 2022 l'euro vaut S=10.55 . L'exportateur cède 1M d'euros au prix S= 10.55")
            st.markdown("- Si au 19 juillet 2022 l'euro vaut S=10.65 .L'exportateur cède 1M d'euros au prix K2= 10.60")

        if liste == 'TUNNEL ASYMETRIQUE (1/m)(IMPORT)':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> Définition </h2>",
                    unsafe_allow_html=True)

                st.markdown(
                    "Le Tunnel asymétrique (1/m) (K1, K2) de nominal N est une stratégie de couverture qui, à l’instar du tunnel symétrique, fixe un couloir délimité par une borne inférieure K1 et une borne supérieure K2, mais présente la particularité de ne pas fixer le montant exact à acheter à l’échéance. Ce montant peut être 1* Nominal ou m * Nominal dépendamment du niveau du sous-jacent.")

            with col2:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> Avantage  </h2>",
                    unsafe_allow_html=True)
                st.markdown("- Couloir largement plus intéressant que celui d’un tunnel symétrique ")
                st.markdown("- Garantie d'un cours maximum d’achat ")
                st.markdown("- Pas de prime à payer")

            with col3:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> PDF  </h2>",
                    unsafe_allow_html=True)
                import os
                import base64


                @st.cache
                def get_binary_file_downloader_html(bin_file, file_label='File'):
                    with open(bin_file, 'rb') as f:
                        data = f.read()
                    bin_str = base64.b64encode(data).decode()
                    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Télécharger   {file_label}</a>'
                    return href


                st.markdown(get_binary_file_downloader_html('t-a-i.pdf', 'fichier PDF'), unsafe_allow_html=True)

            st.markdown(
                "<h2 style='text-align: left ;color:#6cb44c;font-size:35px;'> Exemple  </h2>",
                unsafe_allow_html=True)
            st.markdown(
                "Une société qui importe la marchandise de l'étranger (Europe) à un nominal qui vaut 1M d'euros pour le 19 juillet 2022. La banque lui propose un tunnel asymétrique avec un strike K1=10.50 , K2= 10.60 et m =2")
            st.markdown(
                "- Si au 19  juillet 2022 l'euro vaut S=10.40 . L'importateur achète 2M d'euros au prix K1=10.50")
            st.markdown("- Si au 19  juillet 2022 l'euro vaut S=10.55 . L'importateur achète 1M/2M d'euros au prix S")
            st.markdown(
                "- Si au 19  juillet 2022 l'euro vaut S=10.65 . L'importateur achète 1M d'euros au prix K2=10.60")

        if liste == 'TUNNEL ASYMETRIQUE (1/m)(EXPORT)':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> Définition </h2>",
                    unsafe_allow_html=True)

                st.markdown(
                    "Le Tunnel asymétrique (1/m) (K1, K2) de nominal N est une stratégie de couverture qui, à l’instar du tunnel symétrique, fixe un couloir délimité par une borne inférieure K1 et une borne supérieure K2, mais présente la particularité de ne pas fixer le montant exact à céder à l’échéance. Ce montant peut être 1* Nominal ou m * Nominal dépendamment du niveau du sous-jacent. ")

            with col2:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> Avantage  </h2>",
                    unsafe_allow_html=True)
                st.markdown("- Couloir largement plus intéressant que celui d’un tunnel symétrique")
                st.markdown("- Garantie d’un cours minimum de cession ")
                st.markdown("- Pas de prime à payer")

            with col3:
                st.markdown(
                    "<h2 style='text-align: left ;color:#14448c;font-size:28px;'> PDF  </h2>",
                    unsafe_allow_html=True)
                import os
                import base64


                @st.cache
                def get_binary_file_downloader_html(bin_file, file_label='File'):
                    with open(bin_file, 'rb') as f:
                        data = f.read()
                    bin_str = base64.b64encode(data).decode()
                    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Télécharger   {file_label}</a>'
                    return href


                st.markdown(get_binary_file_downloader_html('t-a-e.pdf', 'fichier PDF'), unsafe_allow_html=True)

            st.markdown(
                "<h2 style='text-align: left ;color:#6cb44c;font-size:35px;'> Exemple  </h2>",
                unsafe_allow_html=True)
            st.markdown(
                "Une société qui exporte une marchandise à l'étranger (Europe) à un nominal qui vaut 1M d'euros pour le 19 juillet 2022. La banque lui propose un Tunnel asymétrique avec un strike K1=10.50 et K2=10.60 ")
            st.markdown("- Si au 19  juillet 2022 l'euro vaut S=10.40 .L'exportateur vent 1M d'euros au prix K1= 10.5")
            st.markdown(
                "- Si au 19  juillet 2022 l'euro vaut S=10.55 .L'exportateur vent 1M/2M d'euros au prix S=10.55")
            st.markdown("- Si au 19  juillet 2022 l'euro vaut S=10.65 .L'exportateur vent 2M d'euros au prix K2=10.60")



    if menu_id == "spot":

        placeholder = st.empty()
        encour=False
        #for seconds in range(10000000):
       # seconds=0
        while True:
            if encour==False:
                encour=True
                try:
                    with placeholder.container():
                        col1, col2, col3 = st.columns([0.8, 1, 1])
                        with col2:
                            st.markdown(

                                "<h1 style='text-align: center; color: #194489;font-size:60px'>  Cotation Spot </h1>",
                                unsafe_allow_html=True)
                            st.header("")
                        workbook = xw.Book("streamlitmodifie1.xlsm")
                        df = workbook.sheets[0].range('A1').options(pd.DataFrame,
                                                                    header=1,
                                                                    # index=True,
                                                                    expand='table').value
                        col1, col2, col3, col4 = st.columns(4)
                        ##########################################################################################"""
                        data_frame = pd.DataFrame(df)
                        EURAUDbid = float(data_frame["Bid"][0])
                        # st.experimental_rerun()
                        EURCADbid = float(data_frame["Bid"][1])
                        EURCHFbid = float(data_frame["Bid"][2])
                        EURDKKbid = float(data_frame["Bid"][3])
                        EURGBPbid = float(data_frame["Bid"][4])
                        EURJPYbid = float(data_frame["Bid"][5])
                        EURNOKbid = float(data_frame["Bid"][6])
                        EURNZDbid = float(data_frame["Bid"][7])
                        EURSEKbid = float(data_frame["Bid"][8])
                        EURUSDbid = float(data_frame["Bid"][9])
                        EURZARbid = float(data_frame["Bid"][10])
                        USDCADbid = float(data_frame["Bid"][11])
                        USDCHFbid = float(data_frame["Bid"][12])
                        USDJPYbid = float(data_frame["Bid"][13])
                        USDNOKbid = float(data_frame["Bid"][14])
                        USDSEKbid = float(data_frame["Bid"][15])
                        USDZARbid = float(data_frame["Bid"][16])

                        EURAUDask = float(data_frame["Ask"][0])
                        EURCADask = float(data_frame["Ask"][1])
                        EURCHFask = float(data_frame["Ask"][2])
                        EURDKKask = float(data_frame["Ask"][3])
                        EURGBPask = float(data_frame["Ask"][4])
                        EURJPYask = float(data_frame["Ask"][5])
                        EURNOKask = float(data_frame["Ask"][6])
                        EURNZDask = float(data_frame["Ask"][7])
                        EURSEKask = float(data_frame["Ask"][8])
                        EURUSDask = float(data_frame["Ask"][9])
                        EURZARask = float(data_frame["Ask"][10])
                        USDCADask = float(data_frame["Ask"][11])
                        USDCHFask = float(data_frame["Ask"][12])
                        USDJPYask = float(data_frame["Ask"][13])
                        USDNOKask = float(data_frame["Ask"][14])
                        USDSEKask = float(data_frame["Ask"][15])
                        USDZARask = float(data_frame["Ask"][16])

                        with col1:
                            st.markdown(

                                "<h2 style='text-align: center; color: green;'> EUR/CAD  </h2>",
                                unsafe_allow_html=True)

                        with col2:
                            st.markdown(

                                "<h2 style='text-align: right; color: green;'> USD/NOK  </h2>",
                                unsafe_allow_html=True)
                        with col4:
                            st.markdown(

                                "<h2 style='text-align: left; color: green;'> EUR/AUD  </h2>",
                                unsafe_allow_html=True)
                        col1, col2, col3, col4, col5, col6 = st.columns(6)

                        col1.metric(("Offre "), format(EURGBPbid,".4f"), delta_color="off")

                        # st_autorefresh(interval=5000, limit=100, key="fizzbuzzcounter")
                        col2.metric("Demande  ", format(EURGBPask,".4f"))

                        col3.metric("Offre", format(USDJPYbid,".4f"))

                        col4.metric("Demande", format(USDJPYask,".4f"))

                        col5.metric("Offre ", format(USDCADbid,".4f"))

                        col6.metric("Offre ", format(USDCADask,".4f"))

                        st.header("")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(

                                "<h2 style='text-align: center; color: green;'> EUR/USD  </h2>",
                                unsafe_allow_html=True)
                            # st.title("EURAUD")
                        with col2:
                            st.markdown(

                                "<h2 style='text-align: right; color: green;'> GBP/USD  </h2>",
                                unsafe_allow_html=True)
                        with col4:
                            st.markdown(

                                "<h2 style='text-align: left; color: green;'> EUR/GBP  </h2>",
                                unsafe_allow_html=True)
                        col1, col2, col3, col4, col5, col6 = st.columns(6)

                        col1.metric("Offre ", format(EURAUDbid,".4f"), delta_color="off")

                        col2.metric("Demande  ", format(EURAUDask,".4f"))

                        col3.metric("Offre", format(EURCADbid,".4f"))

                        col4.metric("Demande", format(EURCADask,".4f"))

                        col5.metric("Offre ", format(EURCHFbid,".4f"))

                        col6.metric("Demande ", format(EURCHFask,".4f"))

                        st.header("")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(

                                "<h2 style='text-align: center; color: green;'> CHF/USD  </h2>",
                                unsafe_allow_html=True)

                        with col2:
                            st.markdown(

                                "<h2 style='text-align: right; color: green;'> EUR/JPY  </h2>",
                                unsafe_allow_html=True)
                        with col4:
                            st.markdown(

                                "<h2 style='text-align: left; color: green;'> EUR/CHF  </h2>",
                                unsafe_allow_html=True)

                        col1, col2, col3, col4, col5, col6 = st.columns(6)

                        col1.metric(("Offre "), format(EURJPYbid,".4f"), delta_color="off")
                        col2.metric("Demande  ", format(EURJPYask,".4f"))
                        col3.metric("Offre", format(EURUSDbid,".4f"))
                        col4.metric("Demande", format(EURUSDask,".4f"))
                        col5.metric("Offre ", format(EURDKKbid,".4f"))
                        col6.metric("Demande ", format(EURDKKask,".4f"))

                        st.header("")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(

                                "<h2 style='text-align: center; color: green;'> EUR/DKK  </h2>",
                                unsafe_allow_html=True)

                        with col2:
                            st.markdown(

                                "<h2 style='text-align: right; color: green;'> EUR/NZD  </h2>",
                                unsafe_allow_html=True)
                        with col4:
                            st.markdown(

                                "<h2 style='text-align: left; color: green;'> JPY/USD  </h2>",
                                unsafe_allow_html=True)

                        col1, col2, col3, col4, col5, col6 = st.columns(6)

                        col1.metric(("Offre "), format(EURNOKbid,".4f"), delta_color="off")
                        col2.metric("Demande  ", format(EURNOKask,".4f"))
                        col3.metric("Offre", format(EURNZDbid,".4f"))
                        col4.metric("Demande", format(EURNZDask,".4f"))
                        col5.metric("Offre ", format(EURZARbid,".4f"))
                        col6.metric("Demande ", format(EURZARask,".4f"))
                        st.header("")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(

                                "<h2 style='text-align: center; color: green;'> EUR/SGD  </h2>",
                                unsafe_allow_html=True)

                        with col2:
                            st.markdown(

                                "<h2 style='text-align: right; color: green;'> EUR/SEK  </h2>",
                                unsafe_allow_html=True)
                        with col4:
                            st.markdown(

                                "<h2 style='text-align: left; color: green;'> EUR/NOK  </h2>",
                                unsafe_allow_html=True)
                        col1, col2, col3, col4, col5, col6 = st.columns(6)

                        col1.metric(("Demande "), format(USDZARbid,".4f"), delta_color="off")
                        col2.metric("Offre  ", format(USDZARask,".4f"))
                        col3.metric("Demande", format(USDSEKbid,".4f"))
                        col4.metric("Offre", format(USDSEKask,".4f"))
                        col5.metric("Demande ", format(USDNOKbid,".4f"))
                        col6.metric("Offre ", format(USDNOKask,".4f"))

                        time.sleep(0.001)
                        encour = False
                except:
                    print("An exception occurred")
                    encour = False
                    break

                    #for seconds in range(200):
                    #time.sleep(1)
                    #driver = webdriver.Chrome(executable_path="C:\\Users\\z0044wmy\\Desktop\\chromedriver_win32\\chromedriver.exe")





    elif menu_id == "taux":
        # dataframe filter
        import os


        option = st.selectbox(
            """ """,
            ('EUR', 'USD', 'GBP', 'CAD','CHF','JPY'))
        if option == "EUR":
            s = sched.scheduler(time.time, time.sleep)

            placeholder = st.empty()

            for seconds in range(200):
                # while True:
                try:
                    with placeholder.container():


                        workbook = xw.Book("streamlitmodifie1.xlsm")
                        df = workbook.sheets[11].range('A1').options(pd.DataFrame,
                                                                    header=1,
                                                                    index=True,
                                                                    expand='table').value

                        #col1, col2, col3 = st.columns(3)
                        # with col1:
                        # st.button("Refresh")
                        #with col3:
                          #  today = date.today()
                          #  v = datetime.now().strftime("%H:%M:%S")

                           # st.write(today, v)
                        #with col2:
                          #  st.markdown("<h1 style='text-align: center; color: green;'> EUR0 </h1>",  unsafe_allow_html=True)

                        col1, col2, col3, col4, col5 = st.columns(5)

                        ##########################################################################################"""
                        data_frame = pd.DataFrame(df)
                        ##########################################################################################"""
                        data_frame = pd.DataFrame(df)

                        one_week = float(data_frame["Taux EURO"][0])
                        One_month = float(data_frame["Taux EURO"][1])
                        three_month = float(data_frame["Taux EURO"][2])
                        six_month = float(data_frame["Taux EURO"][3])
                        one_year = float(data_frame["Taux EURO"][4])

                        col1.metric("Une semaine %  ", format(one_week,".4f"))
                        col2.metric("Un mois %", format(One_month,".4f"))
                        col3.metric("Trois mois %", format(three_month,".4f"))
                        col4.metric("Six mois %", format(six_month,".4f"))
                        col5.metric("Un an %", format(one_year,".4f"))

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.markdown(

                                "<h3 style='text-align: left; color: green;'> Graphique </h3>",
                                unsafe_allow_html=True)
                            df = pd.DataFrame(dict(
                                Maturité=[ '1W', '1M', '3M', '6M', '1Y'],
                                Taux_EUR=df["Taux EURO"]))

                            fig = px.line(
                                df,  # Data Frame
                                x="Maturité",  # Columns from the data frame
                                y="Taux_EUR"
                            )
                            fig.update_traces(line_color="green")
                            fig.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                margin={"t": 1, "b": 1, "r": 1, "l": 1},
                                plot_bgcolor='rgba(0,0,0,0)',
                                    width=500,
                                    height=300


                            )



                            st.plotly_chart(fig)

                        X = [ 7, 30, 3 * 30, 6 * 30, 12 * 30]
                        Y = [ one_week, One_month, three_month, six_month, one_year]
                        with col3:
                            st.markdown(

                                "<h3 style='text-align: left; color: green;'> Maturité à interpoler </h3>",
                                unsafe_allow_html=True)
                            c = float(st.text_input("Saisez la maturité (jours)", value=1))

                            # test value
                            interpolate_x = 7 * 30

                            # Finding the interpolation
                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                            col3.markdown(

                                "<h3 style='text-align: left; color: green;'> Résultat </h3>",
                                unsafe_allow_html=True)
                        with col4:
                            col4.markdown(

                                "<h1 style='text-align: left; color: green;'> </h1>",
                                unsafe_allow_html=True)
                            col4.markdown(

                                "<h1 style='text-align: left; color: green;'> </h1>",
                                unsafe_allow_html=True)
                            col4.markdown(

                                "<h4 style='text-align: left; color: green;'> </h4>",
                                unsafe_allow_html=True)


                            if col4.button(f'Calculer '):
                                col3.write((" Taux d'intèrêt qui correspond à " + " " + str(
                                    format(c, ".0f")) + " " + "jours est :" + " " + str(
                                    format(y_interp(c), ".6f"))))
                            time.sleep(1)
                except:
                      print("An exception occurred")
                break



                     # main_task()





        elif option == "USD":

            s = sched.scheduler(time.time, time.sleep)


           # def main_task():
            placeholder = st.empty()

            for seconds in range(200):
                # while True:
                try:
                    with placeholder.container():
                # print "Doing stuff..."
                # do your stuff
                        workbook = xw.Book("streamlitmodifie1.xlsm")
                        df = workbook.sheets[10].range('A1').options(pd.DataFrame,
                                                                    header=1,
                                                                    index=True,
                                                                    expand='table').value

                        # st.table(df)
                       # col1, col2, col3 = st.columns(3)
                        # with col1:
                        # st.button("Refresh")

                       # with col3:
                          #  today = date.today()
                          #  v = datetime.now().strftime("%H:%M:%S")

                         #   st.write(today, v)
                       # with col2:
                            # st.header("")
                           # st.markdown( "<h1 style='text-align: center; color: green;'> Dollar  </h1>", unsafe_allow_html=True)

                        st.header("")
                        col1, col2, col3, col4,col5 = st.columns(5)
                        with col1:
                            ##########################################################################################"""
                            data_frame = pd.DataFrame(df)
                            over_night = float(data_frame["Taux USD"][0])
                            One_month = float(data_frame["Taux USD"][1])
                            three_month = float(data_frame["Taux USD"][2])
                            six_month = float(data_frame["Taux USD"][3])
                            one_year = float(data_frame["Taux USD"][4])

                            # print(one_week)
                            col1.metric("Pendant la nuit %", format(over_night,".4f"), delta_color="off")
                            col2.metric("Un mois %", format(One_month,".4f"))
                            col3.metric("Trois mois %",format(three_month,".4f") )
                            col4.metric("Six mois %",format(six_month,".4f") )
                            col5.metric("Une année %", format(one_year,".4f") )

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.markdown(

                                "<h3 style='text-align: left; color: green;'> Graphique </h3>",
                                unsafe_allow_html=True)

                            df = pd.DataFrame(dict(
                                Maturité=[' ON', '1M', '3M', '6M','1Y'],
                                Taux_USD=df["Taux USD"]))

                            fig = px.line(
                                df,  # Data Frame
                                x="Maturité",  # Columns from the data frame
                                y="Taux_USD"
                            )
                            fig.update_traces(line_color="green")
                            fig.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                margin={"t": 1, "b": 1, "r": 1, "l": 1},
                                plot_bgcolor='rgba(0,0,0,0)',
                                width=500,
                                height=300

                            )

                            st.plotly_chart(fig)

                        X = [1, 30, 3 * 30, 6 * 30,12*30]
                        Y = [over_night, One_month, three_month, six_month,one_year]
                        with col3:
                            st.markdown(

                                "<h3 style='text-align: left; color: green;'> Maturité à interpoler </h3>",
                                unsafe_allow_html=True)
                            c = float(st.text_input("Saisez la maturité (jours)", value=1))

                            # test value
                            # interpolate_x = 7 * 30

                            # Finding the interpolation
                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                            col3.markdown(

                                "<h3 style='text-align: left; color: green;'> Résultat </h3>",
                                unsafe_allow_html=True)
                            with col4:
                                col4.markdown(

                                    "<h1 style='text-align: left; color: green;'> </h1>",
                                    unsafe_allow_html=True)
                                col4.markdown(

                                    "<h1 style='text-align: left; color: green;'> </h1>",
                                    unsafe_allow_html=True)
                                col4.markdown(

                                    "<h4 style='text-align: left; color: green;'> </h4>",
                                    unsafe_allow_html=True)

                                if col4.button(f'Calculer '):
                                    col3.write((" Taux d'intèrêt qui correspond à " + " " + str(
                                        format(c, ".0f")) + " " + "jours est :" + " " + str(
                                        format(y_interp(c), ".6f"))))
                            time.sleep(1)
                except:
                       print("An exception occurred")
                break


            #main_task()
        elif option == "CAD":
            col1, col2, col3 = st.columns(3)

            s = sched.scheduler(time.time, time.sleep)


            #def main_task():
            placeholder = st.empty()

            for seconds in range(200):
                # while True:
                try:
                    with placeholder.container():

                        workbook = xw.Book("streamlitmodifie1.xlsm")
                        df = workbook.sheets[12].range('A1').options(pd.DataFrame,
                                                                    header=1,
                                                                    index=True,
                                                                    expand='table').value

                        #col1, col2, col3 = st.columns(3)
                        # with col1:
                        #  st.button("Refresh")

                       # with col3:
                        #    today = date.today()
                        #    v = datetime.now().strftime("%H:%M:%S")

                           # st.write(today, v)
                        #with col2:
                        #    st.markdown( "<h1 style='text-align: center; color: green;'> Dollar Canadien  </h1>", unsafe_allow_html=True)

                        col1, col2, col3, col4,col5 = st.columns(5)
                        with col1:
                            ##########################################################################################"""
                            data_frame = pd.DataFrame(df)

                            one_month = float(data_frame["Taux CAD"][0])
                            two_month = float(data_frame["Taux CAD"][1])
                            three_month = float(data_frame["Taux CAD"][2])
                            Six_month = float(data_frame["Taux CAD"][3])
                            One_year = float(data_frame["Taux CAD"][4])

                            # print(one_week)

                            col1.metric("Un mois % ", format(one_month,".4f"))
                            col2.metric("Deux mois % ", format(two_month,".4f"))
                            col3.metric("Trois mois %",format(three_month,".4f") )
                            col4.metric("Six mois %", format(Six_month, ".4f"))
                            col5.metric("Une année %", format(One_year, ".4f"))

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(

                                "<h3 style='text-align: left; color: green;'> Graphique </h3>",
                                unsafe_allow_html=True)

                            df = pd.DataFrame(dict(
                                Maturité=[ '1M', '2M', '3M','6M','1Y'],
                                Taux_CAD=df["Taux CAD"]))

                            fig = px.line(
                                df,  # Data Frame
                                x="Maturité",  # Columns from the data frame
                                y="Taux_CAD"
                            )
                            fig.update_traces(line_color="green")
                            fig.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                margin={"t": 1, "b": 1, "r": 1, "l": 1},
                                plot_bgcolor='rgba(0,0,0,0)',
                                width=500,
                                height=300

                            )
                            st.plotly_chart(fig)

                        X = [ 30, 2 * 30, 3 * 30,6*30,12*30]
                        Y = [one_month, two_month, three_month,Six_month,One_year]
                        with col3:
                            st.markdown(

                                "<h3 style='text-align: left; color: green;'> Maturité à interpoler </h3>",
                                unsafe_allow_html=True)
                            c = float(st.text_input("Saisez la maturité (jours)", value=1))

                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                            col3.markdown(

                                "<h3 style='text-align: left; color: green;'> Résultat </h3>",
                                unsafe_allow_html=True)
                            with col4:
                                col4.markdown(

                                    "<h1 style='text-align: left; color: green;'> </h1>",
                                    unsafe_allow_html=True)
                                col4.markdown(

                                    "<h1 style='text-align: left; color: green;'> </h1>",
                                    unsafe_allow_html=True)
                                col4.markdown(

                                    "<h4 style='text-align: left; color: green;'> </h4>",
                                    unsafe_allow_html=True)

                                if col4.button(f'Calculer '):
                                    col3.write((" Taux d'intèrêt qui correspond à " + " " + str(
                                        format(c, ".0f")) + " " + "jours est :" + " " + str(
                                        format(y_interp(c), ".4f"))))
                            time.sleep(1)
                except:
                       print("An exception occurred")
                break


            #main_task()




        elif option == "GBP":

            s = sched.scheduler(time.time, time.sleep)


            #def main_task():
            placeholder = st.empty()

            for seconds in range(200):
                # while True:
                try:
                    with placeholder.container():

                        workbook = xw.Book("streamlitmodifie1.xlsm")
                        df = workbook.sheets[13].range('A1').options(pd.DataFrame,
                                                                    header=1,
                                                                    index=True,
                                                                    expand='table').value

                        # st.table(df)
                      #  col1, col2, col3 = st.columns(3)
                        # with col1:
                        # st.button("Refresh")

                        #with col3:
                         #   today = date.today()
                         #   v = datetime.now().strftime("%H:%M:%S")

                         #   st.write(today, v)
                        #with col2:
#                            st.markdown( "<h1 style='text-align: center; color: green;'> Livre Sterling </h1>",unsafe_allow_html=True)

                       # st.header("")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            ##########################################################################################"""
                            data_frame = pd.DataFrame(df)

                            One_month = float(data_frame["Taux GBP"][0])
                            three_month = float(data_frame["Taux GBP"][1])
                            six_month = float(data_frame["Taux GBP"][2])
                            # print(one_week)

                            col1.metric("Un mois  %", format(One_month,".4f"))
                            col2.metric("Trois mois %", format(three_month,".4f"))
                            col3.metric("Six mois %", format(six_month,".4f"))

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(

                                "<h3 style='text-align: left; color: green;'> Graphique </h3>",
                                unsafe_allow_html=True)
                            df = pd.DataFrame(dict(
                                Maturité=[ '1M', '3M', '6M'],
                                Taux_GBP=df["Taux GBP"]))

                            fig = px.line(
                                df,  # Data Frame
                                x="Maturité",  # Columns from the data frame
                                y="Taux_GBP"
                            )
                            fig.update_traces(line_color="green")
                            fig.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                margin={"t": 1, "b": 1, "r": 1, "l": 1},
                                plot_bgcolor='rgba(0,0,0,0)',
                                width=500,
                                height=300

                            )
                            st.plotly_chart(fig)

                            # print(one_week)
                        X = [ 30, 3 * 30, 6 * 30]
                        Y = [One_month, three_month, six_month]
                        with col3:
                            st.markdown(

                                "<h3 style='text-align: left; color: green;'> Maturité à interpoler </h3>",
                                unsafe_allow_html=True)
                            c = float(st.text_input("Saisez la maturité (jours)", value=1))

                            # Finding the interpolation
                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                            col3.markdown(

                                "<h3 style='text-align: left; color: green;'> Résultat </h3>",
                                unsafe_allow_html=True)
                            with col4:
                                col4.markdown(

                                    "<h1 style='text-align: left; color: green;'> </h1>",
                                    unsafe_allow_html=True)
                                col4.markdown(

                                    "<h1 style='text-align: left; color: green;'> </h1>",
                                    unsafe_allow_html=True)
                                col4.markdown(

                                    "<h4 style='text-align: left; color: green;'> </h4>",
                                    unsafe_allow_html=True)

                                if col4.button(f'Calculer '):
                                    col3.write((" Taux d'intèrêt qui correspond à " + " " + str(
                                        format(c, ".0f")) + " " + "jours est :" + " " + str(
                                        format(y_interp(c), ".4f"))))
                            time.sleep(1)
                except:
                      print("An exception occurred")
                break


            #main_task()
        elif option == "NOK":
            s = sched.scheduler(time.time, time.sleep)


            #def main_task():
            placeholder = st.empty()

            for seconds in range(200):
                # while True:
                try:
                    with placeholder.container():

                        workbook = xw.Book("streamlitmodifie1.xlsm")
                        df = workbook.sheets[3].range('A1').options(pd.DataFrame,
                                                                    header=1,
                                                                    index=True,
                                                                    expand='table').value

                        col1, col2, col3 = st.columns(3)
                        # with col1:
                        # st.button("Refresh")

                        with col3:
                            today = date.today()
                            v = datetime.now().strftime("%H:%M:%S")

                            st.write(today, v)
                        with col2:
                            st.markdown(

                                "<h1 style='text-align: center; color: green;'> Couronne norvégienne  </h1>",
                                unsafe_allow_html=True)

                        st.header("")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            ##########################################################################################"""
                            data_frame = pd.DataFrame(df)

                            one_week = float(data_frame["Taux NOK"][0])
                            One_month = float(data_frame["Taux NOK"][1])
                            Two_month = float(data_frame["Taux NOK"][2])
                            three_month = float(data_frame["Taux NOK"][3])
                            six_month = float(data_frame["Taux NOK"][4])

                            col1.metric("Une semaine ", one_week, delta_color="off")
                            col2.metric("Un mois  ", One_month)
                            col3.metric("Deux mois ", Two_month)
                            col4.metric("Trois mois", three_month)
                            col5.metric("Six mois", six_month)

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(

                                "<h3 style='text-align: left; color: green;'> Graphique </h3>",
                                unsafe_allow_html=True)
                            df = pd.DataFrame(dict(
                                Maturité=[' 1W', '1M', '2M', '3M', '6M'],
                                Taux_NOK=df["Taux NOK"]))

                            fig = px.line(
                                df,  # Data Frame
                                x="Maturité",  # Columns from the data frame
                                y="Taux_NOK"
                            )
                            fig.update_traces(line_color="green")
                            fig.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                margin={"t": 1, "b": 1, "r": 1, "l": 1},
                                plot_bgcolor='rgba(0,0,0,0)',
                                width=500,
                                height=300

                            )
                            st.plotly_chart(fig)

                        X = [7, 30, 2 * 30, 3 * 30, 6 * 30]
                        Y = [one_week, One_month, Two_month, three_month, six_month]
                        with col3:
                            st.markdown(

                                "<h3 style='text-align: left; color: green;'> Maturité à interpoler </h3>",
                                unsafe_allow_html=True)
                            c = float(st.text_input("Saisez la maturité (jours)", value=1))

                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                            col3.markdown(

                                "<h3 style='text-align: left; color: green;'> Résultat </h3>",
                                unsafe_allow_html=True)
                            with col4:
                                col4.markdown(

                                    "<h1 style='text-align: left; color: green;'> </h1>",
                                    unsafe_allow_html=True)
                                col4.markdown(

                                    "<h1 style='text-align: left; color: green;'> </h1>",
                                    unsafe_allow_html=True)
                                col4.markdown(

                                    "<h4 style='text-align: left; color: green;'> </h4>",
                                    unsafe_allow_html=True)

                                if col4.button(f'Calculer '):
                                    col3.write((" Taux d'intèrêt qui correspond à " + " " + str(
                                        format(c, ".0f")) + " " + "jours est :" + " " + str(
                                        format(y_interp(c), ".6f"))))

                            time.sleep(1)
                except:
                       print("An exception occurred")
                break

                    #main_task()

        elif option == "CHF":
            s = sched.scheduler(time.time, time.sleep)


            #def main_task():
            placeholder = st.empty()

            for seconds in range(200):
                # while True:
                try:
                    with placeholder.container():

                        workbook = xw.Book("streamlitmodifie1.xlsm")
                        df = workbook.sheets[14].range('A1').options(pd.DataFrame,
                                                                    header=1,
                                                                    index=True,
                                                                    expand='table').value

                       # col1, col2, col3 = st.columns(3)
                        # with col1:
                        #  st.button("Refresh")

                       # with col3:
                         #   today = date.today()
                         #   v = datetime.now().strftime("%H:%M:%S")

                          #  st.write(today, v)
                        #with col2:
                         #   st.markdown("<h1 style='text-align: center; color: green;'> Franc Suisse  </h1>",unsafe_allow_html=True)

                        #st.header("")
                        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

                        with col1:
                            ##########################################################################################"""
                            data_frame = pd.DataFrame(df)

                            one_week = float(data_frame["Taux CHF"][0])
                            two_week = float(data_frame["Taux CHF"][1])
                            one_month = float(data_frame["Taux CHF"][2])
                            two_month = float(data_frame["Taux CHF"][3])
                            three_month = float(data_frame["Taux CHF"][4])
                            six_month = float(data_frame["Taux CHF"][5])
                            twelve_month = float(data_frame["Taux CHF"][6])

                            col1.metric("Une semaine % ", format(one_week,".4f"))
                            col2.metric("Deux semaines % ",format(two_week,".4f") )
                            col3.metric("Un mois %", format(one_month,".4f"))
                            col4.metric("Deux mois %", format(two_month,".4f"))
                            col5.metric("Trois mois %", format(three_month,".4f"))
                            col6.metric("Six mois %", format(six_month,".4f"))
                            col7.metric("Un an %", format(twelve_month,".4f"))


                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(

                                "<h3 style='text-align: left; color: green;'> Graphique </h3>",
                                unsafe_allow_html=True)
                            df = pd.DataFrame(dict(
                                Maturité=[ '1W', '2W','1M', '2M', '3M', '6M', "1Y"],
                                Taux_CHF=df["Taux CHF"]))

                            fig = px.line(
                                df,  # Data Frame
                                x="Maturité",  # Columns from the data frame
                                y="Taux_CHF"
                            )
                            fig.update_traces(line_color="green")
                            fig.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                margin={"t": 1, "b": 1, "r": 1, "l": 1},
                                plot_bgcolor='rgba(0,0,0,0)',
                                width=500,
                                height=300

                            )
                            st.plotly_chart(fig, transapancy=False)

                            X = [7,14,30,2 * 30,3 * 30,6 * 30, 12 * 30]
                            Y = [ one_week,two_week,one_month, two_month, three_month, six_month, twelve_month]
                            with col3:
                                st.markdown(

                                    "<h3 style='text-align: left; color: green;'> Maturité à interpoler </h3>",
                                    unsafe_allow_html=True)
                                c = float(st.text_input("Saisez la maturité (jours)", value=1))

                                # test value
                                # interpolate_x = 7 * 30

                                # Finding the interpolation
                                y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                                col3.markdown(

                                    "<h3 style='text-align: left; color: green;'> Résultat </h3>",
                                    unsafe_allow_html=True)
                                with col4:
                                    col4.markdown(

                                        "<h1 style='text-align: left; color: green;'> </h1>",
                                        unsafe_allow_html=True)
                                    col4.markdown(

                                        "<h1 style='text-align: left; color: green;'> </h1>",
                                        unsafe_allow_html=True)
                                    col4.markdown(

                                        "<h4 style='text-align: left; color: green;'> </h4>",
                                        unsafe_allow_html=True)

                                    if col4.button(f'Calculer '):
                                        col3.write((" Taux d'intèrêt qui correspond à " + " " + str(
                                            format(c, ".0f")) + " " + "jours est :" + " " + str(
                                            format(y_interp(c), ".6f"))))
                                time.sleep(1)
                except:
                       print("An exception occurred")
                break

                    #main_task()

        elif option == "JPY":
            s = sched.scheduler(time.time, time.sleep)


            #def main_task():
            placeholder = st.empty()

            for seconds in range(200):
                # while True:
                try:
                    with placeholder.container():

                        workbook = xw.Book("streamlitmodifie1.xlsm")
                        df = workbook.sheets[15].range('A1').options(pd.DataFrame,
                                                                    header=1,
                                                                    index=True,
                                                                    expand='table').value




                        col1, col2, col3, col4, col5, col6,col7 = st.columns(7)

                        with col1:
                            ##########################################################################################"""
                            data_frame = pd.DataFrame(df)
                            one_week = float(data_frame["Taux JPY"][0])
                            two_week = float(data_frame["Taux JPY"][1])
                            one_month = float(data_frame["Taux JPY"][2])
                            two_month = float(data_frame["Taux JPY"][3])
                            three_month = float(data_frame["Taux JPY"][4])
                            six_month = float(data_frame["Taux JPY"][5])
                            one_year = float(data_frame["Taux JPY"][6])

                            # print(one_week)
                        col1.metric(("Un mois %"), format(one_week,".4f"), delta_color="off")
                        col2.metric("Deux mois %", format(two_week,".4f"))
                        col3.metric("Trois mois %",format(one_month,".4f") )
                        col4.metric("Quatres mois %", format(two_month,".4f"))
                        col5.metric("Cinq mois %", format(three_month,".4f"))
                        col6.metric("Six mois %", format(six_month,".4f"))
                        col7.metric("Six mois %",format(one_year,".4f") )


                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(
                                "<h3 style='text-align: left; color: green;'> Graphique  </h3>",
                                unsafe_allow_html=True)
                            df = pd.DataFrame(dict(
                                Maturité=['1W','2W','1M', '2M','3M','6M','1Y'],
                                Taux_JPY=df["Taux JPY"]))

                            fig = px.line(
                                df,  # Data Frame
                                x="Maturité",  # Columns from the data frame
                                y="Taux_JPY"
                            )
                            fig.update_traces(line_color="green")
                            fig.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                margin={"t": 1, "b": 1, "r": 1, "l": 1},
                                plot_bgcolor='rgba(0,0,0,0)',
                                width=500,
                                height=300

                            )
                            st.plotly_chart(fig)

                        col1, col2 = st.columns(2)


                        X = [7,14,30, 2 * 30, 3 * 30,6 * 30,12*30]
                        Y = [one_week,two_week,one_month, two_month, three_month, six_month,one_year]
                        with col3:
                            st.markdown(

                                "<h3 style='text-align: left; color: green;'> Maturité à interpoler </h3>",
                                unsafe_allow_html=True)
                            c = float(st.text_input("Saisez la maturité (jours)", value=1))

                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                            col3.markdown(

                                "<h3 style='text-align: left; color: green;'> Résultat </h3>",
                                unsafe_allow_html=True)
                            with col4:
                                col4.markdown(

                                    "<h1 style='text-align: left; color: green;'> </h1>",
                                    unsafe_allow_html=True)
                                col4.markdown(

                                    "<h1 style='text-align: left; color: green;'> </h1>",
                                    unsafe_allow_html=True)
                                col4.markdown(

                                    "<h4 style='text-align: left; color: green;'> </h4>",
                                    unsafe_allow_html=True)

                                if col4.button(f'Calculer '):
                                    col3.write((" Taux d'intèrêt qui correspond à " + " " + str(
                                        format(c, ".0f")) + " " + "jours est :" + " " + str(
                                        format(y_interp(c), ".6f"))))
                        time.sleep(1)
                except:
                      print("An exception occurred")
                break

                    #main_task()

        elif option == "SEK":
            s = sched.scheduler(time.time, time.sleep)


            #def main_task():
            placeholder = st.empty()

            for seconds in range(200):
                # while True:
                try:
                    with placeholder.container():

                        workbook = xw.Book("streamlitmodifie1.xlsm")
                        df = workbook.sheets[9].range('A1').options(pd.DataFrame,
                                                                    header=1,
                                                                    index=True,
                                                                    expand='table').value

                        col1, col2, col3 = st.columns(3)
                        # with col1:
                        # st.button("Refresh")

                        with col3:
                            today = date.today()
                            v = datetime.now().strftime("%H:%M:%S")

                            st.write(today, v)
                        with col2:
                            st.markdown(

                                "<h1 style='text-align: center; color: green;'> Couronne suédoise  </h1>",
                                unsafe_allow_html=True)

                        st.header("")
                        col1, col2, col3, col4, col5 = st.columns(5)

                        with col1:
                            ##########################################################################################"""
                            data_frame = pd.DataFrame(df)
                            one_week = float(data_frame["Taux SEK"][0])
                            one_month = float(data_frame["Taux SEK"][1])
                            two_month = float(data_frame["Taux SEK"][2])
                            three_month = float(data_frame["Taux SEK"][3])
                            six_month = float(data_frame["Taux SEK"][4])

                            # print(one_week)
                        col1.metric(("Une semaine  "), one_week, delta_color="off")
                        col2.metric("Un mois ", one_month)
                        col3.metric("Deux mois", two_month)
                        col4.metric("Trois mois", three_month)
                        col5.metric("Six mois ", six_month)

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(
                                "<h3 style='text-align: left; color: green;'> Graphe  </h3>",
                                unsafe_allow_html=True)
                            df = pd.DataFrame(dict(
                                Maturité=['1W', '1M', '2M', '3M', '6M'],
                                Taux_SEK=df["Taux SEK"]))

                            fig = px.line(
                                df,  # Data Frame
                                x="Maturité",  # Columns from the data frame
                                y="Taux_SEK"
                            )
                            fig.update_traces(line_color="green")
                            fig.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                margin={"t": 1, "b": 1, "r": 1, "l": 1},
                                plot_bgcolor='rgba(0,0,0,0)',
                                width=500,
                                height=300

                            )
                            st.plotly_chart(fig)

                            # print(one_week)
                        X = [7, 30, 2 * 30, 3 * 30, 6 * 30]
                        Y = [one_week, one_month, two_month, three_month, six_month]
                        with col3:
                            st.markdown(

                                "<h3 style='text-align: left; color: green;'> Maturité à interpoler </h3>",
                                unsafe_allow_html=True)
                            c = float(st.text_input("Saisez la maturité (jours)", value=1))

                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                            col3.markdown(

                                "<h3 style='text-align: left; color: green;'> Résultat </h3>",
                                unsafe_allow_html=True)
                            with col4:
                                col4.markdown(

                                    "<h1 style='text-align: left; color: green;'> </h1>",
                                    unsafe_allow_html=True)
                                col4.markdown(

                                    "<h1 style='text-align: left; color: green;'> </h1>",
                                    unsafe_allow_html=True)
                                col4.markdown(

                                    "<h4 style='text-align: left; color: green;'> </h4>",
                                    unsafe_allow_html=True)

                                if col4.button(f'Calculer '):
                                    col3.write((" Taux d'intèrêt qui correspond à " + " " + str(
                                        format(c, ".0f")) + " " + "jours est :" + " " + str(
                                        format(y_interp(c), ".6f"))))
                            time.sleep(1)
                except:
                       print("An exception occurred")
                break

                    #main_task()


    elif menu_id == "vol":

        paire = st.selectbox(
                    """ Nappe de Volatilité""",
                    ('EUR/USD', 'USD/JPY', 'EUR/CAD', 'USD/CHF', 'EUR/CHF', 'EUR/JPY', 'GBP/USD', 'EUR/GBP', 'USD/CAD'))
        if paire == 'EUR/USD':
            #s = sched.scheduler(time.time, time.sleep)


           # def main_task():
            #placeholder = st.empty()

            #for seconds in range(200):
                # while True:
                #try:
            #with placeholder.container():

            workbook = xw.Book("streamlitmodifie1.xlsm")

            df = workbook.sheets[1].range('A1').options(pd.DataFrame,
                                                         header=1,
                                                         index=True,
                                                         expand='table').value
            df1 = df.style.highlight_null(props="color: transparent;")

            import plotly.graph_objects as go
            # st.header("")
            #st.header("")
            #col1, col2, col3 = st.columns(3)

            #with col2:
                #st.markdown("<h1 style='text-align: center; color: green;'> EUR/USD</h1>",unsafe_allow_html=True)
                #st.header("")

            # with col1:
            # st.button("Refresh")
            #with col3:
               # today = date.today()
               # v = datetime.now().strftime("%H:%M:%S")

               # st.write(today, v)
            #st.header("")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(

                    "<h3 style='text-align: left; color: green;'>  Aperçu   </h3>",
                    unsafe_allow_html=True)
            with col1:

                df1 = df.style.highlight_null(props="color: transparent;")

                data_frame = pd.DataFrame(df)
                over_night = float(data_frame["ATM"][0])
                one_week = float(data_frame["ATM"][1])
                one_month = float(data_frame["ATM"][2])
                two_month = float(data_frame["ATM"][3])
                three_month = float(data_frame["ATM"][4])
                six_month = float(data_frame["ATM"][5])
                one_year = float(data_frame["ATM"][6])
                two_year = float(data_frame["ATM"][7])
                three_year = float(data_frame["ATM"][8])
                five_year = float(data_frame["ATM"][9])
                ten_year = float(data_frame["ATM"][10])
                #############################################################"
                over_night1 = float(data_frame["25 Delta Risk Reversal "][0])
                one_week1 = float(data_frame["25 Delta Risk Reversal "][0])
                one_month1 = float(data_frame["25 Delta Risk Reversal "][1])
                two_month1 = float(data_frame["25 Delta Risk Reversal "][2])
                three_month1 = float(data_frame["25 Delta Risk Reversal "][3])
                six_month1 = float(data_frame["25 Delta Risk Reversal "][4])
                one_year1 = float(data_frame["25 Delta Risk Reversal "][5])
                two_year1 = float(data_frame["25 Delta Risk Reversal "][6])
                three_year1 = float(data_frame["25 Delta Risk Reversal "][7])
                five_year1 = float(data_frame["25 Delta Risk Reversal "][8])
                ten_year1 = float(data_frame["25 Delta Risk Reversal "][10])
                ################################################################""
                over_night2 = float(data_frame["10 Delta Risk Reversal"][0])
                one_week2 = float(data_frame["10 Delta Risk Reversal"][1])
                one_month2 = float(data_frame["10 Delta Risk Reversal"][2])
                two_month2 = float(data_frame["10 Delta Risk Reversal"][3])
                three_month2 = float(data_frame["10 Delta Risk Reversal"][4])
                six_month2 = float(data_frame["10 Delta Risk Reversal"][5])
                one_year2 = float(data_frame["10 Delta Risk Reversal"][6])
                two_year2 = float(data_frame["10 Delta Risk Reversal"][7])
                three_year2 = float(data_frame["10 Delta Risk Reversal"][8])
                five_year2 = float(data_frame["10 Delta Risk Reversal"][9])
                ten_year2 = float(data_frame["10 Delta Risk Reversal"][10])
                ################################################################
                over_night3 = float(data_frame["25 Delta Butterfly"][0])
                one_week3 = float(data_frame["25 Delta Butterfly"][1])
                one_month3 = float(data_frame["25 Delta Butterfly"][2])
                two_month3 = float(data_frame["25 Delta Butterfly"][3])
                three_month3 = float(data_frame["25 Delta Butterfly"][4])
                six_month3 = float(data_frame["25 Delta Butterfly"][5])
                one_year3 = float(data_frame["25 Delta Butterfly"][6])
                two_year3 = float(data_frame["25 Delta Butterfly"][7])
                three_year3 = float(data_frame["25 Delta Butterfly"][8])
                five_year3 = float(data_frame["25 Delta Butterfly"][9])
                ten_year3 = float(data_frame["25 Delta Butterfly"][10])
                ##################################################################
                over_night4 = float(data_frame["10 Delta Butterfly"][0])
                one_week4 = float(data_frame["10 Delta Butterfly"][1])
                one_month4 = float(data_frame["10 Delta Butterfly"][2])
                two_month4 = float(data_frame["10 Delta Butterfly"][3])
                three_month4 = float(data_frame["10 Delta Butterfly"][4])
                six_month4 = float(data_frame["10 Delta Butterfly"][5])
                one_year4 = float(data_frame["10 Delta Butterfly"][6])
                two_year4 = float(data_frame["10 Delta Butterfly"][7])
                three_year4 = float(data_frame["10 Delta Butterfly"][8])
                five_year4 = float(data_frame["10 Delta Butterfly"][9])
                ten_year4 = float(data_frame["10 Delta Butterfly"][10])

                headerColor = 'grey'
                rowEvenColor = 'lightgrey'
                rowOddColor = 'white'
                import plotly.graph_objects as go
                fig = go.Figure(data=[go.Table(
                    header=dict(values=["Volatilité", "ATM", "25D RR", "10D RR", "25D BF", "10D BF"],
                                align=['left', 'center'],
                                line_color='darkslategray',
                                fill_color='#54872E',
                                font=dict(color='black', size=17)),

                    cells=dict(values=[
                        ['Pendant la nuit', 'Une semaine', 'Un mois', 'Deux mois', 'Trois Mois', 'Six mois', 'Un an', 'Deux ans ',
                         'Trois ans', 'Cinq ans', 'Dix ans'],

                        [format(over_night, ".3f"),format(one_week, ".3f"),format(one_month, ".3f"),format(two_month, ".3f"),format(three_month, ".3f"),format(six_month, ".3f"),format(one_year, ".3f"),format(two_year, ".3f"),format(three_year, ".3f"),format(five_year, ".3f"),format(ten_year, ".3f")],
                        [format(over_night1, ".3f"),format(one_week1, ".3f"),format(one_month1, ".3f"),format(two_month1, ".3f"),format(three_month1, ".3f"),format(six_month1, ".3f"),format(one_year1, ".3f"),format(two_year1, ".3f"),format(three_year1, ".3f"),format(five_year1, ".3f"),format(ten_year1, ".3f")],
                        [format(over_night2, ".3f"),format(one_week2, ".3f"),format(one_month2, ".3f"),format(two_month2, ".3f"),format(three_month2, ".3f"),format(six_month2, ".3f"),format(one_year2, ".3f"),format(two_year2, ".3f"),format(three_year2, ".3f"),format(five_year2, ".3f"),format(ten_year2, ".3f")],
                        [format(over_night3, ".3f"),format(one_week3, ".3f"),format(one_month3, ".3f"),format(two_month3, ".3f"),format(three_month3, ".3f"),format(six_month3, ".3f"),format(one_year3, ".3f"),format(two_year3, ".3f"),format(three_year3, ".3f"),format(five_year3, ".3f"),format(ten_year3, ".3f")],
                        [format(over_night4, ".3f"),format(one_week4, ".3f"),format(one_month4, ".3f"),format(two_month4, ".3f"),format(three_month4, ".3f"),format(six_month4, ".3f"),format(one_year4, ".3f"),format(two_year4, ".3f"),format(three_year4, ".3f"),format(five_year4, ".3f"),format(ten_year4, ".3f")]],

                        line_color='white',
                        # 2-D list of colors for alternating rows
                        fill_color='white',
                        align=['left', 'center'],

                        font=dict(color='darkslategray', size=11)

                    ))

                ])
                fig.update_layout(margin=dict(l=1, r=40, b=1, t=1))

                # fig2=fig.replace(np.nan, '', regex=True)
                col1.write(fig)

            #  col1.table(df1)
            #  col1.write( """la volatilité ATM est la valeur de volatilité qui rend le prix implicite d'une option vanille ATM égal au prix du marché de cette option.""")
            with col3:
                st.markdown(

                    "<h3 style='text-align: left; color: green;'>  Maturité à interpoler   </h3>",
                    unsafe_allow_html=True)
                # st.header("")
            with col3:
                volatility = st.selectbox(
                    """Choisissez le type de volatilité """,
                    ('ATM', '25 Delta Risk Reversal', '10 Delta Risk Reversal', '25 Delta Butterfly', '10 Delta Butterfly'))

            with col4:
                st.markdown(

                    "<h1 style='text-align: center; color: green;'>    </h1>",
                    unsafe_allow_html=True)

            st.header("")
            # col4.header("")
            # col4.header("")

            with col4:
                st.markdown(

                    "<h3 style='text-align: center; color: green;'>    </h3>",
                    unsafe_allow_html=True)

            with col4:
                c = float(st.text_input('Saisez la maturité (jours)', value=0))
                if volatility == "ATM":
                    col3.markdown(
                        "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                        unsafe_allow_html=True)
                    data_frame = pd.DataFrame(df)
                    df.dropna(inplace=True)
                    over_night=float(data_frame["ATM"][0])
                    one_week = float(data_frame["ATM"][1])
                    one_month = float(data_frame["ATM"][2])
                    two_month = float(data_frame["ATM"][3])
                    three_month = float(data_frame["ATM"][4])
                    six_month = float(data_frame["ATM"][5])
                    one_year = float(data_frame["ATM"][6])
                    two_year = float(data_frame["ATM"][7])
                    three_year = float(data_frame["ATM"][8])
                    five_year = float(data_frame["ATM"][9])
                    ten_year = float(data_frame["ATM"][10])

                    # col4.metric(("Une semaine  "), one_year, delta_color="off")
                    X = [1,7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                         10 * 12 * 30]
                    Y = [over_night,one_week, one_month, two_month, three_month, six_month, one_year, two_year, three_year,
                         five_year, ten_year]
                    y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')

                    col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                        volatility) + " " + "qui correspond à " + " " + str(
                        format(c, ".0f")) + " " + "jours est :" + " " + str(format(y_interp(c), ".3f"))))


                elif volatility == "25 Delta Risk Reversal":
                    col3.markdown(
                        "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                        unsafe_allow_html=True)
                    data_frame = pd.DataFrame(df)
                    over_night = float(data_frame["25 Delta Risk Reversal "][0])
                    one_week = float(data_frame["25 Delta Risk Reversal "][1])
                    one_month = float(data_frame["25 Delta Risk Reversal "][2])
                    two_month = float(data_frame["25 Delta Risk Reversal "][3])
                    three_month = float(data_frame["25 Delta Risk Reversal "][4])
                    six_month = float(data_frame["25 Delta Risk Reversal "][5])
                    one_year = float(data_frame["25 Delta Risk Reversal "][6])
                    two_year = float(data_frame["25 Delta Risk Reversal "][7])
                    three_year = float(data_frame["25 Delta Risk Reversal "][8])
                    five_year = float(data_frame["25 Delta Risk Reversal "][9])
                    ten_year = float(data_frame["25 Delta Risk Reversal "][10])
                    X = [1,7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                         10 * 12 * 30]
                    Y = [over_night,one_week, one_month, two_month, three_month, six_month, one_year, two_year, three_year,
                         five_year, ten_year]
                    y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                    col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                        volatility) + " " + "qui correspond à " + " " + str(
                        format(c, ".0f")) + " " + "jours est :" + " " + str(
                        format(y_interp(c), ".3f"))))
                elif volatility == "10 Delta Risk Reversal":
                    col3.markdown(
                        "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                        unsafe_allow_html=True)
                    data_frame = pd.DataFrame(df)
                    over_night = float(data_frame["10 Delta Risk Reversal"][0])
                    one_week = float(data_frame["10 Delta Risk Reversal"][1])
                    one_month = float(data_frame["10 Delta Risk Reversal"][2])
                    two_month = float(data_frame["10 Delta Risk Reversal"][3])
                    three_month = float(data_frame["10 Delta Risk Reversal"][4])
                    six_month = float(data_frame["10 Delta Risk Reversal"][5])
                    one_year = float(data_frame["10 Delta Risk Reversal"][6])
                    two_year = float(data_frame["10 Delta Risk Reversal"][7])
                    three_year = float(data_frame["10 Delta Risk Reversal"][8])
                    five_year = float(data_frame["10 Delta Risk Reversal"][9])
                    ten_year = float(data_frame["10 Delta Risk Reversal"][10])
                    X = [1,7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                         10 * 12 * 30]
                    Y = [over_night,one_week, one_month, two_month, three_month, six_month, one_year, two_year, three_year,
                         five_year, ten_year]
                    y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                    col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                        volatility) + " " + "qui correspond à " + " " + str(
                        format(c, ".0f")) + " " + "jours est :" + " " + str(format(y_interp(c), ".3f"))))
                elif volatility == "25 Delta Butterfly":
                    col3.markdown(
                        "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                        unsafe_allow_html=True)
                    data_frame = pd.DataFrame(df)
                    over_night = float(data_frame["25 Delta Butterfly"][0])
                    one_week = float(data_frame["25 Delta Butterfly"][1])
                    one_month = float(data_frame["25 Delta Butterfly"][2])
                    two_month = float(data_frame["25 Delta Butterfly"][3])
                    three_month = float(data_frame["25 Delta Butterfly"][4])
                    six_month = float(data_frame["25 Delta Butterfly"][5])
                    one_year = float(data_frame["25 Delta Butterfly"][6])
                    two_year = float(data_frame["25 Delta Butterfly"][7])
                    three_year = float(data_frame["25 Delta Butterfly"][8])
                    five_year = float(data_frame["25 Delta Butterfly"][9])
                    ten_year = float(data_frame["25 Delta Butterfly"][10])
                    X = [1,7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                         10 * 12 * 30]
                    Y = [over_night,one_week, one_month, two_month, three_month, six_month, one_year, two_year, three_year,
                         five_year, ten_year]
                    y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                    col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                        volatility) + " " + "qui correspond à " + " " + str(
                        format(c, ".0f")) + " " + "jours est :" + " " + str(
                        format(y_interp(c), ".3f"))))
                elif volatility == "10 Delta Butterfly":
                    col3.markdown(
                        "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                        unsafe_allow_html=True)
                    data_frame = pd.DataFrame(df)
                    over_night = float(data_frame["10 Delta Butterfly"][0])
                    one_week = float(data_frame["10 Delta Butterfly"][1])
                    one_month = float(data_frame["10 Delta Butterfly"][2])
                    two_month = float(data_frame["10 Delta Butterfly"][3])
                    three_month = float(data_frame["10 Delta Butterfly"][4])
                    six_month = float(data_frame["10 Delta Butterfly"][5])
                    one_year = float(data_frame["10 Delta Butterfly"][6])
                    two_year = float(data_frame["10 Delta Butterfly"][7])
                    three_year = float(data_frame["10 Delta Butterfly"][8])
                    five_year = float(data_frame["10 Delta Butterfly"][9])
                    ten_year = float(data_frame["10 Delta Butterfly"][10])
                    X = [1,7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                         10 * 12 * 30]
                    Y = [over_night,one_week, one_month, two_month, three_month, six_month, one_year, two_year, three_year,
                         five_year, ten_year]
                    y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                    col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                        volatility) + " " + "qui correspond à " + " " + str(
                        format(c, ".0f")) + " " + "jours est :" + " " + str(
                        format(y_interp(c), ".3f"))))

                            #time.sleep(1)
            #except:
               #    print("An exception occurred")
            #break


        #main_task()
        elif paire == 'EUR/JPY':
            #s = sched.scheduler(time.time, time.sleep)


           # def main_task():
            #placeholder = st.empty()

            #for seconds in range(200):
                # while True:
                #try:
            #with placeholder.container():

            workbook = xw.Book("streamlitmodifie1.xlsm")

            df = workbook.sheets[2].range('A1').options(pd.DataFrame,
                                                         header=1,
                                                         index=True,
                                                         expand='table').value
            df1 = df.style.highlight_null(props="color: transparent;")

            import plotly.graph_objects as go
            # st.header("")

            #col1, col2, col3 = st.columns(3)

           # with col2:
              #  st.markdown( "<h1 style='text-align: center; color: green;'> EUR/JPY</h1>",unsafe_allow_html=True)

            # with col1:
            # st.button("Refresh")
            #with col3:
               # today = date.today()
               # v = datetime.now().strftime("%H:%M:%S")

              #  st.write(today, v)
            #st.header("")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(

                    "<h3 style='text-align: left; color: green;'>  Aperçu   </h3>",
                    unsafe_allow_html=True)
            with col1:

                df1 = df.style.highlight_null(props="color: transparent;")

                data_frame = pd.DataFrame(df)
                over_night = float(data_frame["ATM"][0])
                one_week = float(data_frame["ATM"][1])
                one_month = float(data_frame["ATM"][2])
                two_month = float(data_frame["ATM"][3])
                three_month = float(data_frame["ATM"][4])
                six_month = float(data_frame["ATM"][5])
                one_year = float(data_frame["ATM"][6])
                two_year = float(data_frame["ATM"][7])
                three_year = float(data_frame["ATM"][8])
                five_year = float(data_frame["ATM"][9])
                ten_year = float(data_frame["ATM"][10])
                #############################################################"
                over_night1 = float(data_frame["25 Delta Risk Reversal "][0])
                one_week1 = float(data_frame["25 Delta Risk Reversal "][0])
                one_month1 = float(data_frame["25 Delta Risk Reversal "][1])
                two_month1 = float(data_frame["25 Delta Risk Reversal "][2])
                three_month1 = float(data_frame["25 Delta Risk Reversal "][3])
                six_month1 = float(data_frame["25 Delta Risk Reversal "][4])
                one_year1 = float(data_frame["25 Delta Risk Reversal "][5])
                two_year1 = float(data_frame["25 Delta Risk Reversal "][6])
                three_year1 = float(data_frame["25 Delta Risk Reversal "][7])
                five_year1 = float(data_frame["25 Delta Risk Reversal "][8])
                ten_year1 = float(data_frame["25 Delta Risk Reversal "][10])
                ################################################################""
                over_night2 = float(data_frame["10 Delta Risk Reversal"][0])
                one_week2 = float(data_frame["10 Delta Risk Reversal"][1])
                one_month2 = float(data_frame["10 Delta Risk Reversal"][2])
                two_month2 = float(data_frame["10 Delta Risk Reversal"][3])
                three_month2 = float(data_frame["10 Delta Risk Reversal"][4])
                six_month2 = float(data_frame["10 Delta Risk Reversal"][5])
                one_year2 = float(data_frame["10 Delta Risk Reversal"][6])
                two_year2 = float(data_frame["10 Delta Risk Reversal"][7])
                three_year2 = float(data_frame["10 Delta Risk Reversal"][8])
                five_year2 = float(data_frame["10 Delta Risk Reversal"][9])
                ten_year2 = float(data_frame["10 Delta Risk Reversal"][10])
                ################################################################
                over_night3 = float(data_frame["25 Delta Butterfly"][0])
                one_week3 = float(data_frame["25 Delta Butterfly"][1])
                one_month3 = float(data_frame["25 Delta Butterfly"][2])
                two_month3 = float(data_frame["25 Delta Butterfly"][3])
                three_month3 = float(data_frame["25 Delta Butterfly"][4])
                six_month3 = float(data_frame["25 Delta Butterfly"][5])
                one_year3 = float(data_frame["25 Delta Butterfly"][6])
                two_year3 = float(data_frame["25 Delta Butterfly"][7])
                three_year3 = float(data_frame["25 Delta Butterfly"][8])
                five_year3 = float(data_frame["25 Delta Butterfly"][9])
                ten_year3 = float(data_frame["25 Delta Butterfly"][10])
                ##################################################################
                over_night4 = float(data_frame["10 Delta Butterfly"][0])
                one_week4 = float(data_frame["10 Delta Butterfly"][1])
                one_month4 = float(data_frame["10 Delta Butterfly"][2])
                two_month4 = float(data_frame["10 Delta Butterfly"][3])
                three_month4 = float(data_frame["10 Delta Butterfly"][4])
                six_month4 = float(data_frame["10 Delta Butterfly"][5])
                one_year4 = float(data_frame["10 Delta Butterfly"][6])
                two_year4 = float(data_frame["10 Delta Butterfly"][7])
                three_year4 = float(data_frame["10 Delta Butterfly"][8])
                five_year4 = float(data_frame["10 Delta Butterfly"][9])
                ten_year4 = float(data_frame["10 Delta Butterfly"][10])


                headerColor = 'grey'
                rowEvenColor = 'lightgrey'
                rowOddColor = 'white'
                import plotly.graph_objects as go
                fig = go.Figure(data=[go.Table(
                    header=dict(values=["Volatilité", "ATM", "25D RR", "10D RR", "25D BF", "10D BF"],
                                align=['left', 'center'],
                                line_color='darkslategray',
                                fill_color='#54872E',
                                font=dict(color='black', size=17)),

                    cells=dict(values=[
                        ['Pendant la nuit', 'Une semaine', 'Un mois', 'Deux mois', 'Trois Mois', 'Six mois', 'Un an', 'Deux ans ',
                         'Trois ans', 'Cinq ans', 'Dix ans'],

                        [format(over_night, ".3f"),format(one_week, ".3f"),format(one_month, ".3f"),format(two_month, ".3f"),format(three_month, ".3f"),format(six_month, ".3f"),format(one_year, ".3f"),format(two_year, ".3f"),format(three_year, ".3f"),format(five_year, ".3f"),format(ten_year, ".3f")],
                        [format(over_night1, ".3f"),format(one_week1, ".3f"),format(one_month1, ".3f"),format(two_month1, ".3f"),format(three_month1, ".3f"),format(six_month1, ".3f"),format(one_year1, ".3f"),format(two_year1, ".3f"),format(three_year1, ".3f"),format(five_year1, ".3f"),format(ten_year1, ".3f")],
                        [format(over_night2, ".3f"),format(one_week2, ".3f"),format(one_month2, ".3f"),format(two_month2, ".3f"),format(three_month2, ".3f"),format(six_month2, ".3f"),format(one_year2, ".3f"),format(two_year2, ".3f"),format(three_year2, ".3f"),format(five_year2, ".3f"),format(ten_year2, ".3f")],
                        [format(over_night3, ".3f"),format(one_week3, ".3f"),format(one_month3, ".3f"),format(two_month3, ".3f"),format(three_month3, ".3f"),format(six_month3, ".3f"),format(one_year3, ".3f"),format(two_year3, ".3f"),format(three_year3, ".3f"),format(five_year3, ".3f"),format(ten_year3, ".3f")],
                        [format(over_night4, ".3f"),format(one_week4, ".3f"),format(one_month4, ".3f"),format(two_month4, ".3f"),format(three_month4, ".3f"),format(six_month4, ".3f"),format(one_year4, ".3f"),format(two_year4, ".3f"),format(three_year4, ".3f"),format(five_year4, ".3f"),format(ten_year4, ".3f")]],

                        line_color='white',
                        # 2-D list of colors for alternating rows
                        fill_color='white',
                        align=['left', 'center'],
                        font=dict(color='darkslategray', size=13)

                    ))

                ])
                fig.update_layout(margin=dict(l=1, r=40, b=1, t=1))

                # fig2=fig.replace(np.nan, '', regex=True)
                col1.write(fig)

            #  col1.table(df1)
            #  col1.write( """la volatilité ATM est la valeur de volatilité qui rend le prix implicite d'une option vanille ATM égal au prix du marché de cette option.""")
            with col3:
                st.markdown(

                    "<h3 style='text-align: left; color: green;'>  Maturité à interpoler   </h3>",
                    unsafe_allow_html=True)
                # st.header("")
            with col3:
                volatility = st.selectbox(
                    """Choisissez le type de volatilité """,
                    ('ATM', '25 Delta Risk Reversal', '10 Delta Risk Reversal', '25 Delta Butterfly', '10 Delta Butterfly'))

            with col4:
                st.markdown(

                    "<h1 style='text-align: center; color: green;'>    </h1>",
                    unsafe_allow_html=True)

            st.header("")
            # col4.header("")
            # col4.header("")

            with col4:
                st.markdown(

                    "<h3 style='text-align: center; color: green;'>    </h3>",
                    unsafe_allow_html=True)

            with col4:
                c = float(st.text_input('Saisez la maturité (jours)', value=0))
                if volatility == "ATM":
                    col3.markdown(
                        "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                        unsafe_allow_html=True)
                    data_frame = pd.DataFrame(df)
                    df.dropna(inplace=True)
                    over_night=float(data_frame["ATM"][0])
                    one_week = float(data_frame["ATM"][1])
                    one_month = float(data_frame["ATM"][2])
                    two_month = float(data_frame["ATM"][3])
                    three_month = float(data_frame["ATM"][4])
                    six_month = float(data_frame["ATM"][5])
                    one_year = float(data_frame["ATM"][6])
                    two_year = float(data_frame["ATM"][7])
                    three_year = float(data_frame["ATM"][8])
                    five_year = float(data_frame["ATM"][9])
                    ten_year = float(data_frame["ATM"][10])

                    # col4.metric(("Une semaine  "), one_year, delta_color="off")
                    X = [1,7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                         10 * 12 * 30]
                    Y = [over_night,one_week, one_month, two_month, three_month, six_month, one_year, two_year, three_year,
                         five_year, ten_year]
                    y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')

                    col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                        volatility) + " " + "qui correspond à " + " " + str(
                        format(c, ".0f")) + " " + "jours est :" + " " + str(format(y_interp(c), ".3f"))))


                elif volatility == "25 Delta Risk Reversal":
                    col3.markdown(
                        "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                        unsafe_allow_html=True)
                    data_frame = pd.DataFrame(df)
                    over_night = float(data_frame["25 Delta Risk Reversal "][0])
                    one_week = float(data_frame["25 Delta Risk Reversal "][1])
                    one_month = float(data_frame["25 Delta Risk Reversal "][2])
                    two_month = float(data_frame["25 Delta Risk Reversal "][3])
                    three_month = float(data_frame["25 Delta Risk Reversal "][4])
                    six_month = float(data_frame["25 Delta Risk Reversal "][5])
                    one_year = float(data_frame["25 Delta Risk Reversal "][6])
                    two_year = float(data_frame["25 Delta Risk Reversal "][7])
                    three_year = float(data_frame["25 Delta Risk Reversal "][8])
                    five_year = float(data_frame["25 Delta Risk Reversal "][9])
                    ten_year = float(data_frame["25 Delta Risk Reversal "][10])
                    X = [1,7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                         10 * 12 * 30]
                    Y = [over_night,one_week, one_month, two_month, three_month, six_month, one_year, two_year, three_year,
                         five_year, ten_year]
                    y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                    col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                        volatility) + " " + "qui correspond à " + " " + str(
                        format(c, ".0f")) + " " + "jours est :" + " " + str(
                        format(y_interp(c), ".3f"))))
                elif volatility == "10 Delta Risk Reversal":
                    col3.markdown(
                        "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                        unsafe_allow_html=True)
                    data_frame = pd.DataFrame(df)
                    over_night = float(data_frame["10 Delta Risk Reversal"][0])
                    one_week = float(data_frame["10 Delta Risk Reversal"][1])
                    one_month = float(data_frame["10 Delta Risk Reversal"][2])
                    two_month = float(data_frame["10 Delta Risk Reversal"][3])
                    three_month = float(data_frame["10 Delta Risk Reversal"][4])
                    six_month = float(data_frame["10 Delta Risk Reversal"][5])
                    one_year = float(data_frame["10 Delta Risk Reversal"][6])
                    two_year = float(data_frame["10 Delta Risk Reversal"][7])
                    three_year = float(data_frame["10 Delta Risk Reversal"][8])
                    five_year = float(data_frame["10 Delta Risk Reversal"][9])
                    ten_year = float(data_frame["10 Delta Risk Reversal"][10])
                    X = [1,7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                         10 * 12 * 30]
                    Y = [over_night,one_week, one_month, two_month, three_month, six_month, one_year, two_year, three_year,
                         five_year, ten_year]
                    y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                    col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                        volatility) + " " + "qui correspond à " + " " + str(
                        format(c, ".0f")) + " " + "jours est :" + " " + str(format(y_interp(c), ".3f"))))
                elif volatility == "25 Delta Butterfly":
                    col3.markdown(
                        "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                        unsafe_allow_html=True)
                    data_frame = pd.DataFrame(df)
                    over_night = float(data_frame["25 Delta Butterfly"][0])
                    one_week = float(data_frame["25 Delta Butterfly"][1])
                    one_month = float(data_frame["25 Delta Butterfly"][2])
                    two_month = float(data_frame["25 Delta Butterfly"][3])
                    three_month = float(data_frame["25 Delta Butterfly"][4])
                    six_month = float(data_frame["25 Delta Butterfly"][5])
                    one_year = float(data_frame["25 Delta Butterfly"][6])
                    two_year = float(data_frame["25 Delta Butterfly"][7])
                    three_year = float(data_frame["25 Delta Butterfly"][8])
                    five_year = float(data_frame["25 Delta Butterfly"][9])
                    ten_year = float(data_frame["25 Delta Butterfly"][10])
                    X = [1,7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                         10 * 12 * 30]
                    Y = [over_night,one_week, one_month, two_month, three_month, six_month, one_year, two_year, three_year,
                         five_year, ten_year]
                    y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                    col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                        volatility) + " " + "qui correspond à " + " " + str(
                        format(c, ".0f")) + " " + "jours est :" + " " + str(
                        format(y_interp(c), ".3f"))))
                elif volatility == "10 Delta Butterfly":
                    col3.markdown(
                        "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                        unsafe_allow_html=True)
                    data_frame = pd.DataFrame(df)
                    over_night = float(data_frame["10 Delta Butterfly"][0])
                    one_week = float(data_frame["10 Delta Butterfly"][1])
                    one_month = float(data_frame["10 Delta Butterfly"][2])
                    two_month = float(data_frame["10 Delta Butterfly"][3])
                    three_month = float(data_frame["10 Delta Butterfly"][4])
                    six_month = float(data_frame["10 Delta Butterfly"][5])
                    one_year = float(data_frame["10 Delta Butterfly"][6])
                    two_year = float(data_frame["10 Delta Butterfly"][7])
                    three_year = float(data_frame["10 Delta Butterfly"][8])
                    five_year = float(data_frame["10 Delta Butterfly"][9])
                    ten_year = float(data_frame["10 Delta Butterfly"][10])
                    X = [1,7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                         10 * 12 * 30]
                    Y = [over_night,one_week, one_month, two_month, three_month, six_month, one_year, two_year, three_year,
                         five_year, ten_year]
                    y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                    col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                        volatility) + " " + "qui correspond à " + " " + str(
                        format(c, ".0f")) + " " + "jours est :" + " " + str(
                        format(y_interp(c), ".3f"))))

                            #time.sleep(1)
            #except:
               #    print("An exception occurred")
            #break


        #main_task()
        elif paire == 'EUR/CAD':
            #s = sched.scheduler(time.time, time.sleep)


           # def main_task():
            #placeholder = st.empty()

            #for seconds in range(200):
                # while True:
                #try:
            #with placeholder.container():

            workbook = xw.Book("streamlitmodifie1.xlsm")

            df = workbook.sheets[3].range('A1').options(pd.DataFrame,
                                                         header=1,
                                                         index=True,
                                                         expand='table').value
            df1 = df.style.highlight_null(props="color: transparent;")

            import plotly.graph_objects as go
            # st.header("")
            #st.header("")
           # col1, col2, col3 = st.columns(3)

           # with col2:
               # st.markdown( "<h1 style='text-align: center; color: green;'> EUR/CAD</h1>",  unsafe_allow_html=True)
               # st.header("")

            # with col1:
            # st.button("Refresh")
            #with col3:
               # today = date.today()
               # v = datetime.now().strftime("%H:%M:%S")

                #st.write(today, v)
            #st.header("")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(

                    "<h3 style='text-align: left; color: green;'>  Aperçu   </h3>",
                    unsafe_allow_html=True)
            with col1:

                df1 = df.style.highlight_null(props="color: transparent;")

                data_frame = pd.DataFrame(df)
                over_night = float(data_frame["ATM"][0])
                one_week = float(data_frame["ATM"][1])
                one_month = float(data_frame["ATM"][2])
                two_month = float(data_frame["ATM"][3])
                three_month = float(data_frame["ATM"][4])
                six_month = float(data_frame["ATM"][5])
                one_year = float(data_frame["ATM"][6])
                two_year = float(data_frame["ATM"][7])
                three_year = float(data_frame["ATM"][8])
                five_year = float(data_frame["ATM"][9])

                #############################################################"
                over_night1 = float(data_frame["25 Delta Risk Reversal "][0])
                one_week1 = float(data_frame["25 Delta Risk Reversal "][0])
                one_month1 = float(data_frame["25 Delta Risk Reversal "][1])
                two_month1 = float(data_frame["25 Delta Risk Reversal "][2])
                three_month1 = float(data_frame["25 Delta Risk Reversal "][3])
                six_month1 = float(data_frame["25 Delta Risk Reversal "][4])
                one_year1 = float(data_frame["25 Delta Risk Reversal "][5])
                two_year1 = float(data_frame["25 Delta Risk Reversal "][6])
                three_year1 = float(data_frame["25 Delta Risk Reversal "][7])
                five_year1 = float(data_frame["25 Delta Risk Reversal "][8])

                ################################################################""
                over_night2 = float(data_frame["10 Delta Risk Reversal"][0])
                one_week2 = float(data_frame["10 Delta Risk Reversal"][1])
                one_month2 = float(data_frame["10 Delta Risk Reversal"][2])
                two_month2 = float(data_frame["10 Delta Risk Reversal"][3])
                three_month2 = float(data_frame["10 Delta Risk Reversal"][4])
                six_month2 = float(data_frame["10 Delta Risk Reversal"][5])
                one_year2 = float(data_frame["10 Delta Risk Reversal"][6])
                two_year2 = float(data_frame["10 Delta Risk Reversal"][7])
                three_year2 = float(data_frame["10 Delta Risk Reversal"][8])
                five_year2 = float(data_frame["10 Delta Risk Reversal"][9])

                ################################################################
                over_night3 = float(data_frame["25 Delta Butterfly"][0])
                one_week3 = float(data_frame["25 Delta Butterfly"][1])
                one_month3 = float(data_frame["25 Delta Butterfly"][2])
                two_month3 = float(data_frame["25 Delta Butterfly"][3])
                three_month3 = float(data_frame["25 Delta Butterfly"][4])
                six_month3 = float(data_frame["25 Delta Butterfly"][5])
                one_year3 = float(data_frame["25 Delta Butterfly"][6])
                two_year3 = float(data_frame["25 Delta Butterfly"][7])
                three_year3 = float(data_frame["25 Delta Butterfly"][8])
                five_year3 = float(data_frame["25 Delta Butterfly"][9])

                ##################################################################
                over_night4 = float(data_frame["10 Delta Butterfly"][0])
                one_week4 = float(data_frame["10 Delta Butterfly"][1])
                one_month4 = float(data_frame["10 Delta Butterfly"][2])
                two_month4 = float(data_frame["10 Delta Butterfly"][3])
                three_month4 = float(data_frame["10 Delta Butterfly"][4])
                six_month4 = float(data_frame["10 Delta Butterfly"][5])
                one_year4 = float(data_frame["10 Delta Butterfly"][6])
                two_year4 = float(data_frame["10 Delta Butterfly"][7])
                three_year4 = float(data_frame["10 Delta Butterfly"][8])
                five_year4 = float(data_frame["10 Delta Butterfly"][9])


                # df1.style.format('{:.2f}')
                # st.table(df1.format('{:.2f}'))
                headerColor = 'grey'
                rowEvenColor = 'lightgrey'
                rowOddColor = 'white'
                import plotly.graph_objects as go

                placeholder = st.empty()
                encour = False
                for seconds in range(10000000):
                    # seconds=0
                    # while seconds==True:
                    if encour == False:
                        encour = True
                        with placeholder.container():
                            try:

                                    fig = go.Figure(data=[go.Table(
                                        header=dict(values=["Volatilité", "ATM", "25D RR", "10D RR", "25D BF", "10D BF"],
                                                    align=['left', 'center'],
                                                    line_color='darkslategray',
                                                    fill_color='#54872E',
                                                    font=dict(color='black', size=17)),

                                        cells=dict(values=[
                                            ['Pendant la nuit', 'Une semaine', 'Un mois', 'Deux mois', 'Trois Mois', 'Six mois', 'Un an', 'Deux ans ',
                                             'Trois ans', 'Cinq ans'],

                                            [format(over_night, ".3f"),format(one_week, ".3f"),format(one_month, ".3f"),format(two_month, ".3f"),format(three_month, ".3f"),format(six_month, ".3f"),format(one_year, ".3f"),format(two_year, ".3f"),format(three_year, ".3f"),format(five_year, ".3f")],
                                            [format(over_night1, ".3f"),format(one_week1, ".3f"),format(one_month1, ".3f"),format(two_month1, ".3f"),format(three_month1, ".3f"),format(six_month1, ".3f"),format(one_year1, ".3f"),format(two_year1, ".3f"),format(three_year1, ".3f"),format(five_year1, ".3f")],
                                            [format(over_night2, ".3f"),format(one_week2, ".3f"),format(one_month2, ".3f"),format(two_month2, ".3f"),format(three_month2, ".3f"),format(six_month2, ".3f"),format(one_year2, ".3f"),format(two_year2, ".3f"),format(three_year2, ".3f"),format(five_year2, ".3f")],
                                            [format(over_night3, ".3f"),format(one_week3, ".3f"),format(one_month3, ".3f"),format(two_month3, ".3f"),format(three_month3, ".3f"),format(six_month3, ".3f"),format(one_year3, ".3f"),format(two_year3, ".3f"),format(three_year3, ".3f"),format(five_year3, ".3f")],
                                            [format(over_night4, ".3f"),format(one_week4, ".3f"),format(one_month4, ".3f"),format(two_month4, ".3f"),format(three_month4, ".3f"),format(six_month4, ".3f"),format(one_year4, ".3f"),format(two_year4, ".3f"),format(three_year4, ".3f"),format(five_year4, ".3f")]],

                                            line_color='white',
                                            # 2-D list of colors for alternating rows
                                            fill_color='white',
                                            align=['left', 'center'],
                                            font=dict(color='darkslategray', size=13)

                                        ))

                                    ])
                                    fig.update_layout(margin=dict(l=1, r=40, b=1, t=1))

                                    # fig2=fig.replace(np.nan, '', regex=True)
                                    col1.write(fig)
                                    time.sleep(0.001)

                            except:
                                print("ff")
                                break
            #  col1.table(df1)
            #  col1.write( """la volatilité ATM est la valeur de volatilité qui rend le prix implicite d'une option vanille ATM égal au prix du marché de cette option.""")
            with col3:
                st.markdown(

                    "<h3 style='text-align: left; color: green;'>  Maturité à interpoler   </h3>",
                    unsafe_allow_html=True)
                # st.header("")
            with col3:
                volatility = st.selectbox(
                    """Choisissez la paire de devise """,
                    ('ATM', '25 Delta Risk Reversal', '10 Delta Risk Reversal', '25 Delta Butterfly', '10 Delta Butterfly'))

            with col4:
                st.markdown(

                    "<h1 style='text-align: center; color: green;'>    </h1>",
                    unsafe_allow_html=True)

            st.header("")
            # col4.header("")
            # col4.header("")

            with col4:
                st.markdown(

                    "<h3 style='text-align: center; color: green;'>    </h3>",
                    unsafe_allow_html=True)

            with col4:
                c = float(st.text_input('Saisez la maturité (jours)', value=0))
                if volatility == "ATM":
                    col3.markdown(
                        "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                        unsafe_allow_html=True)
                    data_frame = pd.DataFrame(df)
                    df.dropna(inplace=True)
                    over_night=float(data_frame["ATM"][0])
                    one_week = float(data_frame["ATM"][1])
                    one_month = float(data_frame["ATM"][2])
                    two_month = float(data_frame["ATM"][3])
                    three_month = float(data_frame["ATM"][4])
                    six_month = float(data_frame["ATM"][5])
                    one_year = float(data_frame["ATM"][6])
                    two_year = float(data_frame["ATM"][7])
                    three_year = float(data_frame["ATM"][8])
                    five_year = float(data_frame["ATM"][9])


                    # col4.metric(("Une semaine  "), one_year, delta_color="off")
                    X = [1,7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30
                         ]
                    Y = [over_night,one_week, one_month, two_month, three_month, six_month, one_year, two_year, three_year,
                         five_year]
                    y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')

                    col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                        volatility) + " " + "qui correspond à " + " " + str(
                        format(c, ".0f")) + " " + "jours est :" + " " + str(format(y_interp(c), ".3f"))))


                elif volatility == "25 Delta Risk Reversal":
                    col3.markdown(
                        "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                        unsafe_allow_html=True)
                    data_frame = pd.DataFrame(df)
                    over_night = float(data_frame["25 Delta Risk Reversal "][0])
                    one_week = float(data_frame["25 Delta Risk Reversal "][1])
                    one_month = float(data_frame["25 Delta Risk Reversal "][2])
                    two_month = float(data_frame["25 Delta Risk Reversal "][3])
                    three_month = float(data_frame["25 Delta Risk Reversal "][4])
                    six_month = float(data_frame["25 Delta Risk Reversal "][5])
                    one_year = float(data_frame["25 Delta Risk Reversal "][6])
                    two_year = float(data_frame["25 Delta Risk Reversal "][7])
                    three_year = float(data_frame["25 Delta Risk Reversal "][8])
                    five_year = float(data_frame["25 Delta Risk Reversal "][9])

                    X = [1,7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30
                         ]
                    Y = [over_night,one_week, one_month, two_month, three_month, six_month, one_year, two_year, three_year,
                         five_year]
                    y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                    col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                        volatility) + " " + "qui correspond à " + " " + str(
                        format(c, ".0f")) + " " + "jours est :" + " " + str(
                        format(y_interp(c), ".3f"))))
                elif volatility == "10 Delta Risk Reversal":
                    col3.markdown(
                        "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                        unsafe_allow_html=True)
                    data_frame = pd.DataFrame(df)
                    over_night = float(data_frame["10 Delta Risk Reversal"][0])
                    one_week = float(data_frame["10 Delta Risk Reversal"][1])
                    one_month = float(data_frame["10 Delta Risk Reversal"][2])
                    two_month = float(data_frame["10 Delta Risk Reversal"][3])
                    three_month = float(data_frame["10 Delta Risk Reversal"][4])
                    six_month = float(data_frame["10 Delta Risk Reversal"][5])
                    one_year = float(data_frame["10 Delta Risk Reversal"][6])
                    two_year = float(data_frame["10 Delta Risk Reversal"][7])
                    three_year = float(data_frame["10 Delta Risk Reversal"][8])
                    five_year = float(data_frame["10 Delta Risk Reversal"][9])

                    X = [1,7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30
                         ]
                    Y = [over_night,one_week, one_month, two_month, three_month, six_month, one_year, two_year, three_year,
                         five_year]
                    y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                    col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                        volatility) + " " + "qui correspond à " + " " + str(
                        format(c, ".0f")) + " " + "jours est :" + " " + str(format(y_interp(c), ".3f"))))
                elif volatility == "25 Delta Butterfly":
                    col3.markdown(
                        "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                        unsafe_allow_html=True)
                    data_frame = pd.DataFrame(df)
                    over_night = float(data_frame["25 Delta Butterfly"][0])
                    one_week = float(data_frame["25 Delta Butterfly"][1])
                    one_month = float(data_frame["25 Delta Butterfly"][2])
                    two_month = float(data_frame["25 Delta Butterfly"][3])
                    three_month = float(data_frame["25 Delta Butterfly"][4])
                    six_month = float(data_frame["25 Delta Butterfly"][5])
                    one_year = float(data_frame["25 Delta Butterfly"][6])
                    two_year = float(data_frame["25 Delta Butterfly"][7])
                    three_year = float(data_frame["25 Delta Butterfly"][8])
                    five_year = float(data_frame["25 Delta Butterfly"][9])

                    X = [1,7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30
                       ]
                    Y = [over_night,one_week, one_month, two_month, three_month, six_month, one_year, two_year, three_year,
                         five_year]
                    y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                    col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                        volatility) + " " + "qui correspond à " + " " + str(
                        format(c, ".0f")) + " " + "jours est :" + " " + str(
                        format(y_interp(c), ".3f"))))
                elif volatility == "10 Delta Butterfly":
                    col3.markdown(
                        "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                        unsafe_allow_html=True)
                    data_frame = pd.DataFrame(df)
                    over_night = float(data_frame["10 Delta Butterfly"][0])
                    one_week = float(data_frame["10 Delta Butterfly"][1])
                    one_month = float(data_frame["10 Delta Butterfly"][2])
                    two_month = float(data_frame["10 Delta Butterfly"][3])
                    three_month = float(data_frame["10 Delta Butterfly"][4])
                    six_month = float(data_frame["10 Delta Butterfly"][5])
                    one_year = float(data_frame["10 Delta Butterfly"][6])
                    two_year = float(data_frame["10 Delta Butterfly"][7])
                    three_year = float(data_frame["10 Delta Butterfly"][8])
                    five_year = float(data_frame["10 Delta Butterfly"][9])

                    X = [1,7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30
                         ]
                    Y = [over_night,one_week, one_month, two_month, three_month, six_month, one_year, two_year, three_year,
                         five_year]
                    y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                    col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                        volatility) + " " + "qui correspond à " + " " + str(
                        format(c, ".0f")) + " " + "jours est :" + " " + str(
                        format(y_interp(c), ".3f"))))

                            #time.sleep(1)
            #except:
               #    print("An exception occurred")
            #break
        elif paire == 'EUR/GBP':
                # s = sched.scheduler(time.time, time.sleep)

                # def main_task():
                # placeholder = st.empty()

                # for seconds in range(200):
                # while True:
                # try:
                # with placeholder.container():

                workbook = xw.Book("streamlitmodifie1.xlsm")

                df = workbook.sheets[4].range('A1').options(pd.DataFrame,
                                                            header=1,
                                                            index=True,
                                                            expand='table').value
                df1 = df.style.highlight_null(props="color: transparent;")

                import plotly.graph_objects as go

                # st.header("")
                #st.header("")
                #col1, col2, col3 = st.columns(3)

                #with col2:
                    #st.markdown("<h1 style='text-align: center; color: green;'> EUR/GBP</h1>",unsafe_allow_html=True)
                    #st.header("")

                # with col1:
                # st.button("Refresh")
                #with col3:
                   #  today = date.today()
                  #  v = datetime.now().strftime("%H:%M:%S")

                    #st.write(today, v)
                #st.header("")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(

                        "<h3 style='text-align: left; color: green;'>  Aperçu   </h3>",
                        unsafe_allow_html=True)
                with col1:

                    df1 = df.style.highlight_null(props="color: transparent;")

                    data_frame = pd.DataFrame(df)
                    over_night = float(data_frame["ATM"][0])
                    one_week = float(data_frame["ATM"][1])
                    one_month = float(data_frame["ATM"][2])
                    two_month = float(data_frame["ATM"][3])
                    three_month = float(data_frame["ATM"][4])
                    six_month = float(data_frame["ATM"][5])
                    one_year = float(data_frame["ATM"][6])
                    two_year = float(data_frame["ATM"][7])
                    three_year = float(data_frame["ATM"][8])
                    five_year = float(data_frame["ATM"][9])
                    ten_year = float(data_frame["ATM"][10])
                    #############################################################"
                    over_night1 = float(data_frame["25 Delta Risk Reversal "][0])
                    one_week1 = float(data_frame["25 Delta Risk Reversal "][0])
                    one_month1 = float(data_frame["25 Delta Risk Reversal "][1])
                    two_month1 = float(data_frame["25 Delta Risk Reversal "][2])
                    three_month1 = float(data_frame["25 Delta Risk Reversal "][3])
                    six_month1 = float(data_frame["25 Delta Risk Reversal "][4])
                    one_year1 = float(data_frame["25 Delta Risk Reversal "][5])
                    two_year1 = float(data_frame["25 Delta Risk Reversal "][6])
                    three_year1 = float(data_frame["25 Delta Risk Reversal "][7])
                    five_year1 = float(data_frame["25 Delta Risk Reversal "][8])
                    ten_year1 = float(data_frame["25 Delta Risk Reversal "][10])
                    ################################################################""
                    over_night2 = float(data_frame["10 Delta Risk Reversal"][0])
                    one_week2 = float(data_frame["10 Delta Risk Reversal"][1])
                    one_month2 = float(data_frame["10 Delta Risk Reversal"][2])
                    two_month2 = float(data_frame["10 Delta Risk Reversal"][3])
                    three_month2 = float(data_frame["10 Delta Risk Reversal"][4])
                    six_month2 = float(data_frame["10 Delta Risk Reversal"][5])
                    one_year2 = float(data_frame["10 Delta Risk Reversal"][6])
                    two_year2 = float(data_frame["10 Delta Risk Reversal"][7])
                    three_year2 = float(data_frame["10 Delta Risk Reversal"][8])
                    five_year2 = float(data_frame["10 Delta Risk Reversal"][9])
                    ten_year2 = float(data_frame["10 Delta Risk Reversal"][10])
                    ################################################################
                    over_night3 = float(data_frame["25 Delta Butterfly"][0])
                    one_week3 = float(data_frame["25 Delta Butterfly"][1])
                    one_month3 = float(data_frame["25 Delta Butterfly"][2])
                    two_month3 = float(data_frame["25 Delta Butterfly"][3])
                    three_month3 = float(data_frame["25 Delta Butterfly"][4])
                    six_month3 = float(data_frame["25 Delta Butterfly"][5])
                    one_year3 = float(data_frame["25 Delta Butterfly"][6])
                    two_year3 = float(data_frame["25 Delta Butterfly"][7])
                    three_year3 = float(data_frame["25 Delta Butterfly"][8])
                    five_year3 = float(data_frame["25 Delta Butterfly"][9])
                    ten_year3 = float(data_frame["25 Delta Butterfly"][10])
                    ##################################################################
                    over_night4 = float(data_frame["10 Delta Butterfly"][0])
                    one_week4 = float(data_frame["10 Delta Butterfly"][1])
                    one_month4 = float(data_frame["10 Delta Butterfly"][2])
                    two_month4 = float(data_frame["10 Delta Butterfly"][3])
                    three_month4 = float(data_frame["10 Delta Butterfly"][4])
                    six_month4 = float(data_frame["10 Delta Butterfly"][5])
                    one_year4 = float(data_frame["10 Delta Butterfly"][6])
                    two_year4 = float(data_frame["10 Delta Butterfly"][7])
                    three_year4 = float(data_frame["10 Delta Butterfly"][8])
                    five_year4 = float(data_frame["10 Delta Butterfly"][9])
                    ten_year4 = float(data_frame["10 Delta Butterfly"][10])
                    #######################################################################
                    # one_week5 = float(data_frame["1W"][5])
                    # one_month5 = float(data_frame["1M"][5])
                    # two_month5 = float(data_frame["2M"][5])
                    ##three_month5 = float(data_frame["3M"][5])
                    # six_month5 = float(data_frame["6M"][5])
                    # one_year5 = float(data_frame["1Y"][5])
                    # two_year5 = float(data_frame["2Y"][5])
                    # three_year5 = float(data_frame["3Y"][5])
                    # five_year5 = float(data_frame["5Y"][5])
                    # ten_year5 = float(data_frame["10Y"][5])
                    ###########################################################################"
                    # one_week6 = float(data_frame["1W"][6])
                    # one_month6 = float(data_frame["1M"][6])
                    # two_month6 = float(data_frame["2M"][6])
                    # three_month6 = float(data_frame["3M"][6])
                    # six_month6 = float(data_frame["6M"][6])
                    # one_year6 = float(data_frame["1Y"][6])
                    # two_year6 = float(data_frame["2Y"][6])
                    # three_year6 = float(data_frame["3Y"][6])
                    # five_year6 = float(data_frame["5Y"][6])
                    # ten_year6 = float(data_frame["10Y"][6])
                    ############################################################################"
                    # one_week7 = float(data_frame["1W"][7])
                    # one_month7 = float(data_frame["1M"][7])
                    # two_month7 = float(data_frame["2M"][7])
                    # three_month7 = float(data_frame["3M"][7])
                    # six_month7 = float(data_frame["6M"][7])
                    # one_year7 = float(data_frame["1Y"][7])
                    # two_year7 = float(data_frame["2Y"][7])
                    # three_year7 = float(data_frame["3Y"][7])
                    # five_year7 = float(data_frame["5Y"][7])
                    # ten_year7 = float(data_frame["10Y"][7])
                    ##################################################"
                    # one_week8 = float(data_frame["1W"][8])
                    # one_month8 = float(data_frame["1M"][8])
                    # two_month8 = float(data_frame["2M"][8])
                    # three_month8 = float(data_frame["3M"][8])
                    # six_month8 = float(data_frame["6M"][8])
                    # one_year8 = float(data_frame["1Y"][8])
                    # two_year8 = float(data_frame["2Y"][8])
                    # three_year8 = float(data_frame["3Y"][8])
                    # five_year8 = float(data_frame["5Y"][8])
                    # ten_year8 = float(data_frame["10Y"][8])
                    ###############################################"
                    # one_week9 = float(data_frame["1W"][9])
                    # one_month9 = float(data_frame["1M"][9])
                    # two_month9 = float(data_frame["2M"][9])
                    # three_month9 = float(data_frame["3M"][9])
                    # six_month9 = float(data_frame["6M"][9])
                    # one_year9 = float(data_frame["1Y"][9])
                    # two_year9 = float(data_frame["2Y"][9])
                    # three_year9 = float(data_frame["3Y"][9])
                    # five_year9 = float(data_frame["5Y"][9])
                    # ten_year9 = float(data_frame["10Y"][9])
                    ##############################################
                    # one_week10 = float(data_frame["1W"][10])
                    # one_month10 = float(data_frame["1M"][10])
                    # two_month10 = float(data_frame["2M"][10])
                    # three_month10 = float(data_frame["3M"][10])
                    # six_month10 = float(data_frame["6M"][10])
                    # one_year10 = float(data_frame["1Y"][10])
                    # two_year10 = float(data_frame["2Y"][10])
                    # three_year10 = float(data_frame["3Y"][10])
                    # five_year10 = float(data_frame["5Y"][10])
                    # ten_year10 = float(data_frame["10Y"][10])

                    # df1.style.format('{:.2f}')
                    # st.table(df1.format('{:.2f}'))
                    headerColor = 'grey'
                    rowEvenColor = 'lightgrey'
                    rowOddColor = 'white'
                    import plotly.graph_objects as go

                    fig = go.Figure(data=[go.Table(
                        header=dict(values=["Volatilité", "ATM", "25D RR", "10D RR", "25D BF", "10D BF"],
                                    align=['left', 'center'],
                                    line_color='darkslategray',
                                    fill_color='#54872E',
                                    font=dict(color='black', size=17)),

                        cells=dict(values=[
                            ['Pendant la nuit', 'Une semaine', 'Un mois', 'Deux mois', 'Trois Mois', 'Six mois',
                             'Un an', 'Deux ans ',
                             'Trois ans', 'Cinq ans', 'Dix ans'],

                            [format(over_night, ".3f"), format(one_week, ".3f"), format(one_month, ".3f"),
                             format(two_month, ".3f"), format(three_month, ".3f"), format(six_month, ".3f"),
                             format(one_year, ".3f"), format(two_year, ".3f"), format(three_year, ".3f"),
                             format(five_year, ".3f"), format(ten_year, ".3f")],
                            [format(over_night1, ".3f"), format(one_week1, ".3f"), format(one_month1, ".3f"),
                             format(two_month1, ".3f"), format(three_month1, ".3f"), format(six_month1, ".3f"),
                             format(one_year1, ".3f"), format(two_year1, ".3f"), format(three_year1, ".3f"),
                             format(five_year1, ".3f"), format(ten_year1, ".3f")],
                            [format(over_night2, ".3f"), format(one_week2, ".3f"), format(one_month2, ".3f"),
                             format(two_month2, ".3f"), format(three_month2, ".3f"), format(six_month2, ".3f"),
                             format(one_year2, ".3f"), format(two_year2, ".3f"), format(three_year2, ".3f"),
                             format(five_year2, ".3f"), format(ten_year2, ".3f")],
                            [format(over_night3, ".3f"), format(one_week3, ".3f"), format(one_month3, ".3f"),
                             format(two_month3, ".3f"), format(three_month3, ".3f"), format(six_month3, ".3f"),
                             format(one_year3, ".3f"), format(two_year3, ".3f"), format(three_year3, ".3f"),
                             format(five_year3, ".3f"), format(ten_year3, ".3f")],
                            [format(over_night4, ".3f"), format(one_week4, ".3f"), format(one_month4, ".3f"),
                             format(two_month4, ".3f"), format(three_month4, ".3f"), format(six_month4, ".3f"),
                             format(one_year4, ".3f"), format(two_year4, ".3f"), format(three_year4, ".3f"),
                             format(five_year4, ".3f"), format(ten_year4, ".3f")]],

                            line_color='white',
                            # 2-D list of colors for alternating rows
                            fill_color='white',
                            align=['left', 'center'],
                            font=dict(color='darkslategray', size=13)

                        ))

                    ])
                    fig.update_layout(margin=dict(l=1, r=40, b=1, t=1))

                    # fig2=fig.replace(np.nan, '', regex=True)
                    col1.write(fig)

                #  col1.table(df1)
                #  col1.write( """la volatilité ATM est la valeur de volatilité qui rend le prix implicite d'une option vanille ATM égal au prix du marché de cette option.""")
                with col3:
                    st.markdown(

                        "<h3 style='text-align: left; color: green;'>  Maturité à interpoler   </h3>",
                        unsafe_allow_html=True)
                    # st.header("")
                with col3:
                    volatility = st.selectbox(
                        """Choisissez la paire de devise """,
                        ('ATM', '25 Delta Risk Reversal', '10 Delta Risk Reversal', '25 Delta Butterfly',
                         '10 Delta Butterfly'))

                with col4:
                    st.markdown(

                        "<h1 style='text-align: center; color: green;'>    </h1>",
                        unsafe_allow_html=True)

                st.header("")
                # col4.header("")
                # col4.header("")

                with col4:
                    st.markdown(

                        "<h3 style='text-align: center; color: green;'>    </h3>",
                        unsafe_allow_html=True)

                with col4:
                    c = float(st.text_input('Saisez la maturité (jours)', value=0))
                    if volatility == "ATM":
                        col3.markdown(
                            "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                            unsafe_allow_html=True)
                        data_frame = pd.DataFrame(df)
                        df.dropna(inplace=True)
                        over_night = float(data_frame["ATM"][0])
                        one_week = float(data_frame["ATM"][1])
                        one_month = float(data_frame["ATM"][2])
                        two_month = float(data_frame["ATM"][3])
                        three_month = float(data_frame["ATM"][4])
                        six_month = float(data_frame["ATM"][5])
                        one_year = float(data_frame["ATM"][6])
                        two_year = float(data_frame["ATM"][7])
                        three_year = float(data_frame["ATM"][8])
                        five_year = float(data_frame["ATM"][9])
                        ten_year = float(data_frame["ATM"][10])

                        # col4.metric(("Une semaine  "), one_year, delta_color="off")
                        X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                             10 * 12 * 30]
                        Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                             three_year,
                             five_year, ten_year]
                        y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')

                        col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                            volatility) + " " + "qui correspond à " + " " + str(
                            format(c, ".0f")) + " " + "jours est :" + " " + str(format(y_interp(c), ".3f"))))


                    elif volatility == "25 Delta Risk Reversal":
                        col3.markdown(
                            "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                            unsafe_allow_html=True)
                        data_frame = pd.DataFrame(df)
                        over_night = float(data_frame["25 Delta Risk Reversal "][0])
                        one_week = float(data_frame["25 Delta Risk Reversal "][1])
                        one_month = float(data_frame["25 Delta Risk Reversal "][2])
                        two_month = float(data_frame["25 Delta Risk Reversal "][3])
                        three_month = float(data_frame["25 Delta Risk Reversal "][4])
                        six_month = float(data_frame["25 Delta Risk Reversal "][5])
                        one_year = float(data_frame["25 Delta Risk Reversal "][6])
                        two_year = float(data_frame["25 Delta Risk Reversal "][7])
                        three_year = float(data_frame["25 Delta Risk Reversal "][8])
                        five_year = float(data_frame["25 Delta Risk Reversal "][9])
                        ten_year = float(data_frame["25 Delta Risk Reversal "][10])
                        X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                             10 * 12 * 30]
                        Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                             three_year,
                             five_year, ten_year]
                        y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                        col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                            volatility) + " " + "qui correspond à " + " " + str(
                            format(c, ".0f")) + " " + "jours est :" + " " + str(
                            format(y_interp(c), ".3f"))))
                    elif volatility == "10 Delta Risk Reversal":
                        col3.markdown(
                            "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                            unsafe_allow_html=True)
                        data_frame = pd.DataFrame(df)
                        over_night = float(data_frame["10 Delta Risk Reversal"][0])
                        one_week = float(data_frame["10 Delta Risk Reversal"][1])
                        one_month = float(data_frame["10 Delta Risk Reversal"][2])
                        two_month = float(data_frame["10 Delta Risk Reversal"][3])
                        three_month = float(data_frame["10 Delta Risk Reversal"][4])
                        six_month = float(data_frame["10 Delta Risk Reversal"][5])
                        one_year = float(data_frame["10 Delta Risk Reversal"][6])
                        two_year = float(data_frame["10 Delta Risk Reversal"][7])
                        three_year = float(data_frame["10 Delta Risk Reversal"][8])
                        five_year = float(data_frame["10 Delta Risk Reversal"][9])
                        ten_year = float(data_frame["10 Delta Risk Reversal"][10])
                        X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                             10 * 12 * 30]
                        Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                             three_year,
                             five_year, ten_year]
                        y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                        col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                            volatility) + " " + "qui correspond à " + " " + str(
                            format(c, ".0f")) + " " + "jours est :" + " " + str(format(y_interp(c), ".3f"))))
                    elif volatility == "25 Delta Butterfly":
                        col3.markdown(
                            "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                            unsafe_allow_html=True)
                        data_frame = pd.DataFrame(df)
                        over_night = float(data_frame["25 Delta Butterfly"][0])
                        one_week = float(data_frame["25 Delta Butterfly"][1])
                        one_month = float(data_frame["25 Delta Butterfly"][2])
                        two_month = float(data_frame["25 Delta Butterfly"][3])
                        three_month = float(data_frame["25 Delta Butterfly"][4])
                        six_month = float(data_frame["25 Delta Butterfly"][5])
                        one_year = float(data_frame["25 Delta Butterfly"][6])
                        two_year = float(data_frame["25 Delta Butterfly"][7])
                        three_year = float(data_frame["25 Delta Butterfly"][8])
                        five_year = float(data_frame["25 Delta Butterfly"][9])
                        ten_year = float(data_frame["25 Delta Butterfly"][10])
                        X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                             10 * 12 * 30]
                        Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                             three_year,
                             five_year, ten_year]
                        y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                        col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                            volatility) + " " + "qui correspond à " + " " + str(
                            format(c, ".0f")) + " " + "jours est :" + " " + str(
                            format(y_interp(c), ".3f"))))
                    elif volatility == "10 Delta Butterfly":
                        col3.markdown(
                            "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                            unsafe_allow_html=True)
                        data_frame = pd.DataFrame(df)
                        over_night = float(data_frame["10 Delta Butterfly"][0])
                        one_week = float(data_frame["10 Delta Butterfly"][1])
                        one_month = float(data_frame["10 Delta Butterfly"][2])
                        two_month = float(data_frame["10 Delta Butterfly"][3])
                        three_month = float(data_frame["10 Delta Butterfly"][4])
                        six_month = float(data_frame["10 Delta Butterfly"][5])
                        one_year = float(data_frame["10 Delta Butterfly"][6])
                        two_year = float(data_frame["10 Delta Butterfly"][7])
                        three_year = float(data_frame["10 Delta Butterfly"][8])
                        five_year = float(data_frame["10 Delta Butterfly"][9])
                        ten_year = float(data_frame["10 Delta Butterfly"][10])
                        X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                             10 * 12 * 30]
                        Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                             three_year,
                             five_year, ten_year]
                        y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                        col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                            volatility) + " " + "qui correspond à " + " " + str(
                            format(c, ".0f")) + " " + "jours est :" + " " + str(
                            format(y_interp(c), ".3f"))))

                        # time.sleep(1)
                # except:
                #    print("An exception occurred")
                # break
        elif paire == 'EUR/CHF':
                    # s = sched.scheduler(time.time, time.sleep)

                    # def main_task():
                    # placeholder = st.empty()

                    # for seconds in range(200):
                    # while True:
                    # try:
                    # with placeholder.container():

                    workbook = xw.Book("streamlitmodifie1.xlsm")

                    df = workbook.sheets[5].range('A1').options(pd.DataFrame,
                                                                header=1,
                                                                index=True,
                                                                expand='table').value
                    df1 = df.style.highlight_null(props="color: transparent;")

                    import plotly.graph_objects as go

                    # st.header("")
                    #st.header("")
                   # col1, col2, col3 = st.columns(3)

                    #with col2:
                        #st.markdown("<h1 style='text-align: center; color: green;'> EUR/CHF </h1>",unsafe_allow_html=True)
                        #st.header("")

                    # with col1:
                    # st.button("Refresh")
                    #with col3:
                     #   today = date.today()
                      #  v = datetime.now().strftime("%H:%M:%S")

                     #   st.write(today, v)
                    #st.header("")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(

                            "<h3 style='text-align: left; color: green;'>  Aperçu   </h3>",
                            unsafe_allow_html=True)
                    with col1:

                        df1 = df.style.highlight_null(props="color: transparent;")

                        data_frame = pd.DataFrame(df)
                        over_night = float(data_frame["ATM"][0])
                        one_week = float(data_frame["ATM"][1])
                        one_month = float(data_frame["ATM"][2])
                        two_month = float(data_frame["ATM"][3])
                        three_month = float(data_frame["ATM"][4])
                        six_month = float(data_frame["ATM"][5])
                        one_year = float(data_frame["ATM"][6])
                        two_year = float(data_frame["ATM"][7])
                        three_year = float(data_frame["ATM"][8])

                        #############################################################"
                        over_night1 = float(data_frame["25 Delta Risk Reversal "][0])
                        one_week1 = float(data_frame["25 Delta Risk Reversal "][0])
                        one_month1 = float(data_frame["25 Delta Risk Reversal "][1])
                        two_month1 = float(data_frame["25 Delta Risk Reversal "][2])
                        three_month1 = float(data_frame["25 Delta Risk Reversal "][3])
                        six_month1 = float(data_frame["25 Delta Risk Reversal "][4])
                        one_year1 = float(data_frame["25 Delta Risk Reversal "][5])
                        two_year1 = float(data_frame["25 Delta Risk Reversal "][6])
                        three_year1 = float(data_frame["25 Delta Risk Reversal "][7])

                        ################################################################""
                        over_night2 = float(data_frame["10 Delta Risk Reversal"][0])
                        one_week2 = float(data_frame["10 Delta Risk Reversal"][1])
                        one_month2 = float(data_frame["10 Delta Risk Reversal"][2])
                        two_month2 = float(data_frame["10 Delta Risk Reversal"][3])
                        three_month2 = float(data_frame["10 Delta Risk Reversal"][4])
                        six_month2 = float(data_frame["10 Delta Risk Reversal"][5])
                        one_year2 = float(data_frame["10 Delta Risk Reversal"][6])
                        two_year2 = float(data_frame["10 Delta Risk Reversal"][7])
                        three_year2 = float(data_frame["10 Delta Risk Reversal"][8])

                        ################################################################
                        over_night3 = float(data_frame["25 Delta Butterfly"][0])
                        one_week3 = float(data_frame["25 Delta Butterfly"][1])
                        one_month3 = float(data_frame["25 Delta Butterfly"][2])
                        two_month3 = float(data_frame["25 Delta Butterfly"][3])
                        three_month3 = float(data_frame["25 Delta Butterfly"][4])
                        six_month3 = float(data_frame["25 Delta Butterfly"][5])
                        one_year3 = float(data_frame["25 Delta Butterfly"][6])
                        two_year3 = float(data_frame["25 Delta Butterfly"][7])
                        three_year3 = float(data_frame["25 Delta Butterfly"][8])

                        ##################################################################
                        over_night4 = float(data_frame["10 Delta Butterfly"][0])
                        one_week4 = float(data_frame["10 Delta Butterfly"][1])
                        one_month4 = float(data_frame["10 Delta Butterfly"][2])
                        two_month4 = float(data_frame["10 Delta Butterfly"][3])
                        three_month4 = float(data_frame["10 Delta Butterfly"][4])
                        six_month4 = float(data_frame["10 Delta Butterfly"][5])
                        one_year4 = float(data_frame["10 Delta Butterfly"][6])
                        two_year4 = float(data_frame["10 Delta Butterfly"][7])
                        three_year4 = float(data_frame["10 Delta Butterfly"][8])

                        headerColor = 'grey'
                        rowEvenColor = 'lightgrey'
                        rowOddColor = 'white'
                        import plotly.graph_objects as go

                        fig = go.Figure(data=[go.Table(
                            header=dict(values=["Volatilité", "ATM", "25D RR", "10D RR", "25D BF", "10D BF"],
                                        align=['left', 'center'],
                                        line_color='darkslategray',
                                        fill_color='#54872E',
                                        font=dict(color='black', size=17)),

                            cells=dict(values=[
                                ['Pendant la nuit', 'Une semaine', 'Un mois', 'Deux mois', 'Trois Mois', 'Six mois',
                                 'Un an', 'Deux ans ',
                                 'Trois ans'],

                                [format(over_night, ".3f"), format(one_week, ".3f"), format(one_month, ".3f"),
                                 format(two_month, ".3f"), format(three_month, ".3f"), format(six_month, ".3f"),
                                 format(one_year, ".3f"), format(two_year, ".3f"), format(three_year, ".3f")],
                                [format(over_night1, ".3f"), format(one_week1, ".3f"), format(one_month1, ".3f"),
                                 format(two_month1, ".3f"), format(three_month1, ".3f"), format(six_month1, ".3f"),
                                 format(one_year1, ".3f"), format(two_year1, ".3f"), format(three_year1, ".3f")],
                                [format(over_night2, ".3f"), format(one_week2, ".3f"), format(one_month2, ".3f"),
                                 format(two_month2, ".3f"), format(three_month2, ".3f"), format(six_month2, ".3f"),
                                 format(one_year2, ".3f"), format(two_year2, ".3f"), format(three_year2, ".3f")],
                                [format(over_night3, ".3f"), format(one_week3, ".3f"), format(one_month3, ".3f"),
                                 format(two_month3, ".3f"), format(three_month3, ".3f"), format(six_month3, ".3f"),
                                 format(one_year3, ".3f"), format(two_year3, ".3f"), format(three_year3, ".3f")],
                                [format(over_night4, ".3f"), format(one_week4, ".3f"), format(one_month4, ".3f"),
                                 format(two_month4, ".3f"), format(three_month4, ".3f"), format(six_month4, ".3f"),
                                 format(one_year4, ".3f"), format(two_year4, ".3f"), format(three_year4, ".3f")]],

                                line_color='white',
                                # 2-D list of colors for alternating rows
                                fill_color='white',
                                align=['left', 'center'],
                                font=dict(color='darkslategray', size=13)

                            ))

                        ])
                        fig.update_layout(margin=dict(l=1, r=40, b=1, t=1))

                        # fig2=fig.replace(np.nan, '', regex=True)
                        col1.write(fig)

                    #  col1.table(df1)
                    #  col1.write( """la volatilité ATM est la valeur de volatilité qui rend le prix implicite d'une option vanille ATM égal au prix du marché de cette option.""")
                    with col3:
                        st.markdown(

                            "<h3 style='text-align: left; color: green;'>  Maturité à interpoler   </h3>",
                            unsafe_allow_html=True)
                        # st.header("")
                    with col3:
                        volatility = st.selectbox(
                            """Choisissez la paire de devise """,
                            ('ATM', '25 Delta Risk Reversal', '10 Delta Risk Reversal', '25 Delta Butterfly',
                             '10 Delta Butterfly'))

                    with col4:
                        st.markdown(

                            "<h1 style='text-align: center; color: green;'>    </h1>",
                            unsafe_allow_html=True)

                    st.header("")
                    # col4.header("")
                    # col4.header("")

                    with col4:
                        st.markdown(

                            "<h3 style='text-align: center; color: green;'>    </h3>",
                            unsafe_allow_html=True)

                    with col4:
                        c = float(st.text_input('Saisez la maturité (jours)', value=0))
                        if volatility == "ATM":
                            col3.markdown(
                                "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                                unsafe_allow_html=True)
                            data_frame = pd.DataFrame(df)
                            df.dropna(inplace=True)
                            over_night = float(data_frame["ATM"][0])
                            one_week = float(data_frame["ATM"][1])
                            one_month = float(data_frame["ATM"][2])
                            two_month = float(data_frame["ATM"][3])
                            three_month = float(data_frame["ATM"][4])
                            six_month = float(data_frame["ATM"][5])
                            one_year = float(data_frame["ATM"][6])
                            two_year = float(data_frame["ATM"][7])
                            three_year = float(data_frame["ATM"][8])


                            # col4.metric(("Une semaine  "), one_year, delta_color="off")
                            X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30]
                            Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                                 three_year]
                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')

                            col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                                volatility) + " " + "qui correspond à " + " " + str(
                                format(c, ".0f")) + " " + "jours est :" + " " + str(format(y_interp(c), ".3f"))))


                        elif volatility == "25 Delta Risk Reversal":
                            col3.markdown(
                                "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                                unsafe_allow_html=True)
                            data_frame = pd.DataFrame(df)
                            over_night = float(data_frame["25 Delta Risk Reversal "][0])
                            one_week = float(data_frame["25 Delta Risk Reversal "][1])
                            one_month = float(data_frame["25 Delta Risk Reversal "][2])
                            two_month = float(data_frame["25 Delta Risk Reversal "][3])
                            three_month = float(data_frame["25 Delta Risk Reversal "][4])
                            six_month = float(data_frame["25 Delta Risk Reversal "][5])
                            one_year = float(data_frame["25 Delta Risk Reversal "][6])
                            two_year = float(data_frame["25 Delta Risk Reversal "][7])
                            three_year = float(data_frame["25 Delta Risk Reversal "][8])

                            X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30]
                            Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                                 three_year]
                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                            col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                                volatility) + " " + "qui correspond à " + " " + str(
                                format(c, ".0f")) + " " + "jours est :" + " " + str(
                                format(y_interp(c), ".3f"))))
                        elif volatility == "10 Delta Risk Reversal":
                            col3.markdown(
                                "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                                unsafe_allow_html=True)
                            data_frame = pd.DataFrame(df)
                            over_night = float(data_frame["10 Delta Risk Reversal"][0])
                            one_week = float(data_frame["10 Delta Risk Reversal"][1])
                            one_month = float(data_frame["10 Delta Risk Reversal"][2])
                            two_month = float(data_frame["10 Delta Risk Reversal"][3])
                            three_month = float(data_frame["10 Delta Risk Reversal"][4])
                            six_month = float(data_frame["10 Delta Risk Reversal"][5])
                            one_year = float(data_frame["10 Delta Risk Reversal"][6])
                            two_year = float(data_frame["10 Delta Risk Reversal"][7])
                            three_year = float(data_frame["10 Delta Risk Reversal"][8])

                            X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30]
                            Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                                 three_year]
                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                            col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                                volatility) + " " + "qui correspond à " + " " + str(
                                format(c, ".0f")) + " " + "jours est :" + " " + str(format(y_interp(c), ".3f"))))
                        elif volatility == "25 Delta Butterfly":
                            col3.markdown(
                                "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                                unsafe_allow_html=True)
                            data_frame = pd.DataFrame(df)
                            over_night = float(data_frame["25 Delta Butterfly"][0])
                            one_week = float(data_frame["25 Delta Butterfly"][1])
                            one_month = float(data_frame["25 Delta Butterfly"][2])
                            two_month = float(data_frame["25 Delta Butterfly"][3])
                            three_month = float(data_frame["25 Delta Butterfly"][4])
                            six_month = float(data_frame["25 Delta Butterfly"][5])
                            one_year = float(data_frame["25 Delta Butterfly"][6])
                            two_year = float(data_frame["25 Delta Butterfly"][7])
                            three_year = float(data_frame["25 Delta Butterfly"][8])

                            X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30]
                            Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                                 three_year]
                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                            col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                                volatility) + " " + "qui correspond à " + " " + str(
                                format(c, ".0f")) + " " + "jours est :" + " " + str(
                                format(y_interp(c), ".3f"))))
                        elif volatility == "10 Delta Butterfly":
                            col3.markdown(
                                "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                                unsafe_allow_html=True)
                            data_frame = pd.DataFrame(df)
                            over_night = float(data_frame["10 Delta Butterfly"][0])
                            one_week = float(data_frame["10 Delta Butterfly"][1])
                            one_month = float(data_frame["10 Delta Butterfly"][2])
                            two_month = float(data_frame["10 Delta Butterfly"][3])
                            three_month = float(data_frame["10 Delta Butterfly"][4])
                            six_month = float(data_frame["10 Delta Butterfly"][5])
                            one_year = float(data_frame["10 Delta Butterfly"][6])
                            two_year = float(data_frame["10 Delta Butterfly"][7])
                            three_year = float(data_frame["10 Delta Butterfly"][8])

                            X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30]
                            Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                                 three_year]
                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                            col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                                volatility) + " " + "qui correspond à " + " " + str(
                                format(c, ".0f")) + " " + "jours est :" + " " + str(
                                format(y_interp(c), ".3f"))))

                            # time.sleep(1)
                    # except:
                    #    print("An exception occurred")
                    # break
        elif paire == 'USD/CHF':
                    # s = sched.scheduler(time.time, time.sleep)

                    # def main_task():
                    # placeholder = st.empty()

                    # for seconds in range(200):
                    # while True:
                    # try:
                    # with placeholder.container():

                    workbook = xw.Book("streamlitmodifie1.xlsm")

                    df = workbook.sheets[7].range('A1').options(pd.DataFrame,
                                                                header=1,
                                                                index=True,
                                                                expand='table').value
                    df1 = df.style.highlight_null(props="color: transparent;")

                    import plotly.graph_objects as go

                    # st.header("")
                    #st.header("")
                    #col1, col2, col3 = st.columns(3)

                   # with col2:
                        #st.markdown( "<h1 style='text-align: center; color: green;'> USD/CHF </h1>",unsafe_allow_html=True)
                        #st.header("")

                    # with col1:
                    # st.button("Refresh")
                    #with col3:
                        #today = date.today()
                        #v = datetime.now().strftime("%H:%M:%S")

                        #st.write(today, v)
                    #st.header("")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(

                            "<h3 style='text-align: left; color: green;'>  Aperçu   </h3>",
                            unsafe_allow_html=True)
                    with col1:

                        df1 = df.style.highlight_null(props="color: transparent;")

                        data_frame = pd.DataFrame(df)
                        over_night = float(data_frame["ATM"][0])
                        one_week = float(data_frame["ATM"][1])
                        one_month = float(data_frame["ATM"][2])
                        two_month = float(data_frame["ATM"][3])
                        three_month = float(data_frame["ATM"][4])
                        six_month = float(data_frame["ATM"][5])
                        one_year = float(data_frame["ATM"][6])
                        two_year = float(data_frame["ATM"][7])
                        three_year = float(data_frame["ATM"][8])
                        five_year = float(data_frame["ATM"][9])
                        #ten_year = float(data_frame["ATM"][10])
                        #############################################################"
                        over_night1 = float(data_frame["25 Delta Risk Reversal "][0])
                        one_week1 = float(data_frame["25 Delta Risk Reversal "][0])
                        one_month1 = float(data_frame["25 Delta Risk Reversal "][1])
                        two_month1 = float(data_frame["25 Delta Risk Reversal "][2])
                        three_month1 = float(data_frame["25 Delta Risk Reversal "][3])
                        six_month1 = float(data_frame["25 Delta Risk Reversal "][4])
                        one_year1 = float(data_frame["25 Delta Risk Reversal "][5])
                        two_year1 = float(data_frame["25 Delta Risk Reversal "][6])
                        three_year1 = float(data_frame["25 Delta Risk Reversal "][7])
                        five_year1 = float(data_frame["25 Delta Risk Reversal "][8])
                        #ten_year1 = float(data_frame["25 Delta Risk Reversal "][10])
                        ################################################################""
                        over_night2 = float(data_frame["10 Delta Risk Reversal"][0])
                        one_week2 = float(data_frame["10 Delta Risk Reversal"][1])
                        one_month2 = float(data_frame["10 Delta Risk Reversal"][2])
                        two_month2 = float(data_frame["10 Delta Risk Reversal"][3])
                        three_month2 = float(data_frame["10 Delta Risk Reversal"][4])
                        six_month2 = float(data_frame["10 Delta Risk Reversal"][5])
                        one_year2 = float(data_frame["10 Delta Risk Reversal"][6])
                        two_year2 = float(data_frame["10 Delta Risk Reversal"][7])
                        three_year2 = float(data_frame["10 Delta Risk Reversal"][8])
                        five_year2 = float(data_frame["10 Delta Risk Reversal"][9])
                        #ten_year2 = float(data_frame["10 Delta Risk Reversal"][10])
                        ################################################################
                        over_night3 = float(data_frame["25 Delta Butterfly"][0])
                        one_week3 = float(data_frame["25 Delta Butterfly"][1])
                        one_month3 = float(data_frame["25 Delta Butterfly"][2])
                        two_month3 = float(data_frame["25 Delta Butterfly"][3])
                        three_month3 = float(data_frame["25 Delta Butterfly"][4])
                        six_month3 = float(data_frame["25 Delta Butterfly"][5])
                        one_year3 = float(data_frame["25 Delta Butterfly"][6])
                        two_year3 = float(data_frame["25 Delta Butterfly"][7])
                        three_year3 = float(data_frame["25 Delta Butterfly"][8])
                        five_year3 = float(data_frame["25 Delta Butterfly"][9])
                        #ten_year3 = float(data_frame["25 Delta Butterfly"][10])
                        ##################################################################
                        over_night4 = float(data_frame["10 Delta Butterfly"][0])
                        one_week4 = float(data_frame["10 Delta Butterfly"][1])
                        one_month4 = float(data_frame["10 Delta Butterfly"][2])
                        two_month4 = float(data_frame["10 Delta Butterfly"][3])
                        three_month4 = float(data_frame["10 Delta Butterfly"][4])
                        six_month4 = float(data_frame["10 Delta Butterfly"][5])
                        one_year4 = float(data_frame["10 Delta Butterfly"][6])
                        two_year4 = float(data_frame["10 Delta Butterfly"][7])
                        three_year4 = float(data_frame["10 Delta Butterfly"][8])
                        five_year4 = float(data_frame["10 Delta Butterfly"][9])
                        #ten_year4 = float(data_frame["10 Delta Butterfly"][10])


                        headerColor = 'grey'
                        rowEvenColor = 'lightgrey'
                        rowOddColor = 'white'
                        import plotly.graph_objects as go

                        fig = go.Figure(data=[go.Table(
                            header=dict(values=["Volatilité", "ATM", "25D RR", "10D RR", "25D BF", "10D BF"],
                                        align=['left', 'center'],
                                        line_color='darkslategray',
                                        fill_color='#54872E',
                                        font=dict(color='black', size=17)),

                            cells=dict(values=[
                                ['Pendant la nuit', 'Une semaine', 'Un mois', 'Deux mois', 'Trois Mois', 'Six mois',
                                 'Un an', 'Deux ans ',
                                 'Trois ans', 'Cinq ans'],

                                [format(over_night, ".3f"), format(one_week, ".3f"), format(one_month, ".3f"),
                                 format(two_month, ".3f"), format(three_month, ".3f"), format(six_month, ".3f"),
                                 format(one_year, ".3f"), format(two_year, ".3f"), format(three_year, ".3f"),
                                 format(five_year, ".3f")],
                                [format(over_night1, ".3f"), format(one_week1, ".3f"), format(one_month1, ".3f"),
                                 format(two_month1, ".3f"), format(three_month1, ".3f"), format(six_month1, ".3f"),
                                 format(one_year1, ".3f"), format(two_year1, ".3f"), format(three_year1, ".3f"),
                                 format(five_year1, ".3f")],
                                [format(over_night2, ".3f"), format(one_week2, ".3f"), format(one_month2, ".3f"),
                                 format(two_month2, ".3f"), format(three_month2, ".3f"), format(six_month2, ".3f"),
                                 format(one_year2, ".3f"), format(two_year2, ".3f"), format(three_year2, ".3f"),
                                 format(five_year2, ".3f")],
                                [format(over_night3, ".3f"), format(one_week3, ".3f"), format(one_month3, ".3f"),
                                 format(two_month3, ".3f"), format(three_month3, ".3f"), format(six_month3, ".3f"),
                                 format(one_year3, ".3f"), format(two_year3, ".3f"), format(three_year3, ".3f"),
                                 format(five_year3, ".3f")],
                                [format(over_night4, ".3f"), format(one_week4, ".3f"), format(one_month4, ".3f"),
                                 format(two_month4, ".3f"), format(three_month4, ".3f"), format(six_month4, ".3f"),
                                 format(one_year4, ".3f"), format(two_year4, ".3f"), format(three_year4, ".3f"),
                                 format(five_year4, ".3f")]],

                                line_color='white',
                                # 2-D list of colors for alternating rows
                                fill_color='white',
                                align=['left', 'center'],
                                font=dict(color='darkslategray', size=13)

                            ))

                        ])
                        fig.update_layout(margin=dict(l=1, r=40, b=1, t=1))

                        # fig2=fig.replace(np.nan, '', regex=True)
                        col1.write(fig)

                    #  col1.table(df1)
                    #  col1.write( """la volatilité ATM est la valeur de volatilité qui rend le prix implicite d'une option vanille ATM égal au prix du marché de cette option.""")
                    with col3:
                        st.markdown(

                            "<h3 style='text-align: left; color: green;'>  Maturité à interpoler   </h3>",
                            unsafe_allow_html=True)
                        # st.header("")
                    with col3:
                        volatility = st.selectbox(
                            """Choisissez le type de volatilité """,
                            ('ATM', '25 Delta Risk Reversal', '10 Delta Risk Reversal', '25 Delta Butterfly',
                             '10 Delta Butterfly'))

                    with col4:
                        st.markdown(

                            "<h1 style='text-align: center; color: green;'>    </h1>",
                            unsafe_allow_html=True)

                    st.header("")
                    # col4.header("")
                    # col4.header("")

                    with col4:
                        st.markdown(

                            "<h3 style='text-align: center; color: green;'>    </h3>",
                            unsafe_allow_html=True)

                    with col4:
                        c = float(st.text_input('Saisez la maturité (jours)', value=0))
                        if volatility == "ATM":
                            col3.markdown(
                                "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                                unsafe_allow_html=True)
                            data_frame = pd.DataFrame(df)
                            df.dropna(inplace=True)
                            over_night = float(data_frame["ATM"][0])
                            one_week = float(data_frame["ATM"][1])
                            one_month = float(data_frame["ATM"][2])
                            two_month = float(data_frame["ATM"][3])
                            three_month = float(data_frame["ATM"][4])
                            six_month = float(data_frame["ATM"][5])
                            one_year = float(data_frame["ATM"][6])
                            two_year = float(data_frame["ATM"][7])
                            three_year = float(data_frame["ATM"][8])
                            five_year = float(data_frame["ATM"][9])
                            ten_year = float(data_frame["ATM"][10])

                            # col4.metric(("Une semaine  "), one_year, delta_color="off")
                            X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30]
                            Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                                 three_year,
                                 five_year]
                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')

                            col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                                volatility) + " " + "qui correspond à " + " " + str(
                                format(c, ".0f")) + " " + "jours est :" + " " + str(format(y_interp(c), ".3f"))))


                        elif volatility == "25 Delta Risk Reversal":
                            col3.markdown(
                                "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                                unsafe_allow_html=True)
                            data_frame = pd.DataFrame(df)
                            over_night = float(data_frame["25 Delta Risk Reversal "][0])
                            one_week = float(data_frame["25 Delta Risk Reversal "][1])
                            one_month = float(data_frame["25 Delta Risk Reversal "][2])
                            two_month = float(data_frame["25 Delta Risk Reversal "][3])
                            three_month = float(data_frame["25 Delta Risk Reversal "][4])
                            six_month = float(data_frame["25 Delta Risk Reversal "][5])
                            one_year = float(data_frame["25 Delta Risk Reversal "][6])
                            two_year = float(data_frame["25 Delta Risk Reversal "][7])
                            three_year = float(data_frame["25 Delta Risk Reversal "][8])
                            five_year = float(data_frame["25 Delta Risk Reversal "][9])
                            ten_year = float(data_frame["25 Delta Risk Reversal "][10])
                            X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30]
                            Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                                 three_year,
                                 five_year]
                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                            col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                                volatility) + " " + "qui correspond à " + " " + str(
                                format(c, ".0f")) + " " + "jours est :" + " " + str(
                                format(y_interp(c), ".3f"))))
                        elif volatility == "10 Delta Risk Reversal":
                            col3.markdown(
                                "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                                unsafe_allow_html=True)
                            data_frame = pd.DataFrame(df)
                            over_night = float(data_frame["10 Delta Risk Reversal"][0])
                            one_week = float(data_frame["10 Delta Risk Reversal"][1])
                            one_month = float(data_frame["10 Delta Risk Reversal"][2])
                            two_month = float(data_frame["10 Delta Risk Reversal"][3])
                            three_month = float(data_frame["10 Delta Risk Reversal"][4])
                            six_month = float(data_frame["10 Delta Risk Reversal"][5])
                            one_year = float(data_frame["10 Delta Risk Reversal"][6])
                            two_year = float(data_frame["10 Delta Risk Reversal"][7])
                            three_year = float(data_frame["10 Delta Risk Reversal"][8])
                            five_year = float(data_frame["10 Delta Risk Reversal"][9])
                            ten_year = float(data_frame["10 Delta Risk Reversal"][10])
                            X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30]
                            Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                                 three_year,
                                 five_year]
                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                            col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                                volatility) + " " + "qui correspond à " + " " + str(
                                format(c, ".0f")) + " " + "jours est :" + " " + str(format(y_interp(c), ".3f"))))
                        elif volatility == "25 Delta Butterfly":
                            col3.markdown(
                                "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                                unsafe_allow_html=True)
                            data_frame = pd.DataFrame(df)
                            over_night = float(data_frame["25 Delta Butterfly"][0])
                            one_week = float(data_frame["25 Delta Butterfly"][1])
                            one_month = float(data_frame["25 Delta Butterfly"][2])
                            two_month = float(data_frame["25 Delta Butterfly"][3])
                            three_month = float(data_frame["25 Delta Butterfly"][4])
                            six_month = float(data_frame["25 Delta Butterfly"][5])
                            one_year = float(data_frame["25 Delta Butterfly"][6])
                            two_year = float(data_frame["25 Delta Butterfly"][7])
                            three_year = float(data_frame["25 Delta Butterfly"][8])
                            five_year = float(data_frame["25 Delta Butterfly"][9])
                            ten_year = float(data_frame["25 Delta Butterfly"][10])
                            X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30]
                            Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                                 three_year,
                                 five_year]
                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                            col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                                volatility) + " " + "qui correspond à " + " " + str(
                                format(c, ".0f")) + " " + "jours est :" + " " + str(
                                format(y_interp(c), ".3f"))))
                        elif volatility == "10 Delta Butterfly":
                            col3.markdown(
                                "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                                unsafe_allow_html=True)
                            data_frame = pd.DataFrame(df)
                            over_night = float(data_frame["10 Delta Butterfly"][0])
                            one_week = float(data_frame["10 Delta Butterfly"][1])
                            one_month = float(data_frame["10 Delta Butterfly"][2])
                            two_month = float(data_frame["10 Delta Butterfly"][3])
                            three_month = float(data_frame["10 Delta Butterfly"][4])
                            six_month = float(data_frame["10 Delta Butterfly"][5])
                            one_year = float(data_frame["10 Delta Butterfly"][6])
                            two_year = float(data_frame["10 Delta Butterfly"][7])
                            three_year = float(data_frame["10 Delta Butterfly"][8])
                            five_year = float(data_frame["10 Delta Butterfly"][9])
                            ten_year = float(data_frame["10 Delta Butterfly"][10])
                            X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30]
                            Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                                 three_year,
                                 five_year]
                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                            col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                                volatility) + " " + "qui correspond à " + " " + str(
                                format(c, ".0f")) + " " + "jours est :" + " " + str(
                                format(y_interp(c), ".3f"))))

                            # time.sleep(1)
                    # except:
                    #    print("An exception occurred")
                    # break
        elif paire == 'GBP/USD':
                    # s = sched.scheduler(time.time, time.sleep)

                    # def main_task():
                    # placeholder = st.empty()

                    # for seconds in range(200):
                    # while True:
                    # try:
                    # with placeholder.container():

                    workbook = xw.Book("streamlitmodifie1.xlsm")

                    df = workbook.sheets[6].range('A1').options(pd.DataFrame,
                                                                header=1,
                                                                index=True,
                                                                expand='table').value
                    df1 = df.style.highlight_null(props="color: transparent;")

                    import plotly.graph_objects as go

                    # st.header("")
                   # st.header("")
                    #col1, col2, col3 = st.columns(3)

                   # with col2:
                      #  st.markdown( "<h1 style='text-align: center; color: green;'> GBP/USD"" </h1>",  unsafe_allow_html=True)
                       # st.header("")

                    # with col1:
                    # st.button("Refresh")
                    #with col3:
                       # today = date.today()
                        #v = datetime.now().strftime("%H:%M:%S")

                        #st.write(today, v)
                    #st.header("")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(

                            "<h3 style='text-align: left; color: green;'>  Aperçu   </h3>",
                            unsafe_allow_html=True)
                    with col1:

                        df1 = df.style.highlight_null(props="color: transparent;")

                        data_frame = pd.DataFrame(df)
                        over_night = float(data_frame["ATM"][0])
                        one_week = float(data_frame["ATM"][1])
                        one_month = float(data_frame["ATM"][2])
                        two_month = float(data_frame["ATM"][3])
                        three_month = float(data_frame["ATM"][4])
                        six_month = float(data_frame["ATM"][5])
                        one_year = float(data_frame["ATM"][6])
                        two_year = float(data_frame["ATM"][7])
                        three_year = float(data_frame["ATM"][8])
                        five_year = float(data_frame["ATM"][9])
                        ten_year = float(data_frame["ATM"][10])
                        #############################################################"
                        over_night1 = float(data_frame["25 Delta Risk Reversal "][0])
                        one_week1 = float(data_frame["25 Delta Risk Reversal "][0])
                        one_month1 = float(data_frame["25 Delta Risk Reversal "][1])
                        two_month1 = float(data_frame["25 Delta Risk Reversal "][2])
                        three_month1 = float(data_frame["25 Delta Risk Reversal "][3])
                        six_month1 = float(data_frame["25 Delta Risk Reversal "][4])
                        one_year1 = float(data_frame["25 Delta Risk Reversal "][5])
                        two_year1 = float(data_frame["25 Delta Risk Reversal "][6])
                        three_year1 = float(data_frame["25 Delta Risk Reversal "][7])
                        five_year1 = float(data_frame["25 Delta Risk Reversal "][8])
                        ten_year1 = float(data_frame["25 Delta Risk Reversal "][10])
                        ################################################################""
                        over_night2 = float(data_frame["10 Delta Risk Reversal"][0])
                        one_week2 = float(data_frame["10 Delta Risk Reversal"][1])
                        one_month2 = float(data_frame["10 Delta Risk Reversal"][2])
                        two_month2 = float(data_frame["10 Delta Risk Reversal"][3])
                        three_month2 = float(data_frame["10 Delta Risk Reversal"][4])
                        six_month2 = float(data_frame["10 Delta Risk Reversal"][5])
                        one_year2 = float(data_frame["10 Delta Risk Reversal"][6])
                        two_year2 = float(data_frame["10 Delta Risk Reversal"][7])
                        three_year2 = float(data_frame["10 Delta Risk Reversal"][8])
                        five_year2 = float(data_frame["10 Delta Risk Reversal"][9])
                        ten_year2 = float(data_frame["10 Delta Risk Reversal"][10])
                        ################################################################
                        over_night3 = float(data_frame["25 Delta Butterfly"][0])
                        one_week3 = float(data_frame["25 Delta Butterfly"][1])
                        one_month3 = float(data_frame["25 Delta Butterfly"][2])
                        two_month3 = float(data_frame["25 Delta Butterfly"][3])
                        three_month3 = float(data_frame["25 Delta Butterfly"][4])
                        six_month3 = float(data_frame["25 Delta Butterfly"][5])
                        one_year3 = float(data_frame["25 Delta Butterfly"][6])
                        two_year3 = float(data_frame["25 Delta Butterfly"][7])
                        three_year3 = float(data_frame["25 Delta Butterfly"][8])
                        five_year3 = float(data_frame["25 Delta Butterfly"][9])
                        ten_year3 = float(data_frame["25 Delta Butterfly"][10])
                        ##################################################################
                        over_night4 = float(data_frame["10 Delta Butterfly"][0])
                        one_week4 = float(data_frame["10 Delta Butterfly"][1])
                        one_month4 = float(data_frame["10 Delta Butterfly"][2])
                        two_month4 = float(data_frame["10 Delta Butterfly"][3])
                        three_month4 = float(data_frame["10 Delta Butterfly"][4])
                        six_month4 = float(data_frame["10 Delta Butterfly"][5])
                        one_year4 = float(data_frame["10 Delta Butterfly"][6])
                        two_year4 = float(data_frame["10 Delta Butterfly"][7])
                        three_year4 = float(data_frame["10 Delta Butterfly"][8])
                        five_year4 = float(data_frame["10 Delta Butterfly"][9])
                        ten_year4 = float(data_frame["10 Delta Butterfly"][10])

                        headerColor = 'grey'
                        rowEvenColor = 'lightgrey'
                        rowOddColor = 'white'
                        import plotly.graph_objects as go

                        fig = go.Figure(data=[go.Table(
                            header=dict(values=["Volatilité", "ATM", "25D RR", "10D RR", "25D BF", "10D BF"],
                                        align=['left', 'center'],
                                        line_color='darkslategray',
                                        fill_color='#54872E',
                                        font=dict(color='black', size=17)),

                            cells=dict(values=[
                                ['Pendant la nuit', 'Une semaine', 'Un mois', 'Deux mois', 'Trois Mois', 'Six mois',
                                 'Un an', 'Deux ans ',
                                 'Trois ans', 'Cinq ans', 'Dix ans'],

                                [format(over_night, ".3f"), format(one_week, ".3f"), format(one_month, ".3f"),
                                 format(two_month, ".3f"), format(three_month, ".3f"), format(six_month, ".3f"),
                                 format(one_year, ".3f"), format(two_year, ".3f"), format(three_year, ".3f"),
                                 format(five_year, ".3f"), format(ten_year, ".3f")],
                                [format(over_night1, ".3f"), format(one_week1, ".3f"), format(one_month1, ".3f"),
                                 format(two_month1, ".3f"), format(three_month1, ".3f"), format(six_month1, ".3f"),
                                 format(one_year1, ".3f"), format(two_year1, ".3f"), format(three_year1, ".3f"),
                                 format(five_year1, ".3f"), format(ten_year1, ".3f")],
                                [format(over_night2, ".3f"), format(one_week2, ".3f"), format(one_month2, ".3f"),
                                 format(two_month2, ".3f"), format(three_month2, ".3f"), format(six_month2, ".3f"),
                                 format(one_year2, ".3f"), format(two_year2, ".3f"), format(three_year2, ".3f"),
                                 format(five_year2, ".3f"), format(ten_year2, ".3f")],
                                [format(over_night3, ".3f"), format(one_week3, ".3f"), format(one_month3, ".3f"),
                                 format(two_month3, ".3f"), format(three_month3, ".3f"), format(six_month3, ".3f"),
                                 format(one_year3, ".3f"), format(two_year3, ".3f"), format(three_year3, ".3f"),
                                 format(five_year3, ".3f"), format(ten_year3, ".3f")],
                                [format(over_night4, ".3f"), format(one_week4, ".3f"), format(one_month4, ".3f"),
                                 format(two_month4, ".3f"), format(three_month4, ".3f"), format(six_month4, ".3f"),
                                 format(one_year4, ".3f"), format(two_year4, ".3f"), format(three_year4, ".3f"),
                                 format(five_year4, ".3f"), format(ten_year4, ".3f")]],

                                line_color='white',
                                # 2-D list of colors for alternating rows
                                fill_color='white',
                                align=['left', 'center'],
                                font=dict(color='darkslategray', size=13)

                            ))

                        ])
                        fig.update_layout(margin=dict(l=1, r=40, b=1, t=1))


                        col1.write(fig)

                    #  col1.table(df1)
                    #  col1.write( """la volatilité ATM est la valeur de volatilité qui rend le prix implicite d'une option vanille ATM égal au prix du marché de cette option.""")
                    with col3:
                        st.markdown(

                            "<h3 style='text-align: left; color: green;'>  Maturité à interpoler   </h3>",
                            unsafe_allow_html=True)
                        # st.header("")
                    with col3:
                        volatility = st.selectbox(
                            """Choisissez la paire de devise """,
                            ('ATM', '25 Delta Risk Reversal', '10 Delta Risk Reversal', '25 Delta Butterfly',
                             '10 Delta Butterfly'))

                    with col4:
                        st.markdown(

                            "<h1 style='text-align: center; color: green;'>    </h1>",
                            unsafe_allow_html=True)

                    st.header("")
                    # col4.header("")
                    # col4.header("")

                    with col4:
                        st.markdown(

                            "<h3 style='text-align: center; color: green;'>    </h3>",
                            unsafe_allow_html=True)

                    with col4:
                        c = float(st.text_input('Saisez la maturité (jours)', value=0))
                        if volatility == "ATM":
                            col3.markdown(
                                "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                                unsafe_allow_html=True)
                            data_frame = pd.DataFrame(df)
                            df.dropna(inplace=True)
                            over_night = float(data_frame["ATM"][0])
                            one_week = float(data_frame["ATM"][1])
                            one_month = float(data_frame["ATM"][2])
                            two_month = float(data_frame["ATM"][3])
                            three_month = float(data_frame["ATM"][4])
                            six_month = float(data_frame["ATM"][5])
                            one_year = float(data_frame["ATM"][6])
                            two_year = float(data_frame["ATM"][7])
                            three_year = float(data_frame["ATM"][8])
                            five_year = float(data_frame["ATM"][9])
                            ten_year = float(data_frame["ATM"][10])

                            # col4.metric(("Une semaine  "), one_year, delta_color="off")
                            X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                                 10 * 12 * 30]
                            Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                                 three_year,
                                 five_year, ten_year]
                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')

                            col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                                volatility) + " " + "qui correspond à " + " " + str(
                                format(c, ".0f")) + " " + "jours est :" + " " + str(format(y_interp(c), ".3f"))))


                        elif volatility == "25 Delta Risk Reversal":
                            col3.markdown(
                                "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                                unsafe_allow_html=True)
                            data_frame = pd.DataFrame(df)
                            over_night = float(data_frame["25 Delta Risk Reversal "][0])
                            one_week = float(data_frame["25 Delta Risk Reversal "][1])
                            one_month = float(data_frame["25 Delta Risk Reversal "][2])
                            two_month = float(data_frame["25 Delta Risk Reversal "][3])
                            three_month = float(data_frame["25 Delta Risk Reversal "][4])
                            six_month = float(data_frame["25 Delta Risk Reversal "][5])
                            one_year = float(data_frame["25 Delta Risk Reversal "][6])
                            two_year = float(data_frame["25 Delta Risk Reversal "][7])
                            three_year = float(data_frame["25 Delta Risk Reversal "][8])
                            five_year = float(data_frame["25 Delta Risk Reversal "][9])
                            ten_year = float(data_frame["25 Delta Risk Reversal "][10])
                            X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                                 10 * 12 * 30]
                            Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                                 three_year,
                                 five_year, ten_year]
                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                            col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                                volatility) + " " + "qui correspond à " + " " + str(
                                format(c, ".0f")) + " " + "jours est :" + " " + str(
                                format(y_interp(c), ".3f"))))
                        elif volatility == "10 Delta Risk Reversal":
                            col3.markdown(
                                "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                                unsafe_allow_html=True)
                            data_frame = pd.DataFrame(df)
                            over_night = float(data_frame["10 Delta Risk Reversal"][0])
                            one_week = float(data_frame["10 Delta Risk Reversal"][1])
                            one_month = float(data_frame["10 Delta Risk Reversal"][2])
                            two_month = float(data_frame["10 Delta Risk Reversal"][3])
                            three_month = float(data_frame["10 Delta Risk Reversal"][4])
                            six_month = float(data_frame["10 Delta Risk Reversal"][5])
                            one_year = float(data_frame["10 Delta Risk Reversal"][6])
                            two_year = float(data_frame["10 Delta Risk Reversal"][7])
                            three_year = float(data_frame["10 Delta Risk Reversal"][8])
                            five_year = float(data_frame["10 Delta Risk Reversal"][9])
                            ten_year = float(data_frame["10 Delta Risk Reversal"][10])
                            X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                                 10 * 12 * 30]
                            Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                                 three_year,
                                 five_year, ten_year]
                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                            col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                                volatility) + " " + "qui correspond à " + " " + str(
                                format(c, ".0f")) + " " + "jours est :" + " " + str(format(y_interp(c), ".3f"))))
                        elif volatility == "25 Delta Butterfly":
                            col3.markdown(
                                "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                                unsafe_allow_html=True)
                            data_frame = pd.DataFrame(df)
                            over_night = float(data_frame["25 Delta Butterfly"][0])
                            one_week = float(data_frame["25 Delta Butterfly"][1])
                            one_month = float(data_frame["25 Delta Butterfly"][2])
                            two_month = float(data_frame["25 Delta Butterfly"][3])
                            three_month = float(data_frame["25 Delta Butterfly"][4])
                            six_month = float(data_frame["25 Delta Butterfly"][5])
                            one_year = float(data_frame["25 Delta Butterfly"][6])
                            two_year = float(data_frame["25 Delta Butterfly"][7])
                            three_year = float(data_frame["25 Delta Butterfly"][8])
                            five_year = float(data_frame["25 Delta Butterfly"][9])
                            ten_year = float(data_frame["25 Delta Butterfly"][10])
                            X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                                 10 * 12 * 30]
                            Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                                 three_year,
                                 five_year, ten_year]
                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                            col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                                volatility) + " " + "qui correspond à " + " " + str(
                                format(c, ".0f")) + " " + "jours est :" + " " + str(
                                format(y_interp(c), ".3f"))))
                        elif volatility == "10 Delta Butterfly":
                            col3.markdown(
                                "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                                unsafe_allow_html=True)
                            data_frame = pd.DataFrame(df)
                            over_night = float(data_frame["10 Delta Butterfly"][0])
                            one_week = float(data_frame["10 Delta Butterfly"][1])
                            one_month = float(data_frame["10 Delta Butterfly"][2])
                            two_month = float(data_frame["10 Delta Butterfly"][3])
                            three_month = float(data_frame["10 Delta Butterfly"][4])
                            six_month = float(data_frame["10 Delta Butterfly"][5])
                            one_year = float(data_frame["10 Delta Butterfly"][6])
                            two_year = float(data_frame["10 Delta Butterfly"][7])
                            three_year = float(data_frame["10 Delta Butterfly"][8])
                            five_year = float(data_frame["10 Delta Butterfly"][9])
                            ten_year = float(data_frame["10 Delta Butterfly"][10])
                            X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                                 10 * 12 * 30]
                            Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                                 three_year,
                                 five_year, ten_year]
                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                            col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                                volatility) + " " + "qui correspond à " + " " + str(
                                format(c, ".0f")) + " " + "jours est :" + " " + str(
                                format(y_interp(c), ".3f"))))

                            # time.sleep(1)
                    # except:
                    #    print("An exception occurred")
                    # break
        elif paire == 'USD/CAD':
                    # s = sched.scheduler(time.time, time.sleep)

                    # def main_task():
                    # placeholder = st.empty()

                    # for seconds in range(200):
                    # while True:
                    # try:
                    # with placeholder.container():

                    workbook = xw.Book("streamlitmodifie1.xlsm")

                    df = workbook.sheets[8].range('A1').options(pd.DataFrame,
                                                                header=1,
                                                                index=True,
                                                                expand='table').value
                    df1 = df.style.highlight_null(props="color: transparent;")

                    import plotly.graph_objects as go

                    # st.header("")
                    #st.header("")
                    #col1, col2, col3 = st.columns(3)

                   # with col2:
                       # st.markdown( "<h1 style='text-align: center; color: green;'> USD/CAD </h1>",  unsafe_allow_html=True)
                      #  st.header("")

                    # with col1:
                    # st.button("Refresh")
                    #with col3:
                    #    today = date.today()
                    #    v = datetime.now().strftime("%H:%M:%S")

                      #  st.write(today, v)
                    #st.header("")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(

                            "<h3 style='text-align: left; color: green;'>  Aperçu   </h3>",
                            unsafe_allow_html=True)
                    with col1:

                        df1 = df.style.highlight_null(props="color: transparent;")

                        data_frame = pd.DataFrame(df)
                        over_night = float(data_frame["ATM"][0])
                        one_week = float(data_frame["ATM"][1])
                        one_month = float(data_frame["ATM"][2])
                        two_month = float(data_frame["ATM"][3])
                        three_month = float(data_frame["ATM"][4])
                        six_month = float(data_frame["ATM"][5])
                        one_year = float(data_frame["ATM"][6])
                        two_year = float(data_frame["ATM"][7])
                        three_year = float(data_frame["ATM"][8])
                        five_year = float(data_frame["ATM"][9])

                        #############################################################"
                        over_night1 = float(data_frame["25 Delta Risk Reversal "][0])
                        one_week1 = float(data_frame["25 Delta Risk Reversal "][1])
                        one_month1 = float(data_frame["25 Delta Risk Reversal "][2])
                        two_month1 = float(data_frame["25 Delta Risk Reversal "][3])
                        three_month1 = float(data_frame["25 Delta Risk Reversal "][4])
                        six_month1 = float(data_frame["25 Delta Risk Reversal "][5])
                        one_year1 = float(data_frame["25 Delta Risk Reversal "][6])
                        two_year1 = float(data_frame["25 Delta Risk Reversal "][7])
                        three_year1 = float(data_frame["25 Delta Risk Reversal "][8])
                        five_year1 = float(data_frame["25 Delta Risk Reversal "][9])

                        ################################################################""
                        over_night2 = float(data_frame["10 Delta Risk Reversal"][0])
                        one_week2 = float(data_frame["10 Delta Risk Reversal"][1])
                        one_month2 = float(data_frame["10 Delta Risk Reversal"][2])
                        two_month2 = float(data_frame["10 Delta Risk Reversal"][3])
                        three_month2 = float(data_frame["10 Delta Risk Reversal"][4])
                        six_month2 = float(data_frame["10 Delta Risk Reversal"][5])
                        one_year2 = float(data_frame["10 Delta Risk Reversal"][6])
                        two_year2 = float(data_frame["10 Delta Risk Reversal"][7])
                        three_year2 = float(data_frame["10 Delta Risk Reversal"][8])
                        five_year2 = float(data_frame["10 Delta Risk Reversal"][9])

                        ################################################################
                        over_night3 = float(data_frame["25 Delta Butterfly"][0])
                        one_week3 = float(data_frame["25 Delta Butterfly"][1])
                        one_month3 = float(data_frame["25 Delta Butterfly"][2])
                        two_month3 = float(data_frame["25 Delta Butterfly"][3])
                        three_month3 = float(data_frame["25 Delta Butterfly"][4])
                        six_month3 = float(data_frame["25 Delta Butterfly"][5])
                        one_year3 = float(data_frame["25 Delta Butterfly"][6])
                        two_year3 = float(data_frame["25 Delta Butterfly"][7])
                        three_year3 = float(data_frame["25 Delta Butterfly"][8])
                        five_year3 = float(data_frame["25 Delta Butterfly"][9])

                        ##################################################################
                        over_night4 = float(data_frame["10 Delta Butterfly"][0])
                        one_week4 = float(data_frame["10 Delta Butterfly"][1])
                        one_month4 = float(data_frame["10 Delta Butterfly"][2])
                        two_month4 = float(data_frame["10 Delta Butterfly"][3])
                        three_month4 = float(data_frame["10 Delta Butterfly"][4])
                        six_month4 = float(data_frame["10 Delta Butterfly"][5])
                        one_year4 = float(data_frame["10 Delta Butterfly"][6])
                        two_year4 = float(data_frame["10 Delta Butterfly"][7])
                        three_year4 = float(data_frame["10 Delta Butterfly"][8])
                        five_year4 = float(data_frame["10 Delta Butterfly"][9])


                        headerColor = 'grey'
                        rowEvenColor = 'lightgrey'
                        rowOddColor = 'white'
                        import plotly.graph_objects as go

                        fig = go.Figure(data=[go.Table(
                            header=dict(values=["Volatilité", "ATM", "25D RR", "10D RR", "25D BF", "10D BF"],
                                        align=['left', 'center'],
                                        line_color='darkslategray',
                                        fill_color='#54872E',
                                        font=dict(color='black', size=17)),

                            cells=dict(values=[
                                ['Pendant la nuit', 'Une semaine', 'Un mois', 'Deux mois', 'Trois Mois', 'Six mois',
                                 'Un an', 'Deux ans ',
                                 'Trois ans', 'Cinq ans'],

                                [format(over_night, ".3f"), format(one_week, ".3f"), format(one_month, ".3f"),
                                 format(two_month, ".3f"), format(three_month, ".3f"), format(six_month, ".3f"),
                                 format(one_year, ".3f"), format(two_year, ".3f"), format(three_year, ".3f"),
                                 format(five_year, ".3f")],
                                [format(over_night1, ".3f"), format(one_week1, ".3f"), format(one_month1, ".3f"),
                                 format(two_month1, ".3f"), format(three_month1, ".3f"), format(six_month1, ".3f"),
                                 format(one_year1, ".3f"), format(two_year1, ".3f"), format(three_year1, ".3f"),
                                 format(five_year1, ".3f")],
                                [format(over_night2, ".3f"), format(one_week2, ".3f"), format(one_month2, ".3f"),
                                 format(two_month2, ".3f"), format(three_month2, ".3f"), format(six_month2, ".3f"),
                                 format(one_year2, ".3f"), format(two_year2, ".3f"), format(three_year2, ".3f"),
                                 format(five_year2, ".3f")],
                                [format(over_night3, ".3f"), format(one_week3, ".3f"), format(one_month3, ".3f"),
                                 format(two_month3, ".3f"), format(three_month3, ".3f"), format(six_month3, ".3f"),
                                 format(one_year3, ".3f"), format(two_year3, ".3f"), format(three_year3, ".3f"),
                                 format(five_year3, ".3f")],
                                [format(over_night4, ".3f"), format(one_week4, ".3f"), format(one_month4, ".3f"),
                                 format(two_month4, ".3f"), format(three_month4, ".3f"), format(six_month4, ".3f"),
                                 format(one_year4, ".3f"), format(two_year4, ".3f"), format(three_year4, ".3f"),
                                 format(five_year4, ".3f")]],

                                line_color='white',
                                # 2-D list of colors for alternating rows
                                fill_color='white',
                                align=['left', 'center'],
                                font=dict(color='darkslategray', size=13)

                            ))

                        ])
                        fig.update_layout(margin=dict(l=1, r=40, b=1, t=1))

                        # fig2=fig.replace(np.nan, '', regex=True)
                        col1.write(fig)

                    #  col1.table(df1)
                    #  col1.write( """la volatilité ATM est la valeur de volatilité qui rend le prix implicite d'une option vanille ATM égal au prix du marché de cette option.""")
                    with col3:
                        st.markdown(

                            "<h3 style='text-align: left; color: green;'>  Maturité à interpoler   </h3>",
                            unsafe_allow_html=True)
                        # st.header("")
                    with col3:
                        volatility = st.selectbox(
                        """Choisissez le type de volatilité   """,
                            ('ATM', '25 Delta Risk Reversal', '10 Delta Risk Reversal', '25 Delta Butterfly',
                             '10 Delta Butterfly'))

                    with col4:
                        st.markdown(

                            "<h1 style='text-align: center; color: green;'>    </h1>",
                            unsafe_allow_html=True)

                    st.header("")
                    # col4.header("")
                    # col4.header("")

                    with col4:
                        st.markdown(

                            "<h3 style='text-align: center; color: green;'>    </h3>",
                            unsafe_allow_html=True)

                    with col4:
                        c = float(st.text_input('Saisez la maturité (jours)', value=0))
                        if volatility == "ATM":
                            col3.markdown(
                                "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                                unsafe_allow_html=True)
                            data_frame = pd.DataFrame(df)
                            df.dropna(inplace=True)
                            over_night = float(data_frame["ATM"][0])
                            one_week = float(data_frame["ATM"][1])
                            one_month = float(data_frame["ATM"][2])
                            two_month = float(data_frame["ATM"][3])
                            three_month = float(data_frame["ATM"][4])
                            six_month = float(data_frame["ATM"][5])
                            one_year = float(data_frame["ATM"][6])
                            two_year = float(data_frame["ATM"][7])
                            three_year = float(data_frame["ATM"][8])
                            five_year = float(data_frame["ATM"][9])


                            # col4.metric(("Une semaine  "), one_year, delta_color="off")
                            X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30]
                            Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                                 three_year,
                                 five_year]
                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')

                            col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                                volatility) + " " + "qui correspond à " + " " + str(
                                format(c, ".0f")) + " " + "jours est :" + " " + str(format(y_interp(c), ".3f"))))


                        elif volatility == "25 Delta Risk Reversal":
                            col3.markdown(
                                "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                                unsafe_allow_html=True)
                            data_frame = pd.DataFrame(df)
                            over_night = float(data_frame["25 Delta Risk Reversal "][0])
                            one_week = float(data_frame["25 Delta Risk Reversal "][1])
                            one_month = float(data_frame["25 Delta Risk Reversal "][2])
                            two_month = float(data_frame["25 Delta Risk Reversal "][3])
                            three_month = float(data_frame["25 Delta Risk Reversal "][4])
                            six_month = float(data_frame["25 Delta Risk Reversal "][5])
                            one_year = float(data_frame["25 Delta Risk Reversal "][6])
                            two_year = float(data_frame["25 Delta Risk Reversal "][7])
                            three_year = float(data_frame["25 Delta Risk Reversal "][8])
                            five_year = float(data_frame["25 Delta Risk Reversal "][9])

                            X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30]
                            Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                                 three_year,
                                 five_year]
                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                            col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                                volatility) + " " + "qui correspond à " + " " + str(
                                format(c, ".0f")) + " " + "jours est :" + " " + str(
                                format(y_interp(c), ".3f"))))
                        elif volatility == "10 Delta Risk Reversal":
                            col3.markdown(
                                "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                                unsafe_allow_html=True)
                            data_frame = pd.DataFrame(df)
                            over_night = float(data_frame["10 Delta Risk Reversal"][0])
                            one_week = float(data_frame["10 Delta Risk Reversal"][1])
                            one_month = float(data_frame["10 Delta Risk Reversal"][2])
                            two_month = float(data_frame["10 Delta Risk Reversal"][3])
                            three_month = float(data_frame["10 Delta Risk Reversal"][4])
                            six_month = float(data_frame["10 Delta Risk Reversal"][5])
                            one_year = float(data_frame["10 Delta Risk Reversal"][6])
                            two_year = float(data_frame["10 Delta Risk Reversal"][7])
                            three_year = float(data_frame["10 Delta Risk Reversal"][8])
                            five_year = float(data_frame["10 Delta Risk Reversal"][9])

                            X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30]
                            Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                                 three_year,
                                 five_year]
                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                            col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                                volatility) + " " + "qui correspond à " + " " + str(
                                format(c, ".0f")) + " " + "jours est :" + " " + str(format(y_interp(c), ".3f"))))
                        elif volatility == "25 Delta Butterfly":
                            col3.markdown(
                                "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                                unsafe_allow_html=True)
                            data_frame = pd.DataFrame(df)
                            over_night = float(data_frame["25 Delta Butterfly"][0])
                            one_week = float(data_frame["25 Delta Butterfly"][1])
                            one_month = float(data_frame["25 Delta Butterfly"][2])
                            two_month = float(data_frame["25 Delta Butterfly"][3])
                            three_month = float(data_frame["25 Delta Butterfly"][4])
                            six_month = float(data_frame["25 Delta Butterfly"][5])
                            one_year = float(data_frame["25 Delta Butterfly"][6])
                            two_year = float(data_frame["25 Delta Butterfly"][7])
                            three_year = float(data_frame["25 Delta Butterfly"][8])
                            five_year = float(data_frame["25 Delta Butterfly"][9])

                            X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30]
                            Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                                 three_year,
                                 five_year]
                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                            col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                                volatility) + " " + "qui correspond à " + " " + str(
                                format(c, ".0f")) + " " + "jours est :" + " " + str(
                                format(y_interp(c), ".3f"))))
                        elif volatility == "10 Delta Butterfly":
                            col3.markdown(
                                "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                                unsafe_allow_html=True)
                            data_frame = pd.DataFrame(df)
                            over_night = float(data_frame["10 Delta Butterfly"][0])
                            one_week = float(data_frame["10 Delta Butterfly"][1])
                            one_month = float(data_frame["10 Delta Butterfly"][2])
                            two_month = float(data_frame["10 Delta Butterfly"][3])
                            three_month = float(data_frame["10 Delta Butterfly"][4])
                            six_month = float(data_frame["10 Delta Butterfly"][5])
                            one_year = float(data_frame["10 Delta Butterfly"][6])
                            two_year = float(data_frame["10 Delta Butterfly"][7])
                            three_year = float(data_frame["10 Delta Butterfly"][8])
                            five_year = float(data_frame["10 Delta Butterfly"][9])

                            X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30]
                            Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                                 three_year,
                                 five_year]
                            y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                            col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                                volatility) + " " + "qui correspond à " + " " + str(
                                format(c, ".0f")) + " " + "jours est :" + " " + str(
                                format(y_interp(c), ".3f"))))

                            # time.sleep(1)
                    # except:
                    #    print("An exception occurred")
                    # break
        elif paire == 'USD/JPY':
                # s = sched.scheduler(time.time, time.sleep)

                # def main_task():
                # placeholder = st.empty()

                # for seconds in range(200):
                # while True:
                # try:
                # with placeholder.container():

                workbook = xw.Book("streamlitmodifie1.xlsm")

                df = workbook.sheets[9].range('A1').options(pd.DataFrame,
                                                            header=1,
                                                            index=True,
                                                            expand='table').value
                df1 = df.style.highlight_null(props="color: transparent;")

                import plotly.graph_objects as go

                # st.header("")
               # st.header("")
              #  col1, col2, col3 = st.columns(3)

                #with col2:
                  #  st.markdown( "<h1 style='text-align: center; color: green;'> USD/JPY </h1>",unsafe_allow_html=True)
                    #st.header("")

                # with col1:
                # st.button("Refresh")
                #with col3:
                   # today = date.today()
                  #  v = datetime.now().strftime("%H:%M:%S")

                    #st.write(today, v)
               # st.header("")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(

                        "<h3 style='text-align: left; color: green;'>  Aperçu   </h3>",
                        unsafe_allow_html=True)
                with col1:

                    df1 = df.style.highlight_null(props="color: transparent;")

                    data_frame = pd.DataFrame(df)
                    over_night = float(data_frame["ATM"][0])
                    one_week = float(data_frame["ATM"][1])
                    one_month = float(data_frame["ATM"][2])
                    two_month = float(data_frame["ATM"][3])
                    three_month = float(data_frame["ATM"][4])
                    six_month = float(data_frame["ATM"][5])
                    one_year = float(data_frame["ATM"][6])
                    two_year = float(data_frame["ATM"][7])
                    three_year = float(data_frame["ATM"][8])
                    five_year = float(data_frame["ATM"][9])
                    ten_year = float(data_frame["ATM"][10])
                    #############################################################"
                    over_night1 = float(data_frame["25 Delta Risk Reversal "][0])
                    one_week1 = float(data_frame["25 Delta Risk Reversal "][0])
                    one_month1 = float(data_frame["25 Delta Risk Reversal "][1])
                    two_month1 = float(data_frame["25 Delta Risk Reversal "][2])
                    three_month1 = float(data_frame["25 Delta Risk Reversal "][3])
                    six_month1 = float(data_frame["25 Delta Risk Reversal "][4])
                    one_year1 = float(data_frame["25 Delta Risk Reversal "][5])
                    two_year1 = float(data_frame["25 Delta Risk Reversal "][6])
                    three_year1 = float(data_frame["25 Delta Risk Reversal "][7])
                    five_year1 = float(data_frame["25 Delta Risk Reversal "][8])
                    ten_year1 = float(data_frame["25 Delta Risk Reversal "][10])
                    ################################################################""
                    over_night2 = float(data_frame["10 Delta Risk Reversal"][0])
                    one_week2 = float(data_frame["10 Delta Risk Reversal"][1])
                    one_month2 = float(data_frame["10 Delta Risk Reversal"][2])
                    two_month2 = float(data_frame["10 Delta Risk Reversal"][3])
                    three_month2 = float(data_frame["10 Delta Risk Reversal"][4])
                    six_month2 = float(data_frame["10 Delta Risk Reversal"][5])
                    one_year2 = float(data_frame["10 Delta Risk Reversal"][6])
                    two_year2 = float(data_frame["10 Delta Risk Reversal"][7])
                    three_year2 = float(data_frame["10 Delta Risk Reversal"][8])
                    five_year2 = float(data_frame["10 Delta Risk Reversal"][9])
                    ten_year2 = float(data_frame["10 Delta Risk Reversal"][10])
                    ################################################################
                    over_night3 = float(data_frame["25 Delta Butterfly"][0])
                    one_week3 = float(data_frame["25 Delta Butterfly"][1])
                    one_month3 = float(data_frame["25 Delta Butterfly"][2])
                    two_month3 = float(data_frame["25 Delta Butterfly"][3])
                    three_month3 = float(data_frame["25 Delta Butterfly"][4])
                    six_month3 = float(data_frame["25 Delta Butterfly"][5])
                    one_year3 = float(data_frame["25 Delta Butterfly"][6])
                    two_year3 = float(data_frame["25 Delta Butterfly"][7])
                    three_year3 = float(data_frame["25 Delta Butterfly"][8])
                    five_year3 = float(data_frame["25 Delta Butterfly"][9])
                    ten_year3 = float(data_frame["25 Delta Butterfly"][10])
                    ##################################################################
                    over_night4 = float(data_frame["10 Delta Butterfly"][0])
                    one_week4 = float(data_frame["10 Delta Butterfly"][1])
                    one_month4 = float(data_frame["10 Delta Butterfly"][2])
                    two_month4 = float(data_frame["10 Delta Butterfly"][3])
                    three_month4 = float(data_frame["10 Delta Butterfly"][4])
                    six_month4 = float(data_frame["10 Delta Butterfly"][5])
                    one_year4 = float(data_frame["10 Delta Butterfly"][6])
                    two_year4 = float(data_frame["10 Delta Butterfly"][7])
                    three_year4 = float(data_frame["10 Delta Butterfly"][8])
                    five_year4 = float(data_frame["10 Delta Butterfly"][9])
                    ten_year4 = float(data_frame["10 Delta Butterfly"][10])

                    headerColor = 'grey'
                    rowEvenColor = 'lightgrey'
                    rowOddColor = 'white'
                    import plotly.graph_objects as go

                    fig = go.Figure(data=[go.Table(
                        header=dict(values=["Volatilité", "ATM", "25D RR", "10D RR", "25D BF", "10D BF"],
                                    align=['left', 'center'],
                                    line_color='darkslategray',
                                    fill_color='#54872E',
                                    font=dict(color='black', size=17)),

                        cells=dict(values=[
                            ['Pendant la nuit', 'Une semaine', 'Un mois', 'Deux mois', 'Trois Mois', 'Six mois',
                             'Un an', 'Deux ans ',
                             'Trois ans', 'Cinq ans', 'Dix ans'],

                            [format(over_night, ".3f"), format(one_week, ".3f"), format(one_month, ".3f"),
                             format(two_month, ".3f"), format(three_month, ".3f"), format(six_month, ".3f"),
                             format(one_year, ".3f"), format(two_year, ".3f"), format(three_year, ".3f"),
                             format(five_year, ".3f"), format(ten_year, ".3f")],
                            [format(over_night1, ".3f"), format(one_week1, ".3f"), format(one_month1, ".3f"),
                             format(two_month1, ".3f"), format(three_month1, ".3f"), format(six_month1, ".3f"),
                             format(one_year1, ".3f"), format(two_year1, ".3f"), format(three_year1, ".3f"),
                             format(five_year1, ".3f"), format(ten_year1, ".3f")],
                            [format(over_night2, ".3f"), format(one_week2, ".3f"), format(one_month2, ".3f"),
                             format(two_month2, ".3f"), format(three_month2, ".3f"), format(six_month2, ".3f"),
                             format(one_year2, ".3f"), format(two_year2, ".3f"), format(three_year2, ".3f"),
                             format(five_year2, ".3f"), format(ten_year2, ".3f")],
                            [format(over_night3, ".3f"), format(one_week3, ".3f"), format(one_month3, ".3f"),
                             format(two_month3, ".3f"), format(three_month3, ".3f"), format(six_month3, ".3f"),
                             format(one_year3, ".3f"), format(two_year3, ".3f"), format(three_year3, ".3f"),
                             format(five_year3, ".3f"), format(ten_year3, ".3f")],
                            [format(over_night4, ".3f"), format(one_week4, ".3f"), format(one_month4, ".3f"),
                             format(two_month4, ".3f"), format(three_month4, ".3f"), format(six_month4, ".3f"),
                             format(one_year4, ".3f"), format(two_year4, ".3f"), format(three_year4, ".3f"),
                             format(five_year4, ".3f"), format(ten_year4, ".3f")]],

                            line_color='white',
                            # 2-D list of colors for alternating rows
                            fill_color='white',
                            align=['left', 'center'],
                            font=dict(color='darkslategray', size=13)

                        ))

                    ])
                    fig.update_layout(margin=dict(l=1, r=40, b=1, t=1))

                    # fig2=fig.replace(np.nan, '', regex=True)
                    col1.write(fig)

                #  col1.table(df1)
                #  col1.write( """la volatilité ATM est la valeur de volatilité qui rend le prix implicite d'une option vanille ATM égal au prix du marché de cette option.""")
                with col3:
                    st.markdown(

                        "<h3 style='text-align: left; color: green;'>  Maturité à interpoler   </h3>",
                        unsafe_allow_html=True)
                    # st.header("")
                with col3:
                    volatility = st.selectbox(
                        """Choisissez la paire de devise """,
                        ('ATM', '25 Delta Risk Reversal', '10 Delta Risk Reversal', '25 Delta Butterfly',
                         '10 Delta Butterfly'))

                with col4:
                    st.markdown(

                        "<h1 style='text-align: center; color: green;'>    </h1>",
                        unsafe_allow_html=True)

                st.header("")
                # col4.header("")
                # col4.header("")

                with col4:
                    st.markdown(

                        "<h3 style='text-align: center; color: green;'>    </h3>",
                        unsafe_allow_html=True)

                with col4:
                    c = float(st.text_input('Saisez la maturité (jours)', value=0))
                    if volatility == "ATM":
                        col3.markdown(
                            "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                            unsafe_allow_html=True)
                        data_frame = pd.DataFrame(df)
                        df.dropna(inplace=True)
                        over_night = float(data_frame["ATM"][0])
                        one_week = float(data_frame["ATM"][1])
                        one_month = float(data_frame["ATM"][2])
                        two_month = float(data_frame["ATM"][3])
                        three_month = float(data_frame["ATM"][4])
                        six_month = float(data_frame["ATM"][5])
                        one_year = float(data_frame["ATM"][6])
                        two_year = float(data_frame["ATM"][7])
                        three_year = float(data_frame["ATM"][8])
                        five_year = float(data_frame["ATM"][9])
                        ten_year = float(data_frame["ATM"][10])

                        # col4.metric(("Une semaine  "), one_year, delta_color="off")
                        X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                             10 * 12 * 30]
                        Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                             three_year,
                             five_year, ten_year]
                        y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')

                        col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                            volatility) + " " + "qui correspond à " + " " + str(
                            format(c, ".0f")) + " " + "jours est :" + " " + str(format(y_interp(c), ".3f"))))


                    elif volatility == "25 Delta Risk Reversal":
                        col3.markdown(
                            "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                            unsafe_allow_html=True)
                        data_frame = pd.DataFrame(df)
                        over_night = float(data_frame["25 Delta Risk Reversal "][0])
                        one_week = float(data_frame["25 Delta Risk Reversal "][1])
                        one_month = float(data_frame["25 Delta Risk Reversal "][2])
                        two_month = float(data_frame["25 Delta Risk Reversal "][3])
                        three_month = float(data_frame["25 Delta Risk Reversal "][4])
                        six_month = float(data_frame["25 Delta Risk Reversal "][5])
                        one_year = float(data_frame["25 Delta Risk Reversal "][6])
                        two_year = float(data_frame["25 Delta Risk Reversal "][7])
                        three_year = float(data_frame["25 Delta Risk Reversal "][8])
                        five_year = float(data_frame["25 Delta Risk Reversal "][9])
                        ten_year = float(data_frame["25 Delta Risk Reversal "][10])
                        X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                             10 * 12 * 30]
                        Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                             three_year,
                             five_year, ten_year]
                        y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                        col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                            volatility) + " " + "qui correspond à " + " " + str(
                            format(c, ".0f")) + " " + "jours est :" + " " + str(
                            format(y_interp(c), ".3f"))))
                    elif volatility == "10 Delta Risk Reversal":
                        col3.markdown(
                            "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                            unsafe_allow_html=True)
                        data_frame = pd.DataFrame(df)
                        over_night = float(data_frame["10 Delta Risk Reversal"][0])
                        one_week = float(data_frame["10 Delta Risk Reversal"][1])
                        one_month = float(data_frame["10 Delta Risk Reversal"][2])
                        two_month = float(data_frame["10 Delta Risk Reversal"][3])
                        three_month = float(data_frame["10 Delta Risk Reversal"][4])
                        six_month = float(data_frame["10 Delta Risk Reversal"][5])
                        one_year = float(data_frame["10 Delta Risk Reversal"][6])
                        two_year = float(data_frame["10 Delta Risk Reversal"][7])
                        three_year = float(data_frame["10 Delta Risk Reversal"][8])
                        five_year = float(data_frame["10 Delta Risk Reversal"][9])
                        ten_year = float(data_frame["10 Delta Risk Reversal"][10])
                        X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                             10 * 12 * 30]
                        Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                             three_year,
                             five_year, ten_year]
                        y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                        col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                            volatility) + " " + "qui correspond à " + " " + str(
                            format(c, ".0f")) + " " + "jours est :" + " " + str(format(y_interp(c), ".3f"))))
                    elif volatility == "25 Delta Butterfly":
                        col3.markdown(
                            "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                            unsafe_allow_html=True)
                        data_frame = pd.DataFrame(df)
                        over_night = float(data_frame["25 Delta Butterfly"][0])
                        one_week = float(data_frame["25 Delta Butterfly"][1])
                        one_month = float(data_frame["25 Delta Butterfly"][2])
                        two_month = float(data_frame["25 Delta Butterfly"][3])
                        three_month = float(data_frame["25 Delta Butterfly"][4])
                        six_month = float(data_frame["25 Delta Butterfly"][5])
                        one_year = float(data_frame["25 Delta Butterfly"][6])
                        two_year = float(data_frame["25 Delta Butterfly"][7])
                        three_year = float(data_frame["25 Delta Butterfly"][8])
                        five_year = float(data_frame["25 Delta Butterfly"][9])
                        ten_year = float(data_frame["25 Delta Butterfly"][10])
                        X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                             10 * 12 * 30]
                        Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                             three_year,
                             five_year, ten_year]
                        y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                        col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                            volatility) + " " + "qui correspond à " + " " + str(
                            format(c, ".0f")) + " " + "jours est :" + " " + str(
                            format(y_interp(c), ".3f"))))
                    elif volatility == "10 Delta Butterfly":
                        col3.markdown(
                            "<h3 style='text-align: left; color: green;'>  Resultat  </h3>",
                            unsafe_allow_html=True)
                        data_frame = pd.DataFrame(df)
                        over_night = float(data_frame["10 Delta Butterfly"][0])
                        one_week = float(data_frame["10 Delta Butterfly"][1])
                        one_month = float(data_frame["10 Delta Butterfly"][2])
                        two_month = float(data_frame["10 Delta Butterfly"][3])
                        three_month = float(data_frame["10 Delta Butterfly"][4])
                        six_month = float(data_frame["10 Delta Butterfly"][5])
                        one_year = float(data_frame["10 Delta Butterfly"][6])
                        two_year = float(data_frame["10 Delta Butterfly"][7])
                        three_year = float(data_frame["10 Delta Butterfly"][8])
                        five_year = float(data_frame["10 Delta Butterfly"][9])
                        ten_year = float(data_frame["10 Delta Butterfly"][10])
                        X = [1, 7, 30, 2 * 30, 3 * 30, 6 * 30, 12 * 30, 12 * 2 * 30, 12 * 3 * 30, 12 * 5 * 30,
                             10 * 12 * 30]
                        Y = [over_night, one_week, one_month, two_month, three_month, six_month, one_year, two_year,
                             three_year,
                             five_year, ten_year]
                        y_interp = interpolate.interp1d(X, Y, fill_value='extrapolate')
                        col3.write(("Pour" + " " + str(paire) + " " + "la volatilité " + " " + str(
                            volatility) + " " + "qui correspond à " + " " + str(
                            format(c, ".0f")) + " " + "jours est :" + " " + str(
                            format(y_interp(c), ".3f"))))















    #################################################################################################
    elif menu_id == 'Produits':

        selected = om(menu_title=None,
                      options=['IMPORT', 'EXPORT'],
                      menu_icon='cast', default_index=1, orientation='horizontal',
                      styles={
                          "container": {"padding": "0!important"},
                          "icon": {"color": "", "font-size": "15px"},
                          "nav-link": {"font-size": "12px", "text-align": "middle", "margin": "0px",
                                       "--hover-color": "#eee"},
                          "nav-link-selected": {"background-color": "#4a6da4"}, }
                      )

        if selected == "IMPORT":
            # with st.sidebar:
            # choose = option_menu(menu_title=None,
            #                      options=["Forward","Put Vanille", "Put Participatif ",
            #                               "Tunnel Symétrique", "Tunnel Asymétrique"],
            #                      # icons=['house', 'camera fill', 'kanban', 'book','person lines fill'],
            #                      menu_icon="cast", default_index=1,
            #                      styles={
            #                          "container": {"padding": "5!important", "background-color": "#FFFFFFF"},
            #                          "icon": {"color": "#6cb44c", "font-size": "15px"},
            #                          "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px",
            #                                       "--hover-color": "#6cb44c"},  # eee
            #                          "nav-link-selected": {"background-color": "#6cb44c"},  # DFE8ED
            #
            #                      }
            #                      )

            choose = st.radio(
                label="Choisissez le type de la  stratégie ", horizontal=True, options=
                ('Forward', 'Call Vanille', 'Call Participatif', 'Tunnel Symétrique', 'Tunnel Asymétrique'))
            st.header("")
            if choose == "Forward":
                genre = st.radio(
                    label="Choisissez le type de la  stratégie ", horizontal=True,
                    options=('Stratégie payante', 'Stratégie gratuite'))
                if genre == 'Stratégie payante':
                    st.header("")
                    # st.markdown("-------------------------------------------------------------------------------------------------")

                    st.header("")
                    # st.markdown("-------------------------------------------------------------------------------------------------")

                    st.info("Veuillez  remplir les champs vides")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/MAD', 'USD/MAD'))

                    with col2:

                        N = float(st.text_input('Nominal', value=140000))
                    with col3:
                        s = float(st.text_input('Spot', value=10.090))

                    # pourcentage_paricipation = float(st.text_input('pourcentage de paricipation (%)', value=0))
                    col5, col6, col7, col8 = st.columns(4)
                    with col4:

                        k = float(st.text_input('Strike', value=10.02))
                    with col5:

                        rd = float(st.text_input('Taux doméstique (%)', value=5))
                    with  col6:
                        rf = float(st.text_input('Taux étranger (%)', value=4))

                    with col7:
                        exercise_date = st.date_input('Maturité', min_value=datetime.today() + timedelta(days=1),
                                                      value=datetime.today() + timedelta(days=365))
                    # current_date = st.date_input('Trade date')
                    # echeance=abs(Date-current_date)
                    # v=st.text_input('date',echeance)
                    st.info("(*) : Obligatoire")
                    # Formating selected model parameters
                    if st.button(f"Calculer la prime"):

                        rd1 = rd / 100
                        rf1 = rf / 100
                        T = (exercise_date - datetime.now().date()).days
                        T1 = T / 365

                        # calcul le prix de l'option

                        F = cours_terme(s, T1, rd1, rf1)
                        forward = (k - F) * np.exp(-(rd1 - rf1) * T1)
                        f2 = forward
                        f = N * forward * s

                        col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns([1, 60, 1, 1, 1, 1, 1, 1, 1])
                        with col2:
                            if ticker == 'EUR/MAD':
                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en valeur : " + format(
                                        f2, ".3f") + " </h2>",
                                    unsafe_allow_html=True)
                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
                                        f, ".3f") + " " + "DH" + " </h2>",
                                    unsafe_allow_html=True)

                            elif ticker == 'USD/MAD':
                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en valeur : " + format(
                                        f2, ".3f") + " </h2>",
                                    unsafe_allow_html=True)
                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
                                        f, ".3f") + " " + "DH" + " </h2>",
                                    unsafe_allow_html=True)

                    if st.button(f'Information sur opération '):
                        rd1 = rd / 100
                        rf1 = rf / 100
                        T = (exercise_date - datetime.now().date()).days
                        T1 = T / 365
                        F = cours_terme(s, T1, rd1, rf1)
                        forward = (k - F) * np.exp(-(rd1 - rf1) * T1)
                        f2 = forward
                        f = N * forward * s

                        df = pd.DataFrame(
                            [" Forward ", ticker, format(N, ".2f"), format(s, ".4f"), format(k, ".4f"),
                             format(rd, ".0f") + str("%"), format(rf, ".0f") + str("%"),
                             exercise_date, format(f, ".2f")],
                            index=pd.Index(
                                ["Produit", " Devise", "Nominal", "Spot", "Strike", "Taux doméstique",
                                 "Taux étranger"
                                    , "Maturité", "Prime à payer"]))

                        tdf = df.T
                        # CSS to inject contained in a string
                        hide_table_row_index = """
                                                                                                               <style>
                                                                                                               tbody th {display:none}
                                                                                                               .blank {display:none}
                                                                                                               </style>
                                                                                                               """

                        # Inject CSS with Markdown
                        ############ table recap

                        st.markdown(
                            "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                            unsafe_allow_html=True)

                        st.markdown(hide_table_row_index, unsafe_allow_html=True)
                        st.table(tdf)
                        st.session_state['button'] = False

                        ######### detail

                        st.markdown(
                            "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                            unsafe_allow_html=True)

                        st.write("- Spot " + " " + "  " + ":" + str(format(s, ".4f")))
                        st.write("- Taux doméstique:" + "   " + str(rd) + " " + "%")
                        st.write("- Taux étranger:" + "   " + str(rf) + " " + "%")

                        st.markdown(
                            "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                            unsafe_allow_html=True)

                        st.write("- si  " + "   " + str(ticker), " " + " <" + str(format(k, ".4f")))
                        st.write(" Achat  " + str(format(N, ".2f")), " à " + " " + str(format(k, ".4f")))
                        st.write("- si  " + "   " + str(ticker), " " + " >=" + str(format(k, ".4f")))
                        st.write(" Achat  " + " " + str(format(N, ".2f")), " à" + " " + str(format(k, ".4f")))
                elif genre == 'Stratégie gratuite':
                    st.header("")
                    # st.markdown("-------------------------------------------------------------------------------------------------")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/MAD', 'USD/MAD'))
                        # st.markdown(f" Paire  : {ticker}")
                    with col2:
                        N = float(st.text_input('Nominal', value=140000))
                        # st.markdown(f"Nominal : {N}")
                    with col3:
                        s = float(st.text_input('Spot', value=10.5870))
                        # st.markdown(f"Spot  : {s}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        exercise_date = st.date_input('Maturité', min_value=datetime.today() + timedelta(days=1),
                                                      value=datetime.today() + timedelta(days=365))
                        # st.markdown(f" Date de valeur :{exercise_date}")
                    with col2:
                        rd = float(st.text_input('Taux doméstique (%)',value=5))
                        # st.markdown(f"Taux doméstique : {r} %")
                    with col3:
                        rf = float(st.text_input('Taux étranger (%)',value=4))
                        # st.markdown(f"Taux étranger : {r2} %")
                    # gestion des exceptions
                    #if not all([N, s, rd, rf]):
                       # st.error("Veuillez remplir tous les champs")
                       # button = None
                    #else:
                    if st.button(f"Calculer Strike"):

                        rd1 = rd / 100
                        rf1 = rf / 100
                        T = (exercise_date - datetime.now().date()).days
                        T1 = T / 365

                        # calcul le prix de l'option
                        F = cours_terme(s, T1, rd1, rf1)

                        col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns([1, 60, 1, 1, 1, 1, 1, 1, 1])
                        with col2:
                            st.markdown(
                                "<h2 style='text-align: left ;color:;font-size:20px;'> - Le strike est : " + " " + " " + str(
                                    format(F, '.4f')) + " </h2>",
                                unsafe_allow_html=True)

                    if st.button(f'Information sur opération '):
                            rd1 = rd / 100
                            rf1 = rf / 100
                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365

                            # calcul le prix de l'option
                            F = cours_terme(s, T1, rd1, rf1)
                            df = pd.DataFrame(
                                [" Forward ", ticker, format(N, ".2f"), format(s, ".4f"), format(F, ".4f"),
                                 format(rd, ".0f") + str("%"), format(rf, ".0f") + str("%"),
                                 exercise_date],
                                index=pd.Index(["Produit", " Devise", "Nominal", "Spot", "Strike", "Taux doméstique",
                                                "Taux étranger", "Maturité"]))

                            tdf = df.T
                            # CSS to inject contained in a string
                            hide_table_row_index = """
                                                                                                           <style>
                                                                                                           tbody th {display:none}
                                                                                                           .blank {display:none}
                                                                                                           </style>
                                                                                                           """

                            # Inject CSS with Markdown
                            ############ table recap

                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                unsafe_allow_html=True)

                            st.markdown(hide_table_row_index, unsafe_allow_html=True)
                            st.table(tdf)
                            st.session_state['button'] = False

                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                unsafe_allow_html=True)

                            st.write("- Spot " + " " + "  " + ":" + format(s, ".4f"))
                            st.write("- Taux doméstique:" + "   " + str(rd) + " " + "%")
                            st.write("- Taux étranger:" + "   " + str(rf) + " " + "%")

                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                unsafe_allow_html=True)

                            st.write("- si  " + "   " + str(ticker), " " + " <" + str(format(F, ".4f")))
                            st.write("  Achat  " + str(N), " à " + " " + str(format(F, ".4f")))
                            st.write("- si  " + "   " + str(ticker), " " + " >=" + str(format(F, ".4f")))
                            st.write("  Achat  " + " " + str(N), " à" + " " + str(format(F, ".4f")))
            ###########################################################################################################################################
            # if choose == "p" :
            #     genre = st.radio(
            #         label="Choisissez le type de la  stratégie ", horizontal=True,
            #         options=('Stratégie payante', 'Stratégie gratuite'))
            #     if genre == 'Stratégie payante':
            #         st.header("")
            #         col1, col2, col3, col4 = st.columns(4)
            #         with col1:
            #             ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/MAD', 'USD/MAD'))
            #         # st.markdown(f" Paire  : {ticker}")
            #         with col2:
            #             N = st.number_input('Nominal')
            #             # st.markdown(f"Nominal : {N}")
            #         with col3:
            #             s = st.number_input('Spot')
            #             # st.markdown(f"Spot  : {s}")
            #         with col4:
            #             k = st.number_input('Strike ')
            #             # st.markdown(f"Strike  : {k}")
            #         col1, col2, col3, col4 = st.columns(4)
            #         with col1:
            #             exercise_date = st.date_input('Exercise date', min_value=datetime.today() + timedelta(days=1),
            #                                           value=datetime.today() + timedelta(days=365))
            #             # st.markdown(f" Date de valeur :{exercise_date}")
            #
            #         with col2:
            #             rd = st.number_input('Taux doméstique (%)', 0, 100, 0)
            #             # st.markdown(f"Taux doméstique : {r} %")
            #
            #         with col3:
            #             rf = st.number_input('Taux étranger (%)', 0, 100, 0)
            #             # st.markdown(f"Taux étranger : {r2} %")
            #         with col4:
            #             sigma = st.number_input('Volatilité', 0, 100, 20)
            #             # st.markdown(f" Volatilitée : {sigma} %")
            #
            #         if sigma == 0:
            #             col1, col2 = st.columns(2)
            #             with col1:
            #                 vol1 = st.number_input('Volatilité EUR/USD', 0, 100, 0)
            #                 # st.markdown(f" Volatilitée : {sigma} %")
            #             with col2:
            #                 st.markdown(
            #                     "<h2 style='text-align: left ;color:black;font-size:18px;'> Pondération " + " </h2>",
            #                     unsafe_allow_html=True)
            #                 vol2 = st.number_input('Poids EUR', 0, 100, 60)
            #                 # st.markdown(f" Volatilitée : {sigma} %")
            #             with col2:
            #                 vol3 = st.number_input('Poids USD', 0, 100, 40)
            #
            #         st.markdown( "-------------------------------------------------------------------------------------------")
            #         # gestion des exceptions
            #         if not all([N, s, k, rd, rf]):
            #             st.error("Veuillez remplir tous les champs")
            #             button = None
            #         else:
            #             button = st.button(f"Calculer le prix de l'option")
            #
            #         if st.session_state.get('button') != True:
            #             st.session_state['button'] = button
            #
            #         if st.session_state['button'] == True:
            #             # Formating selected model parameters
            #
            #             if sigma == 0:
            #                 rd1 = rd / 100
            #                 rf1 = rf / 100
            #                 sigma1 = sigma / 100
            #                 sig1 = vol1 / 100
            #                 sig2 = vol2 / 100
            #                 sig3 = vol3 / 100
            #                 T = (exercise_date - datetime.now().date()).days
            #                 T1 = T / 365
            #                 if ticker == 'EUR/MAD':
            #                     volatilite = sig1 * sig3
            #                 elif ticker == 'USD/MAD':
            #                     volatilite = sig1 * sig2
            #
            #                 col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
            #                     [1, 60, 1, 1, 1, 1, 1, 1, 1])
            #                 with col2:
            #
            #                     put = BS_PUT(s, k, T1, rd1, rf1, volatilite)
            #                     p1 = put / k
            #                     p2 = put * N * s
            #
            #                     if ticker == 'EUR/MAD':
            #
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en % du Strike : " + format(
            #                                 p1, ".3f") +  " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
            #                                 p2, ".3f") + " " + "DH" + " </h2>",
            #                             unsafe_allow_html=True)
            #
            #                     elif ticker == 'USD/MAD':
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en % du Nominal : " + format(
            #                                 p1, ".3f") + " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
            #                                 p2, ".3f") + " " + "DH" + " </h2>",
            #                             unsafe_allow_html=True)
            #
            #                 if st.button(f'Information sur opération '):
            #
            #                     volatilite1 = volatilite * 100
            #                     df = pd.DataFrame(
            #                         ["Put Vanille", ticker, format(N, ".2f"), format(s, ".4f"), format(k, ".4f"),
            #                          format(rd, ".0f") + str("%"), format(rf, ".0f") + str("%"),
            #                          format(volatilite1, ".0f") + str("%"), exercise_date, format(p2, ".2f")],
            #                         index=pd.Index(
            #                             ["Produit", " Devise", "Nominal", "Spot", "Strike", "Taux doméstique",
            #                              "Taux étranger", "Volatilitée", "Maturité", "Prime à payer"]))
            #
            #                     tdf = df.T
            #                     # CSS to inject contained in a string
            #                     hide_table_row_index = """
            #                                                             <style>
            #                                                             tbody th {display:none}
            #                                                             .blank {display:none}
            #                                                             </style>
            #                                                             """
            #
            #                     # Inject CSS with Markdown
            #                     ############ table recap
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
            #                         unsafe_allow_html=True)
            #                     st.markdown(hide_table_row_index, unsafe_allow_html=True)
            #                     st.table(tdf)
            #                     st.session_state['button'] = False
            #
            #                     ######### detail
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
            #                         unsafe_allow_html=True)
            #
            #                     st.write("- Spot " + " " + "  " + ":" + str(format(s, ".4f")))
            #                     st.write("- Taux doméstique:" + "   " + str(rd) + " " + "%")
            #                     st.write("- Taux étranger:" + "   " + str(rf) + " " + "%")
            #                     st.write("- Volatilité:" + "   " + str(volatilite) + " " + "%")
            #
            #
            #                     st.header(" ")
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
            #                         unsafe_allow_html=True)
            #                     st.header("")
            #
            #                     st.write("- si  " + "   " + str(ticker), " " + " <" + str(k))
            #                     st.write(" Vendre  " + str(N), " à " + " " + str(k))
            #                     st.write("- si  " + "   " + str(ticker), " " + " >=" + str(s))
            #                     st.write(" Pas d'exercice ")
            #                     st.write(" Vendre  " + " " + str(N), " au prix spot")
            #
            #             else:
            #                 rd1 = rd / 100
            #                 rf1 = rf / 100
            #                 sigma1 = sigma / 100
            #                 T = (exercise_date - datetime.now().date()).days
            #                 T1 = T / 365
            #
            #                 col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
            #                     [1, 60, 1, 1, 1, 1, 1, 1, 1])
            #                 with col2:
            #
            #                     put = BS_PUT(s, k, T1, rd1, rf1, sigma1)
            #                     p1 = put /k
            #                     p2 = put * N * s
            #
            #                     if ticker == 'EUR/MAD':
            #
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en % du Strike : " + format(
            #                                 p1, ".3f") + " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
            #                                 p2, ".3f") + " " + "DH" + " </h2>",
            #                             unsafe_allow_html=True)
            #
            #                     elif ticker == 'USD/MAD':
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en % du Strike : " + format(
            #                                 p1, ".3f") +  " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
            #                                 p2, ".3f") + " " + "DH" + " </h2>",
            #                             unsafe_allow_html=True)
            #
            #                 if st.button(f'Information sur opération '):
            #
            #
            #                     df = pd.DataFrame(
            #                         ["Put Vanille", ticker, format(N, ".2f"), format(s, ".4f"), format(k, ".4f"),
            #                          format(rd, ".0f") + str("%"), format(rf, ".0f") + str("%"),
            #                          format(sigma, ".0f") + str("%"), exercise_date, format(p2, ".2f")],
            #                         index=pd.Index(
            #                             ["Produit", " Devise", "Nominal", "Spot", "Strike", "Taux doméstique",
            #                              "Taux étranger", "Volatilitée", "Maturité", "Prime à payer"]))
            #
            #                     tdf = df.T
            #                     # CSS to inject contained in a string
            #                     hide_table_row_index = """
            #                                                             <style>
            #                                                             tbody th {display:none}
            #                                                             .blank {display:none}
            #                                                             </style>
            #                                                             """
            #
            #                     # Inject CSS with Markdown
            #                     ############ table recap
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
            #                         unsafe_allow_html=True)
            #                     st.markdown(hide_table_row_index, unsafe_allow_html=True)
            #                     st.table(tdf)
            #                     st.session_state['button'] = False
            #
            #                     ######### detail
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
            #                         unsafe_allow_html=True)
            #
            #                     st.write("- Spot " + " " + "  " + ":" + str(format(s, ".4f")))
            #                     st.write("- Taux doméstique:" + "   " + str(rd) + " " + "%")
            #                     st.write("- Taux étranger:" + "   " + str(rf) + " " + "%")
            #                     st.write("- Volatilité:" + "   " + str(sigma) + " " + "%")
            #
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
            #                         unsafe_allow_html=True)
            #
            #                     st.write("- si  " + "   " + str(ticker), " " + " <" + str(format(k, ".4f")))
            #                     st.write(" Vendre  " + str(N), " à " + " " + str(format(k, ".4f")))
            #                     st.write("- si  " + "   " + str(ticker), " " + " >=" + str(format(k, ".4f")))
            #                     st.write(" Pas d'exercice ")
            #                     st.write(" Vendre  " + " " + str(N), " au prix spot")
            #
            #
            #
            #
            #     if genre == 'Stratégie gratuite':
            #
            #         st.header("")
            #         col1, col2, col3 = st.columns(3)
            #         with col1:
            #             ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/MAD', 'USD/MAD'))
            #         # st.markdown(f" Paire  : {ticker}")
            #         with col2:
            #             N = st.number_input('Nominal')
            #             # st.markdown(f"Nominal : {N}")
            #         with col3:
            #             s = st.number_input('Spot')
            #             # st.markdown(f"Spot  : {s}")
            #
            #
            #         col1, col2, col3, col4 = st.columns(4)
            #         with col1:
            #             exercise_date = st.date_input('Exercise date', min_value=datetime.today() + timedelta(days=1),
            #                                           value=datetime.today() + timedelta(days=365))
            #             # st.markdown(f" Date de valeur :{exercise_date}")
            #
            #         with col2:
            #             rd = st.number_input('Taux doméstique (%)', 0, 100, 10)
            #             # st.markdown(f"Taux doméstique : {r} %")
            #
            #         with col3:
            #             rf = st.number_input('Taux étranger (%)', 0, 100, 10)
            #             # st.markdown(f"Taux étranger : {r2} %")
            #         with col4:
            #             sigma = st.number_input('Volatilité', 0, 100, 20)
            #             # st.markdown(f" Volatilitée : {sigma} %")
            #
            #         if sigma == 0:
            #             col1, col2 = st.columns(2)
            #             with col1:
            #                 vol1 = st.number_input('Volatilité EUR/USD', 0, 100, 0)
            #                 # st.markdown(f" Volatilitée : {sigma} %")
            #             with col2:
            #                 st.markdown(
            #                     "<h2 style='text-align: left ;color:black;font-size:18px;'> Pondération " + " </h2>",
            #                     unsafe_allow_html=True)
            #                 vol2 = st.number_input('Poids EUR', 0, 100, 60)
            #                 # st.markdown(f" Volatilitée : {sigma} %")
            #             with col2:
            #                 vol3 = st.number_input('Poids USD', 0, 100, 40)
            #
            #         st.markdown( "-------------------------------------------------------------------------------------------")
            #
            #         # gestion des exceptions
            #         if not all([N, s,  rd, rf]):
            #             st.error("Veuillez remplir tous les champs")
            #             button = None
            #         else:
            #             button = st.button(f"Calculer le prix de l'option")
            #
            #         if st.session_state.get('button') != True:
            #             st.session_state['button'] = button
            #
            #         if st.session_state['button'] == True:
            #
            #             if sigma == 0:
            #                 rd1 = rd / 100
            #                 rf1 = rf / 100
            #                 sigma1 = sigma / 100
            #                 sig1 = vol1 / 100
            #                 sig2 = vol2 / 100
            #                 sig3 = vol3 / 100
            #                 T = (exercise_date - datetime.now().date()).days
            #                 T1 = T / 365
            #
            #                 if ticker == 'EUR/MAD':
            #                     volatilite = sig1 * sig3
            #                 elif ticker == 'USD/MAD':
            #                     volatilite = sig1 * sig2
            #
            #                 col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
            #                     [1, 60, 1, 1, 1, 1, 1, 1, 1])
            #                 with col2:
            #
            #                     def find_strike1(s, rd, rf, T, sigma):
            #                         def objectif_func(k1):
            #                             return  BS_PUT(s, k1, T, rd, rf,sigma)
            #                         sol = optimize.root_scalar(objectif_func, bracket=[1,1e8], method='brentq')
            #                         return sol.root
            #
            #                     k1 = find_strike1(s, rd1, rf1, T1, volatilite)
            #                     put = BS_PUT(s, k1, T1, rd1, rf1, volatilite)
            #                     st.markdown(put)
            #
            #
            #
            #                     p2 = put * N * s
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en % du Strike : " + format(
            #                             k1, ".3f") +  " </h2>",
            #                         unsafe_allow_html=True)
            #             else :
            #                 ##############
            #                 st.markdown(
            #                     "walo")
            #
            #
            #
            #
            #
            #
            #
            #
            #
            #

            ################################################################################################################################
            if choose == "Call Vanille":
                st.header("")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/MAD', 'USD/MAD'))
                # st.markdown(f" Paire  : {ticker}")
                with col2:
                    N = float(st.text_input('Nominal',value=50000))
                    # st.markdown(f"Nominal : {N}")
                with col3:
                    s = float(st.text_input('Spot',value=10.586))
                    # st.markdown(f"Spot  : {s}")
                with col4:
                    k = float(st.text_input('Strike',value=10.50))
                    # st.markdown(f"Strike  : {k}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    exercise_date = st.date_input('Maturité', min_value=datetime.today() + timedelta(days=1),
                                                  value=datetime.today() + timedelta(days=365))
                    # st.markdown(f" Date de valeur :{exercise_date}")

                with col2:
                    rd =  float(st.text_input('Taux doméstrique',value=5))
                    # st.markdown(f"Taux doméstique : {r} %")
                with col3:
                    rf =  float(st.text_input('Taux Etrangé',value=4))
                    # st.markdown(f"Taux étranger : {r2} %")

                input_volatility = st.checkbox("Volatilité EUR/MAD", value=True)
                # st.markdown(f" Volatilitée : {sigma} %")
                if input_volatility:
                    sigma = st.number_input('Volatilité')
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        vol1 = st.number_input('Volatilité EUR/USD')
                        # st.markdown(f" Volatilitée : {sigma} %")
                    with col2:
                        st.markdown(
                            "<h2 style='text-align: left ;color:black;font-size:18px;'> Pondération " + " </h2>",
                            unsafe_allow_html=True)
                        vol2 = st.number_input('Poids EUR', value=60)
                        # st.markdown(f" Volatilitée : {sigma} %")
                    with col2:
                        vol3 = st.number_input('Poids USD', value=40)

                st.markdown(
                    "-------------------------------------------------------------------------------------------")

                # gestion des exceptions
                #if not all([N, s, k, rd, rf]):
                   # st.error("Veuillez remplir tous les champs")
                   # button = None
                #else:
                button = st.button(f"Calculer la prime")

                if st.session_state.get('button') != True:
                    st.session_state['button'] = button

                if st.session_state['button'] == True:
                    # Formating selected model parameters

                    if not input_volatility:
                        rd1 = rd / 100
                        rf1 = rf / 100
                        # sigma1 = sigma / 100
                        sig1 = vol1 / 100
                        sig2 = vol2 / 100
                        sig3 = vol3 / 100
                        T = (exercise_date - datetime.now().date()).days
                        T1 = T / 365
                        if ticker == 'EUR/MAD':
                            volatilite = sig1 * sig3
                        elif ticker == 'USD/MAD':
                            volatilite = sig1 * sig2

                        col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns([1, 60, 1, 1, 1, 1, 1, 1, 1])
                        with col2:

                            put = BS_CALL(s, k, T1, rd1, rf1, volatilite)
                            p1 = put
                            p2 = put * N * s

                            if ticker == 'EUR/MAD':

                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en valeur  : " + format(
                                        p1, ".3f") + " </h2>",
                                    unsafe_allow_html=True)
                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
                                        p2, ".3f") + " " + "DH" + " </h2>",
                                    unsafe_allow_html=True)

                            elif ticker == 'USD/MAD':
                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en valeur : " + format(
                                        p1, ".3f") + " " + "%" + "  </h2>",
                                    unsafe_allow_html=True)
                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
                                        p2, ".3f") + " " + "DH" + " </h2>",
                                    unsafe_allow_html=True)
                        st.header("")
                        if st.button(f'Information sur opération '):
                            volatilite1 = volatilite * 100
                            df = pd.DataFrame(
                                ["Call Vanille", ticker, format(N, ".2f"), format(s, ".4f"), format(k, ".4f"),
                                 format(rd, ".1f") + str("%"), format(rf, ".1f") + str("%"),
                                 format(volatilite1, ".1f") + str("%"), exercise_date, format(p2, ".2f")],
                                index=pd.Index(["Produit", " Devise", "Nominal", "Spot", "Strike", "Taux doméstique",
                                                "Taux étranger", "Volatilité", "Maturité", "Prime à payer"]))

                            tdf = df.T
                            # CSS to inject contained in a string
                            hide_table_row_index = """
                                                                            <style>
                                                                            tbody th {display:none}
                                                                            .blank {display:none}
                                                                            </style>
                                                                            """
                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                unsafe_allow_html=True)
                            st.markdown(hide_table_row_index, unsafe_allow_html=True)
                            st.table(tdf)
                            st.session_state['button'] = False

                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                unsafe_allow_html=True)

                            st.write("- Spot " + " " + "  " + ":" + str(format(s, ".4f")))
                            st.write("- Taux doméstique:" + "   " + str(format(rd, ".1f")) + " " + "%")
                            st.write("- Taux étranger:" + "   " + str(format(rf, ".1f")) + " " + "%")
                            st.write("- Volatilité:" + "   " + str(format(volatilite1, ".1f")) + " " + "%")

                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                unsafe_allow_html=True)

                            st.write("- si  " + "   " + str(ticker), " " + "  >=" + str(format(k, ".4f")))
                            st.write("  Achat  " + str(N), " à " + " " + str(format(k, ".4f")))
                            st.write("- si  " + "   " + str(ticker), " " + "  <" + str(format(k, ".4f")))
                            st.write(" Pas d'exercice ")
                            st.write("  Achat  " + " " + str(N), " au prix spot")

                    else:
                        rd1 = rd / 100
                        rf1 = rf / 100
                        sigma1 = sigma / 100
                        T = (exercise_date - datetime.now().date()).days
                        T1 = T / 365
                        col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns([1, 60, 1, 1, 1, 1, 1, 1, 1])
                        with col2:
                            put = BS_CALL(s, k, T1, rd1, rf1, sigma1)
                            p1 = put
                            p2 = put * N * s

                            if ticker == 'EUR/MAD':

                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en valeur : " + format(
                                        p1, ".3f") + "  </h2>",
                                    unsafe_allow_html=True)
                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
                                        p2, ".3f") + " " + "DH" + " </h2>",
                                    unsafe_allow_html=True)

                            elif ticker == 'USD/MAD':
                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en valeur : " + format(
                                        p1, ".3f") + " " + "%" + " </h2>",
                                    unsafe_allow_html=True)
                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
                                        p2, ".3f") + " " + "DH" + " </h2>",
                                    unsafe_allow_html=True)
                        st.header("")
                        if st.button(f'Information sur opération '):
                            df = pd.DataFrame(
                                ["Call Vanille", ticker, format(N, ".2f"), format(s, ".4f"), format(k, ".4f"),
                                 format(rd, ".1f") + str("%"), format(rf, ".1f") + str("%"),
                                 format(sigma, ".1f") + str("%"),
                                 exercise_date, format(p2, ".2f")],
                                index=pd.Index(
                                    ["Produit", " Devise", "Nominal", "Spot", "Strike", "Taux doméstique",
                                     "Taux étranger",
                                     "Volatilitée", "Maturité", "Prime à payer"]))

                            tdf = df.T
                            # CSS to inject contained in a string
                            hide_table_row_index = """
                                                                            <style>
                                                                            tbody th {display:none}
                                                                            .blank {display:none}
                                                                            </style>
                                                                            """

                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                unsafe_allow_html=True)
                            st.markdown(hide_table_row_index, unsafe_allow_html=True)
                            st.table(tdf)
                            st.session_state['button'] = False

                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                unsafe_allow_html=True)

                            st.write("- Spot " + " " + "  " + ":" + str(format(s, ".4f")))
                            st.write("- Taux doméstique:" + "   " + str(format(rd, ".1f")) + " " + "%")
                            st.write("- Taux étranger:" + "   " + str(format(rf, ".1f")) + " " + "%")
                            st.write("- Volatilité:" + "   " + str(format(sigma, ".1f")) + " " + "%")

                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                unsafe_allow_html=True)

                            st.write("- si  " + "   " + str(ticker), " " + " >=" + str(k))
                            st.write(" Achat  " + str(N), " à " + " " + str(k))
                            st.write("- si  " + "   " + str(ticker), " " + "< " + str(k))
                            st.write(" Pas d'exercice  ")
                            st.write(" Achat  " + " " + str(N), " au prix spot")


            ####################################################################################################################
            elif choose == "Call Participatif":

                genre = st.radio(
                    label="Choisissez le type de la  stratégie ", horizontal=True,
                    options=('Stratégie payante', 'Stratégie gratuite'))
                st.markdown('')
                if genre == 'Stratégie payante':
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/MAD', 'USD/MAD'))
                        # st.markdown(f" Paire  : {ticker}")
                    with col2:
                        N = float(st.text_input('Nominal',value=50000))
                        # st.markdown(f"Nominal : {N}")
                    with col3:
                        Perc = float(st.text_input('Pourcentage optionnel %', value=50))
                        # st.markdown(f"Percentage : {Perc} %")
                    with col4:
                        s = float(st.text_input('Spot',value=10.587))
                        # st.markdown(f"Spot  : {s}")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        k = float(st.text_input('Strike ',value=10.5))
                        # st.markdown(f"Strike  : {k}")
                    with col2:
                        exercise_date = st.date_input('Maturité', min_value=datetime.today() + timedelta(days=1),
                                                      value=datetime.today() + timedelta(days=365))
                        # st.markdown(f" Date de valeur :{exercise_date}")
                    with col3:
                        rd = float(st.text_input('Taux doméstique (%)',value=5))
                        # st.markdown(f"Taux doméstique : {r} %")
                    with col4:
                        rf = float(st.text_input('Taux étranger (%)',value=4))
                        # st.markdown(f"Taux étranger : {r2} %")

                    input_volatility = st.checkbox("Volatilité " + str(ticker), value=True)
                    if input_volatility:
                        sigma = st.number_input('Volatilité ' + str(ticker))
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            vol1 = st.number_input('Volatilité EUR/USD')
                            # st.markdown(f" Volatilitée : {sigma} %")
                        with col2:
                            st.markdown(
                                "<h2 style='text-align: left ;color:black;font-size:18px;'> Pondération " + " </h2>",
                                unsafe_allow_html=True)
                            vol2 = st.number_input('Poids EUR', 0, 100, 60)
                            # st.markdown(f" Volatilitée : {sigma} %")
                        with col2:
                            vol3 = st.number_input('Poids USD', 0, 100, 40)

                    st.markdown(
                        "-------------------------------------------------------------------------------------------")

                    #if not all([s, k, rd, rf]):
                     #   st.error("Veuillez remplir tous les champs")

                      #  button = None
                    #else:
                    button = st.button(f'Calculer la prime ')

                    if st.session_state.get('button') != True:
                        st.session_state['button'] = button

                    if st.session_state['button'] == True:

                        if not input_volatility:
                            rd1 = rd / 100
                            rf1 = rf / 100
                            sig1 = vol1 / 100
                            sig2 = vol2 / 100
                            sig3 = vol3 / 100
                            Perc1 = Perc / 100

                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365
                            if ticker == 'EUR/MAD':
                                volatilite = sig1 * sig3
                            elif ticker == 'USD/MAD':
                                volatilite = sig1 * sig2
                            volatilite1 = volatilite * 100
                            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                                [1, 60, 1, 1, 1, 1, 1, 1, 1])

                            with col2:

                                Ft = cours_terme(s, T1, rd1, rf1)
                                p1 = BS_CALL(s, k, T1, rd1, rf1, volatilite) * Perc1
                                put2 = p1 * (Perc1 * N) * s
                                f = (k - Ft) * np.exp(-(rd1 - rf1) * T1) * (1 - Perc1)
                                forward2 = f * (1 - Perc1) * N * s

                                Prime_partcipative1 = p1 - f
                                Prime_partcipative2 = put2 - forward2

                                if ticker == 'EUR/MAD':
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en valeur : " + format(
                                            Prime_partcipative1, ".3f") + " </h2>",
                                        unsafe_allow_html=True)
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
                                            Prime_partcipative2, ".3f") + " " + "DH" + " </h2>",
                                        unsafe_allow_html=True)

                                elif ticker == 'USD/MAD':
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en valeur : " + format(
                                            Prime_partcipative1, ".3f") + " " + "%" + " </h2>",
                                        unsafe_allow_html=True)
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
                                            Prime_partcipative2, ".3f") + " " + "DH" + " </h2>",
                                        unsafe_allow_html=True)
                            if st.button(f'Information sur opération '):
                                df = pd.DataFrame(
                                    [ticker, format(N, ".2f"), format(s, ".4f"), format(k, ".4f"),
                                     format(Perc, ".0f") + str("%"),
                                     format(Prime_partcipative2, ".2f")],
                                    index=pd.Index(
                                        ["Paire de devise", "Nominal", "Spot", "Strike", "Participation",
                                         "Call Participatif"]))
                                tdf = df.T
                                # CSS to inject contained in a string
                                hide_table_row_index = """
                                                                                            <style>
                                                                                            tbody th {display:none}
                                                                                            .blank {display:none}
                                                                                            </style>
                                                                                            """

                                ######################### detail de strategie

                                # Inject CSS with Markdown
                                ############ table recap

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                    unsafe_allow_html=True)

                                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                                st.table(tdf)
                                st.session_state['button'] = False

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)

                                st.write(
                                    "-Spot " + " " + str(ticker) + "          " + ":" + " " + str(format(s, ".4f")))
                                st.write("-Volatilité:" + "   " + str(format(volatilite1, ".2f")) + "%")
                                st.write("-Taux doméstique:" + "   " + str(format(rd, ".2f")) + "%")
                                st.write("-Taux étranger:" + "   " + str(format(rf, ".2f")) + "%")

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)

                                st.write("- si  ""   " + str(ticker), " ""  > " + str(format(k, ".4f")))
                                st.write(" Achat " + str(N), " à " + " " + str(format(k, ".4f")))
                                st.write("- si  ""   " + str(ticker), " ""< " + str(format(k, ".4f")))
                                st.write(" Achat  " + str(N * Perc1), "au spot du jour")
                                st.write(" Achat " + str(N * (1 - Perc1)), " à " + " " + str(format(k, ".4f")))


                        else:
                            rd1 = rd / 100
                            rf1 = rf / 100
                            sigma1 = sigma / 100
                            r1 = (rd1 - rf1)
                            Perc1 = Perc / 100
                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365

                            Ft = cours_terme(s, T1, rd1, rf1)
                            p2 = BS_CALL(s, k, T1, rd1, rf1, sigma1) * Perc1
                            put3 = p2 * (Perc1 * N)
                            f2 = (k - Ft) * np.exp(-(rd1 - rf1) * T1) * (1 - Perc1)
                            forward3 = f2 * (1 - Perc1) * N * s

                            Prime_partcipative3 = p2 - f2
                            Prime_partcipative4 = put3 - forward3

                            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                                [1, 60, 1, 1, 1, 1, 1, 1, 1])
                            with col2:

                                if ticker == 'EUR/MAD':

                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en valeur : " + format(
                                            Prime_partcipative3, ".3f") + " </h2>",
                                        unsafe_allow_html=True)

                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en MAD : " + format(
                                            Prime_partcipative4, ".3f") + " " + "DH" + " </h2>",
                                        unsafe_allow_html=True)

                                elif ticker == 'USD/MAD':
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en valeur : " + format(
                                            Prime_partcipative3, ".3f") + " </h2>",
                                        unsafe_allow_html=True)
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en MAD : " + format(
                                            Prime_partcipative4, ".3f") + " " + "DH" + " </h2>",
                                        unsafe_allow_html=True)

                            if st.button(f'Information sur opération '):
                                df = pd.DataFrame(
                                    [ticker, format(N, ".2f"), format(s, ".4f"), format(k, ".4f"),
                                     format(Perc, ".0f") + str("%"),
                                     format(Prime_partcipative4, ".2f")],
                                    index=pd.Index(
                                        ["Paire de devise", "Nominal", "Spot", "Strike", "Participation",
                                         "Call Participatif"]))
                                tdf = df.T
                                # CSS to inject contained in a string
                                hide_table_row_index = """
                                                                                            <style>
                                                                                            tbody th {display:none}
                                                                                            .blank {display:none}
                                                                                            </style>
                                                                                            """

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                    unsafe_allow_html=True)
                                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                                st.table(tdf)
                                st.session_state['button'] = False

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)

                                st.write(
                                    "-Spot " + " " + str(ticker) + "          " + ":" + " " + str(format(s, ".4f")))
                                st.write("-Volatilité:" + "   " + str(format(sigma, ".1f")) + "%")
                                st.write("-Taux doméstique:" + "   " + str(format(rd, ".1f")) + "%")
                                st.write("-Taux étranger:" + "   " + str(format(rf, ".1f")) + "%")

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)

                                st.write("- si  ""   " + str(ticker), " "" > " + str(format(k, ".4f")))
                                st.write(" Achat  " + str(N), " à " + " " + str(format(k, ".4f")))
                                st.write("- si  ""   " + str(ticker), " ""< " + str(format(k, ".4f")))
                                st.write(" Achat  " + str(N * Perc1), "au spot du jour")
                                st.write(" Ahat  " + str(format(N * (1 - Perc1), ".4f")),
                                         " à " + " " + str(format(k, ".4f")))

                if genre == 'Stratégie gratuite':

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/MAD', 'USD/MAD'))
                        # st.markdown(f" Paire  : {ticker}")
                    with col2:
                        N = float(st.text_input('Nominal',value=50000))
                        # st.markdown(f"Nominal : {N}")
                    with col3:
                        Perc = float(st.text_input('Pourcentage optionnel %',value=50))
                        # st.markdown(f"Percentage : {Perc} %")
                    with col4:
                        s = float(st.text_input('Spot',value=10.5870))
                        # st.markdown(f"Spot  : {s}")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        exercise_date = st.date_input('Maturité', min_value=datetime.today() + timedelta(days=1),
                                                      value=datetime.today() + timedelta(days=365))
                        # st.markdown(f" Date de valeur :{exercise_date}")
                    with col2:
                        rd = float(st.text_input('Taux doméstique (%)',value=5))
                        # st.markdown(f"Taux doméstique : {r} %")
                    with col3:
                        rf = float(st.text_input('Taux étranger (%)',value=4))
                        # st.markdown(f"Taux étranger : {r2} %")
                    input_volatility = st.checkbox("Volatilité " + str(ticker), value=True)

                    if input_volatility:
                        sigma = st.number_input('Volatilité ' + str(ticker))
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            vol1 = st.number_input('Volatilité EUR/USD')
                            # st.markdown(f" Volatilitée : {sigma} %")
                        with col2:
                            st.markdown(
                                "<h2 style='text-align: left ;color:black;font-size:18px;'> Pondération " + " </h2>",
                                unsafe_allow_html=True)
                            vol2 = st.number_input('Poids EUR', 0, 100, 60)
                            # st.markdown(f" Volatilitée : {sigma} %")
                        with col2:
                            vol3 = st.number_input('Poids USD', 0, 100, 40)

                    st.markdown(
                        "-------------------------------------------------------------------------------------------")
                    #if not all([s, rd, rf]):
                     #   st.error("Veuillez remplir tous les champs")
                    #    button = None
                    #else:
                    button = st.button(f'Calculer le strike ')

                    if st.session_state.get('button') != True:
                        st.session_state['button'] = button

                    if st.session_state['button'] == True:

                        if not input_volatility:
                            rd1 = rd / 100
                            rf1 = rf / 100
                            # sigma1 = sigma / 100
                            sig1 = vol1 / 100
                            sig2 = vol2 / 100
                            sig3 = vol3 / 100
                            Perc1 = Perc / 100
                            Perc2 = 1 - Perc1

                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365
                            if ticker == 'EUR/MAD':
                                volatilite = sig1 * sig3
                            elif ticker == 'USD/MAD':
                                volatilite = sig1 * sig2
                            volatilite1 = volatilite * 100
                            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                                [1, 60, 1, 1, 1, 1, 1, 1, 1])

                            with col2:
                                if Perc1 == 1:
                                    Ft = cours_terme(s, T1, rd1, rf1)
                                    p2 = BS_CALL(s, Ft, T1, rd1, rf1, volatilite)
                                    solution = Ft + p2
                                else:

                                    solution = find_strike2(s, rd1, rf1, T1, volatilite, Perc1, Perc2)

                                # def cours_terme(s, T, rd, rf):
                                #     F = s * np.exp((rd - rf) * T)
                                #     return F
                                #
                                # Ft = cours_terme(s, T1, rd1, rf1)
                                # x0 = s
                                #
                                # def constrainte(k):
                                #     return k - cours_terme(s, T1, rd1, rf1)
                                #
                                # def objective(k):
                                #     g = (N * (1 - Perc1) * ((k - s * np.exp((rd1 - rf1) * T1)) * np.exp(
                                #         -(rd1 - rf1) * T1)) + N * Perc1 * BS_PUT(k, T1, s, rd1, rf1, volatilite)) ** 2
                                #     return g
                                #
                                #
                                # cons = ({'type': 'ineq', 'fun': lambda x: x - s})
                                #
                                # optimize = sci_opt.minimize(objective, x0, method='SLSQP', constraints=cons,
                                #                             options={'disp': True})
                                #
                                # k1 = optimize.x
                                # st.markdown(k1)

                                # Ft = cours_terme(s, T1, rd1, rf1)
                                # k1 = find_strike(s, rd1, rf1, T1, volatilite)
                                #
                                # put = BS_PUT(s, k1, T1, rd1, rf1, volatilite)
                                # forward = (k1 - Ft) * np.exp(-r1 * T1)
                                # b = forward + put

                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Le Strike est : " + format(
                                        solution, ".4f") + " </h2>",
                                    unsafe_allow_html=True)

                            if st.button(f'Information sur opération '):
                                df = pd.DataFrame(
                                    [ticker, format(N, ".2f"), format(s, ".4f"), format(solution, ".4f"),
                                     format(Perc, ".0f") + str("%")
                                     ],
                                    index=pd.Index(
                                        ["Paire de devise", "Nominal", "Spot", "Strike", "Participation"]))
                                tdf = df.T
                                # CSS to inject contained in a string
                                hide_table_row_index = """
                                                                                            <style>
                                                                                            tbody th {display:none}
                                                                                            .blank {display:none}
                                                                                            </style>
                                                                                            """

                                ######################### detail de strategie

                                # Inject CSS with Markdown
                                ############ table recap

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                    unsafe_allow_html=True)

                                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                                st.table(tdf)
                                st.session_state['button'] = False

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)

                                st.write(
                                    "-Spot " + " " + str(ticker) + "          " + ":" + " " + str(format(s, ".4f")))
                                st.write("-Volatilité:" + "   " + str(format(volatilite1, ".2f")) + "%")
                                st.write("-Taux doméstique:" + "   " + str(format(rd, ".2f")) + "%")
                                st.write("-Taux étranger:" + "   " + str(format(rf, ".2f")) + "%")

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)

                                st.write("- si  ""   " + str(ticker), " "" >  " + str(format(solution, ".4f")))
                                st.write(" Achat " + str(N), " à " + " " + str(format(solution, ".4f")))
                                st.write("- si  ""   " + str(ticker), " ""< " + str(format(solution, ".4f")))
                                st.write(" Achat  " + str(format(N * Perc1, ".2f")), "au spot du jour")
                                st.write(" Achat " + str(format(N * (1 - Perc1), ".2f")),
                                         " à " + " " + str(format(solution, ".4f")))


                        else:
                            rd1 = rd / 100
                            rf1 = rf / 100
                            sigma1 = sigma / 100
                            r1 = (rd1 - rf1)
                            Perc1 = Perc / 100
                            Perc2 = 1 - Perc1
                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365

                            if Perc1 == 1:
                                Ft = cours_terme(s, T1, rd1, rf1)
                                p2 = BS_CALL(s, Ft, T1, rd1, rf1, sigma1)
                                solution1 = Ft + p2
                            else:

                                solution1 = find_strike2(s, rd1, rf1, T1, sigma1, Perc1, Perc2)
                            # def cour_t(s, T, rd, rf):
                            #     F = s * np.exp((rd - rf) * T)
                            #     return F
                            #
                            #
                            # Ft = cour_t(s, T1, rd1, rf1)
                            # x0= s
                            #
                            #
                            # def constrainte(k):
                            #     return k - cour_t(s, T1, rd1, rf1)
                            #
                            #
                            # def objective(k):
                            #     g = (N * (1-Perc1) * ((k - s * np.exp((rd1 - rf1) * T1)) * np.exp( -(rd1 - rf1) * T1)) + N * Perc1 * BS_PUT(s, k, T1, rd1 , rf1, sigma1)) ** 2
                            #     return g
                            #
                            #
                            # cons = ({'type': 'ineq', 'fun': lambda x: x - s})
                            #
                            # optimize = sci_opt.minimize(objective, x0, method='SLSQP', constraints=cons, options={'disp': True})
                            #
                            # k2 =optimize.x
                            # st.markdown(k2)
                            # Ft = cours_terme(s, T1, rd1, rf1)
                            # k2 = find_strike(s, rd1, rf1, T1, sigma1)
                            # st.markdown(Ft)
                            # st.markdown(k2)
                            # put = BS_PUT(s, k2, T1, rd1, rf1, sigma)
                            # forward = (k2 - Ft) * np.exp(-r1 * T1)
                            # b = forward + put

                            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                                [1, 60, 1, 1, 1, 1, 1, 1, 1])
                            with col2:

                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Le Strike est : " + format(
                                        solution1, ".4f") + " </h2>",
                                    unsafe_allow_html=True)

                            if st.button(f'Information sur opération '):
                                df = pd.DataFrame(
                                    [ticker, format(N, ".2f"), format(s, ".4f"), format(solution1, ".4f"),
                                     format(Perc, ".0f") + str("%")],
                                    index=pd.Index(
                                        ["Paire de devise", "Nominal", "Spot", "Strike", "Participation"]))
                                tdf = df.T
                                # CSS to inject contained in a string
                                hide_table_row_index = """
                                                                                            <style>
                                                                                            tbody th {display:none}
                                                                                            .blank {display:none}
                                                                                            </style>
                                                                                            """

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                    unsafe_allow_html=True)
                                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                                st.table(tdf)
                                st.session_state['button'] = False

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)

                                st.write(
                                    "-Spot " + " " + str(ticker) + "          " + ":" + " " + str(format(s, ".4f")))
                                st.write("-Volatilité:" + "   " + str(format(sigma, ".1f")) + "%")
                                st.write("-Taux doméstique:" + "   " + str(format(rd, ".1f")) + "%")
                                st.write("-Taux étranger:" + "   " + str(format(rf, ".1f")) + "%")

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)

                                st.write("- si  ""   " + str(ticker), " "" > " + str(format(solution1, ".4f")))
                                st.write(" Achat  " + str(N), " à " + " " + str(format(solution1, ".4f")))
                                st.write("- si  ""   " + str(ticker), " ""< " + str(format(solution1, ".4f")))
                                st.write(" Achat  " + str(format(N * Perc1, ".2f")), "au spot du jour")
                                st.write(" Achat  " + str(format(N * (1 - Perc1), ".2f")),
                                         " à " + " " + str(format(solution1, ".4f")))



            ####################################################################################################################
            #
            # elif choose == "Put Participatif":
            #
            #     st.header(" ")
            #
            #     col1, col2, col3,col4= st.columns(4)
            #     with col1:
            #         ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/USD', 'EUR/MAD', 'USD/MAD'))
            #         # st.markdown(f" Paire  : {ticker}")
            #     with col2:
            #         N = st.number_input('Nominal')
            #         # st.markdown(f"Nominal : {N}")
            #     with col3:
            #         Perc = st.number_input('Pourcentage %')
            #         # st.markdown(f"Percentage : {Perc} %")
            #     with col4 :
            #         s = st.number_input('Spot')
            #         # st.markdown(f"Spot  : {s}")
            #
            #     col1, col2, col3,col4,col5 = st.columns(5)
            #
            #     with col1:
            #         k = st.number_input('Strike ')
            #         # st.markdown(f"Strike  : {k}")
            #     with col2:
            #         exercise_date = st.date_input('Exercise date', min_value=datetime.today() + timedelta(days=1),
            #                                       value=datetime.today() + timedelta(days=365))
            #         # st.markdown(f" Date de valeur :{exercise_date}")
            #     with col3:
            #         rd = st.number_input('Taux doméstique (%)', 0, 100, 10)
            #         # st.markdown(f"Taux doméstique : {r} %")
            #     with col4:
            #         rf = st.number_input('Taux étranger (%)', 0, 100, 10)
            #         # st.markdown(f"Taux étranger : {r2} %")
            #     with col5:
            #         sigma = st.number_input('Volatilité (%)', 0, 100, 20)
            #         # st.markdown(f" Volatilitée : {sigma} %")
            #     if sigma == 0:
            #         col1, col2 = st.columns(2)
            #         with col1:
            #             vol1 = st.number_input('Volatilité EUR/USD', 0, 100, 10)
            #             # st.markdown(f" Volatilitée : {sigma} %")
            #         with col2:
            #             st.markdown(
            #                 "<h2 style='text-align: left ;color:black;font-size:18px;'> Pondération " + " </h2>",
            #                 unsafe_allow_html=True)
            #             vol2 = st.number_input('Poids EUR', 0, 100, 60)
            #             # st.markdown(f" Volatilitée : {sigma} %")
            #         with col2:
            #             vol3 = st.number_input('Poids USD', 0, 100, 40)
            #
            #     st.markdown("-------------------------------------------------------------------------------------------")
            #
            #
            #     genre = st.radio(
            #         "Choisissez le type de la  stratégie ",
            #         ('Stratégie payante', 'Stratégie gratuite'))
            #
            #     if genre == 'Stratégie payante':
            #
            #         # gestion des exceptions
            #
            #         if not all([s, k, rd, rf, Perc]):
            #             st.error("Veuillez remplir tous les champs")
            #
            #             button = None
            #         else:
            #             button = st.button(f'Calculer la prime ')
            #
            #         if st.session_state.get('button') != True:
            #             st.session_state['button'] = button
            #
            #         if st.session_state['button'] == True:
            #
            #             if sigma == 0:
            #                 rd1 = rd / 100
            #                 rf1 = rf / 100
            #                 sigma1 = sigma / 100
            #                 sig1 = vol1 / 100
            #                 sig2 = vol2 / 100
            #                 sig3 = vol3 / 100
            #                 Perc1 = Perc / 100
            #                 r1 = (rd1 - rf1)
            #                 T = (exercise_date - datetime.now().date()).days
            #                 T1 = T / 365
            #
            #                 col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
            #                     [1, 60, 1, 1, 1, 1, 1, 1, 1])
            #
            #                 with col2:
            #                     volatilite = sig1 * sig2
            #                     def cours_terme(s, T1, rd, rf):
            #                         r1 = rd - rf
            #                         F_k = s * np.exp(r1 * T1)
            #                         return F_k
            #
            #                     Ft = cours_terme(s, T1, rd1, rf1)
            #                     p1 = BS_PUT(s, k, T1, rd1, rf1, volatilite)
            #                     put = p1 * (Perc1 * N)
            #                     put2 = p1 * (Perc1 * N) * s
            #                     f = (k - Ft) * np.exp(-r1 * T1)
            #                     forward = f * (1 - Perc1) * N
            #                     forward2 = f * (1 - Perc1) * N* s
            #                     Prime_partcipative1 = put + forward
            #                     Prime_partcipative2 = put2+forward2
            #
            #                     if ticker == 'EUR/MAD':
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en % du Nominal : " + format(
            #                                 Prime_partcipative1, ".3f") + " " + "€" + " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en Devise : " + format(
            #                                 Prime_partcipative2, ".3f") + " " + "DH" + " </h2>",
            #                             unsafe_allow_html=True)
            #
            #                     elif ticker == 'USD/MAD':
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en % du Nominal : " + format(
            #                                 Prime_partcipative1, ".3f") + " " + "$" + " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en Devise : " + format(
            #                                 Prime_partcipative2, ".3f") + " " + "DH" + " </h2>",
            #                             unsafe_allow_html=True)
            #                     elif ticker == 'EUR/USD':
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en % du Nominal : " + format(
            #                                 Prime_partcipative1, ".3f") + " " + "€" + " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en Devise : " + format(
            #                                 Prime_partcipative2, ".3f") + " " + "$" + " </h2>",
            #                             unsafe_allow_html=True)
            #             else:
            #                 rd1 = rd / 100
            #                 rf1 = rf / 100
            #                 sigma1 = sigma / 100
            #                 r1 = (rd1 - rf1)
            #                 Perc1 = Perc / 100
            #                 T = (exercise_date - datetime.now().date()).days
            #                 T1 = T / 365
            #
            #                 def cours_terme(s, T1, rd, rf):
            #                     r1 = rd - rf
            #                     F_k = s * np.exp(r1 * T1)
            #                     return F_k
            #
            #                 Ft = cours_terme(s, T1, rd1, rf1)
            #
            #                 p2 = BS_PUT(s, k, T1, rd1, rf1, sigma1)
            #                 put3= p2 * (Perc1 * N)
            #                 put4 = p2 * (Perc1 * N) * s
            #
            #                 f = (k - Ft) * np.exp(-r1 * T1)
            #                 forward3 = f * (1 - Perc1) * N
            #                 forward4 = f * (1 - Perc1) * N * s
            #
            #                 Prime_partcipative3 = put3 + forward3
            #                 Prime_partcipative4 = put4 + forward4
            #
            #                 col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
            #                     [1, 60, 1, 1, 1, 1, 1, 1, 1])
            #                 with col2:
            #
            #
            #
            #
            #                     if ticker == 'EUR/MAD':
            #
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en % du Nominal : " + format(
            #                                 Prime_partcipative3 , ".3f") + " " + "€" + " </h2>",
            #                             unsafe_allow_html=True)
            #
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en Devise : " + format(
            #                                 Prime_partcipative4, ".3f") + " " + "DH" + " </h2>",
            #                             unsafe_allow_html=True)
            #
            #                     elif ticker == 'USD/MAD':
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en % du Nominal : " + format(
            #                                 Prime_partcipative3, ".3f") + " " + "$" + " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en Devise : " + format(
            #                                 Prime_partcipative4, ".3f") + " " + "DH" + " </h2>",
            #                             unsafe_allow_html=True)
            #                     elif ticker == 'EUR/USD':
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en % du Nominal : " + format(
            #                                 Prime_partcipative3, ".3f") + " " + "€" + " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en Devise : " + format(
            #                                 Prime_partcipative4, ".3f") + " " + "$" + " </h2>",
            #                             unsafe_allow_html=True)
            #
            #
            #             if st.button(f'Information sur opération '):
            #                 if sigma ==0 :
            #                     df = pd.DataFrame(
            #                         [ticker, format(N, ".2f"), format(s, ".4f"), format(k, ".4f"),
            #                          format(Perc, ".0f") + str("%"),
            #                          format(Prime_partcipative2, ".2f")],
            #                         index=pd.Index(
            #                             ["Paire de devise", "Nominal", "Spot", "Strike", "Participation"
            #                                 , "Put Participative"]))
            #                     tdf = df.T
            #                     # CSS to inject contained in a string
            #                     hide_table_row_index = """
            #                                                                         <style>
            #                                                                         tbody th {display:none}
            #                                                                         .blank {display:none}
            #                                                                         </style>
            #                                                                         """
            #
            #                     ######################### detail de strategie
            #
            #                     # Inject CSS with Markdown
            #                     ############ table recap
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
            #                         unsafe_allow_html=True)
            #                     st.header("")
            #                     st.markdown(hide_table_row_index, unsafe_allow_html=True)
            #                     st.table(tdf)
            #                     st.session_state['button'] = False
            #
            #                     ######### detail
            #
            #                     col1, col2, col3 = st.columns(3)
            #                     with col1:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("-Spot " + " " + str(ticker) + "          " + ":" + " " + str(s))
            #                         st.write("-Volatilité:" + "   " + str(volatilite) + "%")
            #                         st.write("-Taux doméstique:" + "   " + str(rd) + "%")
            #                         st.write("-Taux étranger:" + "   " + str(rf) + "%")
            #                         # st.write("-Volatilité:" + "   " + str(sigma))
            #                     st.header("  ")
            #                     with col3:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement a l'échéance </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("- si  ""   " + str(ticker), " "" < " + str(k))
            #                         st.write(" Vous vendez  " + str(N), " à " + " " + str(k))
            #                         st.write("- si  ""   " + str(ticker), " ""> " + str(s))
            #                         st.write(" Vous vendez  " + str(N * Perc1), "au spot du jour")
            #                         st.write(" Vous vendez  " + str(N * (1 - Perc1)), " à " + " " + str(k))
            #
            #
            #                 else :
            #
            #                     df = pd.DataFrame(
            #                         [ticker, format(N, ".2f"), format(s, ".4f"), format(k, ".4f"),
            #                          format(Perc, ".0f") + str("%"),
            #                            format(Prime_partcipative4, ".2f")],
            #                         index=pd.Index(
            #                             ["Paire de devise", "Nominal", "Spot", "Strike", "Participation"
            #                              ,"Put Participative"]))
            #                     tdf = df.T
            #                     # CSS to inject contained in a string
            #                     hide_table_row_index = """
            #                                         <style>
            #                                         tbody th {display:none}
            #                                         .blank {display:none}
            #                                         </style>
            #                                         """
            #
            #                     ######################### detail de strategie
            #
            #                     # Inject CSS with Markdown
            #                     ############ table recap
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
            #                         unsafe_allow_html=True)
            #                     st.header("")
            #                     st.markdown(hide_table_row_index, unsafe_allow_html=True)
            #                     st.table(tdf)
            #                     st.session_state['button'] = False
            #
            #                     ######### detail
            #
            #                     col1, col2, col3 = st.columns(3)
            #                     with col1:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("-Spot " + " " + str(ticker) + "          " + ":" + " " + str(s))
            #                         st.write("-Volatilité:" + "   " + str(sigma) + "%")
            #                         st.write("-Taux doméstique:" + "   " + str(rd) + "%")
            #                         st.write("-Taux étranger:" + "   " + str(rf) + "%")
            #
            #                     st.header("  ")
            #                     with col3:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement a l'échéance </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("- si  ""   " + str(ticker), " "" < " + str(k))
            #                         st.write(" Vous vendez  " + str(N), " à " + " " + str(k))
            #                         st.write("- si  ""   " + str(ticker), " ""> " + str(s))
            #                         st.write(" Vous vendez  " + str(N * Perc1), "au spot du jour")
            #                         st.write(" Vous vendez  " + str(N * (1 - Perc1)), " à " + " " + str(k))
            #
            #
            #
            #     else:
            #         if sigma==0:
            #             # gestion des exceptions
            #             st.markdown("-------------")
            #             if not all([s, k, rd, rf, Perc]):
            #                 st.error("Veuillez remplir tous les champs")
            #                 button = None
            #             else:
            #                 button = st.button(f'Calculer le strike amélioré ')
            #
            #             if st.session_state.get('button') != True:
            #                 st.session_state['button'] = button
            #
            #             if st.session_state['button'] == True:
            #
            #                 # Formating selected model parameters
            #                 rd1 = rd / 100
            #                 rf1 = rf / 100
            #                 r1 = (rd1 - rf1)
            #                 sigma1 = sigma / 100
            #                 sig1 = vol1 / 100
            #                 sig2 = vol2 / 100
            #                 sig3 = vol3 / 100
            #                 T = (exercise_date - datetime.now().date()).days
            #                 T1 = T / 365
            #                 Perc1 = Perc / 100
            #
            #                 if ticker =='EUR/MAD' :
            #                     volatilite = sig1 * sig2
            #                 elif ticker == 'USD/MAD':
            #                     volatilite = sig1 * sig3
            #                 ############"" find strike
            #                 Ft = cours_terme(s, T1, rd1, rf1)
            #                 k1 = find_strike(s, rd1, rf1, T1, volatilite)
            #
            #                 put = BS_PUT(s, k1, T1, rd1, rf1,volatilite )
            #                 forward = (k1 - Ft) * np.exp(-r1 * T1)
            #                 b = forward + put
            #
            #
            #                 st.markdown(
            #                     "<h2 style='text-align: left ;color:black;font-size:20px;'> - Le strike amélioré est : " + " " + " " + str(
            #                         format(k1, '.4f')) + " </h2>",
            #                     unsafe_allow_html=True)
            #
            #                 if st.button(f'Informtion sur opération '):
            #                     df = pd.DataFrame([ticker, format(N, ".2f"), format(s, ".4f"), format(k1, ".4f"),
            #                                        format(Perc, ".0f") + str("%"),format(b, ".2f")],
            #                                       index=pd.Index(
            #                                           ["Paire de devise", "Nominal", "Spot", "Strike amélioré",
            #                                            "Participation", "Prime"]))
            #
            #                     tdf = df.T
            #                     # CSS to inject contained in a string
            #                     hide_table_row_index = """
            #                                                             <style>
            #                                                             tbody th {display:none}
            #                                                             .blank {display:none}
            #                                                             </style>
            #                                                             """
            #
            #
            #                     # Inject CSS with Markdown
            #                     ############ table recap
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
            #                         unsafe_allow_html=True)
            #                     st.header("")
            #                     st.markdown(hide_table_row_index, unsafe_allow_html=True)
            #                     st.table(tdf)
            #                     st.session_state['button'] = False
            #
            #                     ######### detail
            #
            #                     col1, col2, col3 = st.columns(3)
            #                     with col1:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("- Spot " + " " + str(ticker) + "          " + ":" + str(s))
            #                         st.write("- Volatilité:" + "   " + str(format(volatilite,".2f")) + "%")
            #                         st.write("- Taux doméstique:" + "   " + str(rd) + "%")
            #                         st.write("- Taux étranger:" + "   " + str(rf) + "%")
            #                         st.write("- Strike:" + "   " + str(format(k1, ".4f")))
            #                     st.header("  ")
            #                     with col3:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement a l'échéance </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("- si  " + "   " + str(ticker), " " + " < " + str(format(k1, ".4f")))
            #                         st.write(" Vous vendez  " + str(N), " à " + " " + str(format(k1, ".4f")))
            #                         st.write("- si  " + "   " + str(ticker), " " + "> " + str(s))
            #                         st.write(" Vous vendez  " + str(N * Perc1), "au spot du jour")
            #                         st.write(" Vous vendez  " + str(N * (1 - Perc1)),
            #                                  " à " + " " + str(format(k1, ".4f")))
            #
            #         else:
            #             # gestion des exceptions
            #             st.markdown("-------------")
            #             if not all([s, k, rd, rf, Perc]):
            #                 st.error("Veuillez remplir tous les champs")
            #                 button = None
            #             else:
            #                 button = st.button(f'Calculer le strike amélioré ')
            #
            #             if st.session_state.get('button') != True:
            #                 st.session_state['button'] = button
            #
            #             if st.session_state['button'] == True:
            #
            #                 # Formating selected model parameters
            #                 rd1 = rd / 100
            #                 rf1 = rf / 100
            #                 r1 = (rd1 - rf1)
            #                 sigma1 = sigma / 100
            #                 T = (exercise_date - datetime.now().date()).days
            #                 T1 = T / 365
            #                 Perc1 = Perc / 100
            #
            #                 ############"" find strike
            #                 Ft = cours_terme(s, T1, rd1, rf1)
            #                 k1 = find_strike(s, rd1, rf1, T1, sigma1)
            #
            #                 putg = BS_PUT(s, k1, T1, rd1, rf1, sigma1)
            #                 forwardg = (k1 - Ft) * np.exp(-r1 * T1)
            #                 b1 = forwardg + putg
            #
            #
            #                 st.markdown(
            #                     "<h2 style='text-align: left ;color:black;font-size:20px;'> - Le strike amélioré est : " + " " + " " + str(
            #                         format(k1, '.4f')) + " </h2>",
            #                     unsafe_allow_html=True)
            #
            #                 if st.button(f'Informtion sur opération '):
            #                     df = pd.DataFrame([ticker, format(N, ".2f"), format(s, ".4f"), format(k1, ".4f"),
            #                                        format(Perc, ".0f") + str("%"),format(b1, ".2f")],
            #                                       index=pd.Index(
            #                                           ["Paire de devise", "Nominal", "Spot", "Strike amélioré",
            #                                            "Participation", "Prime"]))
            #
            #                     tdf = df.T
            #                     # CSS to inject contained in a string
            #                     hide_table_row_index = """
            #                                     <style>
            #                                     tbody th {display:none}
            #                                     .blank {display:none}
            #                                     </style>
            #                                     """
            #
            #
            #                     ############ table recap
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
            #                         unsafe_allow_html=True)
            #                     st.header("")
            #                     st.markdown(hide_table_row_index, unsafe_allow_html=True)
            #                     st.table(tdf)
            #                     st.session_state['button'] = False
            #
            #                     ######### detail
            #
            #                     col1, col2, col3 = st.columns(3)
            #                     with col1:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("- Spot " + " " + str(ticker) + "          " + ":" + str(s))
            #                         st.write("- Volatilité:" + "   " + str(sigma) + "%")
            #                         st.write("- Taux doméstique:" + "   " + str(rd) + "%")
            #                         st.write("- Taux étranger:" + "   " + str(rf) + "%")
            #                         st.write("- Strike:" + "   " + str(format(k1, ".4f")))
            #                     st.header("  ")
            #                     with col3:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement a l'échéance </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("- si  " + "   " + str(ticker), " " + " < " + str(format(k1, ".4f")))
            #                         st.write(" Vous vendez  " + str(N), " à " + " " + str(format(k1, ".4f")))
            #                         st.write("- si  " + "   " + str(ticker), " " + "> " + str(s))
            #                         st.write(" Vous vendez  " + str(N * Perc1), "au spot du jour")
            #                         st.write(" Vous vendez  " + str(N * (1 - Perc1)), " à " + " " + str(format(k1, ".4f")))

            ############################################################################################################################################################
            elif choose == "Tunnel Symétrique":

                genre = st.radio(
                    label="Choisissez le type de la  stratégie ", horizontal=True,
                    options=('Stratégie payante', 'Stratégie gratuite'))
                st.markdown('')
                if genre == 'Stratégie payante':

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/MAD', 'USD/MAD'))
                        # st.markdown(f" Paire  : {ticker}")
                    with col2:
                        N = float(st.text_input('Nominal',value=50000))
                        # st.markdown(f"Nominal : {N}")
                    with col3:
                        s = float(st.text_input('Spot',value=10.587))
                        # st.markdown(f"Spot  : {s}")
                    with col4:
                        exercise_date = st.date_input('Maturité', min_value=datetime.today() + timedelta(days=1),
                                                      value=datetime.today() + timedelta(days=365))
                        # st.markdown(f" Date de valeur :{exercise_date}")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        k1 = float(st.text_input('Strike k1',value=10.585))
                        # st.markdown(f"Percentage : {k1} %")

                    with col2:
                        k2 = float(st.text_input('Strike k2 ',value=10.5878))
                        # .markdown(f"Strike  : {k2}")
                    with col3:
                        rd = float(st.text_input('Taux doméstique (%)',value= 5))
                        # st.markdown(f"Taux doméstique : {r} %")

                    with col4:
                        rf = float(st.text_input('Taux étranger (%)', value=4))
                        # st.markdown(f"Taux étranger : {r2} %")
                    input_volatility = st.checkbox("Volatilité " + str(ticker), value=True)
                    if input_volatility:
                        sigma = st.number_input('Volatilité ' + str(ticker))
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            vol1 = st.number_input('Volatilité EUR/USD', 0, 100, 10)
                            # st.markdown(f" Volatilitée : {sigma} %")
                        with col2:
                            st.markdown(
                                "<h2 style='text-align: left ;color:black;font-size:18px;'> Pondération " + " </h2>",
                                unsafe_allow_html=True)
                            vol2 = st.number_input('Poids EUR', 0, 100, 60)
                            # st.markdown(f" Volatilitée : {sigma} %")
                        with col2:
                            vol3 = st.number_input('Poids USD', 0, 100, 40)
                    st.markdown('-------------------------------------------------------------------------------------')

                    #if not all([s, k1, k2, rd, rf]):
                      #  st.error("Veuillez remplir tous les champs")

                      #  button = None
                    #else:
                    button = st.button(f'Calculer la prime ')

                    if st.session_state.get('button') != True:
                        st.session_state['button'] = button

                    if st.session_state['button'] == True:

                        if not input_volatility:
                            rd1 = rd / 100
                            rf1 = rf / 100
                            sig1 = vol1 / 100
                            sig2 = vol2 / 100
                            sig3 = vol3 / 100

                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365
                            if ticker == 'EUR/MAD':
                                volatilite = sig1 * sig3
                            elif ticker == 'USD/MAD':
                                volatilite = sig1 * sig2
                            volatilite1 = volatilite * 100

                            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                                [1, 60, 1, 1, 1, 1, 1, 1, 1])

                            with col2:

                                Ft = cours_terme(s, T1, rd1, rf1)
                                c = BS_CALL(s, k2, T1, rd1, rf1, volatilite)
                                p = BS_PUT(s, k1, T1, rd1, rf1, volatilite)

                                tunnel1 = c - p
                                tunnel2 = (c - p) * s * N

                                # Ft = cours_terme(s, T1, rd1, rf1)
                                # p1 = BS_PUT(s, k, T1, rd1, rf1, volatilite)
                                # put2 = p1 * (Perc1 * N) * s
                                # f = (k - Ft) * np.exp(-(rd1 - rf1) * T1)
                                # forward2 = f * (1 - Perc1) * N* s
                                #
                                # Prime_partcipative1 = ((p1 + f)/k)*100
                                # Prime_partcipative2 = put2+forward2

                                if ticker == 'EUR/MAD':
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en valeur : " + format(
                                            tunnel1, ".3f") + " </h2>",
                                        unsafe_allow_html=True)
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
                                            tunnel2, ".3f") + " " + "DH" + " </h2>",
                                        unsafe_allow_html=True)

                                elif ticker == 'USD/MAD':
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en valeur : " + format(
                                            tunnel1, ".3f") + " </h2>",
                                        unsafe_allow_html=True)
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
                                            tunnel2, ".3f") + " " + "DH" + " </h2>",
                                        unsafe_allow_html=True)
                            if st.button(f'Information sur opération '):
                                df = pd.DataFrame(
                                    [ticker, format(N, ".2f"), format(s, ".4f"), format(k1, ".4f"), format(k2, ".4f"),
                                     format(tunnel2, ".2f")],
                                    index=pd.Index(["Paire de devise", "Nominal", "Spot", "Strike k1", "Strike k2",
                                                    "Tunnel Symétrique"]))
                                tdf = df.T
                                # CSS to inject contained in a string
                                hide_table_row_index = """
                                                                                            <style>
                                                                                            tbody th {display:none}
                                                                                            .blank {display:none}
                                                                                            </style>
                                                                                            """

                                ######################### detail de strategie

                                # Inject CSS with Markdown
                                ############ table recap

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                    unsafe_allow_html=True)

                                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                                st.table(tdf)
                                st.session_state['button'] = False

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)

                                st.write(
                                    "-Spot " + " " + str(ticker) + "          " + ":" + " " + str(format(s, ".4f")))
                                st.write("-Volatilité:" + "   " + str(format(volatilite1, ".2f")) + "%")
                                st.write("-Taux doméstique:" + "   " + str(format(rd, ".2f")) + "%")
                                st.write("-Taux étranger:" + "   " + str(format(rf, ".2f")) + "%")

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)

                                st.write("- Si  " + "   " + str(ticker), " " + " > " + str(format(k1, ".4f")))
                                st.write(" Achat  " + str(N), " à " + " " + str(format(k1, ".4f")))

                                st.write("- Si  " + "   " + str(format(k1, ".4f")) + " < " + str(ticker),
                                         " " + " <" + str(format(k2, ".4f")))
                                st.write(" Achat  " + str(N), "au spot du jour")

                                st.write("- Si  " + "   " + str(ticker), " "" < " + str(format(k2, ".4f")))
                                st.write(" Achat  " + str(N), " à " + " " + str(format(k2, ".4f")))



                        else:
                            rd1 = rd / 100
                            rf1 = rf / 100
                            sigma1 = sigma / 100
                            r1 = (rd1 - rf1)

                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365

                            Ft = cours_terme(s, T1, rd1, rf1)
                            c = BS_CALL(s, k2, T1, rd1, rf1, sigma1)
                            p = BS_PUT(s, k1, T1, rd1, rf1, sigma1)

                            tunnel3 = c - p
                            tunnel4 = (c - p) * s * N

                            # Ft = cours_terme(s, T1, rd1, rf1)
                            # p2 = BS_PUT(s, k, T1, rd1, rf1, sigma1)
                            # put3 = p2 * (Perc1 * N)
                            # f2 = (k - Ft) * np.exp(-(rd1 - rf1) * T1)
                            # forward3 = f2* (1 - Perc1) * N * s
                            #
                            # Prime_partcipative3 = ((p2 + f2) / k) * 100
                            # Prime_partcipative4 = put3+ forward3

                            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                                [1, 60, 1, 1, 1, 1, 1, 1, 1])
                            with col2:

                                if ticker == 'EUR/MAD':

                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en valeur : " + format(
                                            tunnel3, ".3f") + " </h2>",
                                        unsafe_allow_html=True)

                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en MAD : " + format(
                                            tunnel4, ".3f") + " " + "DH" + " </h2>",
                                        unsafe_allow_html=True)

                                elif ticker == 'USD/MAD':
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en valeur : " + format(
                                            tunnel3, ".3f") + " </h2>",
                                        unsafe_allow_html=True)
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en MAD : " + format(
                                            tunnel4, ".3f") + " " + "DH" + " </h2>",
                                        unsafe_allow_html=True)

                            if st.button(f'Information sur opération '):
                                df = pd.DataFrame(
                                    [ticker, format(N, ".2f"), format(s, ".4f"), format(k1, ".4f"),
                                     format(k2, ".4f") + str("%"),
                                     format(tunnel4, ".2f")],
                                    index=pd.Index(
                                        ["Paire de devise", "Nominal", "Spot", "Strike k1", "Strike k2",
                                         "Tunnel Symétrique"]))
                                tdf = df.T
                                # CSS to inject contained in a string
                                hide_table_row_index = """
                                                                                            <style>
                                                                                            tbody th {display:none}
                                                                                            .blank {display:none}
                                                                                            </style>
                                                                                            """

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                    unsafe_allow_html=True)
                                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                                st.table(tdf)
                                st.session_state['button'] = False

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)

                                st.write(
                                    "-Spot " + " " + str(ticker) + "          " + ":" + " " + str(format(s, ".4f")))
                                st.write("-Volatilité:" + "   " + str(format(sigma, ".1f")) + "%")
                                st.write("-Taux doméstique:" + "   " + str(format(rd, ".1f")) + "%")
                                st.write("-Taux étranger:" + "   " + str(format(rf, ".1f")) + "%")

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)

                                st.write("- Si  " + "   " + str(ticker), " " + " > " + str(format(k1, ".4f")))
                                st.write(" Achat  " + str(N), " à " + " " + str(format(k1, ".4f")))

                                st.write("- Si  " + "   " + str(format(k1, ".4f")) + " < " + str(ticker),
                                         " " + " <" + str(format(k2, ".4f")))
                                st.write(" Achat  " + str(N), "au spot du jour")

                                st.write("- Si  " + "   " + str(ticker), " "" < " + str(format(k2, ".4f")))
                                st.write(" Achat  " + str(N), " à " + " " + str(format(k2, ".4f")))

                if genre == 'Stratégie gratuite':

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/MAD', 'USD/MAD'))
                        # st.markdown(f" Paire  : {ticker}")
                    with col2:
                        N = float(st.text_input('Nominal',value=50000))
                        # st.markdown(f"Nominal : {N}")
                    with col3:
                        s = float(st.text_input('Spot',value=10.587))
                        # st.markdown(f"Spot  : {s}")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        exercise_date = st.date_input('Maturité', min_value=datetime.today() + timedelta(days=1),
                                                      value=datetime.today() + timedelta(days=365))
                        # st.markdown(f" Date de valeur :{exercise_date}")

                    with col2:
                        rd = float(st.text_input('Taux doméstique (%)',value=5))
                        # st.markdown(f"Taux doméstique : {r} %")

                    with col3:
                        rf = float(st.text_input('Taux étranger (%)', value=4))
                        # st.markdown(f"Taux étranger : {r2} %")
                    input_volatility = st.checkbox("Volatilité " + str(ticker), value=True)

                    if input_volatility:
                        sigma = st.number_input('Volatilité ' + str(ticker))
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            vol1 = st.number_input('Volatilité EUR/USD')
                            # st.markdown(f" Volatilitée : {sigma} %")
                        with col2:
                            st.markdown(
                                "<h2 style='text-align: left ;color:black;font-size:18px;'> Pondération " + " </h2>",
                                unsafe_allow_html=True)
                            vol2 = st.number_input('Poids EUR', 0, 100, 60)
                            # st.markdown(f" Volatilitée : {sigma} %")
                        with col2:
                            vol3 = st.number_input('Poids USD', 0, 100, 40)

                    st.markdown(
                        "-------------------------------------------------------------------------------------------")
                    #if not all([s, rd, rf, N]):
                       # st.error("Veuillez remplir tous les champs")
                       # button = None
                    #else:
                    button = st.button(f'Calculer la strike ')

                    if st.session_state.get('button') != True:
                        st.session_state['button'] = button

                    if st.session_state['button'] == True:

                        if not input_volatility:
                            rd1 = rd / 100
                            rf1 = rf / 100
                            # sigma1 = sigma / 100
                            sig1 = vol1 / 100
                            sig2 = vol2 / 100
                            sig3 = vol3 / 100

                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365
                            if ticker == 'EUR/MAD':
                                volatilite = sig1 * sig3
                            elif ticker == 'USD/MAD':
                                volatilite = sig1 * sig2
                            volatilite1 = volatilite * 100

                            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                                [1, 60, 1, 1, 1, 1, 1, 1, 1])

                            with col2:

                                def cours_terme(s, T1, rd, rf):
                                    r1 = rd - rf
                                    F_k = s * np.exp(r1 * T1)
                                    return F_k


                                Ft = cours_terme(s, T1, rd1, rf1)


                                def BS_CALL1(s, k, T1, rd, rf, volatilite):
                                    d1 = (np.log(s / k) + ((rd - rf) + volatilite ** 2 / 2) * T1) / (
                                                volatilite * np.sqrt(T1))
                                    d2 = d1 - volatilite * np.sqrt(T1)
                                    Call = s * n(d1) * np.exp(-rf * T1) - k * np.exp(-rd * T1) * n(d2)
                                    return Call


                                def BS_PUT1(s, k, T1, rd, rf, volatilite):
                                    d1 = (np.log(s / k) + ((rd - rf) + volatilite ** 2 / 2) * T1) / (
                                                volatilite * np.sqrt(T1))
                                    d2 = d1 - volatilite * np.sqrt(T1)
                                    Put = k * np.exp(-rd * T1) * n(-d2) - s * n(-d1) * np.exp(-rf * T1)
                                    return Put


                                def Ft_gt_k1(k):
                                    k1 = k[0]
                                    return Ft - k1 - 0.0000000000000000000000000000000000001


                                def K2_gt_Ft(k):
                                    k2 = k[1]
                                    return k2 - Ft - 0.0000000000000000000000000000000000001


                                def objective1(k):
                                    k1 = k[0]
                                    k2 = k[1]
                                    return ((BS_CALL(s, k2, T1, rd1, rf1, volatilite) - BS_PUT(s, k1, T1, rd1, rf1,
                                                                                               volatilite))) ** 2


                                k0 = [s, s]

                                con1 = {'type': 'ineq', 'fun': Ft_gt_k1}
                                con2 = {'type': 'ineq', 'fun': K2_gt_Ft}

                                con = [con1, con2]

                                optimize = sci_opt.minimize(objective1, k0, method='SLSQP', constraints=con,
                                                            options={'disp': True})

                                solution1 = optimize.x[0]
                                solution2 = optimize.x[1]

                                # x0 = s
                                #
                                # def constrainte(k):
                                #     return k - cours_terme(s, T1, rd1, rf1)
                                #
                                # def objective(k):
                                #     g = (N * (1 - Perc1) * ((k - s * np.exp((rd1 - rf1) * T1)) * np.exp(
                                #         -(rd1 - rf1) * T1)) + N * Perc1 * BS_PUT(k, T1, s, rd1, rf1, volatilite)) ** 2
                                #     return g
                                #
                                #
                                # cons = ({'type': 'ineq', 'fun': lambda x: x - s})
                                #
                                # optimize = sci_opt.minimize(objective, x0, method='SLSQP', constraints=cons,
                                #                             options={'disp': True})
                                #
                                # k1 = optimize.x
                                # st.markdown(k1)

                                # Ft = cours_terme(s, T1, rd1, rf1)
                                # k1 = find_strike(s, rd1, rf1, T1, volatilite)
                                #
                                # put = BS_PUT(s, k1, T1, rd1, rf1, volatilite)
                                # forward = (k1 - Ft) * np.exp(-r1 * T1)
                                # b = forward + put

                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:20px;'>  Le strike est :  </h2>",
                                    unsafe_allow_html=True)

                                st.markdown(
                                    "<h2 style='text-align: left ;color:;font-size:15px;'>  - K1 = : " + str(
                                        format(solution1, '.4f')) + " </h2>",
                                    unsafe_allow_html=True)

                                st.markdown(
                                    "<h2 style='text-align: left ;color:;font-size:15px;'>  - K2 = : " + str(
                                        format(solution2, '.4f')) + " </h2>",
                                    unsafe_allow_html=True)

                            if st.button(f'Information sur opération '):
                                df = pd.DataFrame(
                                    [ticker, format(N, ".2f"), format(s, ".4f"), format(solution1, ".4f"),
                                     format(solution2, ".4f")],
                                    index=pd.Index(
                                        ["Paire de devise", "Nominal", "Spot", "Strike k1", "Strike k2"]))
                                tdf = df.T
                                # CSS to inject contained in a string
                                hide_table_row_index = """
                                                                                            <style>
                                                                                            tbody th {display:none}
                                                                                            .blank {display:none}
                                                                                            </style>
                                                                                            """

                                ######################### detail de strategie

                                # Inject CSS with Markdown
                                ############ table recap

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                    unsafe_allow_html=True)

                                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                                st.table(tdf)
                                st.session_state['button'] = False

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)

                                st.write(
                                    "-Spot " + " " + str(ticker) + "          " + ":" + " " + str(format(s, ".4f")))
                                st.write("-Volatilité:" + "   " + str(format(volatilite1, ".2f")) + "%")
                                st.write("-Taux doméstique:" + "   " + str(format(rd, ".2f")) + "%")
                                st.write("-Taux étranger:" + "   " + str(format(rf, ".2f")) + "%")

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)

                                st.write("- Si  " + "   " + str(ticker), " " + "> " + str(format(solution1, ".4f")))
                                st.write(" Achat  " + str(N), " à " + " " + str(format(solution1, ".4f")))

                                st.write("- Si  " + "   " + str(format(solution1, ".4f")) + " < " + str(ticker),
                                         " " + " <" + str(format(solution2, ".4f")))
                                st.write(" Achat  " + str(N), "au spot du jour")

                                st.write("- Si  " + "   " + str(ticker), " "" < " + str(format(solution2, ".4f")))
                                st.write(" Achat  " + str(N), " à " + " " + str(format(solution2, ".4f")))

                        else:
                            rd1 = rd / 100
                            rf1 = rf / 100
                            sigma1 = sigma / 100
                            r1 = (rd1 - rf1)

                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365
                            n = norm.cdf


                            def cours_terme(s, T1, rd, rf):
                                r1 = rd - rf
                                F_k = s * np.exp(r1 * T1)
                                return F_k


                            Ft = cours_terme(s, T1, rd1, rf1)


                            def BS_CALL1(s, k, T1, rd, rf, sigma1):
                                d1 = (np.log(s / k) + ((rd - rf) + sigma1 ** 2 / 2) * T1) / (sigma1 * np.sqrt(T1))
                                d2 = d1 - sigma1 * np.sqrt(T1)
                                Call = s * n(d1) * np.exp(-rf * T1) - k * np.exp(-rd * T1) * n(d2)
                                return Call


                            def BS_PUT1(s, k, T1, rd, rf, sigma1):

                                d1 = (np.log(s / k) + ((rd - rf) + sigma1 ** 2 / 2) * T1) / (sigma1 * np.sqrt(T1))
                                d2 = d1 - sigma1 * np.sqrt(T1)
                                Put = k * np.exp(-rd * T1) * n(-d2) - s * n(-d1) * np.exp(-rf * T1)
                                return Put


                            def Ft_gt_k1(k):
                                k1 = k[0]
                                return Ft - k1 - 0.0000000000000000000000000000000000001


                            def K2_gt_Ft(k):
                                k2 = k[1]
                                return k2 - Ft - 0.0000000000000000000000000000000000001


                            def objective1(k):
                                k1 = k[0]
                                k2 = k[1]
                                return ((BS_CALL(s, k2, T1, rd1, rf1, sigma1) - BS_PUT(s, k1, T1, rd1, rf1,
                                                                                       sigma1))) ** 2


                            k0 = [s, s]

                            con1 = {'type': 'ineq', 'fun': Ft_gt_k1}
                            con2 = {'type': 'ineq', 'fun': K2_gt_Ft}

                            con = [con1, con2]

                            optimize = sci_opt.minimize(objective1, k0, method='SLSQP', constraints=con,
                                                        options={'disp': True})

                            solution3 = optimize.x[0]
                            solution4 = optimize.x[1]

                            # def cour_t(s, T, rd, rf):
                            #     F = s * np.exp((rd - rf) * T)
                            #     return F
                            #
                            #
                            # Ft = cour_t(s, T1, rd1, rf1)
                            # x0= s
                            #
                            #
                            # def constrainte(k):
                            #     return k - cour_t(s, T1, rd1, rf1)
                            #
                            #
                            # def objective(k):
                            #     g = (N * (1-Perc1) * ((k - s * np.exp((rd1 - rf1) * T1)) * np.exp( -(rd1 - rf1) * T1)) + N * Perc1 * BS_PUT(s, k, T1, rd1 , rf1, sigma1)) ** 2
                            #     return g
                            #
                            #
                            # cons = ({'type': 'ineq', 'fun': lambda x: x - s})
                            #
                            # optimize = sci_opt.minimize(objective, x0, method='SLSQP', constraints=cons, options={'disp': True})
                            #
                            # k2 =optimize.x
                            # st.markdown(k2)
                            # # Ft = cours_terme(s, T1, rd1, rf1)

                            # k2 = find_strike(s, rd1, rf1, T1, sigma1)
                            # st.markdown(Ft)
                            # st.markdown(k2)
                            # put = BS_PUT(s, k2, T1, rd1, rf1, sigma)
                            # forward = (k2 - Ft) * np.exp(-r1 * T1)
                            # b = forward + put

                            # col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                            #     [1, 60, 1, 1, 1, 1, 1, 1, 1])
                            # with col2:
                            #
                            #     st.markdown(
                            #         "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Le Strike est : " + format(
                            #             k2 , ".3f")+ " </h2>",
                            #         unsafe_allow_html=True)

                            st.markdown(
                                "<h2 style='text-align: left ;color:black;font-size:20px;'>  Le strike est :  </h2>",
                                unsafe_allow_html=True)

                            st.markdown(
                                "<h2 style='text-align: left ;color:;font-size:15px;'>  - K1 = : " + str(
                                    format(solution3, '.4f')) + " </h2>",
                                unsafe_allow_html=True)

                            st.markdown(
                                "<h2 style='text-align: left ;color:;font-size:15px;'>  - K2 = : " + str(
                                    format(solution4, '.4f')) + " </h2>",
                                unsafe_allow_html=True)

                            if st.button(f'Information sur opération'):
                                df = pd.DataFrame(
                                    [ticker, format(N, ".2f"), format(s, ".4f"), format(solution3, ".4f"),
                                     format(solution4, ".4f")],
                                    index=pd.Index(["Paire de devise", "Nominal", "Spot", "Strike k1", "Strike k2"]))
                                tdf1 = df.T
                                # CSS to inject contained in a string
                                hide_table_row_index = """
                                                                                                                            <style>
                                                                                                                            tbody th {display:none}
                                                                                                                            .blank {display:none}
                                                                                                                            </style>
                                                                                                                            """

                                ######################### detail de strategie

                                # Inject CSS with Markdown
                                ############ table recap

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                    unsafe_allow_html=True)

                                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                                st.table(tdf1)
                                st.session_state['button'] = False

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)

                                st.write(
                                    "-Spot " + " " + str(ticker) + "          " + ":" + " " + str(format(s, ".4f")))
                                st.write("-Volatilité:" + "   " + str(format(sigma, ".2f")) + "%")
                                st.write("-Taux doméstique:" + "   " + str(format(rd, ".2f")) + "%")
                                st.write("-Taux étranger:" + "   " + str(format(rf, ".2f")) + "%")

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)

                                st.write("- Si  " + "   " + str(ticker), " " + " > " + str(format(solution3, ".4f")))
                                st.write(" Achat  " + str(N), " à " + " " + str(format(solution3, ".4f")))

                                st.write("- Si  " + "   " + str(format(solution3, ".4f")) + " < " + str(ticker),
                                         " " + " <" + str(format(solution4, ".4f")))
                                st.write(" Achat  " + str(N), "au spot du jour")

                                st.write("- Si  " + "   " + str(ticker), " "" < " + str(format(solution4, ".4f")))
                                st.write(" Achat  " + str(N), " à " + " " + str(format(solution4, ".4f")))






            ############################################################################################################################################################
            # elif choose == "Tunnel Symétrique":
            #
            #     st.header(" ")
            #
            #     col1, col2, col3,col4= st.columns(4)
            #     with col1:
            #         ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/USD', 'EUR/MAD', 'USD/MAD'))
            #         # st.markdown(f" Paire  : {ticker}")
            #     with col2:
            #         N = st.number_input('Nominal')
            #         # st.markdown(f"Nominal : {N}")
            #     with col3:
            #         s = st.number_input('Spot')
            #         # st.markdown(f"Spot  : {s}")
            #     with col4:
            #         exercise_date = st.date_input('Exercise date', min_value=datetime.today() + timedelta(days=1),
            #                                       value=datetime.today() + timedelta(days=365))
            #         # st.markdown(f" Date de valeur :{exercise_date}")
            #
            #     col1, col2, col3,col4,col5 = st.columns(5)
            #
            #     with col1:
            #         k1 = st.number_input('Strike k1')
            #         # st.markdown(f"Percentage : {k1} %")
            #
            #     with col2:
            #         k2 = st.number_input('Strike k2 ')
            #         # .markdown(f"Strike  : {k2}")
            #     with col3:
            #         rd = st.number_input('Taux doméstique (%)', 0, 100, 10)
            #         # st.markdown(f"Taux doméstique : {r} %")
            #
            #     with col4:
            #         rf = st.number_input('Taux étranger (%)', 0, 100, 10)
            #         # st.markdown(f"Taux étranger : {r2} %")
            #
            #     with col5:
            #         sigma = st.number_input('Volatilité (%)', 0, 100, 10)
            #         # st.markdown(f" Volatilitée : {sigma} %")
            #     if sigma == 0:
            #         col1, col2 = st.columns(2)
            #         with col1:
            #             vol1 = st.number_input('Volatilité EUR/USD', 0, 100, 10)
            #             # st.markdown(f" Volatilitée : {sigma} %")
            #         with col2:
            #             st.markdown(
            #                 "<h2 style='text-align: left ;color:black;font-size:18px;'> Pondération " + " </h2>",
            #                 unsafe_allow_html=True)
            #             vol2 = st.number_input('Poids EUR', 0, 100, 60)
            #             # st.markdown(f" Volatilitée : {sigma} %")
            #         with col2:
            #             vol3 = st.number_input('Poids USD', 0, 100, 40)
            #     st.markdown('-------------------------------------------------------------------------------------')
            #     genre = st.radio(
            #         "Choisissez le type de la  stratégie ",
            #         ('Stratégie payante', 'Stratégie gratuite'))
            #
            #     if genre == 'Stratégie payante':
            #
            #         # gestion des exceptions
            #         st.markdown("-------------")
            #         if not all([s, k1, rd, rf, k2]):
            #             st.error("Veuillez remplir tous les champs")
            #             button = None
            #         else:
            #             button = st.button(f'Calculer la prime ')
            #
            #         if st.session_state.get('button') != True:
            #             st.session_state['button'] = button
            #
            #         if st.session_state['button'] == True:
            #
            #             # Formating selected model parameters
            #
            #             rd1 = rd / 100
            #             rf1 = rf / 100
            #             sigma1 = sigma / 100
            #             sig1 = vol1 / 100
            #             sig2 = vol2 / 100
            #             sig3 = vol3 / 100
            #             r1 = (rd1 - rf1)
            #             T = (exercise_date - datetime.now().date()).days
            #             T1 = T / 365
            #             if ticker == 'EUR/MAD':
            #                 volatilite = sig1 * sig2
            #             elif ticker == 'USD/MAD':
            #                 volatilite = sig1 * sig3
            #             else:
            #                 volatilite = sig1
            #
            #             if sigma == 0:
            #
            #                 col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
            #                     [1, 60, 1, 1, 1, 1, 1, 1, 1])
            #
            #                 with col2:
            #
            #
            #                     Ft = cours_terme(s, T1, rd1, rf1)
            #
            #                     c = BS_CALL(s, k2, T1, rd1, rf1, volatilite)
            #
            #                     p = BS_PUT(s, k1, T1, rd1, rf1, volatilite)
            #
            #                     put = p  * N
            #                     call = c  * N
            #
            #                     tunnel = c - p
            #                     tunnel1 = call - put
            #                     tunnel2 = (call - put)*s
            #
            #
            #                     if ticker == 'EUR/MAD':
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en % du Nominal : " + format(
            #                                 tunnel1, ".3f") + " " + "€" + " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en Devise : " + format(
            #                                 tunnel2, ".3f") + " " + "DH" + " </h2>",
            #                             unsafe_allow_html=True)
            #
            #                     elif ticker == 'USD/MAD':
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en % du Nominal : " + format(
            #                                 tunnel1, ".3f") + " " + "$" + " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en Devise : " + format(
            #                                 tunnel2, ".3f") + " " + "DH" + " </h2>",
            #                             unsafe_allow_html=True)
            #                     elif ticker == 'EUR/USD':
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en % du Nominal : " + format(
            #                                 tunnel1, ".3f") + " " + "€" + " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en Devise : " + format(
            #                                 tunnel2, ".3f") + " " + "$" + " </h2>",
            #                             unsafe_allow_html=True)
            #             else:
            #
            #
            #                 col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
            #                     [1, 60, 1, 1, 1, 1, 1, 1, 1])
            #                 with col2:
            #
            #
            #                     Ft = cours_terme(s, T1, rd1, rf1)
            #
            #                     c3 = BS_CALL(s, k2, T1, rd1, rf1, sigma1)
            #
            #                     p3 = BS_PUT(s, k1, T1, rd1, rf1, sigma1)
            #
            #                     put3 = p3 * N
            #                     call3 = c3 * N
            #
            #                     tunnel = c3 - p3
            #                     tunnel3 = call3 - put3
            #                     tunnel4 = (call3 - put3) * s
            #                     if ticker == 'EUR/MAD':
            #
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en % du Nominal : " + format(
            #                                 tunnel3 , ".3f") + " " + "€" + " </h2>",
            #                             unsafe_allow_html=True)
            #
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en Devise : " + format(
            #                                 tunnel4, ".3f") + " " + "DH" + " </h2>",
            #                             unsafe_allow_html=True)
            #
            #                     elif ticker == 'USD/MAD':
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en % du Nominal : " + format(
            #                                 tunnel3, ".3f") + " " + "$" + " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en Devise : " + format(
            #                                 tunnel4, ".3f") + " " + "DH" + " </h2>",
            #                             unsafe_allow_html=True)
            #                     elif ticker == 'EUR/USD':
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en % du Nominal : " + format(
            #                                 tunnel3, ".3f") + " " + "€" + " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en Devise : " + format(
            #                                 tunnel4, ".3f") + " " + "$" + " </h2>",
            #                             unsafe_allow_html=True)
            #
            #
            #
            #             if st.button(f'Information sur opération '):
            #
            #                 if sigma ==0 :
            #
            #                     df = pd.DataFrame([ticker, format(N, ".2f"), format(s, ".4f"), format(k1, ".4f"),
            #                                        format(k2, ".4f"),
            #                                        format(tunnel2, ".2f")],
            #                                       index=pd.Index(
            #                                           ["Paire de devise", "Nominal", "Spot", "Strike k1 ", "Strike k2",
            #                                            " Tunnel Symétrique"]))
            #                     tdf = df.T
            #                     # CSS to inject contained in a string
            #                     hide_table_row_index = """
            #                                                                     <style>
            #                                                                     tbody th {display:none}
            #                                                                     .blank {display:none}
            #                                                                     </style>
            #                                                                     """
            #
            #                     ############ table recap
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
            #                         unsafe_allow_html=True)
            #                     st.header("")
            #                     st.markdown(hide_table_row_index, unsafe_allow_html=True)
            #                     st.table(tdf)
            #                     st.session_state['button'] = False
            #
            #                     ######### detail
            #
            #                     col1, col2, col3 = st.columns(3)
            #                     with col1:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("-Spot " + " " + str(ticker) + "          " + ":" + " " + str(s))
            #                         st.write("-Volatilité:" + "   " + str(volatilite) + "%")
            #                         st.write("-Taux doméstique:" + "   " + str(rd) + "%")
            #                         st.write("-Taux étranger:" + "   " + str(rf) + "%")
            #                         # st.write("-Volatilité:" + "   " + str(sigma))
            #                     st.header("  ")
            #                     with col3:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement a l'échéance </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("- si  " + "   " + str(ticker), " " + " < " + str(k1))
            #                         st.write(" Vous vendez  " + str(N), " à " + " " + str(k1))
            #
            #                         st.write("- si  " + "   " + str(k1) + " < " + str(ticker), " " + " <" + str(s))
            #                         st.write(" Vous vendez  " + str(N), "au spot du jour")
            #
            #                         st.write("- si  " + "   " + str(ticker), " "" > " + str(k2))
            #                         st.write(" Vous vendez  " + str(N), " à " + " " + str(k2))
            #
            #                 else :
            #
            #                     df = pd.DataFrame([ticker, format(N, ".2f"), format(s, ".4f"), format(k1, ".4f"),
            #                                        format(k2, ".4f"),format(tunnel4, ".2f")],
            #                                       index=pd.Index(
            #                                           ["Paire de devise", "Nominal", "Spot", "Strike k1 ", "Strike k2",
            #                                            " Tunnel Symétrique"]))
            #                     tdf = df.T
            #                     # CSS to inject contained in a string
            #                     hide_table_row_index = """
            #                                         <style>
            #                                         tbody th {display:none}
            #                                         .blank {display:none}
            #                                         </style>
            #                                         """
            #
            #                     ############ table recap
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
            #                         unsafe_allow_html=True)
            #                     st.header("")
            #                     st.markdown(hide_table_row_index, unsafe_allow_html=True)
            #                     st.table(tdf)
            #                     st.session_state['button'] = False
            #
            #                     ######### detail
            #
            #                     col1, col2, col3 = st.columns(3)
            #                     with col1:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("-Spot " + " " + str(ticker) + "          " + ":" + " " + str(s))
            #                         st.write("-Volatilité:" + "   " + str(sigma) + "%")
            #                         st.write("-Taux doméstique:" + "   " + str(rd) + "%")
            #                         st.write("-Taux étranger:" + "   " + str(rf) + "%")
            #                         # st.write("-Volatilité:" + "   " + str(sigma))
            #                     st.header("  ")
            #                     with col3:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement a l'échéance </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("- si  " + "   " + str(ticker), " " + " < " + str(k1))
            #                         st.write(" Vous vendez  " + str(N), " à " + " " + str(k1))
            #
            #                         st.write("- si  " + "   " + str(k1) + " < " + str(ticker), " " + " <" + str(s))
            #                         st.write(" Vous vendez  " + str(N), "au spot du jour")
            #
            #                         st.write("- si  " + "   " + str(ticker), " "" > " + str(k2))
            #                         st.write(" Vous vendez  " + str(N), " à " + " " + str(k2))
            #
            #     else:
            #
            #         # gestion des exceptions
            #         st.markdown("-------------")
            #         if not all([s, k1, k2, rd, rf]):
            #             st.error("Veuillez remplir tous les champs")
            #             button = None
            #         else:
            #             button = st.button(f'Calculer le strike amélioré ')
            #
            #         if st.session_state.get('button') != True:
            #             st.session_state['button'] = button
            #
            #         if st.session_state['button'] == True:
            #
            #             if sigma==0 :
            #
            #                 # Formating selected model parameters
            #                 rd1 = rd / 100
            #                 rf1 = rf / 100
            #                 r1 = (rd1 - rf1)
            #                 sigma1 = sigma / 100
            #                 T = (exercise_date - datetime.now().date()).days
            #                 T1 = T / 365
            #                 sig1 = vol1 / 100
            #                 sig2 = vol2 / 100
            #                 sig3 = vol3 / 100
            #
            #                 if ticker == 'EUR/MAD':
            #                     volatilite = sig1 * sig3
            #                 elif ticker == 'USD/MAD':
            #                     volatilite = sig1 * sig2
            #
            #
            #
            #                 ######################### optimisation
            #                 def constrainte(k):
            #                     k1 = k[0]
            #                     k2 = k[1]
            #                     return k2 - k1 - 0.0000000000000000000000000000000000001
            #
            #
            #                 def constrainte_2(k):
            #                     k1 = k[0]
            #                     k2 = k[1]
            #                     return k2 - s - 0.0000000000000000000000000000000000001
            #
            #
            #                 def constrainte_3(k):
            #                     k1 = k[0]
            #                     k2 = k[1]
            #                     return s - k1 - 0.0000000000000000000000000000000000001
            #
            #
            #                 def objective1(k):
            #                     k1 = k[0]
            #                     k2 = k[1]
            #                     return ((BS_CALL(s, k2, T1, rd1, rf1, volatilite)) - BS_PUT(s, k1, T1, rd1, rf1,
            #                                                                             volatilite)) ** 2
            #
            #
            #                 k0 = [s - 0.4, s - 0.2]
            #                 cons = {'type': 'ineq', 'fun': constrainte}
            #                 con2 = {'type': 'ineq', 'fun': constrainte_2}
            #                 con3 = {'type': 'ineq', 'fun': constrainte_3}
            #                 cons1 = [cons, con2, con3]
            #
            #                 optimize = sci_opt.minimize(objective1, k0, method='SLSQP', constraints=cons1,
            #                                             options={'disp': True})
            #
            #                 call2 = BS_CALL(s, optimize.x[1], T1, rd1, rf1, volatilite)
            #                 put2 = BS_PUT(s, optimize.x[0], T1, rd1, rf1, volatilite)
            #
            #                 c = optimize.x[1]
            #                 p = optimize.x[0]
            #                 tunnel1 = call2 - put2
            #
            #
            #                 st.markdown(
            #                     "<h2 style='text-align: left ;color:black;font-size:20px;'>  Le strike amélioré est :  </h2>",
            #                     unsafe_allow_html=True)
            #
            #                 st.markdown(
            #                     "<h2 style='text-align: left ;color:;font-size:15px;'>  - K1 = : " + str(
            #                         format(p, '.4f')) + " </h2>",
            #                     unsafe_allow_html=True)
            #
            #                 st.markdown(
            #                     "<h2 style='text-align: left ;color:;font-size:15px;'>  - K2 = : " + str(
            #                         format(c, '.4f')) + " </h2>",
            #                     unsafe_allow_html=True)
            #
            #                 if st.button(f'Informtion sur opération '):
            #                     df = pd.DataFrame(
            #                         [ticker, format(N, ".2f"), format(s, ".4f"), format(p, ".4f"), format(c, ".4f"),
            #                          format(tunnel1, ".2f")],
            #                         index=pd.Index(
            #                             ["Paire de devise", "Nominal", "Spot", "Strike k1", "Strike k2",
            #                               "Prime"]))
            #
            #                     tdf = df.T
            #                     # CSS to inject contained in a string
            #                     hide_table_row_index = """
            #                                                                     <style>
            #                                                                     tbody th {display:none}
            #                                                                     .blank {display:none}
            #                                                                     </style>
            #                                                                     """
            #
            #                     ############ table recap
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
            #                         unsafe_allow_html=True)
            #                     st.header("")
            #                     st.markdown(hide_table_row_index, unsafe_allow_html=True)
            #                     st.table(tdf)
            #                     st.session_state['button'] = False
            #
            #                     ######### detail
            #
            #                     col1, col2, col3 = st.columns(3)
            #                     with col1:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("-Spot " + " " + str(ticker) + "          " + ":" + " " + str(s))
            #                         st.write("-Volatilité:" + "   " + str(volatilite) + "%")
            #                         st.write("-Taux doméstique:" + "   " + str(rd) + "%")
            #                         st.write("-Taux étranger:" + "   " + str(rf) + "%")
            #
            #                     st.header("  ")
            #                     with col3:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("- si  " + "   " + str(ticker), " " + " < " + str(format(p, ".4f")))
            #                         st.write(" Vous vendez  " + str(N), " à " + " " + str(format(p, ".4f")))
            #
            #                         st.write("- si  " + "   " + str(format(p, ".4f")) + " < " + str(ticker),
            #                                  " " + " <" + str(format(c, ".4f")))
            #                         st.write(" Vous vendez  " + str(N), "au spot du jour")
            #
            #                         st.write("- si  " + "   " + str(ticker), " "" > " + str(format(c, ".4f")))
            #                         st.write(" Vous vendez  " + str(N), " à " + " " + str(format(c, ".4f")))
            #
            #             else :
            #
            #
            #                 # Formating selected model parameters
            #                 rd1 = rd / 100
            #
            #                 rf1 = rf / 100
            #
            #                 r1 = (rd1 - rf1)
            #                 sigma1 = sigma / 100
            #
            #                 T = (exercise_date - datetime.now().date()).days
            #                 T1 = T / 365
            #
            #
            #                 ######################### optimisation
            #                 def constrainte(k):
            #                     k1 = k[0]
            #                     k2 = k[1]
            #                     return k2 - k1 - 0.0000000000000000000000000000000000001
            #
            #
            #                 def constrainte_2(k):
            #                     k1 = k[0]
            #                     k2 = k[1]
            #                     return k2 - s - 0.0000000000000000000000000000000000001
            #
            #
            #                 def constrainte_3(k):
            #                     k1 = k[0]
            #                     k2 = k[1]
            #                     return s - k1 - 0.0000000000000000000000000000000000001
            #
            #
            #                 def objective1(k):
            #                     k1 = k[0]
            #                     k2 = k[1]
            #                     return ((BS_CALL(s, k2, T1, rd1, rf1, sigma1)) - BS_PUT(s, k1, T1, rd1, rf1, sigma1)) ** 2
            #
            #
            #                 k0 = [s - 0.4, s - 0.2]
            #                 cons = {'type': 'ineq', 'fun': constrainte}
            #                 con2 = {'type': 'ineq', 'fun': constrainte_2}
            #                 con3 = {'type': 'ineq', 'fun': constrainte_3}
            #                 cons1 = [cons, con2, con3]
            #
            #                 optimize = sci_opt.minimize(objective1, k0, method='SLSQP', constraints=cons1,
            #                                             options={'disp': True})
            #
            #                 call2 = BS_CALL(s, optimize.x[1], T1, rd1, rf1, sigma1)
            #                 put2 = BS_PUT(s, optimize.x[0], T1, rd1, rf1, sigma1)
            #
            #                 c = optimize.x[1]
            #                 p = optimize.x[0]
            #                 tunnel3 = call2 - put2
            #                 tunnel4 = (call2 - put2)*s
            #
            #                 st.markdown(
            #                     "<h2 style='text-align: left ;color:black;font-size:20px;'>  Le strike amélioré est :  </h2>",
            #                     unsafe_allow_html=True)
            #
            #                 st.markdown(
            #                     "<h2 style='text-align: left ;color:;font-size:15px;'>  - K1 = : " + str(
            #                         format(p, '.4f')) + " </h2>",
            #                     unsafe_allow_html=True)
            #
            #                 st.markdown(
            #                     "<h2 style='text-align: left ;color:;font-size:15px;'>  - K2 = : " + str(
            #                         format(c, '.4f')) + " </h2>",
            #                     unsafe_allow_html=True)
            #
            #                 if st.button(f'Informtion sur opération '):
            #                     df = pd.DataFrame(
            #                         [ticker, format(N, ".2f"), format(s, ".4f"), format(p, ".4f"), format(c, ".4f")
            #                            , format(tunnel4, ".2f")],
            #                         index=pd.Index(
            #                             ["Paire de devise", "Nominal", "Spot", "Strike k1", "Strike k2", "Prime"]))
            #
            #                     tdf = df.T
            #                     # CSS to inject contained in a string
            #                     hide_table_row_index = """
            #                                             <style>
            #                                             tbody th {display:none}
            #                                             .blank {display:none}
            #                                             </style>
            #                                             """
            #
            #
            #                     ############ table recap
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
            #                         unsafe_allow_html=True)
            #                     st.header("")
            #                     st.markdown(hide_table_row_index, unsafe_allow_html=True)
            #                     st.table(tdf)
            #                     st.session_state['button'] = False
            #
            #                     ######### detail
            #
            #                     col1, col2, col3 = st.columns(3)
            #                     with col1:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("-Spot " + " " + str(ticker) + "          " + ":" + " " + str(s))
            #                         st.write("-Volatilité:" + "   " + str(sigma) + "%")
            #                         st.write("-Taux doméstique:" + "   " + str(rd) + "%")
            #                         st.write("-Taux étranger:" + "   " + str(rf) + "%")
            #
            #                     st.header("  ")
            #                     with col3:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("- si  " + "   " + str(ticker), " " + " < " + str(format(p, ".4f")))
            #                         st.write(" Vous vendez  " + str(N), " à " + " " + str(format(p, ".4f")))
            #
            #                         st.write("- si  " + "   " + str(format(p, ".4f")) + " < " + str(ticker),
            #                                  " " + " <" + str(format(c, ".4f")))
            #                         st.write(" Vous vendez  " + str(N), "au spot du jour")
            #
            #                         st.write("- si  " + "   " + str(ticker), " "" > " + str(format(c, ".4f")))
            #                         st.write(" Vous vendez  " + str(N), " à " + " " + str(format(c, ".4f")))

            ######################################################################################################################################################

            elif choose == "Tunnel Asymétrique":

                genre = st.radio(
                    label="Choisissez le type de la  stratégie ", horizontal=True,
                    options=('Stratégie payante', 'Stratégie gratuite'))
                st.markdown('')
                if genre == 'Stratégie payante':

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/MAD', 'USD/MAD'))
                        # st.markdown(f" Paire  : {ticker}")
                    with col2:
                        N = float(st.text_input('Nominal',value=50000))
                        # st.markdown(f"Nominal : {N}")
                    with col3:
                        s = float(st.text_input('Spot',value=10.587))
                        # st.markdown(f"Spot  : {s}")
                    with col4:
                        exercise_date = st.date_input('Maturité', min_value=datetime.today() + timedelta(days=1),
                                                      value=datetime.today() + timedelta(days=365))
                        # st.markdown(f" Date de valeur :{exercise_date}")

                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        m = float(st.text_input("Coefficient d'asymétrie",value=2))
                    with col2:
                        k1 = float(st.text_input('Strike k1',value=10.585))
                        # st.markdown(f"Percentage : {k1} %")

                    with col3:
                        k2 = float(st.text_input('Strike k2 ',value=10.5875))
                        # .markdown(f"Strike  : {k2}")
                    with col4:
                        rd = float(st.text_input('Taux doméstique (%)',value=5))
                        # st.markdown(f"Taux doméstique : {r} %")

                    with col5:
                        rf =float(st.text_input('Taux étranger (%)', value=4))
                        # st.markdown(f"Taux étranger : {r2} %")

                    input_volatility = st.checkbox("Volatilité " + str(ticker), value=True)
                    if input_volatility:
                        sigma = st.number_input('Volatilité ' + str(ticker))
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            vol1 = st.number_input('Volatilité EUR/USD', 0, 100, 10)
                            # st.markdown(f" Volatilitée : {sigma} %")
                        with col2:
                            st.markdown(
                                "<h2 style='text-align: left ;color:black;font-size:18px;'> Pondération " + " </h2>",
                                unsafe_allow_html=True)
                            vol2 = st.number_input('Poids EUR', 0, 100, 60)
                            # st.markdown(f" Volatilitée : {sigma} %")
                        with col2:
                            vol3 = st.number_input('Poids USD', 0, 100, 40)
                    st.markdown('-------------------------------------------------------------------------------------')

                    if not all([s, k1, k2, rd, rf, m]):
                        st.error("Veuillez remplir tous les champs")

                        button = None
                    else:
                        button = st.button(f'Calculer la prime ')

                    if st.session_state.get('button') != True:
                        st.session_state['button'] = button

                    if st.session_state['button'] == True:

                        if not input_volatility:
                            rd1 = rd / 100
                            rf1 = rf / 100
                            sig1 = vol1 / 100
                            sig2 = vol2 / 100
                            sig3 = vol3 / 100

                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365
                            if ticker == 'EUR/MAD':
                                volatilite = sig1 * sig3
                            elif ticker == 'USD/MAD':
                                volatilite = sig1 * sig2
                            volatilite1 = volatilite * 100

                            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                                [1, 60, 1, 1, 1, 1, 1, 1, 1])

                            with col2:

                                Ft = cours_terme(s, T1, rd1, rf1)
                                c = m * BS_CALL(s, k2, T1, rd1, rf1, volatilite)
                                p = BS_PUT(s, k1, T1, rd1, rf1, volatilite)

                                tunnel1 = (c - p)
                                tunnel2 = (c - p) * s * N

                                # Ft = cours_terme(s, T1, rd1, rf1)
                                # p1 = BS_PUT(s, k, T1, rd1, rf1, volatilite)
                                # put2 = p1 * (Perc1 * N) * s
                                # f = (k - Ft) * np.exp(-(rd1 - rf1) * T1)
                                # forward2 = f * (1 - Perc1) * N* s
                                #
                                # Prime_partcipative1 = ((p1 + f)/k)*100
                                # Prime_partcipative2 = put2+forward2

                                if ticker == 'EUR/MAD':
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en valeur : " + format(
                                            tunnel1, ".3f") + " </h2>",
                                        unsafe_allow_html=True)
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
                                            tunnel2, ".3f") + " " + "DH" + " </h2>",
                                        unsafe_allow_html=True)

                                elif ticker == 'USD/MAD':
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en valeur : " + format(
                                            tunnel1, ".3f") + " </h2>",
                                        unsafe_allow_html=True)
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
                                            tunnel2, ".3f") + " " + "DH" + " </h2>",
                                        unsafe_allow_html=True)
                            if st.button(f'Information sur opération '):
                                df = pd.DataFrame(
                                    [ticker, format(N, ".2f"), format(s, ".4f"), format(k1, ".4f"), format(k2, ".4f"),
                                     format(tunnel2, ".2f")],
                                    index=pd.Index(["Paire de devise", "Nominal", "Spot", "Strike k1", "Strike k2",
                                                    "Tunnel Asymétrique"]))
                                tdf = df.T
                                # CSS to inject contained in a string
                                hide_table_row_index = """
                                                                                                <style>
                                                                                                tbody th {display:none}
                                                                                                .blank {display:none}
                                                                                                </style>
                                                                                                """

                                ######################### detail de strategie

                                # Inject CSS with Markdown
                                ############ table recap

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                    unsafe_allow_html=True)

                                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                                st.table(tdf)
                                st.session_state['button'] = False

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)

                                st.write(
                                    "-Spot " + " " + str(ticker) + "          " + ":" + " " + str(format(s, ".4f")))
                                st.write("-Volatilité:" + "   " + str(format(volatilite1, ".2f")) + "%")
                                st.write("-Taux doméstique:" + "   " + str(format(rd, ".2f")) + "%")
                                st.write("-Taux étranger:" + "   " + str(format(rf, ".2f")) + "%")

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)

                                st.write("- si  " + "   " + str(ticker), " " + " > " + str(format(k1, ".4f")))
                                st.write(" Achat  " + str(N), " à " + " " + str(format(k1, ".4f")))

                                st.write("- si  " + "   " + str(format(k1, ".4f")) + " < " + str(ticker),
                                         " " + " <" + str(format(k2, ".4f")))
                                st.write(" Achat  " + str(N), "au spot du jour")

                                st.write("- si  " + "   " + str(ticker), " ""   < " + str(format(k2, ".4f")))
                                st.write(" Achat  " + str(m * N), " à " + " " + str(format(k2, ".4f")))



                        else:
                            rd1 = rd / 100
                            rf1 = rf / 100
                            sigma1 = sigma / 100
                            r1 = (rd1 - rf1)

                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365

                            Ft = cours_terme(s, T1, rd1, rf1)
                            c = m * BS_CALL(s, k2, T1, rd1, rf1, sigma1)
                            p = BS_PUT(s, k1, T1, rd1, rf1, sigma1)

                            tunnel3 = (c - p)
                            tunnel4 = (c - p) * s * N

                            # Ft = cours_terme(s, T1, rd1, rf1)
                            # p2 = BS_PUT(s, k, T1, rd1, rf1, sigma1)
                            # put3 = p2 * (Perc1 * N)
                            # f2 = (k - Ft) * np.exp(-(rd1 - rf1) * T1)
                            # forward3 = f2* (1 - Perc1) * N * s
                            #
                            # Prime_partcipative3 = ((p2 + f2) / k) * 100
                            # Prime_partcipative4 = put3+ forward3

                            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                                [1, 60, 1, 1, 1, 1, 1, 1, 1])
                            with col2:

                                if ticker == 'EUR/MAD':

                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en valeur: " + format(
                                            tunnel3, ".3f") + " </h2>",
                                        unsafe_allow_html=True)

                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en MAD : " + format(
                                            tunnel4, ".3f") + " " + "DH" + " </h2>",
                                        unsafe_allow_html=True)

                                elif ticker == 'USD/MAD':
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en valeur : " + format(
                                            tunnel3, ".3f") + " </h2>",
                                        unsafe_allow_html=True)
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en MAD : " + format(
                                            tunnel4, ".3f") + " " + "DH" + " </h2>",
                                        unsafe_allow_html=True)

                            if st.button(f'Information sur opération '):
                                df = pd.DataFrame(
                                    [ticker, format(N, ".2f"), format(s, ".4f"), format(k1, ".4f"),
                                     format(k2, ".4f") + str("%"),
                                     format(tunnel4, ".2f")],
                                    index=pd.Index(
                                        ["Paire de devise", "Nominal", "Spot", "Strike k1", "Strike k2",
                                         "Tunnel Symétrique"]))
                                tdf = df.T
                                # CSS to inject contained in a string
                                hide_table_row_index = """
                                                                                                <style>
                                                                                                tbody th {display:none}
                                                                                                .blank {display:none}
                                                                                                </style>
                                                                                                """

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                    unsafe_allow_html=True)
                                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                                st.table(tdf)
                                st.session_state['button'] = False

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)

                                st.write(
                                    "-Spot " + " " + str(ticker) + "          " + ":" + " " + str(format(s, ".4f")))
                                st.write("-Volatilité:" + "   " + str(format(sigma, ".1f")) + "%")
                                st.write("-Taux doméstique:" + "   " + str(format(rd, ".1f")) + "%")
                                st.write("-Taux étranger:" + "   " + str(format(rf, ".1f")) + "%")

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)

                                st.write("- Si  " + "   " + str(ticker), " " + " < " + str(format(k1, ".4f")))
                                st.write(" Achat  " + str(m*N), " à " + " " + str(format(k1, ".4f")))

                                st.write("- Si  " + "   " + str(format(k1, ".4f")) + " < " + str(ticker),
                                         " " + " <" + str(format(k2, ".4f")))
                                st.write(" Achat  " + str(N), "au spot du jour")

                                st.write("- Si  " + "   " + str(ticker), " "" >" + str(format(k2, ".4f")))
                                st.write(" Achat  " + str( N), " à " + " " + str(format(k2, ".4f")))

                if genre == 'Stratégie gratuite':

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/MAD', 'USD/MAD'))
                        # st.markdown(f" Paire  : {ticker}")
                    with col2:
                        N = float(st.text_input('Nominal',value=50000))
                        # st.markdown(f"Nominal : {N}")
                    with col3:
                        s = float(st.text_input('Spot',value=10.587))
                        # st.markdown(f"Spot  : {s}")

                    with col4:
                        exercise_date = st.date_input('Maturité', min_value=datetime.today() + timedelta(days=1),
                                                      value=datetime.today() + timedelta(days=365))
                        # st.markdown(f" Date de valeur :{exercise_date}")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        m =float(st.text_input("Coefficient d'asymétrie",value=2))

                    with col2:
                        rd = float(st.text_input('Taux doméstique (%)',value=5))
                        # st.markdown(f"Taux doméstique : {r} %")

                    with col3:
                        rf = float(st.text_input('Taux étranger (%)',value=4))
                        # st.markdown(f"Taux étranger : {r2} %")
                    input_volatility = st.checkbox("Volatilité " + str(ticker), value=True)

                    if input_volatility:
                        sigma = st.number_input('Volatilité ' + str(ticker))
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            vol1 = st.number_input('Volatilité EUR/USD')
                            # st.markdown(f" Volatilitée : {sigma} %")
                        with col2:
                            st.markdown(
                                "<h2 style='text-align: left ;color:black;font-size:18px;'> Pondération " + " </h2>",
                                unsafe_allow_html=True)
                            vol2 = st.number_input('Poids EUR', 0, 100, 60)
                            # st.markdown(f" Volatilitée : {sigma} %")
                        with col2:
                            vol3 = st.number_input('Poids USD', 0, 100, 40)

                    st.markdown(
                        "-------------------------------------------------------------------------------------------")
                    if not all([s, rd, rf, N, m]):
                        st.error("Veuillez remplir tous les champs")
                        button = None
                    else:
                        button = st.button(f'Calculer le strike ')

                    if st.session_state.get('button') != True:
                        st.session_state['button'] = button

                    if st.session_state['button'] == True:

                        if not input_volatility:
                            rd1 = rd / 100
                            rf1 = rf / 100
                            # sigma1 = sigma / 100
                            sig1 = vol1 / 100
                            sig2 = vol2 / 100
                            sig3 = vol3 / 100

                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365
                            if ticker == 'EUR/MAD':
                                volatilite = sig1 * sig3
                            elif ticker == 'USD/MAD':
                                volatilite = sig1 * sig2
                            volatilite1 = volatilite * 100

                            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                                [1, 60, 1, 1, 1, 1, 1, 1, 1])

                            with col2:

                                def cours_terme(s, T1, rd, rf):
                                    r1 = rd - rf
                                    F_k = s * np.exp(r1 * T1)
                                    return F_k


                                Ft = cours_terme(s, T1, rd1, rf1)


                                def BS_CALL1(s, k, T1, rd, rf, volatilite):
                                    d1 = (np.log(s / k) + ((rd - rf) + volatilite ** 2 / 2) * T1) / (
                                            volatilite * np.sqrt(T1))
                                    d2 = d1 - volatilite * np.sqrt(T1)
                                    Call = s * n(d1) * np.exp(-rf * T1) - k * np.exp(-rd * T1) * n(d2)
                                    return Call


                                def BS_PUT1(s, k, T1, rd, rf, volatilite):
                                    d1 = (np.log(s / k) + ((rd - rf) + volatilite ** 2 / 2) * T1) / (
                                                volatilite * np.sqrt(T1))
                                    d2 = d1 - volatilite * np.sqrt(T1)
                                    Put = k * np.exp(-rd * T1) * n(-d2) - s * n(-d1) * np.exp(-rf * T1)
                                    return Put


                                def Ft_gt_k1(k):
                                    k1 = k[0]
                                    return Ft - k1 - 0.0000000000000000000000000000000000001


                                def K2_gt_Ft(k):
                                    k2 = k[1]
                                    return k2 - Ft - 0.0000000000000000000000000000000000001


                                def objective1(k):
                                    k1 = k[0]
                                    k2 = k[1]
                                    return ((BS_PUT(s, k2, T1, rd1, rf1, volatilite) - m * BS_CALL(s, k1, T1, rd1, rf1,
                                                                                                   volatilite))) ** 2


                                k0 = [s, s]

                                con1 = {'type': 'ineq', 'fun': Ft_gt_k1}
                                con2 = {'type': 'ineq', 'fun': K2_gt_Ft}

                                con = [con1, con2]

                                optimize = sci_opt.minimize(objective1, k0, method='SLSQP', constraints=con,
                                                            options={'disp': True})

                                solution1 = optimize.x[0]
                                solution2 = optimize.x[1]

                                # x0 = s
                                #
                                # def constrainte(k):
                                #     return k - cours_terme(s, T1, rd1, rf1)
                                #
                                # def objective(k):
                                #     g = (N * (1 - Perc1) * ((k - s * np.exp((rd1 - rf1) * T1)) * np.exp(
                                #         -(rd1 - rf1) * T1)) + N * Perc1 * BS_PUT(k, T1, s, rd1, rf1, volatilite)) ** 2
                                #     return g
                                #
                                #
                                # cons = ({'type': 'ineq', 'fun': lambda x: x - s})
                                #
                                # optimize = sci_opt.minimize(objective, x0, method='SLSQP', constraints=cons,
                                #                             options={'disp': True})
                                #
                                # k1 = optimize.x
                                # st.markdown(k1)

                                # Ft = cours_terme(s, T1, rd1, rf1)
                                # k1 = find_strike(s, rd1, rf1, T1, volatilite)
                                #
                                # put = BS_PUT(s, k1, T1, rd1, rf1, volatilite)
                                # forward = (k1 - Ft) * np.exp(-r1 * T1)
                                # b = forward + put

                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:20px;'>  Le strike est :  </h2>",
                                    unsafe_allow_html=True)

                                st.markdown(
                                    "<h2 style='text-align: left ;color:;font-size:15px;'>  - K1 = : " + str(
                                        format(solution1, '.4f')) + " </h2>",
                                    unsafe_allow_html=True)

                                st.markdown(
                                    "<h2 style='text-align: left ;color:;font-size:15px;'>  - K2 = : " + str(
                                        format(solution2, '.4f')) + " </h2>",
                                    unsafe_allow_html=True)

                            if st.button(f'Information sur opération '):
                                df = pd.DataFrame(
                                    [ticker, format(N, ".2f"), format(s, ".4f"), format(solution1, ".4f"),
                                     format(solution2, ".4f")],
                                    index=pd.Index(
                                        ["Paire de devise", "Nominal", "Spot", "Strike k1", "Strike k2"]))
                                tdf = df.T
                                # CSS to inject contained in a string
                                hide_table_row_index = """
                                                                                                <style>
                                                                                                tbody th {display:none}
                                                                                                .blank {display:none}
                                                                                                </style>
                                                                                                """

                                ######################### detail de strategie

                                # Inject CSS with Markdown
                                ############ table recap

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                    unsafe_allow_html=True)

                                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                                st.table(tdf)
                                st.session_state['button'] = False

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)

                                st.write(
                                    "-Spot " + " " + str(ticker) + "          " + ":" + " " + str(format(s, ".4f")))
                                st.write("-Volatilité:" + "   " + str(format(volatilite1, ".2f")) + "%")
                                st.write("-Taux doméstique:" + "   " + str(format(rd, ".2f")) + "%")
                                st.write("-Taux étranger:" + "   " + str(format(rf, ".2f")) + "%")

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)

                                st.write("- Si  " + "   " + str(ticker), " " + "  > " + str(format(solution1, ".4f")))
                                st.write(" Achat  " + str(m * N), " à " + " " + str(format(solution1, ".4f")))

                                st.write("- Si  " + "   " + str(format(solution1, ".4f")) + " < " + str(ticker),
                                         " " + " <" + str(format(solution2, ".4f")))
                                st.write(" Achat  " + str(N), "au spot du jour")

                                st.write("- Si  " + "   " + str(ticker), " "" < " + str(format(solution2, ".4f")))
                                st.write(" Achat  " + str(N), " à " + " " + str(format(solution2, ".4f")))


                        else:
                            rd1 = rd / 100
                            rf1 = rf / 100
                            sigma1 = sigma / 100
                            r1 = (rd1 - rf1)

                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365
                            n = norm.cdf


                            def cours_terme(s, T1, rd, rf):
                                r1 = rd - rf
                                F_k = s * np.exp(r1 * T1)
                                return F_k


                            Ft = cours_terme(s, T1, rd1, rf1)


                            def BS_CALL1(s, k, T1, rd, rf, sigma1):
                                d1 = (np.log(s / k) + ((rd - rf) + sigma1 ** 2 / 2) * T1) / (sigma1 * np.sqrt(T1))
                                d2 = d1 - sigma1 * np.sqrt(T1)
                                Call = s * n(d1) * np.exp(-rf * T1) - k * np.exp(-rd * T1) * n(d2)
                                return Call


                            def BS_PUT1(s, k, T1, rd, rf, sigma1):

                                d1 = (np.log(s / k) + ((rd - rf) + sigma1 ** 2 / 2) * T1) / (sigma1 * np.sqrt(T1))
                                d2 = d1 - sigma1 * np.sqrt(T1)
                                Put = k * np.exp(-rd * T1) * n(-d2) - s * n(-d1) * np.exp(-rf * T1)
                                return Put


                            def Ft_gt_k1(k):
                                k1 = k[0]
                                return Ft - k1 - 0.0000000000000000000000000000000000001


                            def K2_gt_Ft(k):
                                k2 = k[1]
                                return k2 - Ft - 0.0000000000000000000000000000000000001


                            def objective1(k):
                                k1 = k[0]
                                k2 = k[1]
                                return ((BS_PUT(s, k2, T1, rd1, rf1, sigma1) - m * BS_CALL(s, k1, T1, rd1, rf1,
                                                                                           sigma1))) ** 2


                            k0 = [s, s]

                            con1 = {'type': 'ineq', 'fun': Ft_gt_k1}
                            con2 = {'type': 'ineq', 'fun': K2_gt_Ft}

                            con = [con1, con2]

                            optimize = sci_opt.minimize(objective1, k0, method='SLSQP', constraints=con,
                                                        options={'disp': True})

                            solution3 = optimize.x[0]
                            solution4 = optimize.x[1]

                            # def cour_t(s, T, rd, rf):
                            #     F = s * np.exp((rd - rf) * T)
                            #     return F
                            #
                            #
                            # Ft = cour_t(s, T1, rd1, rf1)
                            # x0= s
                            #
                            #
                            # def constrainte(k):
                            #     return k - cour_t(s, T1, rd1, rf1)
                            #
                            #
                            # def objective(k):
                            #     g = (N * (1-Perc1) * ((k - s * np.exp((rd1 - rf1) * T1)) * np.exp( -(rd1 - rf1) * T1)) + N * Perc1 * BS_PUT(s, k, T1, rd1 , rf1, sigma1)) ** 2
                            #     return g
                            #
                            #
                            # cons = ({'type': 'ineq', 'fun': lambda x: x - s})
                            #
                            # optimize = sci_opt.minimize(objective, x0, method='SLSQP', constraints=cons, options={'disp': True})
                            #
                            # k2 =optimize.x
                            # st.markdown(k2)
                            # # Ft = cours_terme(s, T1, rd1, rf1)

                            # k2 = find_strike(s, rd1, rf1, T1, sigma1)
                            # st.markdown(Ft)
                            # st.markdown(k2)
                            # put = BS_PUT(s, k2, T1, rd1, rf1, sigma)
                            # forward = (k2 - Ft) * np.exp(-r1 * T1)
                            # b = forward + put

                            # col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                            #     [1, 60, 1, 1, 1, 1, 1, 1, 1])
                            # with col2:
                            #
                            #     st.markdown(
                            #         "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Le Strike est : " + format(
                            #             k2 , ".3f")+ " </h2>",
                            #         unsafe_allow_html=True)

                            st.markdown(
                                "<h2 style='text-align: left ;color:black;font-size:20px;'>  Le strike est :  </h2>",
                                unsafe_allow_html=True)

                            st.markdown(
                                "<h2 style='text-align: left ;color:;font-size:15px;'>  - K1 = : " + str(
                                    format(solution3, '.4f')) + " </h2>",
                                unsafe_allow_html=True)

                            st.markdown(
                                "<h2 style='text-align: left ;color:;font-size:15px;'>  - K2 = : " + str(
                                    format(solution4, '.4f')) + " </h2>",
                                unsafe_allow_html=True)

                            if st.button(f'Information sur opération'):
                                df = pd.DataFrame(
                                    [ticker, format(N, ".2f"), format(s, ".4f"), format(solution3, ".4f"),
                                     format(solution4, ".4f")],
                                    index=pd.Index(["Paire de devise", "Nominal", "Spot", "Strike k1", "Strike k2"]))
                                tdf1 = df.T
                                # CSS to inject contained in a string
                                hide_table_row_index = """
                                                                                                                                <style>
                                                                                                                                tbody th {display:none}
                                                                                                                                .blank {display:none}
                                                                                                                                </style>
                                                                                                                                """

                                ######################### detail de strategie

                                # Inject CSS with Markdown
                                ############ table recap

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                    unsafe_allow_html=True)

                                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                                st.table(tdf1)
                                st.session_state['button'] = False

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)

                                st.write(
                                    "-Spot " + " " + str(ticker) + "          " + ":" + " " + str(format(s, ".4f")))
                                st.write("-Volatilité:" + "   " + str(format(sigma, ".2f")) + "%")
                                st.write("-Taux doméstique:" + "   " + str(format(rd, ".2f")) + "%")
                                st.write("-Taux étranger:" + "   " + str(format(rf, ".2f")) + "%")

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)

                                st.write("- Si  " + "   " + str(ticker), " " + " > " + str(format(solution3, ".4f")))
                                st.write(" Achat  " + str(m * N), " à " + " " + str(format(solution3, ".4f")))

                                st.write("- Si  " + "   " + str(format(solution3, ".4f")) + " < " + str(ticker),
                                         " " + " <" + str(format(solution4, ".4f")))
                                st.write(" Achat  " + str(N), "au spot du jour")

                                st.write("- Si  " + "   " + str(ticker), " ""<  " + str(format(solution4, ".4f")))
                                st.write(" Achat  " + str(N), " à " + " " + str(format(solution4, ".4f")))

            #######################################################################################################################################################
            if choose == "p":

                st.header(" ")

                ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/USD', 'EUR/MAD', 'USD/MAD'))
                # st.markdown(f" Paire  : {ticker}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    N = st.number_input('Nominal')
                    # st.markdown(f"Nominal : {N}")
                with col2:
                    s = st.number_input('Spot')
                    # st.markdown(f"Spot  : {s}")
                with col3:
                    PERC = st.number_input('m')
                    # st.markdown(f" Paire  : {ticker}")

                col1, col2, col3 = st.columns(3)

                with col1:
                    k1 = st.number_input('Strike k1')
                    # st.markdown(f"Percentage : {k1} %")

                with col2:
                    k2 = st.number_input('Strike k2 ')
                    # .markdown(f"Strike  : {k2}")
                with col3:
                    exercise_date = st.date_input('Exercise date', min_value=datetime.today() + timedelta(days=1),
                                                  value=datetime.today() + timedelta(days=365))
                    # st.markdown(f" Date de valeur :{exercise_date}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    rd = st.slider('Taux doméstique (%)', 0, 100, 10)
                    # st.markdown(f"Taux doméstique : {r} %")

                with col2:
                    rf = st.slider('Taux étranger (%)', 0, 100, 10)
                    # st.markdown(f"Taux étranger : {r2} %")

                with col3:
                    sigma = st.slider('Sigma (%)', 0, 100, 20)
                    # st.markdown(f" Volatilitée : {sigma} %")

                genre = st.radio(
                    "Choisissez le type de la  stratégie ",
                    ('Stratégie payante', 'Stratégie gratuite'))

                if genre == 'Stratégie payante':

                    # gestion des exceptions
                    st.markdown("-------------")
                    if not all([s, k1, rd, rf, sigma, k2]):
                        st.error("Veuillez remplir tous les champs")
                        button = None
                    else:
                        button = st.button(f'Calculer la prime ')

                    if st.session_state.get('button') != True:
                        st.session_state['button'] = button

                    if st.session_state['button'] == True:

                        # Formating selected model parameters
                        rd1 = rd / 100
                        rf1 = rf / 100
                        sigma1 = sigma / 100
                        T = (exercise_date - datetime.now().date()).days
                        T1 = T / 365

                        # le calcul des primes
                        c = BS_CALL(s, k2, T1, rd1, rf1, sigma1)

                        p = BS_PUT(s, k1, T1, rd1, rf1, sigma1)
                        # c1 = 1/c
                        # p1=1/p
                        put = p * N
                        call = c * N
                        # forward = s - k * np.exp(-r1 * T1)
                        # t1 =c1-p1
                        tunnel = p - c
                        tunnel1 = put - call

                        # if Prime_partcipative < 0:
                        #     Prime_partcipative = 0

                        # tableau des prime en %
                        df = pd.DataFrame([format(p, ".3f"), format(c, ".3f"), format(tunnel, ".2f")],
                                          index=pd.Index(
                                              ["Put Vanille", " Call Vanille", "Tunnel Symétrique"]))

                        # tableau des primes en nominal
                        df1 = pd.DataFrame([format(put, ".3f"), format(call, ".3f"), format(tunnel1, ".2f")],
                                           index=pd.Index(
                                               ["Put Vanille", " Call Vanille", "Tunnel Symétrique"]))

                        ###############################"

                        tdf = df.T

                        # CSS to inject contained in a string
                        hide_table_row_index = """
                                                                        <style>
                                                                        tbody th {display:none}
                                                                        .blank {display:none}
                                                                        </style>
                                                                        """

                        # Inject CSS with Markdown

                        st.markdown(
                            "<h2 style='text-align: left ;color:#4a6da4;font-size:22px;'> - Tableau des primes en % </h2>",
                            unsafe_allow_html=True)
                        st.header("")
                        st.markdown(hide_table_row_index, unsafe_allow_html=True)
                        st.table(tdf)

                        #################################"

                        tdf1 = df1.T

                        # CSS to inject contained in a string
                        hide_table_row_index = """
                                                                                           <style>
                                                                                           tbody th {display:none}
                                                                                           .blank {display:none}
                                                                                           </style>
                                                                                           """

                        # Inject CSS with Markdown

                        st.markdown(
                            "<h2 style='text-align: left ;color:#4a6da4;font-size:22px;'> - Tableau des primes en nominal </h2>",
                            unsafe_allow_html=True)
                        st.header("")
                        st.markdown(hide_table_row_index, unsafe_allow_html=True)
                        st.table(tdf1)

                        # st.markdown(
                        # "<h2 style='text-align: left ;color:black;font-size:18px;'> Le prix du Put  :   </h2>",
                        # unsafe_allow_html=True)

                        # la prime du put avec le strike imposé
                        # put=BS_PUT(s, k, T1, r, sigma)
                        # st.markdown(f'   - Le prix du put est: {put}' )

                        # st.markdown(
                        # "<h2 style='text-align: left ;color:black;font-size:18px;'> Le prix du forward :   </h2>",
                        # unsafe_allow_html=True)

                        # la valeur du forward avec le strike imposé
                        # forward = s-k*np.exp(-r*T1)
                        # st.markdown(f'   - Le prix du forward  : {forward}')

                        # la prime payé

                        # Prime_partcipative =put + forward
                        # st.markdown(f'-Total Prime a payer: {Prime_partcipative}')

                        if st.button(f'Information sur opération '):
                            df = pd.DataFrame([ticker, format(N, ".2f"), format(s, ".4f"), format(k1, ".4f"),
                                               format(k2, ".4f"), format(put, ".2f"), format(call, ".2f"),
                                               format(tunnel1, ".2f")],
                                              index=pd.Index(
                                                  ["Paire de devise", "Nominal", "Spot", "Strike k1 ", "Strike k2",
                                                   "Put Vanille", "Call Vanille", " Tunnel Symétrique"]))
                            tdf = df.T
                            # CSS to inject contained in a string
                            hide_table_row_index = """
                                                        <style>
                                                        tbody th {display:none}
                                                        .blank {display:none}
                                                        </style>
                                                        """

                            ######################### detail de strategie
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.header(" ")
                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Détail de la strategie </h2>",
                                    unsafe_allow_html=True)
                                st.header("")

                                st.write("- Achat : Put Vanille")
                                st.write("- Nominal:" + "   " + str(N))
                                st.write("- Prix d'éxercice :" + "   " + str(k1))
                                st.write("- Date d'échéance:" + "   " + str(exercise_date))

                            st.header("  ")
                            with col3:
                                st.header(" ")
                                st.header(" ")
                                st.header(" ")
                                # st.markdown(
                                # "<h2 style='text-align: left ;color:green;font-size:14px;'> Dénouement a l'échéance </h2>",
                                # unsafe_allow_html=True)
                                st.header("")

                                st.write("- Vente : Call Vanille  ")
                                st.write("- Nominal:" + "   " + str(N))
                                st.write("- Prix d'éxercice :" + "   " + str(k2))
                                st.write("- Date d'échéance:" + "   " + str(exercise_date))

                            # Inject CSS with Markdown
                            ############ table recap

                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                unsafe_allow_html=True)
                            st.header("")
                            st.markdown(hide_table_row_index, unsafe_allow_html=True)
                            st.table(tdf)
                            st.session_state['button'] = False

                            ######### detail

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.header(" ")
                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)
                                st.header("")

                                st.write("-Spot " + " " + str(ticker) + "          " + ":" + " " + str(s))
                                st.write("-Volatilité:" + "   " + str(sigma) + "%")
                                st.write("-Taux doméstique:" + "   " + str(rd) + "%")
                                st.write("-Taux étranger:" + "   " + str(rf) + "%")
                                # st.write("-Volatilité:" + "   " + str(sigma))
                            st.header("  ")
                            with col3:
                                st.header(" ")
                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement a l'échéance </h2>",
                                    unsafe_allow_html=True)
                                st.header("")

                                st.write("- si  " + "   " + str(ticker), " " + " < " + str(k1))
                                st.write(" Vous vendez  " + str(N), " à " + " " + str(k1))

                                st.write("- si  " + "   " + str(k1) + " < " + str(ticker), " " + " <" + str(s))
                                st.write(" Vous vendez  " + str(N), "au spot du jour")

                                st.write("- si  " + "   " + str(ticker), " "" > " + str(k2))
                                st.write(" Vous vendez  " + str(N), " à " + " " + str(k2))

                else:

                    # gestion des exceptions
                    st.markdown("-------------")
                    if not all([s, k1, k2, rd, rf, sigma]):
                        st.error(" Veuillez remplir tous les champs ")
                        button = None
                    else:
                        button = st.button(f'Calculer le strike amélioré ')

                    if st.session_state.get('button') != True:
                        st.session_state['button'] = button

                    if st.session_state['button'] == True:

                        # Formating selected model parameters
                        rd1 = rd / 100
                        # st.markdown(rd1)
                        rf1 = rf / 100
                        # st.markdown(rf1)
                        r1 = (rd1 - rf1)
                        sigma1 = sigma / 100
                        # st.markdown(sigma1)
                        T = (exercise_date - datetime.now().date()).days
                        T1 = T / 365


                        ######################### optimisation
                        def constrainte(k):
                            k1 = k[0]
                            k2 = k[1]
                            return k2 - k1 - 0.0000000000000000000000000000000000001


                        def constrainte_2(k):
                            k1 = k[0]
                            k2 = k[1]
                            return k2 - s - 0.0000000000000000000000000000000000001


                        def constrainte_3(k):
                            k1 = k[0]
                            k2 = k[1]
                            return s - k1 - 0.0000000000000000000000000000000000001


                        def objective1(k):
                            k1 = k[0]
                            k2 = k[1]
                            return ((BS_PUT(s, k2, T1, rd1, rf1, sigma1)) - BS_CALL(s, k1, T1, rd1, rf1, sigma1)) ** 2


                        k0 = [s - 0.4, s - 0.2]
                        cons = {'type': 'ineq', 'fun': constrainte}
                        con2 = {'type': 'ineq', 'fun': constrainte_2}
                        con3 = {'type': 'ineq', 'fun': constrainte_3}
                        cons1 = [cons, con2, con3]

                        optimize = sci_opt.minimize(objective1, k0, method='SLSQP', constraints=cons1,
                                                    options={'disp': True})

                        call2 = BS_CALL(s, optimize.x[1], T1, rd1, rf1, sigma1)
                        put2 = BS_PUT(s, optimize.x[0], T1, rd1, rf1, sigma1)

                        c = optimize.x[1]
                        p = optimize.x[0]
                        tunnel = call2 - put2
                        # j = (put / 100) * (Perc1 * N)

                        st.markdown(
                            "<h2 style='text-align: left ;color:black;font-size:20px;'>  Le strike amélioré est :  </h2>",
                            unsafe_allow_html=True)

                        st.markdown(
                            "<h2 style='text-align: left ;color:;font-size:15px;'>  - K1 = : " + str(
                                format(p, '.4f')) + " </h2>",
                            unsafe_allow_html=True)

                        st.markdown(
                            "<h2 style='text-align: left ;color:;font-size:15px;'>  - K2 = : " + str(
                                format(c, '.4f')) + " </h2>",
                            unsafe_allow_html=True)

                        if st.button(f'Informtion sur opération '):
                            df = pd.DataFrame(
                                [ticker, format(N, ".2f"), format(s, ".4f"), format(p, ".4f"), format(c, ".4f")
                                    , format(call2, ".3f"), format(put2, ".3f"),
                                 format(tunnel, ".2f")],
                                index=pd.Index(
                                    ["Paire de devise", "Nominal", "Spot", "Strike k1", "Strike k2",
                                     "Call Vanille", "Put vanille", "Prime"]))

                            tdf = df.T
                            # CSS to inject contained in a string
                            hide_table_row_index = """
                                                            <style>
                                                            tbody th {display:none}
                                                            .blank {display:none}
                                                            </style>
                                                            """
                            # df1 = pd.DataFrame(
                            #     [ticker, format(N, ".2f"), format(s, ".2f"), format(k, ".2f"), format(Perc, ".2f"),
                            #      format(p, ".2f"), format(f, ".2f"), format(p-p, ".2f")],
                            #     index=pd.Index(["Paire de devise", "Nominal", "Spot", "Strike amélioré", "Participation",
                            #                     "Put Vanille", "Forward", "Prime"]))
                            #
                            # tdf1 = df1.T
                            # # CSS to inject contained in a string
                            # hide_table_row_index = """
                            #                                     <style>
                            #                                     tbody th {display:none}
                            #                                     .blank {display:none}
                            #                                     </style>
                            #                                     """

                            ######################### detail de strategie
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.header(" ")
                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Détail de la stratégie </h2>",
                                    unsafe_allow_html=True)
                                st.header("")

                                st.write("- Achat : Put ")
                                st.write("- Nominal:" + "   " + str(N))
                                st.write("- Prix d'éxercice k1 :" + "   " + str(format(k1, ".4f")))
                                st.write("- Date d'échéance:" + "   " + str(exercise_date))

                            st.header("  ")
                            with col3:
                                st.header(" ")
                                st.header(" ")
                                # st.markdown(
                                # "<h2 style='text-align: left ;color:green;font-size:14px;'> Dénouement a l'échéance </h2>",
                                # unsafe_allow_html=True)
                                st.header("")
                                st.header("")
                                st.write("- Vente : Call ")
                                st.write("- Nominal:" + "   " + str(N))
                                st.write("- Prix d'éxercice k2  :" + "   " + str(format(k2, ".4f")))
                                st.write("- Date d'échéance:" + "   " + str(exercise_date))

                            # Inject CSS with Markdown
                            ############ table recap

                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                unsafe_allow_html=True)
                            st.header("")
                            st.markdown(hide_table_row_index, unsafe_allow_html=True)
                            st.table(tdf)
                            st.session_state['button'] = False

                            ######### detail

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.header(" ")
                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)
                                st.header("")

                                st.write("-Spot " + " " + str(ticker) + "          " + ":" + " " + str(s))
                                st.write("-Volatilité:" + "   " + str(sigma) + "%")
                                st.write("-Taux doméstique:" + "   " + str(rd) + "%")
                                st.write("-Taux étranger:" + "   " + str(rf) + "%")

                            st.header("  ")
                            with col3:
                                st.header(" ")
                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)
                                st.header("")

                                st.write("- si  " + "   " + str(ticker), " " + " < " + str(format(p, ".4f")))
                                st.write(" Vous vendez  " + str(N), " à " + " " + str(format(p, ".4f")))

                                st.write("- si  " + "   " + str(format(p, ".4f")) + " < " + str(ticker),
                                         " " + " <" + str(format(c, ".4f")))
                                st.write(" Vous vendez  " + str(N), "au spot du jour")

                                st.write("- si  " + "   " + str(ticker), " "" > " + str(format(c, ".4f")))
                                st.write(" Vous vendez  " + str(N), " à " + " " + str(format(c, ".4f")))

        if selected == "EXPORT":
            # with st.sidebar:
            # choose = option_menu(menu_title=None,
            #                      options=["Forward","Put Vanille", "Put Participatif ",
            #                               "Tunnel Symétrique", "Tunnel Asymétrique"],
            #                      # icons=['house', 'camera fill', 'kanban', 'book','person lines fill'],
            #                      menu_icon="cast", default_index=1,
            #                      styles={
            #                          "container": {"padding": "5!important", "background-color": "#FFFFFFF"},
            #                          "icon": {"color": "#6cb44c", "font-size": "15px"},
            #                          "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px",
            #                                       "--hover-color": "#6cb44c"},  # eee
            #                          "nav-link-selected": {"background-color": "#6cb44c"},  # DFE8ED
            #
            #                      }
            #                      )

            choose = st.radio(
                label="Choisissez le type de la  stratégie ", horizontal=True, options=
                ('Forward', 'Put Vanille', 'Put Participatif', 'Tunnel Symétrique', 'Tunnel Asymétrique'))
            st.header("")
            if choose == "Forward":
                genre = st.radio(
                    label="Choisissez le type de la  stratégie ", horizontal=True,
                    options=('Stratégie payante', 'Stratégie gratuite'))
                if genre == 'Stratégie payante':
                    st.header("")
                    # st.markdown("-------------------------------------------------------------------------------------------------")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/MAD', 'USD/MAD'))
                        # st.markdown(f" Paire  : {ticker}")
                    with col2:
                        N = st.number_input('Nominal')
                        # st.markdown(f"Nominal : {N}")
                    with col3:
                        s = st.number_input('Spot')
                        # st.markdown(f"Spot  : {s}")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        k = st.number_input('Strike')
                        # st.markdown(f"Strike  : {k}")
                    with col2:
                        exercise_date = st.date_input('Maturité', min_value=datetime.today() + timedelta(days=1),
                                                      value=datetime.today() + timedelta(days=365))
                        # st.markdown(f" Date de valeur :{exercise_date}")
                    with col3:
                        rd = st.number_input('Taux doméstique (%)')
                        # st.markdown(f"Taux doméstique : {r} %")

                    with col4:
                        rf = st.number_input('Taux étranger (%)')
                        # st.markdown(f"Taux étranger : {r2} %")

                    if not all([N, s, rd, rf, k]):
                        st.error("Veuillez remplir tous les champs")
                        button = None
                    else:
                        button = st.button(f"Calculer la prime ")

                    if st.session_state.get('button') != True:
                        st.session_state['button'] = button

                    if st.session_state['button'] == True:

                        # Formating selected model parameters

                        rd1 = rd / 100
                        rf1 = rf / 100
                        T = (exercise_date - datetime.now().date()).days
                        T1 = T / 365

                        # calcul le prix de l'option

                        F = cours_terme(s, T1, rd1, rf1)
                        forward = (k - F) * np.exp(-(rd1 - rf1) * T1)
                        f2 = forward
                        f = N * forward * s

                        col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns([1, 60, 1, 1, 1, 1, 1, 1, 1])
                        with col2:
                            if ticker == 'EUR/MAD':
                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en valeur  : " + format(
                                        f2, ".3f") + " </h2>",
                                    unsafe_allow_html=True)
                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
                                        f, ".3f") + " " + "DH" + " </h2>",
                                    unsafe_allow_html=True)

                            elif ticker == 'USD/MAD':
                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en valeur  : " + format(
                                        f2, ".3f") + "%" + " </h2>",
                                    unsafe_allow_html=True)
                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
                                        f, ".3f") + " " + "DH" + " </h2>",
                                    unsafe_allow_html=True)

                        if st.button(f'Information sur opération '):
                            df = pd.DataFrame(
                                [" Forward ", ticker, format(N, ".2f"), format(s, ".4f"), format(k, ".4f"),
                                 format(rd, ".0f") + str("%"), format(rf, ".0f") + str("%"),
                                 exercise_date, format(f, ".2f")],
                                index=pd.Index(
                                    ["Produit", " Devise", "Nominal", "Spot", "Strike", "Taux doméstique",
                                     "Taux étranger"
                                        , "Maturité", "Prime à payer"]))

                            tdf = df.T
                            # CSS to inject contained in a string
                            hide_table_row_index = """
                                                                                    <style>
                                                                                    tbody th {display:none}
                                                                                    .blank {display:none}
                                                                                    </style>
                                                                                    """

                            # Inject CSS with Markdown
                            ############ table recap

                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                unsafe_allow_html=True)

                            st.markdown(hide_table_row_index, unsafe_allow_html=True)
                            st.table(tdf)
                            st.session_state['button'] = False

                            ######### detail

                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                unsafe_allow_html=True)

                            st.write("- Spot " + " " + "  " + ":" + str(format(s, ".4f")))
                            st.write("- Taux doméstique:" + "   " + str(rd) + " " + "%")
                            st.write("- Taux étranger:" + "   " + str(rf) + " " + "%")

                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                unsafe_allow_html=True)

                            st.write("- si  " + "   " + str(ticker), " " + " <" + str(format(k, ".4f")))
                            st.write(" Vendre  " + str(format(N, ".2f")), " à " + " " + str(format(k, ".4f")))
                            st.write("- si  " + "   " + str(ticker), " " + " >=" + str(format(k, ".4f")))
                            st.write(" Vendre  " + " " + str(format(N, ".2f")), " à" + " " + str(format(k, ".4f")))
                elif genre == 'Stratégie gratuite':
                    st.header("")
                    # st.markdown("-------------------------------------------------------------------------------------------------")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/MAD', 'USD/MAD'))
                        # st.markdown(f" Paire  : {ticker}")
                    with col2:
                        N = st.number_input('Nominal')
                        # st.markdown(f"Nominal : {N}")
                    with col3:
                        s = st.number_input('Spot')
                        # st.markdown(f"Spot  : {s}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        exercise_date = st.date_input('Maturité', min_value=datetime.today() + timedelta(days=1),
                                                      value=datetime.today() + timedelta(days=365))
                        # st.markdown(f" Date de valeur :{exercise_date}")
                    with col2:
                        rd = st.number_input('Taux doméstique (%)')
                        # st.markdown(f"Taux doméstique : {r} %")
                    with col3:
                        rf = st.number_input('Taux étranger (%)')
                        # st.markdown(f"Taux étranger : {r2} %")
                    # gestion des exceptions
                    if not all([N, s, rd, rf]):
                        st.error("Veuillez remplir tous les champs")
                        button = None
                    else:
                        button = st.button(f"Calculer le strike")

                    if st.session_state.get('button') != True:
                        st.session_state['button'] = button

                    if st.session_state['button'] == True:

                        # Formating selected model parameters

                        rd1 = rd / 100
                        rf1 = rf / 100
                        T = (exercise_date - datetime.now().date()).days
                        T1 = T / 365

                        # calcul le prix de l'option
                        F = cours_terme(s, T1, rd1, rf1)

                        col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns([1, 60, 1, 1, 1, 1, 1, 1, 1])
                        with col2:
                            st.markdown(
                                "<h2 style='text-align: left ;color:;font-size:20px;'> - Le strike est : " + " " + " " + str(
                                    format(F, '.4f')) + " </h2>",
                                unsafe_allow_html=True)

                        if st.button(f'Information sur opération '):
                            df = pd.DataFrame(
                                [" Forward ", ticker, format(N, ".2f"), format(s, ".4f"), format(F, ".4f"),
                                 format(rd, ".0f") + str("%"), format(rf, ".0f") + str("%"),
                                 exercise_date],
                                index=pd.Index(["Produit", " Devise", "Nominal", "Spot", "Strike", "Taux doméstique",
                                                "Taux étranger", "Maturité"]))

                            tdf = df.T
                            # CSS to inject contained in a string
                            hide_table_row_index = """
                                                                                                       <style>
                                                                                                       tbody th {display:none}
                                                                                                       .blank {display:none}
                                                                                                       </style>
                                                                                                       """

                            # Inject CSS with Markdown
                            ############ table recap

                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                unsafe_allow_html=True)

                            st.markdown(hide_table_row_index, unsafe_allow_html=True)
                            st.table(tdf)
                            st.session_state['button'] = False

                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                unsafe_allow_html=True)

                            st.write("- Spot " + " " + "  " + ":" + format(s, ".4f"))
                            st.write("- Taux doméstique:" + "   " + str(rd) + " " + "%")
                            st.write("- Taux étranger:" + "   " + str(rf) + " " + "%")

                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                unsafe_allow_html=True)

                            st.write("- si  " + "   " + str(ticker), " " + " <" + str(format(F, ".4f")))
                            st.write("  Vendre  " + str(N), " à " + " " + str(format(F, ".4f")))
                            st.write("- si  " + "   " + str(ticker), " " + " >=" + str(format(F, ".4f")))
                            st.write("  Vendre  " + " " + str(N), " à" + " " + str(format(F, ".4f")))
            ###########################################################################################################################################
            # if choose == "p" :
            #     genre = st.radio(
            #         label="Choisissez le type de la  stratégie ", horizontal=True,
            #         options=('Stratégie payante', 'Stratégie gratuite'))
            #     if genre == 'Stratégie payante':
            #         st.header("")
            #         col1, col2, col3, col4 = st.columns(4)
            #         with col1:
            #             ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/MAD', 'USD/MAD'))
            #         # st.markdown(f" Paire  : {ticker}")
            #         with col2:
            #             N = st.number_input('Nominal')
            #             # st.markdown(f"Nominal : {N}")
            #         with col3:
            #             s = st.number_input('Spot')
            #             # st.markdown(f"Spot  : {s}")
            #         with col4:
            #             k = st.number_input('Strike ')
            #             # st.markdown(f"Strike  : {k}")
            #         col1, col2, col3, col4 = st.columns(4)
            #         with col1:
            #             exercise_date = st.date_input('Exercise date', min_value=datetime.today() + timedelta(days=1),
            #                                           value=datetime.today() + timedelta(days=365))
            #             # st.markdown(f" Date de valeur :{exercise_date}")
            #
            #         with col2:
            #             rd = st.number_input('Taux doméstique (%)', 0, 100, 0)
            #             # st.markdown(f"Taux doméstique : {r} %")
            #
            #         with col3:
            #             rf = st.number_input('Taux étranger (%)', 0, 100, 0)
            #             # st.markdown(f"Taux étranger : {r2} %")
            #         with col4:
            #             sigma = st.number_input('Volatilité', 0, 100, 20)
            #             # st.markdown(f" Volatilitée : {sigma} %")
            #
            #         if sigma == 0:
            #             col1, col2 = st.columns(2)
            #             with col1:
            #                 vol1 = st.number_input('Volatilité EUR/USD', 0, 100, 0)
            #                 # st.markdown(f" Volatilitée : {sigma} %")
            #             with col2:
            #                 st.markdown(
            #                     "<h2 style='text-align: left ;color:black;font-size:18px;'> Pondération " + " </h2>",
            #                     unsafe_allow_html=True)
            #                 vol2 = st.number_input('Poids EUR', 0, 100, 60)
            #                 # st.markdown(f" Volatilitée : {sigma} %")
            #             with col2:
            #                 vol3 = st.number_input('Poids USD', 0, 100, 40)
            #
            #         st.markdown( "-------------------------------------------------------------------------------------------")
            #         # gestion des exceptions
            #         if not all([N, s, k, rd, rf]):
            #             st.error("Veuillez remplir tous les champs")
            #             button = None
            #         else:
            #             button = st.button(f"Calculer le prix de l'option")
            #
            #         if st.session_state.get('button') != True:
            #             st.session_state['button'] = button
            #
            #         if st.session_state['button'] == True:
            #             # Formating selected model parameters
            #
            #             if sigma == 0:
            #                 rd1 = rd / 100
            #                 rf1 = rf / 100
            #                 sigma1 = sigma / 100
            #                 sig1 = vol1 / 100
            #                 sig2 = vol2 / 100
            #                 sig3 = vol3 / 100
            #                 T = (exercise_date - datetime.now().date()).days
            #                 T1 = T / 365
            #                 if ticker == 'EUR/MAD':
            #                     volatilite = sig1 * sig3
            #                 elif ticker == 'USD/MAD':
            #                     volatilite = sig1 * sig2
            #
            #                 col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
            #                     [1, 60, 1, 1, 1, 1, 1, 1, 1])
            #                 with col2:
            #
            #                     put = BS_PUT(s, k, T1, rd1, rf1, volatilite)
            #                     p1 = put / k
            #                     p2 = put * N * s
            #
            #                     if ticker == 'EUR/MAD':
            #
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en % du Strike : " + format(
            #                                 p1, ".3f") +  " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
            #                                 p2, ".3f") + " " + "DH" + " </h2>",
            #                             unsafe_allow_html=True)
            #
            #                     elif ticker == 'USD/MAD':
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en % du Nominal : " + format(
            #                                 p1, ".3f") + " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
            #                                 p2, ".3f") + " " + "DH" + " </h2>",
            #                             unsafe_allow_html=True)
            #
            #                 if st.button(f'Information sur opération '):
            #
            #                     volatilite1 = volatilite * 100
            #                     df = pd.DataFrame(
            #                         ["Put Vanille", ticker, format(N, ".2f"), format(s, ".4f"), format(k, ".4f"),
            #                          format(rd, ".0f") + str("%"), format(rf, ".0f") + str("%"),
            #                          format(volatilite1, ".0f") + str("%"), exercise_date, format(p2, ".2f")],
            #                         index=pd.Index(
            #                             ["Produit", " Devise", "Nominal", "Spot", "Strike", "Taux doméstique",
            #                              "Taux étranger", "Volatilitée", "Maturité", "Prime à payer"]))
            #
            #                     tdf = df.T
            #                     # CSS to inject contained in a string
            #                     hide_table_row_index = """
            #                                                             <style>
            #                                                             tbody th {display:none}
            #                                                             .blank {display:none}
            #                                                             </style>
            #                                                             """
            #
            #                     # Inject CSS with Markdown
            #                     ############ table recap
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
            #                         unsafe_allow_html=True)
            #                     st.markdown(hide_table_row_index, unsafe_allow_html=True)
            #                     st.table(tdf)
            #                     st.session_state['button'] = False
            #
            #                     ######### detail
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
            #                         unsafe_allow_html=True)
            #
            #                     st.write("- Spot " + " " + "  " + ":" + str(format(s, ".4f")))
            #                     st.write("- Taux doméstique:" + "   " + str(rd) + " " + "%")
            #                     st.write("- Taux étranger:" + "   " + str(rf) + " " + "%")
            #                     st.write("- Volatilité:" + "   " + str(volatilite) + " " + "%")
            #
            #
            #                     st.header(" ")
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
            #                         unsafe_allow_html=True)
            #                     st.header("")
            #
            #                     st.write("- si  " + "   " + str(ticker), " " + " <" + str(k))
            #                     st.write(" Vendre  " + str(N), " à " + " " + str(k))
            #                     st.write("- si  " + "   " + str(ticker), " " + " >=" + str(s))
            #                     st.write(" Pas d'exercice ")
            #                     st.write(" Vendre  " + " " + str(N), " au prix spot")
            #
            #             else:
            #                 rd1 = rd / 100
            #                 rf1 = rf / 100
            #                 sigma1 = sigma / 100
            #                 T = (exercise_date - datetime.now().date()).days
            #                 T1 = T / 365
            #
            #                 col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
            #                     [1, 60, 1, 1, 1, 1, 1, 1, 1])
            #                 with col2:
            #
            #                     put = BS_PUT(s, k, T1, rd1, rf1, sigma1)
            #                     p1 = put /k
            #                     p2 = put * N * s
            #
            #                     if ticker == 'EUR/MAD':
            #
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en % du Strike : " + format(
            #                                 p1, ".3f") + " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
            #                                 p2, ".3f") + " " + "DH" + " </h2>",
            #                             unsafe_allow_html=True)
            #
            #                     elif ticker == 'USD/MAD':
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en % du Strike : " + format(
            #                                 p1, ".3f") +  " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
            #                                 p2, ".3f") + " " + "DH" + " </h2>",
            #                             unsafe_allow_html=True)
            #
            #                 if st.button(f'Information sur opération '):
            #
            #
            #                     df = pd.DataFrame(
            #                         ["Put Vanille", ticker, format(N, ".2f"), format(s, ".4f"), format(k, ".4f"),
            #                          format(rd, ".0f") + str("%"), format(rf, ".0f") + str("%"),
            #                          format(sigma, ".0f") + str("%"), exercise_date, format(p2, ".2f")],
            #                         index=pd.Index(
            #                             ["Produit", " Devise", "Nominal", "Spot", "Strike", "Taux doméstique",
            #                              "Taux étranger", "Volatilitée", "Maturité", "Prime à payer"]))
            #
            #                     tdf = df.T
            #                     # CSS to inject contained in a string
            #                     hide_table_row_index = """
            #                                                             <style>
            #                                                             tbody th {display:none}
            #                                                             .blank {display:none}
            #                                                             </style>
            #                                                             """
            #
            #                     # Inject CSS with Markdown
            #                     ############ table recap
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
            #                         unsafe_allow_html=True)
            #                     st.markdown(hide_table_row_index, unsafe_allow_html=True)
            #                     st.table(tdf)
            #                     st.session_state['button'] = False
            #
            #                     ######### detail
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
            #                         unsafe_allow_html=True)
            #
            #                     st.write("- Spot " + " " + "  " + ":" + str(format(s, ".4f")))
            #                     st.write("- Taux doméstique:" + "   " + str(rd) + " " + "%")
            #                     st.write("- Taux étranger:" + "   " + str(rf) + " " + "%")
            #                     st.write("- Volatilité:" + "   " + str(sigma) + " " + "%")
            #
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
            #                         unsafe_allow_html=True)
            #
            #                     st.write("- si  " + "   " + str(ticker), " " + " <" + str(format(k, ".4f")))
            #                     st.write(" Vendre  " + str(N), " à " + " " + str(format(k, ".4f")))
            #                     st.write("- si  " + "   " + str(ticker), " " + " >=" + str(format(k, ".4f")))
            #                     st.write(" Pas d'exercice ")
            #                     st.write(" Vendre  " + " " + str(N), " au prix spot")
            #
            #
            #
            #
            #     if genre == 'Stratégie gratuite':
            #
            #         st.header("")
            #         col1, col2, col3 = st.columns(3)
            #         with col1:
            #             ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/MAD', 'USD/MAD'))
            #         # st.markdown(f" Paire  : {ticker}")
            #         with col2:
            #             N = st.number_input('Nominal')
            #             # st.markdown(f"Nominal : {N}")
            #         with col3:
            #             s = st.number_input('Spot')
            #             # st.markdown(f"Spot  : {s}")
            #
            #
            #         col1, col2, col3, col4 = st.columns(4)
            #         with col1:
            #             exercise_date = st.date_input('Exercise date', min_value=datetime.today() + timedelta(days=1),
            #                                           value=datetime.today() + timedelta(days=365))
            #             # st.markdown(f" Date de valeur :{exercise_date}")
            #
            #         with col2:
            #             rd = st.number_input('Taux doméstique (%)', 0, 100, 10)
            #             # st.markdown(f"Taux doméstique : {r} %")
            #
            #         with col3:
            #             rf = st.number_input('Taux étranger (%)', 0, 100, 10)
            #             # st.markdown(f"Taux étranger : {r2} %")
            #         with col4:
            #             sigma = st.number_input('Volatilité', 0, 100, 20)
            #             # st.markdown(f" Volatilitée : {sigma} %")
            #
            #         if sigma == 0:
            #             col1, col2 = st.columns(2)
            #             with col1:
            #                 vol1 = st.number_input('Volatilité EUR/USD', 0, 100, 0)
            #                 # st.markdown(f" Volatilitée : {sigma} %")
            #             with col2:
            #                 st.markdown(
            #                     "<h2 style='text-align: left ;color:black;font-size:18px;'> Pondération " + " </h2>",
            #                     unsafe_allow_html=True)
            #                 vol2 = st.number_input('Poids EUR', 0, 100, 60)
            #                 # st.markdown(f" Volatilitée : {sigma} %")
            #             with col2:
            #                 vol3 = st.number_input('Poids USD', 0, 100, 40)
            #
            #         st.markdown( "-------------------------------------------------------------------------------------------")
            #
            #         # gestion des exceptions
            #         if not all([N, s,  rd, rf]):
            #             st.error("Veuillez remplir tous les champs")
            #             button = None
            #         else:
            #             button = st.button(f"Calculer le prix de l'option")
            #
            #         if st.session_state.get('button') != True:
            #             st.session_state['button'] = button
            #
            #         if st.session_state['button'] == True:
            #
            #             if sigma == 0:
            #                 rd1 = rd / 100
            #                 rf1 = rf / 100
            #                 sigma1 = sigma / 100
            #                 sig1 = vol1 / 100
            #                 sig2 = vol2 / 100
            #                 sig3 = vol3 / 100
            #                 T = (exercise_date - datetime.now().date()).days
            #                 T1 = T / 365
            #
            #                 if ticker == 'EUR/MAD':
            #                     volatilite = sig1 * sig3
            #                 elif ticker == 'USD/MAD':
            #                     volatilite = sig1 * sig2
            #
            #                 col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
            #                     [1, 60, 1, 1, 1, 1, 1, 1, 1])
            #                 with col2:
            #
            #                     def find_strike1(s, rd, rf, T, sigma):
            #                         def objectif_func(k1):
            #                             return  BS_PUT(s, k1, T, rd, rf,sigma)
            #                         sol = optimize.root_scalar(objectif_func, bracket=[1,1e8], method='brentq')
            #                         return sol.root
            #
            #                     k1 = find_strike1(s, rd1, rf1, T1, volatilite)
            #                     put = BS_PUT(s, k1, T1, rd1, rf1, volatilite)
            #                     st.markdown(put)
            #
            #
            #
            #                     p2 = put * N * s
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en % du Strike : " + format(
            #                             k1, ".3f") +  " </h2>",
            #                         unsafe_allow_html=True)
            #             else :
            #                 ##############
            #                 st.markdown(
            #                     "walo")
            #
            #
            #
            #
            #
            #
            #
            #
            #
            #

            ################################################################################################################################
            if choose == "Put Vanille":
                st.header("")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/MAD', 'USD/MAD'))
                # st.markdown(f" Paire  : {ticker}")
                with col2:
                    N = st.number_input('Nominal')
                    # st.markdown(f"Nominal : {N}")
                with col3:
                    s = st.number_input('Spot')
                    # st.markdown(f"Spot  : {s}")
                with col4:
                    k = st.number_input('Strike ')
                    # st.markdown(f"Strike  : {k}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    exercise_date = st.date_input('Maturité', min_value=datetime.today() + timedelta(days=1),
                                                  value=datetime.today() + timedelta(days=365))
                    # st.markdown(f" Date de valeur :{exercise_date}")

                with col2:
                    rd = st.number_input('Taux doméstique (%)')
                    # st.markdown(f"Taux doméstique : {r} %")
                with col3:
                    rf = st.number_input('Taux étranger (%)')
                    # st.markdown(f"Taux étranger : {r2} %")

                input_volatility = st.checkbox("Volatilité EUR/MAD", value=True)
                # st.markdown(f" Volatilitée : {sigma} %")
                if input_volatility:
                    sigma = st.number_input('Volatilité')
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        vol1 = st.number_input('Volatilité EUR/USD')
                        # st.markdown(f" Volatilitée : {sigma} %")
                    with col2:
                        st.markdown(
                            "<h2 style='text-align: left ;color:black;font-size:18px;'> Pondération " + " </h2>",
                            unsafe_allow_html=True)
                        vol2 = st.number_input('Poids EUR', value=60)
                        # st.markdown(f" Volatilitée : {sigma} %")
                    with col2:
                        vol3 = st.number_input('Poids USD', value=40)

                st.markdown(
                    "-------------------------------------------------------------------------------------------")

                # gestion des exceptions
                if not all([N, s, k, rd, rf]):
                    st.error("Veuillez remplir tous les champs")
                    button = None
                else:
                    button = st.button(f"Calculer la prime")

                if st.session_state.get('button') != True:
                    st.session_state['button'] = button

                if st.session_state['button'] == True:
                    # Formating selected model parameters

                    if not input_volatility:
                        rd1 = rd / 100
                        rf1 = rf / 100
                        # sigma1 = sigma / 100
                        sig1 = vol1 / 100
                        sig2 = vol2 / 100
                        sig3 = vol3 / 100
                        T = (exercise_date - datetime.now().date()).days
                        T1 = T / 365
                        if ticker == 'EUR/MAD':
                            volatilite = sig1 * sig3
                        elif ticker == 'USD/MAD':
                            volatilite = sig1 * sig2

                        col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns([1, 60, 1, 1, 1, 1, 1, 1, 1])
                        with col2:

                            put = BS_PUT(s, k, T1, rd1, rf1, volatilite)
                            p1 = put
                            p2 = put * N * s

                            if ticker == 'EUR/MAD':

                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en valeur : " + format(
                                        p1, ".3f") + " </h2>",
                                    unsafe_allow_html=True)
                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
                                        p2, ".3f") + " " + "DH" + " </h2>",
                                    unsafe_allow_html=True)

                            elif ticker == 'USD/MAD':
                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en valeur : " + format(
                                        p1, ".3f") + " " + "%" + "  </h2>",
                                    unsafe_allow_html=True)
                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
                                        p2, ".3f") + " " + "DH" + " </h2>",
                                    unsafe_allow_html=True)
                        st.header("")
                        if st.button(f'Information sur opération '):
                            volatilite1 = volatilite * 100
                            df = pd.DataFrame(
                                ["Put Vanille", ticker, format(N, ".2f"), format(s, ".4f"), format(k, ".4f"),
                                 format(rd, ".1f") + str("%"), format(rf, ".1f") + str("%"),
                                 format(volatilite1, ".1f") + str("%"), exercise_date, format(p2, ".2f")],
                                index=pd.Index(["Produit", " Devise", "Nominal", "Spot", "Strike", "Taux doméstique",
                                                "Taux étranger", "Volatilité", "Maturité", "Prime à payer"]))

                            tdf = df.T
                            # CSS to inject contained in a string
                            hide_table_row_index = """
                                                                        <style>
                                                                        tbody th {display:none}
                                                                        .blank {display:none}
                                                                        </style>
                                                                        """
                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                unsafe_allow_html=True)
                            st.markdown(hide_table_row_index, unsafe_allow_html=True)
                            st.table(tdf)
                            st.session_state['button'] = False

                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                unsafe_allow_html=True)

                            st.write("- Spot " + " " + "  " + ":" + str(format(s, ".4f")))
                            st.write("- Taux doméstique:" + "   " + str(format(rd, ".1f")) + " " + "%")
                            st.write("- Taux étranger:" + "   " + str(format(rf, ".1f")) + " " + "%")
                            st.write("- Volatilité:" + "   " + str(format(volatilite1, ".1f")) + " " + "%")

                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                unsafe_allow_html=True)

                            st.write("- si  " + "   " + str(ticker), " " + " <" + str(format(k, ".4f")))
                            st.write("  Vendre  " + str(N), " à " + " " + str(format(k, ".4f")))
                            st.write("- si  " + "   " + str(ticker), " " + " >=" + str(format(k, ".4f")))
                            st.write(" Pas d'exercice ")
                            st.write("  Vendre  " + " " + str(N), " au prix spot")

                    else:
                        rd1 = rd / 100
                        rf1 = rf / 100
                        sigma1 = sigma / 100
                        T = (exercise_date - datetime.now().date()).days
                        T1 = T / 365
                        col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns([1, 60, 1, 1, 1, 1, 1, 1, 1])
                        with col2:
                            put = BS_PUT(s, k, T1, rd1, rf1, sigma1)
                            p1 = put
                            p2 = put * N * s

                            if ticker == 'EUR/MAD':

                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en valeur : " + format(
                                        p1, ".3f") + "  </h2>",
                                    unsafe_allow_html=True)
                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
                                        p2, ".3f") + " " + "DH" + " </h2>",
                                    unsafe_allow_html=True)

                            elif ticker == 'USD/MAD':
                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en valeur : " + format(
                                        p1, ".3f") + " " + "%" + " </h2>",
                                    unsafe_allow_html=True)
                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
                                        p2, ".3f") + " " + "DH" + " </h2>",
                                    unsafe_allow_html=True)
                        st.header("")
                        if st.button(f'Information sur opération '):
                            df = pd.DataFrame(
                                ["Put Vanille", ticker, format(N, ".2f"), format(s, ".4f"), format(k, ".4f"),
                                 format(rd, ".1f") + str("%"), format(rf, ".1f") + str("%"),
                                 format(sigma, ".1f") + str("%"),
                                 exercise_date, format(p2, ".2f")],
                                index=pd.Index(
                                    ["Produit", " Devise", "Nominal", "Spot", "Strike", "Taux doméstique",
                                     "Taux étranger",
                                     "Volatilitée", "Maturité", "Prime à payer"]))

                            tdf = df.T
                            # CSS to inject contained in a string
                            hide_table_row_index = """
                                                                        <style>
                                                                        tbody th {display:none}
                                                                        .blank {display:none}
                                                                        </style>
                                                                        """

                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                unsafe_allow_html=True)
                            st.markdown(hide_table_row_index, unsafe_allow_html=True)
                            st.table(tdf)
                            st.session_state['button'] = False

                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                unsafe_allow_html=True)

                            st.write("- Spot " + " " + "  " + ":" + str(format(s, ".4f")))
                            st.write("- Taux doméstique:" + "   " + str(format(rd, ".1f")) + " " + "%")
                            st.write("- Taux étranger:" + "   " + str(format(rf, ".1f")) + " " + "%")
                            st.write("- Volatilité:" + "   " + str(format(sigma, ".1f")) + " " + "%")

                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                unsafe_allow_html=True)

                            st.write("- si  " + "   " + str(ticker), " " + " <" + str(k))
                            st.write(" Vendre  " + str(N), " à " + " " + str(k))
                            st.write("- si  " + "   " + str(ticker), " " + " >=" + str(k))
                            st.write(" Pas d'exercice  ")
                            st.write(" Vendre  " + " " + str(N), " au prix spot")


            ####################################################################################################################
            elif choose == "Put Participatif":

                genre = st.radio(
                    label="Choisissez le type de la  stratégie ", horizontal=True,
                    options=('Stratégie payante', 'Stratégie gratuite'))
                st.markdown('')
                if genre == 'Stratégie payante':
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/MAD', 'USD/MAD'))
                        # st.markdown(f" Paire  : {ticker}")
                    with col2:
                        N = st.number_input('Nominal')
                        # st.markdown(f"Nominal : {N}")
                    with col3:
                        Perc = st.number_input('Pourcentage optionnel %', 0, 100)
                        # st.markdown(f"Percentage : {Perc} %")
                    with col4:
                        s = st.number_input('Spot')
                        # st.markdown(f"Spot  : {s}")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        k = st.number_input('Strike ')
                        # st.markdown(f"Strike  : {k}")
                    with col2:
                        exercise_date = st.date_input('Maturité', min_value=datetime.today() + timedelta(days=1),
                                                      value=datetime.today() + timedelta(days=365))
                        # st.markdown(f" Date de valeur :{exercise_date}")
                    with col3:
                        rd = st.number_input('Taux doméstique (%)')
                        # st.markdown(f"Taux doméstique : {r} %")
                    with col4:
                        rf = st.number_input('Taux étranger (%)')
                        # st.markdown(f"Taux étranger : {r2} %")

                    input_volatility = st.checkbox("Volatilité " + str(ticker), value=True)
                    if input_volatility:
                        sigma = st.number_input('Volatilité ' + str(ticker))
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            vol1 = st.number_input('Volatilité EUR/USD')
                            # st.markdown(f" Volatilitée : {sigma} %")
                        with col2:
                            st.markdown(
                                "<h2 style='text-align: left ;color:black;font-size:18px;'> Pondération " + " </h2>",
                                unsafe_allow_html=True)
                            vol2 = st.number_input('Poids EUR', 0, 100, 60)
                            # st.markdown(f" Volatilitée : {sigma} %")
                        with col2:
                            vol3 = st.number_input('Poids USD', 0, 100, 40)

                    st.markdown(
                        "-------------------------------------------------------------------------------------------")

                    if not all([s, k, rd, rf, Perc]):
                        st.error("Veuillez remplir tous les champs")

                        button = None
                    else:
                        button = st.button(f'Calculer la prime ')

                    if st.session_state.get('button') != True:
                        st.session_state['button'] = button

                    if st.session_state['button'] == True:

                        if not input_volatility:
                            rd1 = rd / 100
                            rf1 = rf / 100
                            sig1 = vol1 / 100
                            sig2 = vol2 / 100
                            sig3 = vol3 / 100
                            Perc1 = Perc / 100

                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365
                            if ticker == 'EUR/MAD':
                                volatilite = sig1 * sig3
                            elif ticker == 'USD/MAD':
                                volatilite = sig1 * sig2
                            volatilite1 = volatilite * 100
                            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                                [1, 60, 1, 1, 1, 1, 1, 1, 1])

                            with col2:

                                Ft = cours_terme(s, T1, rd1, rf1)
                                p1 = BS_PUT(s, k, T1, rd1, rf1, volatilite) * Perc1
                                put2 = p1 * (Perc1 * N) * s
                                f = (k - Ft) * np.exp(-(rd1 - rf1) * T1) * (1 - Perc1)
                                forward2 = f * (1 - Perc1) * N * s

                                Prime_partcipative1 = p1 + f
                                Prime_partcipative2 = put2 + forward2

                                if ticker == 'EUR/MAD':
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en valeur : " + format(
                                            Prime_partcipative1, ".3f") + " </h2>",
                                        unsafe_allow_html=True)
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
                                            Prime_partcipative2, ".3f") + " " + "DH" + " </h2>",
                                        unsafe_allow_html=True)

                                elif ticker == 'USD/MAD':
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en valeur : " + format(
                                            Prime_partcipative1, ".3f") + " " + "%" + " </h2>",
                                        unsafe_allow_html=True)
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
                                            Prime_partcipative2, ".3f") + " " + "DH" + " </h2>",
                                        unsafe_allow_html=True)
                            if st.button(f'Information sur opération '):
                                df = pd.DataFrame(
                                    [ticker, format(N, ".2f"), format(s, ".4f"), format(k, ".4f"),
                                     format(Perc, ".0f") + str("%"),
                                     format(Prime_partcipative2, ".2f")],
                                    index=pd.Index(
                                        ["Paire de devise", "Nominal", "Spot", "Strike", "Participation",
                                         "Put Participative"]))
                                tdf = df.T
                                # CSS to inject contained in a string
                                hide_table_row_index = """
                                                                                        <style>
                                                                                        tbody th {display:none}
                                                                                        .blank {display:none}
                                                                                        </style>
                                                                                        """

                                ######################### detail de strategie

                                # Inject CSS with Markdown
                                ############ table recap

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                    unsafe_allow_html=True)

                                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                                st.table(tdf)
                                st.session_state['button'] = False

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)

                                st.write(
                                    "-Spot " + " " + str(ticker) + "          " + ":" + " " + str(format(s, ".4f")))
                                st.write("-Volatilité:" + "   " + str(format(volatilite1, ".2f")) + "%")
                                st.write("-Taux doméstique:" + "   " + str(format(rd, ".2f")) + "%")
                                st.write("-Taux étranger:" + "   " + str(format(rf, ".2f")) + "%")

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)

                                st.write("- si  ""   " + str(ticker), " "" < " + str(format(k, ".4f")))
                                st.write(" Vendre " + str(N), " à " + " " + str(format(k, ".4f")))
                                st.write("- si  ""   " + str(ticker), " ""> " + str(format(k, ".4f")))
                                st.write(" Vendre  " + str(N * Perc1), "au spot du jour")
                                st.write(" Vendre " + str(N * (1 - Perc1)), " à " + " " + str(format(k, ".4f")))


                        else:
                            rd1 = rd / 100
                            rf1 = rf / 100
                            sigma1 = sigma / 100
                            r1 = (rd1 - rf1)
                            Perc1 = Perc / 100
                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365

                            Ft = cours_terme(s, T1, rd1, rf1)
                            p2 = BS_PUT(s, k, T1, rd1, rf1, sigma1) * Perc1
                            put3 = p2 * (Perc1 * N)
                            f2 = (k - Ft) * np.exp(-(rd1 - rf1) * T1) * (1 - Perc1)
                            forward3 = f2 * (1 - Perc1) * N * s

                            Prime_partcipative3 = p2 + f2
                            Prime_partcipative4 = put3 + forward3

                            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                                [1, 60, 1, 1, 1, 1, 1, 1, 1])
                            with col2:

                                if ticker == 'EUR/MAD':

                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en valeur : " + format(
                                            Prime_partcipative3, ".3f") + " </h2>",
                                        unsafe_allow_html=True)

                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en MAD : " + format(
                                            Prime_partcipative4, ".3f") + " " + "DH" + " </h2>",
                                        unsafe_allow_html=True)

                                elif ticker == 'USD/MAD':
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en valeur : " + format(
                                            Prime_partcipative3, ".3f") + " </h2>",
                                        unsafe_allow_html=True)
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en MAD : " + format(
                                            Prime_partcipative4, ".3f") + " " + "DH" + " </h2>",
                                        unsafe_allow_html=True)

                            if st.button(f'Information sur opération '):
                                df = pd.DataFrame(
                                    [ticker, format(N, ".2f"), format(s, ".4f"), format(k, ".4f"),
                                     format(Perc, ".0f") + str("%"),
                                     format(Prime_partcipative4, ".2f")],
                                    index=pd.Index(
                                        ["Paire de devise", "Nominal", "Spot", "Strike", "Participation",
                                         "Put Participative"]))
                                tdf = df.T
                                # CSS to inject contained in a string
                                hide_table_row_index = """
                                                                                        <style>
                                                                                        tbody th {display:none}
                                                                                        .blank {display:none}
                                                                                        </style>
                                                                                        """

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                    unsafe_allow_html=True)
                                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                                st.table(tdf)
                                st.session_state['button'] = False

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)

                                st.write(
                                    "-Spot " + " " + str(ticker) + "          " + ":" + " " + str(format(s, ".4f")))
                                st.write("-Volatilité:" + "   " + str(format(sigma, ".1f")) + "%")
                                st.write("-Taux doméstique:" + "   " + str(format(rd, ".1f")) + "%")
                                st.write("-Taux étranger:" + "   " + str(format(rf, ".1f")) + "%")

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)

                                st.write("- si  ""   " + str(ticker), " "" < " + str(format(k, ".4f")))
                                st.write(" Vendre  " + str(N), " à " + " " + str(format(k, ".4f")))
                                st.write("- si  ""   " + str(ticker), " ""> " + str(format(k, ".4f")))
                                st.write(" Vendre  " + str(N * Perc1), "au spot du jour")
                                st.write(" Vendre  " + str(format(N * (1 - Perc1), ".4f")),
                                         " à " + " " + str(format(k, ".4f")))

                if genre == 'Stratégie gratuite':

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/MAD', 'USD/MAD'))
                        # st.markdown(f" Paire  : {ticker}")
                    with col2:
                        N = st.number_input('Nominal')
                        # st.markdown(f"Nominal : {N}")
                    with col3:
                        Perc = st.number_input('Pourcentage optionnel %')
                        # st.markdown(f"Percentage : {Perc} %")
                    with col4:
                        s = st.number_input('Spot')
                        # st.markdown(f"Spot  : {s}")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        exercise_date = st.date_input('Maturité', min_value=datetime.today() + timedelta(days=1),
                                                      value=datetime.today() + timedelta(days=365))
                        # st.markdown(f" Date de valeur :{exercise_date}")
                    with col2:
                        rd = st.number_input('Taux doméstique (%)')
                        # st.markdown(f"Taux doméstique : {r} %")
                    with col3:
                        rf = st.number_input('Taux étranger (%)')
                        # st.markdown(f"Taux étranger : {r2} %")
                    input_volatility = st.checkbox("Volatilité " + str(ticker), value=True)

                    if input_volatility:
                        sigma = st.number_input('Volatilité ' + str(ticker))
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            vol1 = st.number_input('Volatilité EUR/USD')
                            # st.markdown(f" Volatilitée : {sigma} %")
                        with col2:
                            st.markdown(
                                "<h2 style='text-align: left ;color:black;font-size:18px;'> Pondération " + " </h2>",
                                unsafe_allow_html=True)
                            vol2 = st.number_input('Poids EUR', 0, 100, 60)
                            # st.markdown(f" Volatilitée : {sigma} %")
                        with col2:
                            vol3 = st.number_input('Poids USD', 0, 100, 40)

                    st.markdown(
                        "-------------------------------------------------------------------------------------------")
                    if not all([s, rd, rf, Perc]):
                        st.error("Veuillez remplir tous les champs")
                        button = None
                    else:
                        button = st.button(f'Calculer le strike ')

                    if st.session_state.get('button') != True:
                        st.session_state['button'] = button

                    if st.session_state['button'] == True:

                        if not input_volatility:
                            rd1 = rd / 100
                            rf1 = rf / 100
                            # sigma1 = sigma / 100
                            sig1 = vol1 / 100
                            sig2 = vol2 / 100
                            sig3 = vol3 / 100
                            Perc1 = Perc / 100
                            Perc2 = 1 - Perc1

                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365
                            if ticker == 'EUR/MAD':
                                volatilite = sig1 * sig3
                            elif ticker == 'USD/MAD':
                                volatilite = sig1 * sig2
                            volatilite1 = volatilite * 100
                            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                                [1, 60, 1, 1, 1, 1, 1, 1, 1])

                            with col2:

                                if Perc1 == 1:
                                    Ft = cours_terme(s, T1, rd1, rf1)
                                    p2 = BS_PUT(s, Ft, T1, rd1, rf1, volatilite)
                                    solution = Ft - p2
                                else:
                                    solution = find_strike1(s, rd1, rf1, T1, volatilite, Perc1, Perc2)

                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Le Strike est : " + format(
                                        solution, ".4f") + " </h2>",
                                    unsafe_allow_html=True)

                            if st.button(f'Information sur opération '):
                                df = pd.DataFrame(
                                    [ticker, format(N, ".2f"), format(s, ".4f"), format(solution, ".4f"),
                                     format(Perc, ".0f") + str("%")
                                     ],
                                    index=pd.Index(
                                        ["Paire de devise", "Nominal", "Spot", "Strike", "Participation"]))
                                tdf = df.T
                                # CSS to inject contained in a string
                                hide_table_row_index = """
                                                                                        <style>
                                                                                        tbody th {display:none}
                                                                                        .blank {display:none}
                                                                                        </style>
                                                                                        """

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                    unsafe_allow_html=True)

                                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                                st.table(tdf)
                                st.session_state['button'] = False

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)

                                st.write(
                                    "-Spot " + " " + str(ticker) + "          " + ":" + " " + str(format(s, ".4f")))
                                st.write("-Volatilité:" + "   " + str(format(volatilite1, ".2f")) + "%")
                                st.write("-Taux doméstique:" + "   " + str(format(rd, ".2f")) + "%")
                                st.write("-Taux étranger:" + "   " + str(format(rf, ".2f")) + "%")

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)

                                st.write("- si  ""   " + str(ticker), " "" < " + str(format(solution, ".4f")))
                                st.write(" Vendre " + str(N), " à " + " " + str(format(solution, ".4f")))
                                st.write("- si  ""   " + str(ticker), " ""> " + str(format(solution, ".4f")))
                                st.write(" Vendre  " + str(format(N * Perc1, ".2f")), "au spot du jour")
                                st.write(" Vendre " + str(format(N * (1 - Perc1), ".2f")),
                                         " à " + " " + str(format(solution, ".4f")))


                        else:
                            rd1 = rd / 100
                            rf1 = rf / 100
                            sigma1 = sigma / 100
                            r1 = (rd1 - rf1)
                            Perc1 = Perc / 100
                            Perc2 = 1 - Perc1
                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365

                            if Perc1 == 1:
                                Ft = cours_terme(s, T1, rd1, rf1)
                                p2 = BS_PUT(s, Ft, T1, rd1, rf1, sigma1)
                                solution1 = Ft - p2
                            else:
                                solution1 = find_strike1(s, rd1, rf1, T1, sigma1, Perc1, Perc2)

                            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                                [1, 60, 1, 1, 1, 1, 1, 1, 1])
                            with col2:

                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Le Strike est : " + format(
                                        solution1, ".4f") + " </h2>",
                                    unsafe_allow_html=True)

                            if st.button(f'Information sur opération '):
                                df = pd.DataFrame(
                                    [ticker, format(N, ".2f"), format(s, ".4f"), format(solution1, ".4f"),
                                     format(Perc, ".0f") + str("%")],
                                    index=pd.Index(
                                        ["Paire de devise", "Nominal", "Spot", "Strike", "Participation"]))
                                tdf = df.T
                                # CSS to inject contained in a string
                                hide_table_row_index = """
                                                                                        <style>
                                                                                        tbody th {display:none}
                                                                                        .blank {display:none}
                                                                                        </style>
                                                                                        """

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                    unsafe_allow_html=True)
                                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                                st.table(tdf)
                                st.session_state['button'] = False

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)

                                st.write(
                                    "-Spot " + " " + str(ticker) + "          " + ":" + " " + str(format(s, ".4f")))
                                st.write("-Volatilité:" + "   " + str(format(sigma, ".1f")) + "%")
                                st.write("-Taux doméstique:" + "   " + str(format(rd, ".1f")) + "%")
                                st.write("-Taux étranger:" + "   " + str(format(rf, ".1f")) + "%")

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)

                                st.write("- si  ""   " + str(ticker), " "" < " + str(format(solution1, ".4f")))
                                st.write(" Vendre  " + str(N), " à " + " " + str(format(solution1, ".4f")))
                                st.write("- si  ""   " + str(ticker), " ""> " + str(format(solution1, ".4f")))
                                st.write(" Vendre  " + str(format(N * Perc1, ".2f")), "au spot du jour")
                                st.write(" Vendre  " + str(format(N * (1 - Perc1), ".2f")),
                                         " à " + " " + str(format(solution1, ".4f")))



            ####################################################################################################################
            #
            # elif choose == "Put Participatif":
            #
            #     st.header(" ")
            #
            #     col1, col2, col3,col4= st.columns(4)
            #     with col1:
            #         ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/USD', 'EUR/MAD', 'USD/MAD'))
            #         # st.markdown(f" Paire  : {ticker}")
            #     with col2:
            #         N = st.number_input('Nominal')
            #         # st.markdown(f"Nominal : {N}")
            #     with col3:
            #         Perc = st.number_input('Pourcentage %')
            #         # st.markdown(f"Percentage : {Perc} %")
            #     with col4 :
            #         s = st.number_input('Spot')
            #         # st.markdown(f"Spot  : {s}")
            #
            #     col1, col2, col3,col4,col5 = st.columns(5)
            #
            #     with col1:
            #         k = st.number_input('Strike ')
            #         # st.markdown(f"Strike  : {k}")
            #     with col2:
            #         exercise_date = st.date_input('Exercise date', min_value=datetime.today() + timedelta(days=1),
            #                                       value=datetime.today() + timedelta(days=365))
            #         # st.markdown(f" Date de valeur :{exercise_date}")
            #     with col3:
            #         rd = st.number_input('Taux doméstique (%)', 0, 100, 10)
            #         # st.markdown(f"Taux doméstique : {r} %")
            #     with col4:
            #         rf = st.number_input('Taux étranger (%)', 0, 100, 10)
            #         # st.markdown(f"Taux étranger : {r2} %")
            #     with col5:
            #         sigma = st.number_input('Volatilité (%)', 0, 100, 20)
            #         # st.markdown(f" Volatilitée : {sigma} %")
            #     if sigma == 0:
            #         col1, col2 = st.columns(2)
            #         with col1:
            #             vol1 = st.number_input('Volatilité EUR/USD', 0, 100, 10)
            #             # st.markdown(f" Volatilitée : {sigma} %")
            #         with col2:
            #             st.markdown(
            #                 "<h2 style='text-align: left ;color:black;font-size:18px;'> Pondération " + " </h2>",
            #                 unsafe_allow_html=True)
            #             vol2 = st.number_input('Poids EUR', 0, 100, 60)
            #             # st.markdown(f" Volatilitée : {sigma} %")
            #         with col2:
            #             vol3 = st.number_input('Poids USD', 0, 100, 40)
            #
            #     st.markdown("-------------------------------------------------------------------------------------------")
            #
            #
            #     genre = st.radio(
            #         "Choisissez le type de la  stratégie ",
            #         ('Stratégie payante', 'Stratégie gratuite'))
            #
            #     if genre == 'Stratégie payante':
            #
            #         # gestion des exceptions
            #
            #         if not all([s, k, rd, rf, Perc]):
            #             st.error("Veuillez remplir tous les champs")
            #
            #             button = None
            #         else:
            #             button = st.button(f'Calculer la prime ')
            #
            #         if st.session_state.get('button') != True:
            #             st.session_state['button'] = button
            #
            #         if st.session_state['button'] == True:
            #
            #             if sigma == 0:
            #                 rd1 = rd / 100
            #                 rf1 = rf / 100
            #                 sigma1 = sigma / 100
            #                 sig1 = vol1 / 100
            #                 sig2 = vol2 / 100
            #                 sig3 = vol3 / 100
            #                 Perc1 = Perc / 100
            #                 r1 = (rd1 - rf1)
            #                 T = (exercise_date - datetime.now().date()).days
            #                 T1 = T / 365
            #
            #                 col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
            #                     [1, 60, 1, 1, 1, 1, 1, 1, 1])
            #
            #                 with col2:
            #                     volatilite = sig1 * sig2
            #                     def cours_terme(s, T1, rd, rf):
            #                         r1 = rd - rf
            #                         F_k = s * np.exp(r1 * T1)
            #                         return F_k
            #
            #                     Ft = cours_terme(s, T1, rd1, rf1)
            #                     p1 = BS_PUT(s, k, T1, rd1, rf1, volatilite)
            #                     put = p1 * (Perc1 * N)
            #                     put2 = p1 * (Perc1 * N) * s
            #                     f = (k - Ft) * np.exp(-r1 * T1)
            #                     forward = f * (1 - Perc1) * N
            #                     forward2 = f * (1 - Perc1) * N* s
            #                     Prime_partcipative1 = put + forward
            #                     Prime_partcipative2 = put2+forward2
            #
            #                     if ticker == 'EUR/MAD':
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en % du Nominal : " + format(
            #                                 Prime_partcipative1, ".3f") + " " + "€" + " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en Devise : " + format(
            #                                 Prime_partcipative2, ".3f") + " " + "DH" + " </h2>",
            #                             unsafe_allow_html=True)
            #
            #                     elif ticker == 'USD/MAD':
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en % du Nominal : " + format(
            #                                 Prime_partcipative1, ".3f") + " " + "$" + " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en Devise : " + format(
            #                                 Prime_partcipative2, ".3f") + " " + "DH" + " </h2>",
            #                             unsafe_allow_html=True)
            #                     elif ticker == 'EUR/USD':
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en % du Nominal : " + format(
            #                                 Prime_partcipative1, ".3f") + " " + "€" + " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en Devise : " + format(
            #                                 Prime_partcipative2, ".3f") + " " + "$" + " </h2>",
            #                             unsafe_allow_html=True)
            #             else:
            #                 rd1 = rd / 100
            #                 rf1 = rf / 100
            #                 sigma1 = sigma / 100
            #                 r1 = (rd1 - rf1)
            #                 Perc1 = Perc / 100
            #                 T = (exercise_date - datetime.now().date()).days
            #                 T1 = T / 365
            #
            #                 def cours_terme(s, T1, rd, rf):
            #                     r1 = rd - rf
            #                     F_k = s * np.exp(r1 * T1)
            #                     return F_k
            #
            #                 Ft = cours_terme(s, T1, rd1, rf1)
            #
            #                 p2 = BS_PUT(s, k, T1, rd1, rf1, sigma1)
            #                 put3= p2 * (Perc1 * N)
            #                 put4 = p2 * (Perc1 * N) * s
            #
            #                 f = (k - Ft) * np.exp(-r1 * T1)
            #                 forward3 = f * (1 - Perc1) * N
            #                 forward4 = f * (1 - Perc1) * N * s
            #
            #                 Prime_partcipative3 = put3 + forward3
            #                 Prime_partcipative4 = put4 + forward4
            #
            #                 col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
            #                     [1, 60, 1, 1, 1, 1, 1, 1, 1])
            #                 with col2:
            #
            #
            #
            #
            #                     if ticker == 'EUR/MAD':
            #
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en % du Nominal : " + format(
            #                                 Prime_partcipative3 , ".3f") + " " + "€" + " </h2>",
            #                             unsafe_allow_html=True)
            #
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en Devise : " + format(
            #                                 Prime_partcipative4, ".3f") + " " + "DH" + " </h2>",
            #                             unsafe_allow_html=True)
            #
            #                     elif ticker == 'USD/MAD':
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en % du Nominal : " + format(
            #                                 Prime_partcipative3, ".3f") + " " + "$" + " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en Devise : " + format(
            #                                 Prime_partcipative4, ".3f") + " " + "DH" + " </h2>",
            #                             unsafe_allow_html=True)
            #                     elif ticker == 'EUR/USD':
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en % du Nominal : " + format(
            #                                 Prime_partcipative3, ".3f") + " " + "€" + " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en Devise : " + format(
            #                                 Prime_partcipative4, ".3f") + " " + "$" + " </h2>",
            #                             unsafe_allow_html=True)
            #
            #
            #             if st.button(f'Information sur opération '):
            #                 if sigma ==0 :
            #                     df = pd.DataFrame(
            #                         [ticker, format(N, ".2f"), format(s, ".4f"), format(k, ".4f"),
            #                          format(Perc, ".0f") + str("%"),
            #                          format(Prime_partcipative2, ".2f")],
            #                         index=pd.Index(
            #                             ["Paire de devise", "Nominal", "Spot", "Strike", "Participation"
            #                                 , "Put Participative"]))
            #                     tdf = df.T
            #                     # CSS to inject contained in a string
            #                     hide_table_row_index = """
            #                                                                         <style>
            #                                                                         tbody th {display:none}
            #                                                                         .blank {display:none}
            #                                                                         </style>
            #                                                                         """
            #
            #                     ######################### detail de strategie
            #
            #                     # Inject CSS with Markdown
            #                     ############ table recap
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
            #                         unsafe_allow_html=True)
            #                     st.header("")
            #                     st.markdown(hide_table_row_index, unsafe_allow_html=True)
            #                     st.table(tdf)
            #                     st.session_state['button'] = False
            #
            #                     ######### detail
            #
            #                     col1, col2, col3 = st.columns(3)
            #                     with col1:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("-Spot " + " " + str(ticker) + "          " + ":" + " " + str(s))
            #                         st.write("-Volatilité:" + "   " + str(volatilite) + "%")
            #                         st.write("-Taux doméstique:" + "   " + str(rd) + "%")
            #                         st.write("-Taux étranger:" + "   " + str(rf) + "%")
            #                         # st.write("-Volatilité:" + "   " + str(sigma))
            #                     st.header("  ")
            #                     with col3:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement a l'échéance </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("- si  ""   " + str(ticker), " "" < " + str(k))
            #                         st.write(" Vous vendez  " + str(N), " à " + " " + str(k))
            #                         st.write("- si  ""   " + str(ticker), " ""> " + str(s))
            #                         st.write(" Vous vendez  " + str(N * Perc1), "au spot du jour")
            #                         st.write(" Vous vendez  " + str(N * (1 - Perc1)), " à " + " " + str(k))
            #
            #
            #                 else :
            #
            #                     df = pd.DataFrame(
            #                         [ticker, format(N, ".2f"), format(s, ".4f"), format(k, ".4f"),
            #                          format(Perc, ".0f") + str("%"),
            #                            format(Prime_partcipative4, ".2f")],
            #                         index=pd.Index(
            #                             ["Paire de devise", "Nominal", "Spot", "Strike", "Participation"
            #                              ,"Put Participative"]))
            #                     tdf = df.T
            #                     # CSS to inject contained in a string
            #                     hide_table_row_index = """
            #                                         <style>
            #                                         tbody th {display:none}
            #                                         .blank {display:none}
            #                                         </style>
            #                                         """
            #
            #                     ######################### detail de strategie
            #
            #                     # Inject CSS with Markdown
            #                     ############ table recap
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
            #                         unsafe_allow_html=True)
            #                     st.header("")
            #                     st.markdown(hide_table_row_index, unsafe_allow_html=True)
            #                     st.table(tdf)
            #                     st.session_state['button'] = False
            #
            #                     ######### detail
            #
            #                     col1, col2, col3 = st.columns(3)
            #                     with col1:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("-Spot " + " " + str(ticker) + "          " + ":" + " " + str(s))
            #                         st.write("-Volatilité:" + "   " + str(sigma) + "%")
            #                         st.write("-Taux doméstique:" + "   " + str(rd) + "%")
            #                         st.write("-Taux étranger:" + "   " + str(rf) + "%")
            #
            #                     st.header("  ")
            #                     with col3:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement a l'échéance </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("- si  ""   " + str(ticker), " "" < " + str(k))
            #                         st.write(" Vous vendez  " + str(N), " à " + " " + str(k))
            #                         st.write("- si  ""   " + str(ticker), " ""> " + str(s))
            #                         st.write(" Vous vendez  " + str(N * Perc1), "au spot du jour")
            #                         st.write(" Vous vendez  " + str(N * (1 - Perc1)), " à " + " " + str(k))
            #
            #
            #
            #     else:
            #         if sigma==0:
            #             # gestion des exceptions
            #             st.markdown("-------------")
            #             if not all([s, k, rd, rf, Perc]):
            #                 st.error("Veuillez remplir tous les champs")
            #                 button = None
            #             else:
            #                 button = st.button(f'Calculer le strike amélioré ')
            #
            #             if st.session_state.get('button') != True:
            #                 st.session_state['button'] = button
            #
            #             if st.session_state['button'] == True:
            #
            #                 # Formating selected model parameters
            #                 rd1 = rd / 100
            #                 rf1 = rf / 100
            #                 r1 = (rd1 - rf1)
            #                 sigma1 = sigma / 100
            #                 sig1 = vol1 / 100
            #                 sig2 = vol2 / 100
            #                 sig3 = vol3 / 100
            #                 T = (exercise_date - datetime.now().date()).days
            #                 T1 = T / 365
            #                 Perc1 = Perc / 100
            #
            #                 if ticker =='EUR/MAD' :
            #                     volatilite = sig1 * sig2
            #                 elif ticker == 'USD/MAD':
            #                     volatilite = sig1 * sig3
            #                 ############"" find strike
            #                 Ft = cours_terme(s, T1, rd1, rf1)
            #                 k1 = find_strike(s, rd1, rf1, T1, volatilite)
            #
            #                 put = BS_PUT(s, k1, T1, rd1, rf1,volatilite )
            #                 forward = (k1 - Ft) * np.exp(-r1 * T1)
            #                 b = forward + put
            #
            #
            #                 st.markdown(
            #                     "<h2 style='text-align: left ;color:black;font-size:20px;'> - Le strike amélioré est : " + " " + " " + str(
            #                         format(k1, '.4f')) + " </h2>",
            #                     unsafe_allow_html=True)
            #
            #                 if st.button(f'Informtion sur opération '):
            #                     df = pd.DataFrame([ticker, format(N, ".2f"), format(s, ".4f"), format(k1, ".4f"),
            #                                        format(Perc, ".0f") + str("%"),format(b, ".2f")],
            #                                       index=pd.Index(
            #                                           ["Paire de devise", "Nominal", "Spot", "Strike amélioré",
            #                                            "Participation", "Prime"]))
            #
            #                     tdf = df.T
            #                     # CSS to inject contained in a string
            #                     hide_table_row_index = """
            #                                                             <style>
            #                                                             tbody th {display:none}
            #                                                             .blank {display:none}
            #                                                             </style>
            #                                                             """
            #
            #
            #                     # Inject CSS with Markdown
            #                     ############ table recap
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
            #                         unsafe_allow_html=True)
            #                     st.header("")
            #                     st.markdown(hide_table_row_index, unsafe_allow_html=True)
            #                     st.table(tdf)
            #                     st.session_state['button'] = False
            #
            #                     ######### detail
            #
            #                     col1, col2, col3 = st.columns(3)
            #                     with col1:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("- Spot " + " " + str(ticker) + "          " + ":" + str(s))
            #                         st.write("- Volatilité:" + "   " + str(format(volatilite,".2f")) + "%")
            #                         st.write("- Taux doméstique:" + "   " + str(rd) + "%")
            #                         st.write("- Taux étranger:" + "   " + str(rf) + "%")
            #                         st.write("- Strike:" + "   " + str(format(k1, ".4f")))
            #                     st.header("  ")
            #                     with col3:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement a l'échéance </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("- si  " + "   " + str(ticker), " " + " < " + str(format(k1, ".4f")))
            #                         st.write(" Vous vendez  " + str(N), " à " + " " + str(format(k1, ".4f")))
            #                         st.write("- si  " + "   " + str(ticker), " " + "> " + str(s))
            #                         st.write(" Vous vendez  " + str(N * Perc1), "au spot du jour")
            #                         st.write(" Vous vendez  " + str(N * (1 - Perc1)),
            #                                  " à " + " " + str(format(k1, ".4f")))
            #
            #         else:
            #             # gestion des exceptions
            #             st.markdown("-------------")
            #             if not all([s, k, rd, rf, Perc]):
            #                 st.error("Veuillez remplir tous les champs")
            #                 button = None
            #             else:
            #                 button = st.button(f'Calculer le strike amélioré ')
            #
            #             if st.session_state.get('button') != True:
            #                 st.session_state['button'] = button
            #
            #             if st.session_state['button'] == True:
            #
            #                 # Formating selected model parameters
            #                 rd1 = rd / 100
            #                 rf1 = rf / 100
            #                 r1 = (rd1 - rf1)
            #                 sigma1 = sigma / 100
            #                 T = (exercise_date - datetime.now().date()).days
            #                 T1 = T / 365
            #                 Perc1 = Perc / 100
            #
            #                 ############"" find strike
            #                 Ft = cours_terme(s, T1, rd1, rf1)
            #                 k1 = find_strike(s, rd1, rf1, T1, sigma1)
            #
            #                 putg = BS_PUT(s, k1, T1, rd1, rf1, sigma1)
            #                 forwardg = (k1 - Ft) * np.exp(-r1 * T1)
            #                 b1 = forwardg + putg
            #
            #
            #                 st.markdown(
            #                     "<h2 style='text-align: left ;color:black;font-size:20px;'> - Le strike amélioré est : " + " " + " " + str(
            #                         format(k1, '.4f')) + " </h2>",
            #                     unsafe_allow_html=True)
            #
            #                 if st.button(f'Informtion sur opération '):
            #                     df = pd.DataFrame([ticker, format(N, ".2f"), format(s, ".4f"), format(k1, ".4f"),
            #                                        format(Perc, ".0f") + str("%"),format(b1, ".2f")],
            #                                       index=pd.Index(
            #                                           ["Paire de devise", "Nominal", "Spot", "Strike amélioré",
            #                                            "Participation", "Prime"]))
            #
            #                     tdf = df.T
            #                     # CSS to inject contained in a string
            #                     hide_table_row_index = """
            #                                     <style>
            #                                     tbody th {display:none}
            #                                     .blank {display:none}
            #                                     </style>
            #                                     """
            #
            #
            #                     ############ table recap
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
            #                         unsafe_allow_html=True)
            #                     st.header("")
            #                     st.markdown(hide_table_row_index, unsafe_allow_html=True)
            #                     st.table(tdf)
            #                     st.session_state['button'] = False
            #
            #                     ######### detail
            #
            #                     col1, col2, col3 = st.columns(3)
            #                     with col1:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("- Spot " + " " + str(ticker) + "          " + ":" + str(s))
            #                         st.write("- Volatilité:" + "   " + str(sigma) + "%")
            #                         st.write("- Taux doméstique:" + "   " + str(rd) + "%")
            #                         st.write("- Taux étranger:" + "   " + str(rf) + "%")
            #                         st.write("- Strike:" + "   " + str(format(k1, ".4f")))
            #                     st.header("  ")
            #                     with col3:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement a l'échéance </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("- si  " + "   " + str(ticker), " " + " < " + str(format(k1, ".4f")))
            #                         st.write(" Vous vendez  " + str(N), " à " + " " + str(format(k1, ".4f")))
            #                         st.write("- si  " + "   " + str(ticker), " " + "> " + str(s))
            #                         st.write(" Vous vendez  " + str(N * Perc1), "au spot du jour")
            #                         st.write(" Vous vendez  " + str(N * (1 - Perc1)), " à " + " " + str(format(k1, ".4f")))

            ############################################################################################################################################################
            elif choose == "Tunnel Symétrique":

                genre = st.radio(
                    label="Choisissez le type de la  stratégie ", horizontal=True,
                    options=('Stratégie payante', 'Stratégie gratuite'))
                st.markdown('')
                if genre == 'Stratégie payante':

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/MAD', 'USD/MAD'))
                        # st.markdown(f" Paire  : {ticker}")
                    with col2:
                        N = st.number_input('Nominal')
                        # st.markdown(f"Nominal : {N}")
                    with col3:
                        s = st.number_input('Spot')
                        # st.markdown(f"Spot  : {s}")
                    with col4:
                        exercise_date = st.date_input('Maturité', min_value=datetime.today() + timedelta(days=1),
                                                      value=datetime.today() + timedelta(days=365))
                        # st.markdown(f" Date de valeur :{exercise_date}")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        k1 = st.number_input('Strike k1')
                        # st.markdown(f"Percentage : {k1} %")

                    with col2:
                        k2 = st.number_input('Strike k2 ')
                        # .markdown(f"Strike  : {k2}")
                    with col3:
                        rd = st.number_input('Taux doméstique (%)', 0, 100, 10)
                        # st.markdown(f"Taux doméstique : {r} %")

                    with col4:
                        rf = st.number_input('Taux étranger (%)', 0, 100, 10)
                        # st.markdown(f"Taux étranger : {r2} %")
                    input_volatility = st.checkbox("Volatilité " + str(ticker), value=True)
                    if input_volatility:
                        sigma = st.number_input('Volatilité ' + str(ticker))
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            vol1 = st.number_input('Volatilité EUR/USD', 0, 100, 10)
                            # st.markdown(f" Volatilitée : {sigma} %")
                        with col2:
                            st.markdown(
                                "<h2 style='text-align: left ;color:black;font-size:18px;'> Pondération " + " </h2>",
                                unsafe_allow_html=True)
                            vol2 = st.number_input('Poids EUR', 0, 100, 60)
                            # st.markdown(f" Volatilitée : {sigma} %")
                        with col2:
                            vol3 = st.number_input('Poids USD', 0, 100, 40)
                    st.markdown('-------------------------------------------------------------------------------------')

                    if not all([s, k1, k2, rd, rf]):
                        st.error("Veuillez remplir tous les champs")

                        button = None
                    else:
                        button = st.button(f'Calculer la prime ')

                    if st.session_state.get('button') != True:
                        st.session_state['button'] = button

                    if st.session_state['button'] == True:

                        if not input_volatility:
                            rd1 = rd / 100
                            rf1 = rf / 100
                            sig1 = vol1 / 100
                            sig2 = vol2 / 100
                            sig3 = vol3 / 100

                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365
                            if ticker == 'EUR/MAD':
                                volatilite = sig1 * sig3
                            elif ticker == 'USD/MAD':
                                volatilite = sig1 * sig2
                            volatilite1 = volatilite * 100

                            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                                [1, 60, 1, 1, 1, 1, 1, 1, 1])

                            with col2:

                                Ft = cours_terme(s, T1, rd1, rf1)
                                c = BS_CALL(s, k2, T1, rd1, rf1, volatilite)
                                p = BS_PUT(s, k1, T1, rd1, rf1, volatilite)

                                tunnel1 = p - c
                                tunnel2 = (p - c) * s * N

                                # Ft = cours_terme(s, T1, rd1, rf1)
                                # p1 = BS_PUT(s, k, T1, rd1, rf1, volatilite)
                                # put2 = p1 * (Perc1 * N) * s
                                # f = (k - Ft) * np.exp(-(rd1 - rf1) * T1)
                                # forward2 = f * (1 - Perc1) * N* s
                                #
                                # Prime_partcipative1 = ((p1 + f)/k)*100
                                # Prime_partcipative2 = put2+forward2

                                if ticker == 'EUR/MAD':
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en valeur : " + format(
                                            tunnel1, ".3f") + " </h2>",
                                        unsafe_allow_html=True)
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
                                            tunnel2, ".3f") + " " + "DH" + " </h2>",
                                        unsafe_allow_html=True)

                                elif ticker == 'USD/MAD':
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en valeur : " + format(
                                            tunnel1, ".3f") + " </h2>",
                                        unsafe_allow_html=True)
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
                                            tunnel2, ".3f") + " " + "DH" + " </h2>",
                                        unsafe_allow_html=True)
                            if st.button(f'Information sur opération '):
                                df = pd.DataFrame(
                                    [ticker, format(N, ".2f"), format(s, ".4f"), format(k1, ".4f"), format(k2, ".4f"),
                                     format(tunnel2, ".2f")],
                                    index=pd.Index(["Paire de devise", "Nominal", "Spot", "Strike k1", "Strike k2",
                                                    "Tunnel Symétrique"]))
                                tdf = df.T
                                # CSS to inject contained in a string
                                hide_table_row_index = """
                                                                                        <style>
                                                                                        tbody th {display:none}
                                                                                        .blank {display:none}
                                                                                        </style>
                                                                                        """

                                ######################### detail de strategie

                                # Inject CSS with Markdown
                                ############ table recap

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                    unsafe_allow_html=True)

                                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                                st.table(tdf)
                                st.session_state['button'] = False

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)

                                st.write(
                                    "-Spot " + " " + str(ticker) + "          " + ":" + " " + str(format(s, ".4f")))
                                st.write("-Volatilité:" + "   " + str(format(volatilite1, ".2f")) + "%")
                                st.write("-Taux doméstique:" + "   " + str(format(rd, ".2f")) + "%")
                                st.write("-Taux étranger:" + "   " + str(format(rf, ".2f")) + "%")

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)

                                st.write("- si  " + "   " + str(ticker), " " + " < " + str(format(k1, ".4f")))
                                st.write(" Vendre  " + str(N), " à " + " " + str(format(k1, ".4f")))

                                st.write("- si  " + "   " + str(format(k1, ".4f")) + " < " + str(ticker),
                                         " " + " <" + str(format(k2, ".4f")))
                                st.write(" Vendre  " + str(N), "au spot du jour")

                                st.write("- si  " + "   " + str(ticker), " "" > " + str(format(k2, ".4f")))
                                st.write(" Vendre  " + str(N), " à " + " " + str(format(k2, ".4f")))



                        else:
                            rd1 = rd / 100
                            rf1 = rf / 100
                            sigma1 = sigma / 100
                            r1 = (rd1 - rf1)

                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365

                            Ft = cours_terme(s, T1, rd1, rf1)
                            c = BS_CALL(s, k2, T1, rd1, rf1, sigma1)
                            p = BS_PUT(s, k1, T1, rd1, rf1, sigma1)

                            tunnel3 = p - c
                            tunnel4 = (p - c) * s * N

                            # Ft = cours_terme(s, T1, rd1, rf1)
                            # p2 = BS_PUT(s, k, T1, rd1, rf1, sigma1)
                            # put3 = p2 * (Perc1 * N)
                            # f2 = (k - Ft) * np.exp(-(rd1 - rf1) * T1)
                            # forward3 = f2* (1 - Perc1) * N * s
                            #
                            # Prime_partcipative3 = ((p2 + f2) / k) * 100
                            # Prime_partcipative4 = put3+ forward3

                            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                                [1, 60, 1, 1, 1, 1, 1, 1, 1])
                            with col2:

                                if ticker == 'EUR/MAD':

                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en valeur : " + format(
                                            tunnel3, ".3f") + " </h2>",
                                        unsafe_allow_html=True)

                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en MAD : " + format(
                                            tunnel4, ".3f") + " " + "DH" + " </h2>",
                                        unsafe_allow_html=True)

                                elif ticker == 'USD/MAD':
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en valeur : " + format(
                                            tunnel3, ".3f") + " </h2>",
                                        unsafe_allow_html=True)
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en MAD : " + format(
                                            tunnel4, ".3f") + " " + "DH" + " </h2>",
                                        unsafe_allow_html=True)

                            if st.button(f'Information sur opération '):
                                df = pd.DataFrame(
                                    [ticker, format(N, ".2f"), format(s, ".4f"), format(k1, ".4f"),
                                     format(k2, ".4f") + str("%"),
                                     format(tunnel4, ".2f")],
                                    index=pd.Index(
                                        ["Paire de devise", "Nominal", "Spot", "Strike k1", "Strike k2",
                                         "Tunnel Symétrique"]))
                                tdf = df.T
                                # CSS to inject contained in a string
                                hide_table_row_index = """
                                                                                        <style>
                                                                                        tbody th {display:none}
                                                                                        .blank {display:none}
                                                                                        </style>
                                                                                        """

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                    unsafe_allow_html=True)
                                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                                st.table(tdf)
                                st.session_state['button'] = False

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)

                                st.write(
                                    "-Spot " + " " + str(ticker) + "          " + ":" + " " + str(format(s, ".4f")))
                                st.write("-Volatilité:" + "   " + str(format(sigma, ".1f")) + "%")
                                st.write("-Taux doméstique:" + "   " + str(format(rd, ".1f")) + "%")
                                st.write("-Taux étranger:" + "   " + str(format(rf, ".1f")) + "%")

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)

                                st.write("- Si  " + "   " + str(ticker), " " + " < " + str(format(k1, ".4f")))
                                st.write(" Vendre  " + str(N), " à " + " " + str(format(k1, ".4f")))

                                st.write("- Si  " + "   " + str(format(k1, ".4f")) + " < " + str(ticker),
                                         " " + " <" + str(format(k2, ".4f")))
                                st.write(" Vendre  " + str(N), "au spot du jour")

                                st.write("- Si  " + "   " + str(ticker), " "" > " + str(format(k2, ".4f")))
                                st.write(" Vendre  " + str(N), " à " + " " + str(format(k2, ".4f")))

                if genre == 'Stratégie gratuite':

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/MAD', 'USD/MAD'))
                        # st.markdown(f" Paire  : {ticker}")
                    with col2:
                        N = st.number_input('Nominal')
                        # st.markdown(f"Nominal : {N}")
                    with col3:
                        s = st.number_input('Spot')
                        # st.markdown(f"Spot  : {s}")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        exercise_date = st.date_input('Maturité', min_value=datetime.today() + timedelta(days=1),
                                                      value=datetime.today() + timedelta(days=365))
                        # st.markdown(f" Date de valeur :{exercise_date}")

                    with col2:
                        rd = st.number_input('Taux doméstique (%)', 0, 100, 10)
                        # st.markdown(f"Taux doméstique : {r} %")

                    with col3:
                        rf = st.number_input('Taux étranger (%)', 0, 100, 10)
                        # st.markdown(f"Taux étranger : {r2} %")
                    input_volatility = st.checkbox("Volatilité " + str(ticker), value=True)

                    if input_volatility:
                        sigma = st.number_input('Volatilité ' + str(ticker))
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            vol1 = st.number_input('Volatilité EUR/USD')
                            # st.markdown(f" Volatilitée : {sigma} %")
                        with col2:
                            st.markdown(
                                "<h2 style='text-align: left ;color:black;font-size:18px;'> Pondération " + " </h2>",
                                unsafe_allow_html=True)
                            vol2 = st.number_input('Poids EUR', 0, 100, 60)
                            # st.markdown(f" Volatilitée : {sigma} %")
                        with col2:
                            vol3 = st.number_input('Poids USD', 0, 100, 40)

                    st.markdown(
                        "-------------------------------------------------------------------------------------------")
                    if not all([s, rd, rf, N]):
                        st.error("Veuillez remplir tous les champs")
                        button = None
                    else:
                        button = st.button(f'Calculer le strike ')

                    if st.session_state.get('button') != True:
                        st.session_state['button'] = button

                    if st.session_state['button'] == True:

                        if not input_volatility:
                            rd1 = rd / 100
                            rf1 = rf / 100
                            # sigma1 = sigma / 100
                            sig1 = vol1 / 100
                            sig2 = vol2 / 100
                            sig3 = vol3 / 100

                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365
                            if ticker == 'EUR/MAD':
                                volatilite = sig1 * sig3
                            elif ticker == 'USD/MAD':
                                volatilite = sig1 * sig2
                            volatilite1 = volatilite * 100

                            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                                [1, 60, 1, 1, 1, 1, 1, 1, 1])

                            with col2:

                                def cours_terme(s, T1, rd, rf):
                                    r1 = rd - rf
                                    F_k = s * np.exp(r1 * T1)
                                    return F_k


                                Ft = cours_terme(s, T1, rd1, rf1)


                                def BS_CALL1(s, k, T1, rd, rf, volatilite):
                                    d1 = (np.log(s / k) + ((rd - rf) + volatilite ** 2 / 2) * T1) / (
                                                volatilite * np.sqrt(T1))
                                    d2 = d1 - volatilite * np.sqrt(T1)
                                    Call = s * n(d1) * np.exp(-rf * T1) - k * np.exp(-rd * T1) * n(d2)
                                    return Call


                                def BS_PUT1(s, k, T1, rd, rf, volatilite):
                                    d1 = (np.log(s / k) + ((rd - rf) + volatilite ** 2 / 2) * T1) / (
                                                volatilite * np.sqrt(T1))
                                    d2 = d1 - volatilite * np.sqrt(T1)
                                    Put = k * np.exp(-rd * T1) * n(-d2) - s * n(-d1) * np.exp(-rf * T1)
                                    return Put


                                def Ft_gt_k1(k):
                                    k1 = k[0]
                                    return Ft - k1 - 0.0000000000000000000000000000000000001


                                def K2_gt_Ft(k):
                                    k2 = k[1]
                                    return k2 - Ft - 0.0000000000000000000000000000000000001


                                def objective1(k):
                                    k1 = k[0]
                                    k2 = k[1]
                                    return ((BS_PUT(s, k1, T1, rd1, rf1, volatilite) - BS_CALL(s, k2, T1, rd1, rf1,
                                                                                               volatilite))) ** 2


                                k0 = [s, s]

                                con1 = {'type': 'ineq', 'fun': Ft_gt_k1}
                                con2 = {'type': 'ineq', 'fun': K2_gt_Ft}

                                con = [con1, con2]

                                optimize = sci_opt.minimize(objective1, k0, method='SLSQP', constraints=con,
                                                            options={'disp': True})

                                solution1 = optimize.x[0]
                                solution2 = optimize.x[1]

                                # x0 = s
                                #
                                # def constrainte(k):
                                #     return k - cours_terme(s, T1, rd1, rf1)
                                #
                                # def objective(k):
                                #     g = (N * (1 - Perc1) * ((k - s * np.exp((rd1 - rf1) * T1)) * np.exp(
                                #         -(rd1 - rf1) * T1)) + N * Perc1 * BS_PUT(k, T1, s, rd1, rf1, volatilite)) ** 2
                                #     return g
                                #
                                #
                                # cons = ({'type': 'ineq', 'fun': lambda x: x - s})
                                #
                                # optimize = sci_opt.minimize(objective, x0, method='SLSQP', constraints=cons,
                                #                             options={'disp': True})
                                #
                                # k1 = optimize.x
                                # st.markdown(k1)

                                # Ft = cours_terme(s, T1, rd1, rf1)
                                # k1 = find_strike(s, rd1, rf1, T1, volatilite)
                                #
                                # put = BS_PUT(s, k1, T1, rd1, rf1, volatilite)
                                # forward = (k1 - Ft) * np.exp(-r1 * T1)
                                # b = forward + put

                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:20px;'>  Le strike est :  </h2>",
                                    unsafe_allow_html=True)

                                st.markdown(
                                    "<h2 style='text-align: left ;color:;font-size:15px;'>  - K1 = : " + str(
                                        format(solution1, '.4f')) + " </h2>",
                                    unsafe_allow_html=True)

                                st.markdown(
                                    "<h2 style='text-align: left ;color:;font-size:15px;'>  - K2 = : " + str(
                                        format(solution2, '.4f')) + " </h2>",
                                    unsafe_allow_html=True)

                            if st.button(f'Information sur opération '):
                                df = pd.DataFrame(
                                    [ticker, format(N, ".2f"), format(s, ".4f"), format(solution1, ".4f"),
                                     format(solution2, ".4f")],
                                    index=pd.Index(
                                        ["Paire de devise", "Nominal", "Spot", "Strike k1", "Strike k2"]))
                                tdf = df.T
                                # CSS to inject contained in a string
                                hide_table_row_index = """
                                                                                        <style>
                                                                                        tbody th {display:none}
                                                                                        .blank {display:none}
                                                                                        </style>
                                                                                        """

                                ######################### detail de strategie

                                # Inject CSS with Markdown
                                ############ table recap

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                    unsafe_allow_html=True)

                                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                                st.table(tdf)
                                st.session_state['button'] = False

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)

                                st.write(
                                    "-Spot " + " " + str(ticker) + "          " + ":" + " " + str(format(s, ".4f")))
                                st.write("-Volatilité:" + "   " + str(format(volatilite1, ".2f")) + "%")
                                st.write("-Taux doméstique:" + "   " + str(format(rd, ".2f")) + "%")
                                st.write("-Taux étranger:" + "   " + str(format(rf, ".2f")) + "%")

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)

                                st.write("- Si  " + "   " + str(ticker), " " + " < " + str(format(solution1, ".4f")))
                                st.write(" Vendre  " + str(N), " à " + " " + str(format(solution1, ".4f")))

                                st.write("- Si  " + "   " + str(format(solution1, ".4f")) + " < " + str(ticker),
                                         " " + " <" + str(format(solution2, ".4f")))
                                st.write(" Vendre  " + str(N), "au spot du jour")

                                st.write("- Si  " + "   " + str(ticker), " "" > " + str(format(solution2, ".4f")))
                                st.write(" Vendre  " + str(N), " à " + " " + str(format(solution2, ".4f")))

                        else:
                            rd1 = rd / 100
                            rf1 = rf / 100
                            sigma1 = sigma / 100
                            r1 = (rd1 - rf1)

                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365
                            n = norm.cdf


                            def cours_terme(s, T1, rd, rf):
                                r1 = rd - rf
                                F_k = s * np.exp(r1 * T1)
                                return F_k


                            Ft = cours_terme(s, T1, rd1, rf1)


                            def BS_CALL1(s, k, T1, rd, rf, sigma1):
                                d1 = (np.log(s / k) + ((rd - rf) + sigma1 ** 2 / 2) * T1) / (sigma1 * np.sqrt(T1))
                                d2 = d1 - sigma1 * np.sqrt(T1)
                                Call = s * n(d1) * np.exp(-rf * T1) - k * np.exp(-rd * T1) * n(d2)
                                return Call


                            def BS_PUT1(s, k, T1, rd, rf, sigma1):

                                d1 = (np.log(s / k) + ((rd - rf) + sigma1 ** 2 / 2) * T1) / (sigma1 * np.sqrt(T1))
                                d2 = d1 - sigma1 * np.sqrt(T1)
                                Put = k * np.exp(-rd * T1) * n(-d2) - s * n(-d1) * np.exp(-rf * T1)
                                return Put


                            def Ft_gt_k1(k):
                                k1 = k[0]
                                return Ft - k1 - 0.0000000000000000000000000000000000001


                            def K2_gt_Ft(k):
                                k2 = k[1]
                                return k2 - Ft - 0.0000000000000000000000000000000000001


                            def objective1(k):
                                k1 = k[0]
                                k2 = k[1]
                                return ((BS_PUT(s, k1, T1, rd1, rf1, sigma1) - BS_CALL(s, k2, T1, rd1, rf1,
                                                                                       sigma1))) ** 2


                            k0 = [s, s]

                            con1 = {'type': 'ineq', 'fun': Ft_gt_k1}
                            con2 = {'type': 'ineq', 'fun': K2_gt_Ft}

                            con = [con1, con2]

                            optimize = sci_opt.minimize(objective1, k0, method='SLSQP', constraints=con,
                                                        options={'disp': True})

                            solution3 = optimize.x[0]
                            solution4 = optimize.x[1]

                            # def cour_t(s, T, rd, rf):
                            #     F = s * np.exp((rd - rf) * T)
                            #     return F
                            #
                            #
                            # Ft = cour_t(s, T1, rd1, rf1)
                            # x0= s
                            #
                            #
                            # def constrainte(k):
                            #     return k - cour_t(s, T1, rd1, rf1)
                            #
                            #
                            # def objective(k):
                            #     g = (N * (1-Perc1) * ((k - s * np.exp((rd1 - rf1) * T1)) * np.exp( -(rd1 - rf1) * T1)) + N * Perc1 * BS_PUT(s, k, T1, rd1 , rf1, sigma1)) ** 2
                            #     return g
                            #
                            #
                            # cons = ({'type': 'ineq', 'fun': lambda x: x - s})
                            #
                            # optimize = sci_opt.minimize(objective, x0, method='SLSQP', constraints=cons, options={'disp': True})
                            #
                            # k2 =optimize.x
                            # st.markdown(k2)
                            # # Ft = cours_terme(s, T1, rd1, rf1)

                            # k2 = find_strike(s, rd1, rf1, T1, sigma1)
                            # st.markdown(Ft)
                            # st.markdown(k2)
                            # put = BS_PUT(s, k2, T1, rd1, rf1, sigma)
                            # forward = (k2 - Ft) * np.exp(-r1 * T1)
                            # b = forward + put

                            # col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                            #     [1, 60, 1, 1, 1, 1, 1, 1, 1])
                            # with col2:
                            #
                            #     st.markdown(
                            #         "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Le Strike est : " + format(
                            #             k2 , ".3f")+ " </h2>",
                            #         unsafe_allow_html=True)

                            st.markdown(
                                "<h2 style='text-align: left ;color:black;font-size:20px;'>  Le strike est :  </h2>",
                                unsafe_allow_html=True)

                            st.markdown(
                                "<h2 style='text-align: left ;color:;font-size:15px;'>  - K1 = : " + str(
                                    format(solution3, '.4f')) + " </h2>",
                                unsafe_allow_html=True)

                            st.markdown(
                                "<h2 style='text-align: left ;color:;font-size:15px;'>  - K2 = : " + str(
                                    format(solution4, '.4f')) + " </h2>",
                                unsafe_allow_html=True)

                            if st.button(f'Information sur opération'):
                                df = pd.DataFrame(
                                    [ticker, format(N, ".2f"), format(s, ".4f"), format(solution3, ".4f"),
                                     format(solution4, ".4f")],
                                    index=pd.Index(["Paire de devise", "Nominal", "Spot", "Strike k1", "Strike k2"]))
                                tdf1 = df.T
                                # CSS to inject contained in a string
                                hide_table_row_index = """
                                                                                                                        <style>
                                                                                                                        tbody th {display:none}
                                                                                                                        .blank {display:none}
                                                                                                                        </style>
                                                                                                                        """

                                ######################### detail de strategie

                                # Inject CSS with Markdown
                                ############ table recap

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                    unsafe_allow_html=True)

                                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                                st.table(tdf1)
                                st.session_state['button'] = False

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)

                                st.write(
                                    "-Spot " + " " + str(ticker) + "          " + ":" + " " + str(format(s, ".4f")))
                                st.write("-Volatilité:" + "   " + str(format(sigma, ".2f")) + "%")
                                st.write("-Taux doméstique:" + "   " + str(format(rd, ".2f")) + "%")
                                st.write("-Taux étranger:" + "   " + str(format(rf, ".2f")) + "%")

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)

                                st.write("- Si  " + "   " + str(ticker), " " + " < " + str(format(solution3, ".4f")))
                                st.write(" Vendre  " + str(N), " à " + " " + str(format(solution3, ".4f")))

                                st.write("- Si  " + "   " + str(format(solution3, ".4f")) + " < " + str(ticker),
                                         " " + " <" + str(format(solution4, ".4f")))
                                st.write(" Vendre  " + str(N), "au spot du jour")

                                st.write("- Si  " + "   " + str(ticker), " "" > " + str(format(solution4, ".4f")))
                                st.write(" Vendre  " + str(N), " à " + " " + str(format(solution4, ".4f")))






            ############################################################################################################################################################
            # elif choose == "Tunnel Symétrique":
            #
            #     st.header(" ")
            #
            #     col1, col2, col3,col4= st.columns(4)
            #     with col1:
            #         ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/USD', 'EUR/MAD', 'USD/MAD'))
            #         # st.markdown(f" Paire  : {ticker}")
            #     with col2:
            #         N = st.number_input('Nominal')
            #         # st.markdown(f"Nominal : {N}")
            #     with col3:
            #         s = st.number_input('Spot')
            #         # st.markdown(f"Spot  : {s}")
            #     with col4:
            #         exercise_date = st.date_input('Exercise date', min_value=datetime.today() + timedelta(days=1),
            #                                       value=datetime.today() + timedelta(days=365))
            #         # st.markdown(f" Date de valeur :{exercise_date}")
            #
            #     col1, col2, col3,col4,col5 = st.columns(5)
            #
            #     with col1:
            #         k1 = st.number_input('Strike k1')
            #         # st.markdown(f"Percentage : {k1} %")
            #
            #     with col2:
            #         k2 = st.number_input('Strike k2 ')
            #         # .markdown(f"Strike  : {k2}")
            #     with col3:
            #         rd = st.number_input('Taux doméstique (%)', 0, 100, 10)
            #         # st.markdown(f"Taux doméstique : {r} %")
            #
            #     with col4:
            #         rf = st.number_input('Taux étranger (%)', 0, 100, 10)
            #         # st.markdown(f"Taux étranger : {r2} %")
            #
            #     with col5:
            #         sigma = st.number_input('Volatilité (%)', 0, 100, 10)
            #         # st.markdown(f" Volatilitée : {sigma} %")
            #     if sigma == 0:
            #         col1, col2 = st.columns(2)
            #         with col1:
            #             vol1 = st.number_input('Volatilité EUR/USD', 0, 100, 10)
            #             # st.markdown(f" Volatilitée : {sigma} %")
            #         with col2:
            #             st.markdown(
            #                 "<h2 style='text-align: left ;color:black;font-size:18px;'> Pondération " + " </h2>",
            #                 unsafe_allow_html=True)
            #             vol2 = st.number_input('Poids EUR', 0, 100, 60)
            #             # st.markdown(f" Volatilitée : {sigma} %")
            #         with col2:
            #             vol3 = st.number_input('Poids USD', 0, 100, 40)
            #     st.markdown('-------------------------------------------------------------------------------------')
            #     genre = st.radio(
            #         "Choisissez le type de la  stratégie ",
            #         ('Stratégie payante', 'Stratégie gratuite'))
            #
            #     if genre == 'Stratégie payante':
            #
            #         # gestion des exceptions
            #         st.markdown("-------------")
            #         if not all([s, k1, rd, rf, k2]):
            #             st.error("Veuillez remplir tous les champs")
            #             button = None
            #         else:
            #             button = st.button(f'Calculer la prime ')
            #
            #         if st.session_state.get('button') != True:
            #             st.session_state['button'] = button
            #
            #         if st.session_state['button'] == True:
            #
            #             # Formating selected model parameters
            #
            #             rd1 = rd / 100
            #             rf1 = rf / 100
            #             sigma1 = sigma / 100
            #             sig1 = vol1 / 100
            #             sig2 = vol2 / 100
            #             sig3 = vol3 / 100
            #             r1 = (rd1 - rf1)
            #             T = (exercise_date - datetime.now().date()).days
            #             T1 = T / 365
            #             if ticker == 'EUR/MAD':
            #                 volatilite = sig1 * sig2
            #             elif ticker == 'USD/MAD':
            #                 volatilite = sig1 * sig3
            #             else:
            #                 volatilite = sig1
            #
            #             if sigma == 0:
            #
            #                 col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
            #                     [1, 60, 1, 1, 1, 1, 1, 1, 1])
            #
            #                 with col2:
            #
            #
            #                     Ft = cours_terme(s, T1, rd1, rf1)
            #
            #                     c = BS_CALL(s, k2, T1, rd1, rf1, volatilite)
            #
            #                     p = BS_PUT(s, k1, T1, rd1, rf1, volatilite)
            #
            #                     put = p  * N
            #                     call = c  * N
            #
            #                     tunnel = c - p
            #                     tunnel1 = call - put
            #                     tunnel2 = (call - put)*s
            #
            #
            #                     if ticker == 'EUR/MAD':
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en % du Nominal : " + format(
            #                                 tunnel1, ".3f") + " " + "€" + " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en Devise : " + format(
            #                                 tunnel2, ".3f") + " " + "DH" + " </h2>",
            #                             unsafe_allow_html=True)
            #
            #                     elif ticker == 'USD/MAD':
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en % du Nominal : " + format(
            #                                 tunnel1, ".3f") + " " + "$" + " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en Devise : " + format(
            #                                 tunnel2, ".3f") + " " + "DH" + " </h2>",
            #                             unsafe_allow_html=True)
            #                     elif ticker == 'EUR/USD':
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en % du Nominal : " + format(
            #                                 tunnel1, ".3f") + " " + "€" + " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en Devise : " + format(
            #                                 tunnel2, ".3f") + " " + "$" + " </h2>",
            #                             unsafe_allow_html=True)
            #             else:
            #
            #
            #                 col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
            #                     [1, 60, 1, 1, 1, 1, 1, 1, 1])
            #                 with col2:
            #
            #
            #                     Ft = cours_terme(s, T1, rd1, rf1)
            #
            #                     c3 = BS_CALL(s, k2, T1, rd1, rf1, sigma1)
            #
            #                     p3 = BS_PUT(s, k1, T1, rd1, rf1, sigma1)
            #
            #                     put3 = p3 * N
            #                     call3 = c3 * N
            #
            #                     tunnel = c3 - p3
            #                     tunnel3 = call3 - put3
            #                     tunnel4 = (call3 - put3) * s
            #                     if ticker == 'EUR/MAD':
            #
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en % du Nominal : " + format(
            #                                 tunnel3 , ".3f") + " " + "€" + " </h2>",
            #                             unsafe_allow_html=True)
            #
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en Devise : " + format(
            #                                 tunnel4, ".3f") + " " + "DH" + " </h2>",
            #                             unsafe_allow_html=True)
            #
            #                     elif ticker == 'USD/MAD':
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en % du Nominal : " + format(
            #                                 tunnel3, ".3f") + " " + "$" + " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en Devise : " + format(
            #                                 tunnel4, ".3f") + " " + "DH" + " </h2>",
            #                             unsafe_allow_html=True)
            #                     elif ticker == 'EUR/USD':
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en % du Nominal : " + format(
            #                                 tunnel3, ".3f") + " " + "€" + " </h2>",
            #                             unsafe_allow_html=True)
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en Devise : " + format(
            #                                 tunnel4, ".3f") + " " + "$" + " </h2>",
            #                             unsafe_allow_html=True)
            #
            #
            #
            #             if st.button(f'Information sur opération '):
            #
            #                 if sigma ==0 :
            #
            #                     df = pd.DataFrame([ticker, format(N, ".2f"), format(s, ".4f"), format(k1, ".4f"),
            #                                        format(k2, ".4f"),
            #                                        format(tunnel2, ".2f")],
            #                                       index=pd.Index(
            #                                           ["Paire de devise", "Nominal", "Spot", "Strike k1 ", "Strike k2",
            #                                            " Tunnel Symétrique"]))
            #                     tdf = df.T
            #                     # CSS to inject contained in a string
            #                     hide_table_row_index = """
            #                                                                     <style>
            #                                                                     tbody th {display:none}
            #                                                                     .blank {display:none}
            #                                                                     </style>
            #                                                                     """
            #
            #                     ############ table recap
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
            #                         unsafe_allow_html=True)
            #                     st.header("")
            #                     st.markdown(hide_table_row_index, unsafe_allow_html=True)
            #                     st.table(tdf)
            #                     st.session_state['button'] = False
            #
            #                     ######### detail
            #
            #                     col1, col2, col3 = st.columns(3)
            #                     with col1:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("-Spot " + " " + str(ticker) + "          " + ":" + " " + str(s))
            #                         st.write("-Volatilité:" + "   " + str(volatilite) + "%")
            #                         st.write("-Taux doméstique:" + "   " + str(rd) + "%")
            #                         st.write("-Taux étranger:" + "   " + str(rf) + "%")
            #                         # st.write("-Volatilité:" + "   " + str(sigma))
            #                     st.header("  ")
            #                     with col3:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement a l'échéance </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("- si  " + "   " + str(ticker), " " + " < " + str(k1))
            #                         st.write(" Vous vendez  " + str(N), " à " + " " + str(k1))
            #
            #                         st.write("- si  " + "   " + str(k1) + " < " + str(ticker), " " + " <" + str(s))
            #                         st.write(" Vous vendez  " + str(N), "au spot du jour")
            #
            #                         st.write("- si  " + "   " + str(ticker), " "" > " + str(k2))
            #                         st.write(" Vous vendez  " + str(N), " à " + " " + str(k2))
            #
            #                 else :
            #
            #                     df = pd.DataFrame([ticker, format(N, ".2f"), format(s, ".4f"), format(k1, ".4f"),
            #                                        format(k2, ".4f"),format(tunnel4, ".2f")],
            #                                       index=pd.Index(
            #                                           ["Paire de devise", "Nominal", "Spot", "Strike k1 ", "Strike k2",
            #                                            " Tunnel Symétrique"]))
            #                     tdf = df.T
            #                     # CSS to inject contained in a string
            #                     hide_table_row_index = """
            #                                         <style>
            #                                         tbody th {display:none}
            #                                         .blank {display:none}
            #                                         </style>
            #                                         """
            #
            #                     ############ table recap
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
            #                         unsafe_allow_html=True)
            #                     st.header("")
            #                     st.markdown(hide_table_row_index, unsafe_allow_html=True)
            #                     st.table(tdf)
            #                     st.session_state['button'] = False
            #
            #                     ######### detail
            #
            #                     col1, col2, col3 = st.columns(3)
            #                     with col1:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("-Spot " + " " + str(ticker) + "          " + ":" + " " + str(s))
            #                         st.write("-Volatilité:" + "   " + str(sigma) + "%")
            #                         st.write("-Taux doméstique:" + "   " + str(rd) + "%")
            #                         st.write("-Taux étranger:" + "   " + str(rf) + "%")
            #                         # st.write("-Volatilité:" + "   " + str(sigma))
            #                     st.header("  ")
            #                     with col3:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement a l'échéance </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("- si  " + "   " + str(ticker), " " + " < " + str(k1))
            #                         st.write(" Vous vendez  " + str(N), " à " + " " + str(k1))
            #
            #                         st.write("- si  " + "   " + str(k1) + " < " + str(ticker), " " + " <" + str(s))
            #                         st.write(" Vous vendez  " + str(N), "au spot du jour")
            #
            #                         st.write("- si  " + "   " + str(ticker), " "" > " + str(k2))
            #                         st.write(" Vous vendez  " + str(N), " à " + " " + str(k2))
            #
            #     else:
            #
            #         # gestion des exceptions
            #         st.markdown("-------------")
            #         if not all([s, k1, k2, rd, rf]):
            #             st.error("Veuillez remplir tous les champs")
            #             button = None
            #         else:
            #             button = st.button(f'Calculer le strike amélioré ')
            #
            #         if st.session_state.get('button') != True:
            #             st.session_state['button'] = button
            #
            #         if st.session_state['button'] == True:
            #
            #             if sigma==0 :
            #
            #                 # Formating selected model parameters
            #                 rd1 = rd / 100
            #                 rf1 = rf / 100
            #                 r1 = (rd1 - rf1)
            #                 sigma1 = sigma / 100
            #                 T = (exercise_date - datetime.now().date()).days
            #                 T1 = T / 365
            #                 sig1 = vol1 / 100
            #                 sig2 = vol2 / 100
            #                 sig3 = vol3 / 100
            #
            #                 if ticker == 'EUR/MAD':
            #                     volatilite = sig1 * sig3
            #                 elif ticker == 'USD/MAD':
            #                     volatilite = sig1 * sig2
            #
            #
            #
            #                 ######################### optimisation
            #                 def constrainte(k):
            #                     k1 = k[0]
            #                     k2 = k[1]
            #                     return k2 - k1 - 0.0000000000000000000000000000000000001
            #
            #
            #                 def constrainte_2(k):
            #                     k1 = k[0]
            #                     k2 = k[1]
            #                     return k2 - s - 0.0000000000000000000000000000000000001
            #
            #
            #                 def constrainte_3(k):
            #                     k1 = k[0]
            #                     k2 = k[1]
            #                     return s - k1 - 0.0000000000000000000000000000000000001
            #
            #
            #                 def objective1(k):
            #                     k1 = k[0]
            #                     k2 = k[1]
            #                     return ((BS_CALL(s, k2, T1, rd1, rf1, volatilite)) - BS_PUT(s, k1, T1, rd1, rf1,
            #                                                                             volatilite)) ** 2
            #
            #
            #                 k0 = [s - 0.4, s - 0.2]
            #                 cons = {'type': 'ineq', 'fun': constrainte}
            #                 con2 = {'type': 'ineq', 'fun': constrainte_2}
            #                 con3 = {'type': 'ineq', 'fun': constrainte_3}
            #                 cons1 = [cons, con2, con3]
            #
            #                 optimize = sci_opt.minimize(objective1, k0, method='SLSQP', constraints=cons1,
            #                                             options={'disp': True})
            #
            #                 call2 = BS_CALL(s, optimize.x[1], T1, rd1, rf1, volatilite)
            #                 put2 = BS_PUT(s, optimize.x[0], T1, rd1, rf1, volatilite)
            #
            #                 c = optimize.x[1]
            #                 p = optimize.x[0]
            #                 tunnel1 = call2 - put2
            #
            #
            #                 st.markdown(
            #                     "<h2 style='text-align: left ;color:black;font-size:20px;'>  Le strike amélioré est :  </h2>",
            #                     unsafe_allow_html=True)
            #
            #                 st.markdown(
            #                     "<h2 style='text-align: left ;color:;font-size:15px;'>  - K1 = : " + str(
            #                         format(p, '.4f')) + " </h2>",
            #                     unsafe_allow_html=True)
            #
            #                 st.markdown(
            #                     "<h2 style='text-align: left ;color:;font-size:15px;'>  - K2 = : " + str(
            #                         format(c, '.4f')) + " </h2>",
            #                     unsafe_allow_html=True)
            #
            #                 if st.button(f'Informtion sur opération '):
            #                     df = pd.DataFrame(
            #                         [ticker, format(N, ".2f"), format(s, ".4f"), format(p, ".4f"), format(c, ".4f"),
            #                          format(tunnel1, ".2f")],
            #                         index=pd.Index(
            #                             ["Paire de devise", "Nominal", "Spot", "Strike k1", "Strike k2",
            #                               "Prime"]))
            #
            #                     tdf = df.T
            #                     # CSS to inject contained in a string
            #                     hide_table_row_index = """
            #                                                                     <style>
            #                                                                     tbody th {display:none}
            #                                                                     .blank {display:none}
            #                                                                     </style>
            #                                                                     """
            #
            #                     ############ table recap
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
            #                         unsafe_allow_html=True)
            #                     st.header("")
            #                     st.markdown(hide_table_row_index, unsafe_allow_html=True)
            #                     st.table(tdf)
            #                     st.session_state['button'] = False
            #
            #                     ######### detail
            #
            #                     col1, col2, col3 = st.columns(3)
            #                     with col1:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("-Spot " + " " + str(ticker) + "          " + ":" + " " + str(s))
            #                         st.write("-Volatilité:" + "   " + str(volatilite) + "%")
            #                         st.write("-Taux doméstique:" + "   " + str(rd) + "%")
            #                         st.write("-Taux étranger:" + "   " + str(rf) + "%")
            #
            #                     st.header("  ")
            #                     with col3:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("- si  " + "   " + str(ticker), " " + " < " + str(format(p, ".4f")))
            #                         st.write(" Vous vendez  " + str(N), " à " + " " + str(format(p, ".4f")))
            #
            #                         st.write("- si  " + "   " + str(format(p, ".4f")) + " < " + str(ticker),
            #                                  " " + " <" + str(format(c, ".4f")))
            #                         st.write(" Vous vendez  " + str(N), "au spot du jour")
            #
            #                         st.write("- si  " + "   " + str(ticker), " "" > " + str(format(c, ".4f")))
            #                         st.write(" Vous vendez  " + str(N), " à " + " " + str(format(c, ".4f")))
            #
            #             else :
            #
            #
            #                 # Formating selected model parameters
            #                 rd1 = rd / 100
            #
            #                 rf1 = rf / 100
            #
            #                 r1 = (rd1 - rf1)
            #                 sigma1 = sigma / 100
            #
            #                 T = (exercise_date - datetime.now().date()).days
            #                 T1 = T / 365
            #
            #
            #                 ######################### optimisation
            #                 def constrainte(k):
            #                     k1 = k[0]
            #                     k2 = k[1]
            #                     return k2 - k1 - 0.0000000000000000000000000000000000001
            #
            #
            #                 def constrainte_2(k):
            #                     k1 = k[0]
            #                     k2 = k[1]
            #                     return k2 - s - 0.0000000000000000000000000000000000001
            #
            #
            #                 def constrainte_3(k):
            #                     k1 = k[0]
            #                     k2 = k[1]
            #                     return s - k1 - 0.0000000000000000000000000000000000001
            #
            #
            #                 def objective1(k):
            #                     k1 = k[0]
            #                     k2 = k[1]
            #                     return ((BS_CALL(s, k2, T1, rd1, rf1, sigma1)) - BS_PUT(s, k1, T1, rd1, rf1, sigma1)) ** 2
            #
            #
            #                 k0 = [s - 0.4, s - 0.2]
            #                 cons = {'type': 'ineq', 'fun': constrainte}
            #                 con2 = {'type': 'ineq', 'fun': constrainte_2}
            #                 con3 = {'type': 'ineq', 'fun': constrainte_3}
            #                 cons1 = [cons, con2, con3]
            #
            #                 optimize = sci_opt.minimize(objective1, k0, method='SLSQP', constraints=cons1,
            #                                             options={'disp': True})
            #
            #                 call2 = BS_CALL(s, optimize.x[1], T1, rd1, rf1, sigma1)
            #                 put2 = BS_PUT(s, optimize.x[0], T1, rd1, rf1, sigma1)
            #
            #                 c = optimize.x[1]
            #                 p = optimize.x[0]
            #                 tunnel3 = call2 - put2
            #                 tunnel4 = (call2 - put2)*s
            #
            #                 st.markdown(
            #                     "<h2 style='text-align: left ;color:black;font-size:20px;'>  Le strike amélioré est :  </h2>",
            #                     unsafe_allow_html=True)
            #
            #                 st.markdown(
            #                     "<h2 style='text-align: left ;color:;font-size:15px;'>  - K1 = : " + str(
            #                         format(p, '.4f')) + " </h2>",
            #                     unsafe_allow_html=True)
            #
            #                 st.markdown(
            #                     "<h2 style='text-align: left ;color:;font-size:15px;'>  - K2 = : " + str(
            #                         format(c, '.4f')) + " </h2>",
            #                     unsafe_allow_html=True)
            #
            #                 if st.button(f'Informtion sur opération '):
            #                     df = pd.DataFrame(
            #                         [ticker, format(N, ".2f"), format(s, ".4f"), format(p, ".4f"), format(c, ".4f")
            #                            , format(tunnel4, ".2f")],
            #                         index=pd.Index(
            #                             ["Paire de devise", "Nominal", "Spot", "Strike k1", "Strike k2", "Prime"]))
            #
            #                     tdf = df.T
            #                     # CSS to inject contained in a string
            #                     hide_table_row_index = """
            #                                             <style>
            #                                             tbody th {display:none}
            #                                             .blank {display:none}
            #                                             </style>
            #                                             """
            #
            #
            #                     ############ table recap
            #
            #                     st.markdown(
            #                         "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
            #                         unsafe_allow_html=True)
            #                     st.header("")
            #                     st.markdown(hide_table_row_index, unsafe_allow_html=True)
            #                     st.table(tdf)
            #                     st.session_state['button'] = False
            #
            #                     ######### detail
            #
            #                     col1, col2, col3 = st.columns(3)
            #                     with col1:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("-Spot " + " " + str(ticker) + "          " + ":" + " " + str(s))
            #                         st.write("-Volatilité:" + "   " + str(sigma) + "%")
            #                         st.write("-Taux doméstique:" + "   " + str(rd) + "%")
            #                         st.write("-Taux étranger:" + "   " + str(rf) + "%")
            #
            #                     st.header("  ")
            #                     with col3:
            #                         st.header(" ")
            #                         st.markdown(
            #                             "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
            #                             unsafe_allow_html=True)
            #                         st.header("")
            #
            #                         st.write("- si  " + "   " + str(ticker), " " + " < " + str(format(p, ".4f")))
            #                         st.write(" Vous vendez  " + str(N), " à " + " " + str(format(p, ".4f")))
            #
            #                         st.write("- si  " + "   " + str(format(p, ".4f")) + " < " + str(ticker),
            #                                  " " + " <" + str(format(c, ".4f")))
            #                         st.write(" Vous vendez  " + str(N), "au spot du jour")
            #
            #                         st.write("- si  " + "   " + str(ticker), " "" > " + str(format(c, ".4f")))
            #                         st.write(" Vous vendez  " + str(N), " à " + " " + str(format(c, ".4f")))

            ######################################################################################################################################################

            elif choose == "Tunnel Asymétrique":

                genre = st.radio(
                    label="Choisissez le type de la  stratégie ", horizontal=True,
                    options=('Stratégie payante', 'Stratégie gratuite'))
                st.markdown('')
                if genre == 'Stratégie payante':

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/MAD', 'USD/MAD'))
                        # st.markdown(f" Paire  : {ticker}")
                    with col2:
                        N = st.number_input('Nominal')
                        # st.markdown(f"Nominal : {N}")
                    with col3:
                        s = st.number_input('Spot')
                        # st.markdown(f"Spot  : {s}")
                    with col4:
                        exercise_date = st.date_input('Maturité', min_value=datetime.today() + timedelta(days=1),
                                                      value=datetime.today() + timedelta(days=365))
                        # st.markdown(f" Date de valeur :{exercise_date}")

                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        m = st.number_input("Coefficient d'asymétrie")
                    with col2:
                        k1 = st.number_input('Strike k1')
                        # st.markdown(f"Percentage : {k1} %")

                    with col3:
                        k2 = st.number_input('Strike k2 ')
                        # .markdown(f"Strike  : {k2}")
                    with col4:
                        rd = st.number_input('Taux doméstique (%)', 0, 100, 10)
                        # st.markdown(f"Taux doméstique : {r} %")

                    with col5:
                        rf = st.number_input('Taux étranger (%)', 0, 100, 10)
                        # st.markdown(f"Taux étranger : {r2} %")

                    input_volatility = st.checkbox("Volatilité " + str(ticker), value=True)
                    if input_volatility:
                        sigma = st.number_input('Volatilité ' + str(ticker))
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            vol1 = st.number_input('Volatilité EUR/USD', 0, 100, 10)
                            # st.markdown(f" Volatilitée : {sigma} %")
                        with col2:
                            st.markdown(
                                "<h2 style='text-align: left ;color:black;font-size:18px;'> Pondération " + " </h2>",
                                unsafe_allow_html=True)
                            vol2 = st.number_input('Poids EUR', 0, 100, 60)
                            # st.markdown(f" Volatilitée : {sigma} %")
                        with col2:
                            vol3 = st.number_input('Poids USD', 0, 100, 40)
                    st.markdown('-------------------------------------------------------------------------------------')

                    if not all([s, k1, k2, rd, rf, m]):
                        st.error("Veuillez remplir tous les champs")

                        button = None
                    else:
                        button = st.button(f'Calculer la prime ')

                    if st.session_state.get('button') != True:
                        st.session_state['button'] = button

                    if st.session_state['button'] == True:

                        if not input_volatility:
                            rd1 = rd / 100
                            rf1 = rf / 100
                            sig1 = vol1 / 100
                            sig2 = vol2 / 100
                            sig3 = vol3 / 100

                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365
                            if ticker == 'EUR/MAD':
                                volatilite = sig1 * sig3
                            elif ticker == 'USD/MAD':
                                volatilite = sig1 * sig2
                            volatilite1 = volatilite * 100

                            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                                [1, 60, 1, 1, 1, 1, 1, 1, 1])

                            with col2:

                                Ft = cours_terme(s, T1, rd1, rf1)
                                c = m * BS_CALL(s, k2, T1, rd1, rf1, volatilite)
                                p = BS_PUT(s, k1, T1, rd1, rf1, volatilite)

                                tunnel1 = (p - c)
                                tunnel2 = (p - c) * s * N

                                # Ft = cours_terme(s, T1, rd1, rf1)
                                # p1 = BS_PUT(s, k, T1, rd1, rf1, volatilite)
                                # put2 = p1 * (Perc1 * N) * s
                                # f = (k - Ft) * np.exp(-(rd1 - rf1) * T1)
                                # forward2 = f * (1 - Perc1) * N* s
                                #
                                # Prime_partcipative1 = ((p1 + f)/k)*100
                                # Prime_partcipative2 = put2+forward2

                                if ticker == 'EUR/MAD':
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en valeur : " + format(
                                            tunnel1, ".3f") + " </h2>",
                                        unsafe_allow_html=True)
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
                                            tunnel2, ".3f") + " " + "DH" + " </h2>",
                                        unsafe_allow_html=True)

                                elif ticker == 'USD/MAD':
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en valeur : " + format(
                                            tunnel1, ".3f") + " </h2>",
                                        unsafe_allow_html=True)
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:18px;'>    - Prime en MAD : " + format(
                                            tunnel2, ".3f") + " " + "DH" + " </h2>",
                                        unsafe_allow_html=True)
                            if st.button(f'Information sur opération '):
                                df = pd.DataFrame(
                                    [ticker, format(N, ".2f"), format(s, ".4f"), format(k1, ".4f"), format(k2, ".4f"),
                                     format(tunnel2, ".2f")],
                                    index=pd.Index(["Paire de devise", "Nominal", "Spot", "Strike k1", "Strike k2",
                                                    "Tunnel Asymétrique"]))
                                tdf = df.T
                                # CSS to inject contained in a string
                                hide_table_row_index = """
                                                                                            <style>
                                                                                            tbody th {display:none}
                                                                                            .blank {display:none}
                                                                                            </style>
                                                                                            """

                                ######################### detail de strategie

                                # Inject CSS with Markdown
                                ############ table recap

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                    unsafe_allow_html=True)

                                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                                st.table(tdf)
                                st.session_state['button'] = False

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)

                                st.write(
                                    "-Spot " + " " + str(ticker) + "          " + ":" + " " + str(format(s, ".4f")))
                                st.write("-Volatilité:" + "   " + str(format(volatilite1, ".2f")) + "%")
                                st.write("-Taux doméstique:" + "   " + str(format(rd, ".2f")) + "%")
                                st.write("-Taux étranger:" + "   " + str(format(rf, ".2f")) + "%")

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)

                                st.write("- si  " + "   " + str(ticker), " " + " < " + str(format(k1, ".4f")))
                                st.write(" Vendre  " + str(N), " à " + " " + str(format(k1, ".4f")))

                                st.write("- si  " + "   " + str(format(k1, ".4f")) + " < " + str(ticker),
                                         " " + " <" + str(format(k2, ".4f")))
                                st.write(" Vendre  " + str(N), "au spot du jour")

                                st.write("- si  " + "   " + str(ticker), " "" > " + str(format(k2, ".4f")))
                                st.write(" Vendre  " + str(m * N), " à " + " " + str(format(k2, ".4f")))



                        else:
                            rd1 = rd / 100
                            rf1 = rf / 100
                            sigma1 = sigma / 100
                            r1 = (rd1 - rf1)

                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365

                            Ft = cours_terme(s, T1, rd1, rf1)
                            c = m * BS_CALL(s, k2, T1, rd1, rf1, sigma1)
                            p = BS_PUT(s, k1, T1, rd1, rf1, sigma1)

                            tunnel3 = (p - c)
                            tunnel4 = (p - c) * s * N

                            # Ft = cours_terme(s, T1, rd1, rf1)
                            # p2 = BS_PUT(s, k, T1, rd1, rf1, sigma1)
                            # put3 = p2 * (Perc1 * N)
                            # f2 = (k - Ft) * np.exp(-(rd1 - rf1) * T1)
                            # forward3 = f2* (1 - Perc1) * N * s
                            #
                            # Prime_partcipative3 = ((p2 + f2) / k) * 100
                            # Prime_partcipative4 = put3+ forward3

                            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                                [1, 60, 1, 1, 1, 1, 1, 1, 1])
                            with col2:

                                if ticker == 'EUR/MAD':

                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en valeur : " + format(
                                            tunnel3, ".3f") + " </h2>",
                                        unsafe_allow_html=True)

                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en MAD : " + format(
                                            tunnel4, ".3f") + " " + "DH" + " </h2>",
                                        unsafe_allow_html=True)

                                elif ticker == 'USD/MAD':
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en valeur : " + format(
                                            tunnel3, ".3f") + " </h2>",
                                        unsafe_allow_html=True)
                                    st.markdown(
                                        "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Prime en MAD : " + format(
                                            tunnel4, ".3f") + " " + "DH" + " </h2>",
                                        unsafe_allow_html=True)

                            if st.button(f'Information sur opération '):
                                df = pd.DataFrame(
                                    [ticker, format(N, ".2f"), format(s, ".4f"), format(k1, ".4f"),
                                     format(k2, ".4f"),
                                     format(tunnel4, ".2f")],
                                    index=pd.Index(
                                        ["Paire de devise", "Nominal", "Spot", "Strike k1", "Strike k2",
                                         "Tunnel Symétrique"]))
                                tdf = df.T
                                # CSS to inject contained in a string
                                hide_table_row_index = """
                                                                                            <style>
                                                                                            tbody th {display:none}
                                                                                            .blank {display:none}
                                                                                            </style>
                                                                                            """

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                    unsafe_allow_html=True)
                                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                                st.table(tdf)
                                st.session_state['button'] = False

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)

                                st.write(
                                    "-Spot " + " " + str(ticker) + "          " + ":" + " " + str(format(s, ".4f")))
                                st.write("-Volatilité:" + "   " + str(format(sigma, ".1f")) + "%")
                                st.write("-Taux doméstique:" + "   " + str(format(rd, ".1f")) + "%")
                                st.write("-Taux étranger:" + "   " + str(format(rf, ".1f")) + "%")

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)

                                st.write("- Si  " + "   " + str(ticker), " " + " < " + str(format(k1, ".4f")))
                                st.write(" Vendre  " + str(N), " à " + " " + str(format(k1, ".4f")))

                                st.write("- Si  " + "   " + str(format(k1, ".4f")) + " < " + str(ticker),
                                         " " + " <" + str(format(k2, ".4f")))
                                st.write(" Vendre  " + str(N), "au spot du jour")

                                st.write("- Si  " + "   " + str(ticker), " "" > " + str(format(k2, ".4f")))
                                st.write(" Vendre  " + str(m * N), " à " + " " + str(format(k2, ".4f")))

                if genre == 'Stratégie gratuite':

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/MAD', 'USD/MAD'))
                        # st.markdown(f" Paire  : {ticker}")
                    with col2:
                        N = st.number_input('Nominal')
                        # st.markdown(f"Nominal : {N}")
                    with col3:
                        s = st.number_input('Spot')
                        # st.markdown(f"Spot  : {s}")

                    with col4:
                        exercise_date = st.date_input('Maturité', min_value=datetime.today() + timedelta(days=1),
                                                      value=datetime.today() + timedelta(days=365))
                        # st.markdown(f" Date de valeur :{exercise_date}")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        m = st.number_input("Coefficient d'asymétrie")

                    with col2:
                        rd = st.number_input('Taux doméstique (%)')
                        # st.markdown(f"Taux doméstique : {r} %")

                    with col3:
                        rf = st.number_input('Taux étranger (%)')
                        # st.markdown(f"Taux étranger : {r2} %")
                    input_volatility = st.checkbox("Volatilité " + str(ticker), value=True)

                    if input_volatility:
                        sigma = st.number_input('Volatilité ' + str(ticker))
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            vol1 = st.number_input('Volatilité EUR/USD')
                            # st.markdown(f" Volatilitée : {sigma} %")
                        with col2:
                            st.markdown(
                                "<h2 style='text-align: left ;color:black;font-size:18px;'> Pondération " + " </h2>",
                                unsafe_allow_html=True)
                            vol2 = st.number_input('Poids EUR', 0, 100, 60)
                            # st.markdown(f" Volatilitée : {sigma} %")
                        with col2:
                            vol3 = st.number_input('Poids USD', 0, 100, 40)

                    st.markdown(
                        "-------------------------------------------------------------------------------------------")
                    if not all([s, rd, rf, N, m]):
                        st.error("Veuillez remplir tous les champs")
                        button = None
                    else:
                        button = st.button(f'Calculer la prime ')

                    if st.session_state.get('button') != True:
                        st.session_state['button'] = button

                    if st.session_state['button'] == True:

                        if not input_volatility:
                            rd1 = rd / 100
                            rf1 = rf / 100
                            # sigma1 = sigma / 100
                            sig1 = vol1 / 100
                            sig2 = vol2 / 100
                            sig3 = vol3 / 100

                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365
                            if ticker == 'EUR/MAD':
                                volatilite = sig1 * sig3
                            elif ticker == 'USD/MAD':
                                volatilite = sig1 * sig2
                            volatilite1 = volatilite * 100

                            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                                [1, 60, 1, 1, 1, 1, 1, 1, 1])

                            with col2:

                                def cours_terme(s, T1, rd, rf):
                                    r1 = rd - rf
                                    F_k = s * np.exp(r1 * T1)
                                    return F_k


                                Ft = cours_terme(s, T1, rd1, rf1)


                                def BS_CALL1(s, k, T1, rd, rf, volatilite):
                                    d1 = (np.log(s / k) + ((rd - rf) + volatilite ** 2 / 2) * T1) / (
                                            volatilite * np.sqrt(T1))
                                    d2 = d1 - volatilite * np.sqrt(T1)
                                    Call = s * n(d1) * np.exp(-rf * T1) - k * np.exp(-rd * T1) * n(d2)
                                    return Call


                                def BS_PUT1(s, k, T1, rd, rf, volatilite):
                                    d1 = (np.log(s / k) + ((rd - rf) + volatilite ** 2 / 2) * T1) / (
                                                volatilite * np.sqrt(T1))
                                    d2 = d1 - volatilite * np.sqrt(T1)
                                    Put = k * np.exp(-rd * T1) * n(-d2) - s * n(-d1) * np.exp(-rf * T1)
                                    return Put


                                def Ft_gt_k1(k):
                                    k1 = k[0]
                                    return Ft - k1 - 0.0000000000000000000000000000000000001


                                def K2_gt_Ft(k):
                                    k2 = k[1]
                                    return k2 - Ft - 0.0000000000000000000000000000000000001


                                def objective1(k):
                                    k1 = k[0]
                                    k2 = k[1]
                                    return ((BS_PUT(s, k1, T1, rd1, rf1, volatilite) - m * BS_CALL(s, k2, T1, rd1, rf1,
                                                                                                   volatilite))) ** 2


                                # k0 = [s, s]
                                k0 = [Ft, Ft]
                                con1 = {'type': 'ineq', 'fun': Ft_gt_k1}
                                con2 = {'type': 'ineq', 'fun': K2_gt_Ft}

                                con = [con1, con2]

                                optimize = sci_opt.minimize(objective1, k0, method='SLSQP', constraints=con,
                                                            options={'disp': True})

                                solution1 = optimize.x[0]
                                solution2 = optimize.x[1]

                                # x0 = s
                                #
                                # def constrainte(k):
                                #     return k - cours_terme(s, T1, rd1, rf1)
                                #
                                # def objective(k):
                                #     g = (N * (1 - Perc1) * ((k - s * np.exp((rd1 - rf1) * T1)) * np.exp(
                                #         -(rd1 - rf1) * T1)) + N * Perc1 * BS_PUT(k, T1, s, rd1, rf1, volatilite)) ** 2
                                #     return g
                                #
                                #
                                # cons = ({'type': 'ineq', 'fun': lambda x: x - s})
                                #
                                # optimize = sci_opt.minimize(objective, x0, method='SLSQP', constraints=cons,
                                #                             options={'disp': True})
                                #
                                # k1 = optimize.x
                                # st.markdown(k1)

                                # Ft = cours_terme(s, T1, rd1, rf1)
                                # k1 = find_strike(s, rd1, rf1, T1, volatilite)
                                #
                                # put = BS_PUT(s, k1, T1, rd1, rf1, volatilite)
                                # forward = (k1 - Ft) * np.exp(-r1 * T1)
                                # b = forward + put

                                st.markdown(
                                    "<h2 style='text-align: left ;color:black;font-size:20px;'>  Le strike est :  </h2>",
                                    unsafe_allow_html=True)

                                st.markdown(
                                    "<h2 style='text-align: left ;color:;font-size:15px;'>  - K1 = : " + str(
                                        format(solution1, '.4f')) + " </h2>",
                                    unsafe_allow_html=True)

                                st.markdown(
                                    "<h2 style='text-align: left ;color:;font-size:15px;'>  - K2 = : " + str(
                                        format(solution2, '.4f')) + " </h2>",
                                    unsafe_allow_html=True)

                            if st.button(f'Information sur opération '):
                                df = pd.DataFrame(
                                    [ticker, format(N, ".2f"), format(s, ".4f"), format(solution1, ".4f"),
                                     format(solution2, ".4f")],
                                    index=pd.Index(
                                        ["Paire de devise", "Nominal", "Spot", "Strike k1", "Strike k2"]))
                                tdf = df.T
                                # CSS to inject contained in a string
                                hide_table_row_index = """
                                                                                            <style>
                                                                                            tbody th {display:none}
                                                                                            .blank {display:none}
                                                                                            </style>
                                                                                            """

                                ######################### detail de strategie

                                # Inject CSS with Markdown
                                ############ table recap

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                    unsafe_allow_html=True)

                                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                                st.table(tdf)
                                st.session_state['button'] = False

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)

                                st.write(
                                    "-Spot " + " " + str(ticker) + "          " + ":" + " " + str(format(s, ".4f")))
                                st.write("-Volatilité:" + "   " + str(format(volatilite1, ".2f")) + "%")
                                st.write("-Taux doméstique:" + "   " + str(format(rd, ".2f")) + "%")
                                st.write("-Taux étranger:" + "   " + str(format(rf, ".2f")) + "%")

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)

                                st.write("- Si  " + "   " + str(ticker), " " + " < " + str(format(solution1, ".4f")))
                                st.write(" Vendre  " + str(N), " à " + " " + str(format(solution1, ".4f")))

                                st.write("- Si  " + "   " + str(format(solution1, ".4f")) + " < " + str(ticker),
                                         " " + " <" + str(format(solution2, ".4f")))
                                st.write(" Vendre  " + str(N), "au spot du jour")

                                st.write("- Si  " + "   " + str(ticker), " "" > " + str(format(solution2, ".4f")))
                                st.write(" Vendre  " + str(m * N), " à " + " " + str(format(solution2, ".4f")))

                        else:
                            rd1 = rd / 100
                            rf1 = rf / 100
                            sigma1 = sigma / 100
                            r1 = (rd1 - rf1)

                            T = (exercise_date - datetime.now().date()).days
                            T1 = T / 365
                            n = norm.cdf


                            def cours_terme(s, T1, rd, rf):
                                r1 = rd - rf
                                F_k = s * np.exp(r1 * T1)
                                return F_k


                            Ft = cours_terme(s, T1, rd1, rf1)


                            def BS_CALL1(s, k, T1, rd, rf, sigma1):
                                d1 = (np.log(s / k) + ((rd - rf) + sigma1 ** 2 / 2) * T1) / (sigma1 * np.sqrt(T1))
                                d2 = d1 - sigma1 * np.sqrt(T1)
                                Call = s * n(d1) * np.exp(-rf * T1) - k * np.exp(-rd * T1) * n(d2)
                                return Call


                            def BS_PUT1(s, k, T1, rd, rf, sigma1):

                                d1 = (np.log(s / k) + ((rd - rf) + sigma1 ** 2 / 2) * T1) / (sigma1 * np.sqrt(T1))
                                d2 = d1 - sigma1 * np.sqrt(T1)
                                Put = k * np.exp(-rd * T1) * n(-d2) - s * n(-d1) * np.exp(-rf * T1)
                                return Put


                            def Ft_gt_k1(k):
                                k1 = k[0]
                                return Ft - k1 - 0.0000000000000000000000000000000000001


                            def K2_gt_Ft(k):
                                k2 = k[1]
                                return k2 - Ft - 0.0000000000000000000000000000000000001


                            def objective1(k):
                                k1 = k[0]
                                k2 = k[1]
                                return ((BS_PUT(s, k1, T1, rd1, rf1, sigma1) - m * BS_CALL(s, k2, T1, rd1, rf1,
                                                                                           sigma1))) ** 2


                            k0 = [s, s]
                            # k0 = [Ft, Ft]

                            con1 = {'type': 'ineq', 'fun': Ft_gt_k1}
                            con2 = {'type': 'ineq', 'fun': K2_gt_Ft}

                            con = [con1, con2]

                            optimize = sci_opt.minimize(objective1, k0, method='SLSQP', constraints=con,
                                                        options={'disp': True})

                            solution3 = optimize.x[0]
                            solution4 = optimize.x[1]

                            # def cour_t(s, T, rd, rf):
                            #     F = s * np.exp((rd - rf) * T)
                            #     return F
                            #
                            #
                            # Ft = cour_t(s, T1, rd1, rf1)
                            # x0= s
                            #
                            #
                            # def constrainte(k):
                            #     return k - cour_t(s, T1, rd1, rf1)
                            #
                            #
                            # def objective(k):
                            #     g = (N * (1-Perc1) * ((k - s * np.exp((rd1 - rf1) * T1)) * np.exp( -(rd1 - rf1) * T1)) + N * Perc1 * BS_PUT(s, k, T1, rd1 , rf1, sigma1)) ** 2
                            #     return g
                            #
                            #
                            # cons = ({'type': 'ineq', 'fun': lambda x: x - s})
                            #
                            # optimize = sci_opt.minimize(objective, x0, method='SLSQP', constraints=cons, options={'disp': True})
                            #
                            # k2 =optimize.x
                            # st.markdown(k2)
                            # # Ft = cours_terme(s, T1, rd1, rf1)

                            # k2 = find_strike(s, rd1, rf1, T1, sigma1)
                            # st.markdown(Ft)
                            # st.markdown(k2)
                            # put = BS_PUT(s, k2, T1, rd1, rf1, sigma)
                            # forward = (k2 - Ft) * np.exp(-r1 * T1)
                            # b = forward + put

                            # col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(
                            #     [1, 60, 1, 1, 1, 1, 1, 1, 1])
                            # with col2:
                            #
                            #     st.markdown(
                            #         "<h2 style='text-align: left ;color:black;font-size:20px;'>    - Le Strike est : " + format(
                            #             k2 , ".3f")+ " </h2>",
                            #         unsafe_allow_html=True)

                            st.markdown(
                                "<h2 style='text-align: left ;color:black;font-size:20px;'>  Le strike est :  </h2>",
                                unsafe_allow_html=True)

                            st.markdown(
                                "<h2 style='text-align: left ;color:;font-size:15px;'>  - K1 = : " + str(
                                    format(solution3, '.4f')) + " </h2>",
                                unsafe_allow_html=True)

                            st.markdown(
                                "<h2 style='text-align: left ;color:;font-size:15px;'>  - K2 = : " + str(
                                    format(solution4, '.4f')) + " </h2>",
                                unsafe_allow_html=True)

                            if st.button(f'Information sur opération'):
                                df = pd.DataFrame(
                                    [ticker, format(N, ".2f"), format(s, ".4f"), format(solution3, ".4f"),
                                     format(solution4, ".4f")],
                                    index=pd.Index(["Paire de devise", "Nominal", "Spot", "Strike k1", "Strike k2"]))
                                tdf1 = df.T
                                # CSS to inject contained in a string
                                hide_table_row_index = """
                                                                                                                            <style>
                                                                                                                            tbody th {display:none}
                                                                                                                            .blank {display:none}
                                                                                                                            </style>
                                                                                                                            """

                                ######################### detail de strategie

                                # Inject CSS with Markdown
                                ############ table recap

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                    unsafe_allow_html=True)

                                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                                st.table(tdf1)
                                st.session_state['button'] = False

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)

                                st.write(
                                    "-Spot " + " " + str(ticker) + "          " + ":" + " " + str(format(s, ".4f")))
                                st.write("-Volatilité:" + "   " + str(format(sigma, ".2f")) + "%")
                                st.write("-Taux doméstique:" + "   " + str(format(rd, ".2f")) + "%")
                                st.write("-Taux étranger:" + "   " + str(format(rf, ".2f")) + "%")

                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)

                                st.write("- Si  " + "   " + str(ticker), " " + " < " + str(format(solution3, ".4f")))
                                st.write(" Vendre  " + str(N), " à " + " " + str(format(solution3, ".4f")))

                                st.write("- Si  " + "   " + str(format(solution3, ".4f")) + " < " + str(ticker),
                                         " " + " <" + str(format(solution4, ".4f")))
                                st.write(" Vendre  " + str(N), "au spot du jour")

                                st.write("- Si  " + "   " + str(ticker), " "" > " + str(format(solution4, ".4f")))
                                st.write(" Vendre  " + str(m * N), " à " + " " + str(format(solution4, ".4f")))

            #######################################################################################################################################################
            if choose == "p":

                st.header(" ")

                ticker = st.selectbox('Sélectionner votre paire de devise', ('EUR/USD', 'EUR/MAD', 'USD/MAD'))
                # st.markdown(f" Paire  : {ticker}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    N = st.number_input('Nominal')
                    # st.markdown(f"Nominal : {N}")
                with col2:
                    s = st.number_input('Spot')
                    # st.markdown(f"Spot  : {s}")
                with col3:
                    PERC = st.number_input('m')
                    # st.markdown(f" Paire  : {ticker}")

                col1, col2, col3 = st.columns(3)

                with col1:
                    k1 = st.number_input('Strike k1')
                    # st.markdown(f"Percentage : {k1} %")

                with col2:
                    k2 = st.number_input('Strike k2 ')
                    # .markdown(f"Strike  : {k2}")
                with col3:
                    exercise_date = st.date_input('Exercise date', min_value=datetime.today() + timedelta(days=1),
                                                  value=datetime.today() + timedelta(days=365))
                    # st.markdown(f" Date de valeur :{exercise_date}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    rd = st.slider('Taux doméstique (%)', 0, 100, 10)
                    # st.markdown(f"Taux doméstique : {r} %")

                with col2:
                    rf = st.slider('Taux étranger (%)', 0, 100, 10)
                    # st.markdown(f"Taux étranger : {r2} %")

                with col3:
                    sigma = st.slider('Sigma (%)', 0, 100, 20)
                    # st.markdown(f" Volatilitée : {sigma} %")

                genre = st.radio(
                    "Choisissez le type de la  stratégie ",
                    ('Stratégie payante', 'Stratégie gratuite'))

                if genre == 'Stratégie payante':

                    # gestion des exceptions
                    st.markdown("-------------")
                    if not all([s, k1, rd, rf, sigma, k2]):
                        st.error("Veuillez remplir tous les champs")
                        button = None
                    else:
                        button = st.button(f'Calculer la prime ')

                    if st.session_state.get('button') != True:
                        st.session_state['button'] = button

                    if st.session_state['button'] == True:

                        # Formating selected model parameters
                        rd1 = rd / 100
                        rf1 = rf / 100
                        sigma1 = sigma / 100
                        T = (exercise_date - datetime.now().date()).days
                        T1 = T / 365

                        # le calcul des primes
                        c = BS_CALL(s, k2, T1, rd1, rf1, sigma1)

                        p = BS_PUT(s, k1, T1, rd1, rf1, sigma1)
                        # c1 = 1/c
                        # p1=1/p
                        put = p * N
                        call = c * N
                        # forward = s - k * np.exp(-r1 * T1)
                        # t1 =c1-p1
                        tunnel = p - c
                        tunnel1 = put - call

                        # if Prime_partcipative < 0:
                        #     Prime_partcipative = 0

                        # tableau des prime en %
                        df = pd.DataFrame([format(p, ".3f"), format(c, ".3f"), format(tunnel, ".2f")],
                                          index=pd.Index(
                                              ["Put Vanille", " Call Vanille", "Tunnel Symétrique"]))

                        # tableau des primes en nominal
                        df1 = pd.DataFrame([format(put, ".3f"), format(call, ".3f"), format(tunnel1, ".2f")],
                                           index=pd.Index(
                                               ["Put Vanille", " Call Vanille", "Tunnel Symétrique"]))

                        ###############################"

                        tdf = df.T

                        # CSS to inject contained in a string
                        hide_table_row_index = """
                                                                    <style>
                                                                    tbody th {display:none}
                                                                    .blank {display:none}
                                                                    </style>
                                                                    """

                        # Inject CSS with Markdown

                        st.markdown(
                            "<h2 style='text-align: left ;color:#4a6da4;font-size:22px;'> - Tableau des primes en % </h2>",
                            unsafe_allow_html=True)
                        st.header("")
                        st.markdown(hide_table_row_index, unsafe_allow_html=True)
                        st.table(tdf)

                        #################################"

                        tdf1 = df1.T

                        # CSS to inject contained in a string
                        hide_table_row_index = """
                                                                                       <style>
                                                                                       tbody th {display:none}
                                                                                       .blank {display:none}
                                                                                       </style>
                                                                                       """

                        # Inject CSS with Markdown

                        st.markdown(
                            "<h2 style='text-align: left ;color:#4a6da4;font-size:22px;'> - Tableau des primes en nominal </h2>",
                            unsafe_allow_html=True)
                        st.header("")
                        st.markdown(hide_table_row_index, unsafe_allow_html=True)
                        st.table(tdf1)

                        # st.markdown(
                        # "<h2 style='text-align: left ;color:black;font-size:18px;'> Le prix du Put  :   </h2>",
                        # unsafe_allow_html=True)

                        # la prime du put avec le strike imposé
                        # put=BS_PUT(s, k, T1, r, sigma)
                        # st.markdown(f'   - Le prix du put est: {put}' )

                        # st.markdown(
                        # "<h2 style='text-align: left ;color:black;font-size:18px;'> Le prix du forward :   </h2>",
                        # unsafe_allow_html=True)

                        # la valeur du forward avec le strike imposé
                        # forward = s-k*np.exp(-r*T1)
                        # st.markdown(f'   - Le prix du forward  : {forward}')

                        # la prime payé

                        # Prime_partcipative =put + forward
                        # st.markdown(f'-Total Prime a payer: {Prime_partcipative}')

                        if st.button(f'Information sur opération '):
                            df = pd.DataFrame([ticker, format(N, ".2f"), format(s, ".4f"), format(k1, ".4f"),
                                               format(k2, ".4f"), format(put, ".2f"), format(call, ".2f"),
                                               format(tunnel1, ".2f")],
                                              index=pd.Index(
                                                  ["Paire de devise", "Nominal", "Spot", "Strike k1 ", "Strike k2",
                                                   "Put Vanille", "Call Vanille", " Tunnel Symétrique"]))
                            tdf = df.T
                            # CSS to inject contained in a string
                            hide_table_row_index = """
                                                    <style>
                                                    tbody th {display:none}
                                                    .blank {display:none}
                                                    </style>
                                                    """

                            ######################### detail de strategie
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.header(" ")
                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Détail de la strategie </h2>",
                                    unsafe_allow_html=True)
                                st.header("")

                                st.write("- Achat : Put Vanille")
                                st.write("- Nominal:" + "   " + str(N))
                                st.write("- Prix d'éxercice :" + "   " + str(k1))
                                st.write("- Date d'échéance:" + "   " + str(exercise_date))

                            st.header("  ")
                            with col3:
                                st.header(" ")
                                st.header(" ")
                                st.header(" ")
                                # st.markdown(
                                # "<h2 style='text-align: left ;color:green;font-size:14px;'> Dénouement a l'échéance </h2>",
                                # unsafe_allow_html=True)
                                st.header("")

                                st.write("- Vente : Call Vanille  ")
                                st.write("- Nominal:" + "   " + str(N))
                                st.write("- Prix d'éxercice :" + "   " + str(k2))
                                st.write("- Date d'échéance:" + "   " + str(exercise_date))

                            # Inject CSS with Markdown
                            ############ table recap

                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                unsafe_allow_html=True)
                            st.header("")
                            st.markdown(hide_table_row_index, unsafe_allow_html=True)
                            st.table(tdf)
                            st.session_state['button'] = False

                            ######### detail

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.header(" ")
                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)
                                st.header("")

                                st.write("-Spot " + " " + str(ticker) + "          " + ":" + " " + str(s))
                                st.write("-Volatilité:" + "   " + str(sigma) + "%")
                                st.write("-Taux doméstique:" + "   " + str(rd) + "%")
                                st.write("-Taux étranger:" + "   " + str(rf) + "%")
                                # st.write("-Volatilité:" + "   " + str(sigma))
                            st.header("  ")
                            with col3:
                                st.header(" ")
                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement a l'échéance </h2>",
                                    unsafe_allow_html=True)
                                st.header("")

                                st.write("- si  " + "   " + str(ticker), " " + " < " + str(k1))
                                st.write(" Vous vendez  " + str(N), " à " + " " + str(k1))

                                st.write("- si  " + "   " + str(k1) + " < " + str(ticker), " " + " <" + str(s))
                                st.write(" Vous vendez  " + str(N), "au spot du jour")

                                st.write("- si  " + "   " + str(ticker), " "" > " + str(k2))
                                st.write(" Vous vendez  " + str(N), " à " + " " + str(k2))

                else:

                    # gestion des exceptions
                    st.markdown("-------------")
                    if not all([s, k1, k2, rd, rf, sigma]):
                        st.error(" Veuillez remplir tous les champs ")
                        button = None
                    else:
                        button = st.button(f'Calculer le strike amélioré ')

                    if st.session_state.get('button') != True:
                        st.session_state['button'] = button

                    if st.session_state['button'] == True:

                        # Formating selected model parameters
                        rd1 = rd / 100
                        # st.markdown(rd1)
                        rf1 = rf / 100
                        # st.markdown(rf1)
                        r1 = (rd1 - rf1)
                        sigma1 = sigma / 100
                        # st.markdown(sigma1)
                        T = (exercise_date - datetime.now().date()).days
                        T1 = T / 365


                        ######################### optimisation
                        def constrainte(k):
                            k1 = k[0]
                            k2 = k[1]
                            return k2 - k1 - 0.0000000000000000000000000000000000001


                        def constrainte_2(k):
                            k1 = k[0]
                            k2 = k[1]
                            return k2 - s - 0.0000000000000000000000000000000000001


                        def constrainte_3(k):
                            k1 = k[0]
                            k2 = k[1]
                            return s - k1 - 0.0000000000000000000000000000000000001


                        def objective1(k):
                            k1 = k[0]
                            k2 = k[1]
                            return ((BS_PUT(s, k2, T1, rd1, rf1, sigma1)) - BS_CALL(s, k1, T1, rd1, rf1, sigma1)) ** 2


                        k0 = [s - 0.4, s - 0.2]
                        cons = {'type': 'ineq', 'fun': constrainte}
                        con2 = {'type': 'ineq', 'fun': constrainte_2}
                        con3 = {'type': 'ineq', 'fun': constrainte_3}
                        cons1 = [cons, con2, con3]

                        optimize = sci_opt.minimize(objective1, k0, method='SLSQP', constraints=cons1,
                                                    options={'disp': True})

                        call2 = BS_CALL(s, optimize.x[1], T1, rd1, rf1, sigma1)
                        put2 = BS_PUT(s, optimize.x[0], T1, rd1, rf1, sigma1)

                        c = optimize.x[1]
                        p = optimize.x[0]
                        tunnel = call2 - put2
                        # j = (put / 100) * (Perc1 * N)

                        st.markdown(
                            "<h2 style='text-align: left ;color:black;font-size:20px;'>  Le strike amélioré est :  </h2>",
                            unsafe_allow_html=True)

                        st.markdown(
                            "<h2 style='text-align: left ;color:;font-size:15px;'>  - K1 = : " + str(
                                format(p, '.4f')) + " </h2>",
                            unsafe_allow_html=True)

                        st.markdown(
                            "<h2 style='text-align: left ;color:;font-size:15px;'>  - K2 = : " + str(
                                format(c, '.4f')) + " </h2>",
                            unsafe_allow_html=True)

                        if st.button(f'Informtion sur opération '):
                            df = pd.DataFrame(
                                [ticker, format(N, ".2f"), format(s, ".4f"), format(p, ".4f"), format(c, ".4f")
                                    , format(call2, ".3f"), format(put2, ".3f"),
                                 format(tunnel, ".2f")],
                                index=pd.Index(
                                    ["Paire de devise", "Nominal", "Spot", "Strike k1", "Strike k2",
                                     "Call Vanille", "Put vanille", "Prime"]))

                            tdf = df.T
                            # CSS to inject contained in a string
                            hide_table_row_index = """
                                                        <style>
                                                        tbody th {display:none}
                                                        .blank {display:none}
                                                        </style>
                                                        """
                            # df1 = pd.DataFrame(
                            #     [ticker, format(N, ".2f"), format(s, ".2f"), format(k, ".2f"), format(Perc, ".2f"),
                            #      format(p, ".2f"), format(f, ".2f"), format(p-p, ".2f")],
                            #     index=pd.Index(["Paire de devise", "Nominal", "Spot", "Strike amélioré", "Participation",
                            #                     "Put Vanille", "Forward", "Prime"]))
                            #
                            # tdf1 = df1.T
                            # # CSS to inject contained in a string
                            # hide_table_row_index = """
                            #                                     <style>
                            #                                     tbody th {display:none}
                            #                                     .blank {display:none}
                            #                                     </style>
                            #                                     """

                            ######################### detail de strategie
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.header(" ")
                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Détail de la stratégie </h2>",
                                    unsafe_allow_html=True)
                                st.header("")

                                st.write("- Achat : Put ")
                                st.write("- Nominal:" + "   " + str(N))
                                st.write("- Prix d'éxercice k1 :" + "   " + str(format(k1, ".4f")))
                                st.write("- Date d'échéance:" + "   " + str(exercise_date))

                            st.header("  ")
                            with col3:
                                st.header(" ")
                                st.header(" ")
                                # st.markdown(
                                # "<h2 style='text-align: left ;color:green;font-size:14px;'> Dénouement a l'échéance </h2>",
                                # unsafe_allow_html=True)
                                st.header("")
                                st.header("")
                                st.write("- Vente : Call ")
                                st.write("- Nominal:" + "   " + str(N))
                                st.write("- Prix d'éxercice k2  :" + "   " + str(format(k2, ".4f")))
                                st.write("- Date d'échéance:" + "   " + str(exercise_date))

                            # Inject CSS with Markdown
                            ############ table recap

                            st.markdown(
                                "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Résultat </h2>",
                                unsafe_allow_html=True)
                            st.header("")
                            st.markdown(hide_table_row_index, unsafe_allow_html=True)
                            st.table(tdf)
                            st.session_state['button'] = False

                            ######### detail

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.header(" ")
                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Données de marché </h2>",
                                    unsafe_allow_html=True)
                                st.header("")

                                st.write("-Spot " + " " + str(ticker) + "          " + ":" + " " + str(s))
                                st.write("-Volatilité:" + "   " + str(sigma) + "%")
                                st.write("-Taux doméstique:" + "   " + str(rd) + "%")
                                st.write("-Taux étranger:" + "   " + str(rf) + "%")

                            st.header("  ")
                            with col3:
                                st.header(" ")
                                st.markdown(
                                    "<h2 style='text-align: left ;color:#6cb44c;font-size:28px;'> Dénouement à l'échéance </h2>",
                                    unsafe_allow_html=True)
                                st.header("")

                                st.write("- si  " + "   " + str(ticker), " " + " < " + str(format(p, ".4f")))
                                st.write(" Vous vendez  " + str(N), " à " + " " + str(format(p, ".4f")))

                                st.write("- si  " + "   " + str(format(p, ".4f")) + " < " + str(ticker),
                                         " " + " <" + str(format(c, ".4f")))
                                st.write(" Vous vendez  " + str(N), "au spot du jour")

                                st.write("- si  " + "   " + str(ticker), " "" > " + str(format(c, ".4f")))
                                st.write(" Vous vendez  " + str(N), " à " + " " + str(format(c, ".4f")))

    if menu_id == 'contact':
        st.markdown('- Tél : +212 5 22 23 76 02')
        st.markdown('- Fax : +212 5 22 36 87 84')
        st.markdown('- Email : info@atlascapital.ma')
        st.markdown('- 88 rue El Marrakchi, Quartier Hippodrome, 20100 Casablanca Maroc')

        st.header("  ")
        m = folium.Map(location=[33.590548, -7.652014], zoom_start=15)
        folium.Marker(location=[33.590548, -7.652014], popup='Rue Al Morrakchi').add_to(m)
        st_data = st_folium(m, width=1000)

    if menu_id == 'Déconnexion':
        col1, col2, col3, col4 = st.columns([6, 1, 1, 1])
        with col1:
            authenticator.logout("Déconnexion", "main")

        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
        with col1:
            image = Image.open('cet.png')
            st.image(image, width=1000)















