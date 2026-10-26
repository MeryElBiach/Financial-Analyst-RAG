"""Paramètres de l'étape d'analyse (LLM, recherche, requêtes par type d'analyse)."""
import os
from dotenv import load_dotenv
load_dotenv()

# --- LLM -------------------------------------------------------------------
# Modèle Gemini. À changer dans .env (GEMINI_MODEL=...) sans toucher au code.
# Les modèles "gemini-2.5-*" sont classés "legacy" par Google : on prend un modèle récent.
# Si l'offre gratuite refuse "gemini-3.8-flash", essayer "gemini-3.5-flash-lite".
MODELE_LLM = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# --- Citations ---------------------------------------------------------------
# Dans Qdrant, la page 0 est la couverture (numérotation FinanceBench).
# Les citations affichées utilisent la numérotation d'un lecteur PDF : page + 1.
# Mettre 0 pour afficher la numérotation FinanceBench.
DECALAGE_PAGE = 1

# --- Recherche ---------------------------------------------------------------
K_PAR_REQUETE = 4  # nombre de passages gardés pour chaque requête (analyses)
K_QUESTION = 8  # nombre de passages gardés pour une question libre

# Valeurs du champ metadata.doc_type qui désignent un rapport annuel (voir ingestion.py).
TYPES_ANNUELS = ["10k", "10k_annualreport"]

# Un rapport de l'année N contient aussi les chiffres de N-1 (colonnes de comparaison).
# Un rapport de N+1 contient ceux de N et N-1 : on le cherche aussi pour les ratios.
ANNEES_SUPPLEMENTAIRES = {"rentabilite": 1, "croissance": 1, "swot": 0}

# Requêtes en anglais, car les rapports FinanceBench sont en anglais.
REQUETES_FINANCES = [
    "consolidated statement of income net sales revenues cost of sales operating income net income",
    "gross profit gross margin operating margin net income attributable to shareholders",
    "consolidated balance sheet total equity shareholders equity",
    "results of operations net sales percent change compared to prior year",
]

REQUETES = {
    "rentabilite": REQUETES_FINANCES,
    "croissance": REQUETES_FINANCES,
    "swot": [
        "business overview segments products customers competitive position strengths",
        "risk factors competition regulation litigation supply chain",
        "growth strategy opportunities new products markets investment",
        "challenges declining sales margin pressure costs debt",
        "restructuring charges impairment decrease in operating income segment decline",
        "legal proceedings litigation liabilities accrued product liability environmental",
    ],
}

TYPES_ANALYSE = list(REQUETES)