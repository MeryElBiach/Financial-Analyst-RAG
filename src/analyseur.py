"""Analyse d'une entreprise à partir des rapports annuels indexés dans Qdrant.

Trois types d'analyse : rentabilité, croissance, matrice SWOT.

Principe pour la rentabilité et la croissance (le LLM ne calcule jamais) :
  1. recherche des passages utiles dans Qdrant (filtrée sur l'entreprise et l'année) ;
  2. le LLM EXTRAIT les montants, avec leur document et leur page ;
  3. Python vérifie que chaque montant s'écrit bien dans la page citée, puis CALCULE les ratios ;
  4. le LLM rédige un commentaire à partir des chiffres et des ratios.

Utilisation :
    python src/analyseur.py 3M 2018                 (analyse complète : les trois volets)
    python src/analyseur.py 3M 2018 rentabilite
    python src/analyseur.py 3M 2018 croissance
    python src/analyseur.py 3M 2018 swot
    python src/analyseur.py 3M 2018 --json
"""
import argparse
from analyse_config import ANNEES_SUPPLEMENTAIRES, REQUETES, TYPES_ANALYSE
from llm import charger_llm
from prompts import PROMPT_COMMENTAIRE, PROMPT_EXTRACTION, PROMPT_SWOT
from ratios import croissance, marge, marge_brute, roe
from recherche import (
    annees_indexees, chercher, citation_existe, formater_passages, ouvrir_base, source,
    trouver_entreprise, valeur_dans_passages,
)

from schemas import Extraction, LigneChiffre, LigneRatio, ResultatAnalyse, ResultatComplet, Swot

TITRES = {"rentabilite": "rentabilité", "croissance": "croissance", "swot": "matrice SWOT"}

CHAMPS = {
    "chiffre_affaires": "Chiffre d'affaires",
    "cout_des_ventes": "Coût des ventes",
    "resultat_operationnel": "Résultat opérationnel",
    "resultat_net": "Résultat net",
    "capitaux_propres": "Capitaux propres",
}
CHAMPS_PAR_ANALYSE = {
    "rentabilite": list(CHAMPS),
    "croissance": ["chiffre_affaires", "resultat_operationnel", "resultat_net"],
}


# --- Outils de mise en forme ---------------------------------------------------------
def _texte(reponse):
    """Texte d'une réponse du LLM (le contenu peut être une chaîne ou une liste de parties)."""
    contenu = getattr(reponse, "content", reponse)
    if isinstance(contenu, str):
        return contenu.strip()
    if isinstance(contenu, list):
        return "".join(p if isinstance(p, str) else p.get("text", "") for p in contenu).strip()
    return str(contenu)


def _valider(reponse, schema):
    """Le LLM peut renvoyer un objet du bon type, un dictionnaire ou rien."""
    if isinstance(reponse, schema):
        return reponse
    if isinstance(reponse, dict):
        return schema.model_validate(reponse)
    raise RuntimeError("Le LLM n'a pas renvoyé de données exploitables. Relancer l'analyse.")


def _fmt(valeur, pourcentage=False):
    """Montant (32,765) ou pourcentage (16.33 %) ; 'non disponible' si la valeur manque."""
    if valeur is None:
        return "non disponible"
    if pourcentage:
        return f"{valeur:.2f} %"
    return f"{int(valeur):,}" if float(valeur).is_integer() else f"{valeur:,.2f}"


# --- Rentabilité et croissance -----------------------------------------------------------
def _ratios_rentabilite(exercices):
    definitions = [
        ("Marge brute", "(CA - coût des ventes) / CA",
         lambda c: marge_brute(c.chiffre_affaires.valeur, c.cout_des_ventes.valeur)),
        ("Marge opérationnelle", "résultat opérationnel / CA",
         lambda c: marge(c.resultat_operationnel.valeur, c.chiffre_affaires.valeur)),
        ("Marge nette", "résultat net / CA",
         lambda c: marge(c.resultat_net.valeur, c.chiffre_affaires.valeur)),
        ("ROE", "résultat net / capitaux propres de fin d'exercice",
         lambda c: roe(c.resultat_net.valeur, c.capitaux_propres.valeur)),
    ]
    return [
        LigneRatio(nom=nom, annee=annee, valeur=calcul(chiffres), formule=formule)
        for nom, formule, calcul in definitions
        for chiffres, annee in exercices
    ]


def _ratios_croissance(annee, courante, precedente):
    definitions = [
        ("Croissance du chiffre d'affaires", "chiffre_affaires"),
        ("Croissance du résultat opérationnel", "resultat_operationnel"),
        ("Croissance du résultat net", "resultat_net"),
    ]
    return [
        LigneRatio(
            nom=nom, annee=annee, formule="(N - N-1) / |N-1|",
            valeur=croissance(getattr(courante, champ).valeur, getattr(precedente, champ).valeur),
        )
        for nom, champ in definitions
    ]


def _extraire(entreprise, annee, passages, llm):
    """Le LLM lit les passages et renvoie les chiffres des deux exercices (sans rien calculer)."""
    messages = PROMPT_EXTRACTION.format_messages(
        entreprise=entreprise, annee=annee, annee_precedente=annee - 1, passages=passages
    )
    return _valider(llm.with_structured_output(Extraction).invoke(messages), Extraction)


def _analyse_chiffree(resultat, annee, docs, passages, llm, extraction=None):
    if extraction is None:
        extraction = _extraire(resultat.entreprise, annee, passages, llm)
    resultat.unite_montants = extraction.unite
    courante, precedente = extraction.annee_courante, extraction.annee_precedente
    exercices = [(courante, annee), (precedente, annee - 1)]

    # Chiffres extraits, avec vérification dans la page citée
    for champ in CHAMPS_PAR_ANALYSE[resultat.type_analyse]:
        for chiffres, a in exercices:
            mesure = getattr(chiffres, champ)
            verifiee = valeur_dans_passages(mesure.valeur, mesure.doc_name, mesure.page, docs)
            cite = f"{mesure.doc_name}, p. {mesure.page}" if mesure.doc_name and mesure.page else None
            resultat.chiffres.append(LigneChiffre(
                nom=CHAMPS[champ], annee=a, valeur=mesure.valeur, source=cite, verifiee=verifiee
            ))
            if mesure.valeur is None:
                resultat.avertissements.append(f"Donnée introuvable : {CHAMPS[champ]} {a}.")
            elif not verifiee:
                resultat.avertissements.append(
                    f"Valeur non retrouvée dans la page citée, à vérifier : {CHAMPS[champ]} {a}."
                )

    # Ratios calculés par Python
    if resultat.type_analyse == "rentabilite":
        resultat.ratios = _ratios_rentabilite(exercices)
    else:
        resultat.ratios = _ratios_croissance(annee, courante, precedente)

    # Commentaire rédigé par le LLM, seulement s'il existe au moins un ratio
    if all(r.valeur is None for r in resultat.ratios):
        resultat.commentaire = (
            "Analyse impossible : les chiffres nécessaires n'ont pas été trouvés dans les passages."
        )
        return
    messages = PROMPT_COMMENTAIRE.format_messages(
        sujet=TITRES[resultat.type_analyse], entreprise=resultat.entreprise, annee=annee,
        unite=extraction.unite or "non précisée",
        chiffres="\n".join(f"- {c.nom} {c.annee} : {_fmt(c.valeur)}" for c in resultat.chiffres),
        ratios="\n".join(f"- {r.nom} {r.annee} : {_fmt(r.valeur, True)}" for r in resultat.ratios),
    )
    resultat.commentaire = _texte(llm.invoke(messages))


# --- SWOT ---------------------------------------------------------------------------------
def _analyse_swot(resultat, annee, docs, passages, llm):
    messages = PROMPT_SWOT.format_messages(
        entreprise=resultat.entreprise, annee=annee, passages=passages
    )
    swot = _valider(llm.with_structured_output(Swot).invoke(messages), Swot)
    for categorie in ("forces", "faiblesses", "opportunites", "menaces"):
        for element in getattr(swot, categorie):
            if not (element.doc_name and element.page
                    and citation_existe(element.doc_name, element.page, docs)):
                resultat.avertissements.append(
                    f"Citation non retrouvée ({categorie}) : « {element.texte[:60]} »"
                )
    resultat.swot = swot
    resultat.commentaire = swot.conclusion


# --- Point d'entrée ---------------------------------------------------------------------------
def analyser(entreprise, annee, type_analyse, base=None, llm=None):
    """Analyse une entreprise pour un exercice. Renvoie un ResultatAnalyse.
    'base' (Qdrant) et 'llm' peuvent être fournis pour éviter de les recharger à chaque appel."""
    if type_analyse not in TYPES_ANALYSE:
        raise ValueError(f"Type d'analyse inconnu : {type_analyse}. Choix : {', '.join(TYPES_ANALYSE)}")
    nom = trouver_entreprise(entreprise)
    base = base if base is not None else ouvrir_base()
    llm = llm if llm is not None else charger_llm()

    # Quels rapports annuels de cette entreprise sont réellement dans Qdrant ?
    voulues = [annee + i for i in range(ANNEES_SUPPLEMENTAIRES[type_analyse] + 1)]
    annees = annees_indexees(base, nom, voulues)
    if not annees:
        disponibles = annees_indexees(base, nom, range(2008, 2027))
        raise ValueError(
            f"Aucun rapport annuel de {nom} indexé pour {' ou '.join(map(str, voulues))}. "
            f"Années indexées : {', '.join(map(str, disponibles)) or 'aucune'}."
        )

    docs = chercher(base, nom, annees, REQUETES[type_analyse])
    if not docs:
        raise ValueError(f"Aucun passage trouvé dans Qdrant pour {nom} ({annees}). L'indexation est-elle terminée ?")
    passages = formater_passages(docs)

    resultat = ResultatAnalyse(
        type_analyse=type_analyse, entreprise=nom, annee=annee,
        sources=sorted({source(d) for d in docs}),
    )
    if type_analyse == "swot":
        _analyse_swot(resultat, annee, docs, passages, llm)
    else:
        _analyse_chiffree(resultat, annee, docs, passages, llm)
    return resultat


def analyser_complet(entreprise, annee, base=None, llm=None):
    """Analyse complète : rentabilité, croissance et SWOT en un seul rapport.
    Les chiffres de la rentabilité et de la croissance viennent des mêmes passages : ils sont
    cherchés et extraits une seule fois. Si un volet échoue, les autres sont quand même rendus
    et l'échec est signalé dans les avertissements."""
    nom = trouver_entreprise(entreprise)
    base = base if base is not None else ouvrir_base()
    llm = llm if llm is not None else charger_llm()
    rapport = ResultatComplet(entreprise=nom, annee=annee)

    voulues = [annee + i for i in range(ANNEES_SUPPLEMENTAIRES["rentabilite"] + 1)]
    annees_chiffres = annees_indexees(base, nom, voulues)
    annees_swot = annees_indexees(base, nom, [annee])
    if not annees_chiffres and not annees_swot:
        disponibles = annees_indexees(base, nom, range(2008, 2027))
        raise ValueError(
            f"Aucun rapport annuel de {nom} indexé pour {' ou '.join(map(str, voulues))}. "
            f"Années indexées : {', '.join(map(str, disponibles)) or 'aucune'}."
        )

    # Volets chiffrés : une recherche et une extraction pour les deux
    if annees_chiffres:
        try:
            docs = chercher(base, nom, annees_chiffres, REQUETES["rentabilite"])
            passages = formater_passages(docs)
            sources = sorted({source(d) for d in docs})
            extraction = _extraire(nom, annee, passages, llm)
            for typ in ("rentabilite", "croissance"):
                try:
                    resultat = ResultatAnalyse(type_analyse=typ, entreprise=nom, annee=annee, sources=sources)
                    _analyse_chiffree(resultat, annee, docs, passages, llm, extraction)
                    setattr(rapport, typ, resultat)
                except Exception as erreur:
                    rapport.avertissements.append(f"Volet {TITRES[typ]} impossible : {erreur}")
        except Exception as erreur:
            rapport.avertissements.append(f"Volets rentabilité et croissance impossibles : {erreur}")
    else:
        rapport.avertissements.append(
            f"Rentabilité et croissance : aucun rapport annuel de {nom} indexé pour "
            f"{' ou '.join(map(str, voulues))}."
        )

    # SWOT
    if annees_swot:
        try:
            docs = chercher(base, nom, annees_swot, REQUETES["swot"])
            resultat = ResultatAnalyse(
                type_analyse="swot", entreprise=nom, annee=annee,
                sources=sorted({source(d) for d in docs}),
            )
            _analyse_swot(resultat, annee, docs, formater_passages(docs), llm)
            rapport.swot = resultat
        except Exception as erreur:
            rapport.avertissements.append(f"Volet SWOT impossible : {erreur}")
    else:
        rapport.avertissements.append(f"SWOT : aucun rapport annuel de {nom} indexé pour {annee}.")

    if not (rapport.rentabilite or rapport.croissance or rapport.swot):
        raise RuntimeError("L'analyse a échoué : " + " | ".join(rapport.avertissements))
    return rapport


def en_texte_complet(rapport):
    """Présentation lisible du rapport complet dans le terminal."""
    morceaux = [
        f"##### ANALYSE COMPLÈTE : {rapport.entreprise}, exercice {rapport.annee} #####"
    ]
    for volet in (rapport.rentabilite, rapport.croissance, rapport.swot):
        if volet:
            morceaux.append(en_texte(volet))
    if rapport.avertissements:
        morceaux.append("Avertissements généraux\n" + "\n".join(f"  ! {a}" for a in rapport.avertissements))
    return "\n\n".join(morceaux)


def en_texte(resultat):
    """Présentation lisible du résultat dans le terminal."""
    lignes = [f"=== Analyse ({TITRES[resultat.type_analyse]}) : {resultat.entreprise}, {resultat.annee} ==="]
    if resultat.unite_montants:
        lignes.append(f"Unité des montants : {resultat.unite_montants}")
    if resultat.chiffres:
        lignes += ["", "Chiffres"]
        for c in resultat.chiffres:
            etat = "vérifié" if c.verifiee else "à vérifier"
            lignes.append(f"  {c.annee}  {c.nom} : {_fmt(c.valeur)}  ({c.source or 'source absente'}) [{etat}]")
    if resultat.ratios:
        lignes += ["", "Ratios (calculés par Python)"]
        for r in resultat.ratios:
            lignes.append(f"  {r.annee}  {r.nom} : {_fmt(r.valeur, True)}   <- {r.formule}")
    if resultat.swot:
        for titre, elements in [("Forces", resultat.swot.forces), ("Faiblesses", resultat.swot.faiblesses),
                                ("Opportunités", resultat.swot.opportunites), ("Menaces", resultat.swot.menaces)]:
            lignes += ["", titre]
            lignes += [f"  - {e.texte} ({e.doc_name}, p. {e.page})" for e in elements] or ["  (aucun élément étayé)"]
    lignes += ["", "Commentaire" if resultat.swot is None else "Conclusion", resultat.commentaire]
    if resultat.avertissements:
        lignes += ["", "Avertissements"] + [f"  ! {a}" for a in resultat.avertissements]
    lignes += ["", f"Passages utilisés : {' ; '.join(resultat.sources)}"]
    return "\n".join(lignes)


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="Analyse financière d'une entreprise FinanceBench")
    parseur.add_argument("entreprise", help="par exemple : 3M, Apple, Amazon")
    parseur.add_argument("annee", type=int, help="exercice analysé, par exemple 2018")
    parseur.add_argument(
        "type_analyse", nargs="?", default="complet", choices=TYPES_ANALYSE + ["complet"],
        help="complet (par défaut) = rentabilité + croissance + SWOT",
    )
    parseur.add_argument("--json", action="store_true", help="affiche le résultat en JSON")
    args = parseur.parse_args()

    print("Chargement des modèles et connexion à Qdrant...")
    if args.type_analyse == "complet":
        resultat = analyser_complet(args.entreprise, args.annee)
        print(resultat.model_dump_json(indent=2) if args.json else en_texte_complet(resultat))
    else:
        resultat = analyser(args.entreprise, args.annee, args.type_analyse)
        print(resultat.model_dump_json(indent=2) if args.json else en_texte(resultat))