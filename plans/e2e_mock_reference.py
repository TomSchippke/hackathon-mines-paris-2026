"""Test de bout en bout sur données synthétiques, LLM et embeddings simulés.

Lancé depuis un dossier de travail vide (chemins relatifs : context/, outputs/), avec le repo dans PYTHONPATH.
"""
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.update(AZURE_OPENAI_CHAT_DEPLOYMENT="mock", AZURE_OPENAI_EMBEDDING_DEPLOYMENT="mock",
                  AZURE_OPENAI_API_KEY="x", AZURE_OPENAI_ENDPOINT="x", AZURE_OPENAI_API_VERSION="x")

# --- données synthétiques ---------------------------------------------------------------------
CAUSES = {
    "alpha": {"surchauffe": ("la surchauffe du boîtier par forte chaleur ambiante pendant la journée", "dégât mineur"),
              "rongeurs": ("des rongeurs ont attaqué les gaines du câblage sous les rangées", "perte d'activité"),
              "salissure": ("une accumulation de salissure sur les panneaux réduit le rendement attendu", "perte d'activité"),
              "orage": ("un orage violent a provoqué une surtension sur les armoires électriques", "dégât majeur"),
              "usure": ("une usure avancée des connecteurs après des années de service intensif", "dégât mineur")},
    "beta": {"rafales": ("des rafales très fortes ont dépassé le seuil et déclenché la mise en sécurité", "arrêt sûr"),
             "foudroiement": ("un foudroiement direct a frappé le mât et endommagé les capteurs", "dégât majeur"),
             "graissage": ("un défaut de graissage du roulement principal non suivi par le plan", "dégât majeur"),
             "fissure": ("une fissure du socle béton due au tassement du terrain meuble", "dégât mineur"),
             "givre": ("du givre accumulé sur le rotor a déséquilibré la machine en hiver", "perte d'activité")},
}
SITES = {"alpha": ["Site A-1", "Site A-2", "Site A-3"], "beta": ["Site B-1", "Site B-2", "Site B-3"]}
rng = random.Random(0)
rows, iid = [], 0
for seg, causes in CAUSES.items():
    names = list(causes)
    for k, site in enumerate(SITES[seg]):
        weights = [3 if i % 3 == k else 1 for i in range(len(names))]
        for _ in range(110):
            iid += 1
            c = rng.choices(names, weights)[0]
            unknown = rng.random() < 0.1
            text = (f"Intervention sur équipement de production du site. {'Origine non identifiée, arrêt constaté par le technicien de garde.' if unknown else causes[c][0].capitalize() + '.'} "
                    f"Le technicien a sécurisé la zone et rédigé le rapport complet de production.")
            rows.append({"incident_id": iid, "site": site, "asset_type": seg, "equipment_id": f"EQ{iid}",
                         "date": f"202{rng.randint(1, 4)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                         "severity": "faible" if causes[c][1] == "arrêt sûr" else rng.choice(["faible", "moyenne", "grave"]),
                         "downtime_hours": round(rng.uniform(1, 40), 1),
                         "estimated_cost_eur": round(rng.uniform(500, 20000) * (3 if causes[c][1] == "dégât majeur" else 1), 2),
                         "weather_condition": rng.choice(["clair", "pluie", "vent"]), "technician_id": f"T{k}",
                         "text": text, "_cause": None if unknown else c, "_seg": seg})
df = pd.DataFrame(rows)
truth = df.drop_duplicates("text").set_index("text")[["_cause", "_seg"]].to_dict("index")
Path("outputs/1_extraction").mkdir(parents=True, exist_ok=True)
df.drop(columns=["_cause", "_seg"]).to_csv("outputs/1_extraction/incidents_full.csv", index=False)
Path("context").mkdir(exist_ok=True)
Path("context/contexte.md").write_text("Le client exploite 6 sites : 3 de type alpha et 3 de type beta.", encoding="utf-8")

# --- LLM simulé ---------------------------------------------------------------------------------
FACTOR_OF = {"surchauffe": "chaleur", "rongeurs": "animaux", "salissure": "encrassement", "orage": "orage",
             "usure": "usure des pièces", "rafales": "vent fort", "foudroiement": "orage", "graissage": "défaut de maintenance",
             "fissure": "sol / fondations", "givre": "givre"}
CADRAGE = {
    "domaine": "Le client exploite des sites de type alpha et beta.",
    "segments": [{"valeur": "alpha", "libelle": "installations alpha"}, {"valeur": "beta", "libelle": "installations bêta"}],
    "facteurs": [{"nom": n, "mots_cles": kw, "type": "programme d'inspection",
                  "mesure": f"Inspecter chaque trimestre les équipements exposés au facteur {n} et corriger les écarts",
                  "mecanisme": f"le facteur {n} est repéré et traité avant de provoquer une panne"}
                 for n, kw in [("chaleur", ["chaleur", "surchauff"]), ("animaux", ["rongeur", "animaux"]),
                               ("encrassement", ["salissure", "encrass"]), ("orage", ["orage", "foudr"]),
                               ("usure des pièces", ["usure"]), ("vent fort", ["rafale", "vent fort"]),
                               ("défaut de maintenance", ["graissage"]), ("sol / fondations", ["fissure", "tassement"]),
                               ("givre", ["givre"])]],
    "composants": ["boîtier", "câblage", "panneau", "armoire", "rotor", "mât", "socle"],
    "categories_consequence": [{"nom": n, "fonctionnement_normal": n == "arrêt de sécurité"} for n in
                               ["perte d'activité", "dégât mineur", "dégât majeur", "arrêt de sécurité", "aucun impact"]],
    "severites": ["faible", "moyenne", "grave"],
    "profils": [{"segment": "beta", "id": p, "description": d, "attentes": a} for p, d, a in [
        ("X", "site très venté", [{"critere": "vent fort", "sens": "élevé", "raison": "très venté"}]),
        ("Y", "site exposé aux orages", [{"critere": "orage", "sens": "élevé", "raison": "orages"},
                                         {"critere": "inconnu", "sens": "élevé", "raison": "test d'attente invalide"}]),
        ("Z", "terrain meuble", [{"critere": "sol / fondations", "sens": "élevé", "raison": "sol meuble"}])]],
}
CONS = {"dégât mineur": "dégât mineur", "dégât majeur": "dégât majeur", "perte d'activité": "perte d'activité",
        "arrêt sûr": "arrêt de sécurité"}


def dummy(schema):
    t = schema.get("type")
    if "enum" in schema:
        return schema["enum"][0]
    if t == "object":
        return {k: dummy(v) for k, v in schema["properties"].items()}
    if t == "array":
        return []
    if t == "boolean":
        return False
    return "Texte de rédaction simulée sans chiffre."


calls = {"cadrage": 0, "extraction": 0, "redaction": 0, "jugement": 0, "agent": 0}


def mock_chat(messages, *, role, tools=None, response_format=None, temperature=0.0):
    if tools:
        calls["agent"] += 1
        return {"content": "Je réfléchis.", "tool_calls": []}
    name = response_format["json_schema"]["name"]
    if name == "cadrage":
        calls["cadrage"] += 1
        assert "alpha" in messages[1]["content"] and "Descriptions d'incidents" in messages[1]["content"]
        return {"content": json.dumps(CADRAGE, ensure_ascii=False), "tool_calls": []}
    if name == "analyse_incident":
        calls["extraction"] += 1
        user = messages[1]["content"]
        assert "Site A" not in user and "Site B" not in user and "EQ" not in user, "site ou identifiant montré au LLM"
        text = user.split("Description : ", 1)[1]
        c = truth[text]["_cause"]
        seg_causes = CAUSES[truth[text]["_seg"]]
        out = dummy(response_format["json_schema"]["schema"])
        if not isinstance(c, str):
            out.update(cause_type="inconnue", cause_racine="cause non précisée dans le rapport", facteur_principal="indéterminé",
                       consequence="arrêt constaté", categorie_consequence="perte d'activité")
        else:
            phrase = seg_causes[c][0]
            out.update(cause_type="explicite", cause_racine=phrase, indice=" ".join(phrase.split()[:12]),
                       facteur_principal=FACTOR_OF[c], consequence=f"{seg_causes[c][1]} sur l'équipement",
                       categorie_consequence=CONS[seg_causes[c][1]], confiance="haute",
                       maintenance_suspectee="probable" if c in ("usure", "graissage") else "non")
        return {"content": json.dumps(out, ensure_ascii=False), "tool_calls": []}
    calls["redaction" if name == "redaction" else "jugement"] += 1
    return {"content": json.dumps(dummy(response_format["json_schema"]["schema"]), ensure_ascii=False), "tool_calls": []}


def mock_embed(texts, cache_root):
    out = np.zeros((len(texts), 128), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"\w{4,}", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 128] += 1
        out[i] += np.random.default_rng(i).normal(0, 0.05, 128)
    return out


import llm.client, llm.agent, framing.pipeline, cause_extraction.pipeline, analysis_agent.writing, analysis_agent.judge  # noqa
import topic_modeling.embeddings  # noqa
for m in (llm.client, llm.agent, framing.pipeline, cause_extraction.pipeline, analysis_agent.writing, analysis_agent.judge):
    m.chat = mock_chat
topic_modeling.embeddings.embed_texts = mock_embed

from framing.pipeline import run as run_framing  # noqa
from analysis_agent.pipeline import run as run_analysis  # noqa
from review_agent.pipeline import run as run_review  # noqa

run_framing()
assert run_analysis(max_turns=1)
run_review(max_turns=1)
import webapp.export_html as wx  # noqa
wf = sys.modules["figures"]
wf.ANALYSIS, wf.REVIEW = Path("outputs/2_analyse").resolve(), Path("outputs/3_verification").resolve()
wx.OUTPUT = wf.ANALYSIS / "rapport_interactif.html"
print("export :", wx.build())
print("appels simulés :", calls)
