"""Du texte d'un rapport aux valeurs finales, avec contrôles. Rien ici ne relit les PDF : tout part
du texte (couche texte d'un PDF digital, ou lecture docTR en cache pour un scan).
"""

from __future__ import annotations

import pandas as pd

from .config import RATE_BAND_MARGIN
from .parsing import clean_fields, snap, split_fields


def build_reference(text_rows: pd.DataFrame, digital: list[tuple[str, str]]) -> dict:
    """Référence tirée des rapports sans OCR (lignes `text` du CSV + PDF digitaux) :
    - valeurs connues de sévérité et de météo (pour corriger une valeur mal lue) ;
    - plage de coût horaire (€/h) par (type d'installation, sévérité) (pour repérer un chiffre mal lu).

    `digital` : liste de (texte du PDF, site). La référence s'enrichit si on ajoute des rapports.
    """
    raw = [split_fields(text) for text, _ in digital]
    vocab = {c: sorted(set(text_rows[c].dropna()) | {r[c] for r in raw if r.get(c)})
             for c in ("severity", "weather_condition")}
    parsed = pd.DataFrame([clean_fields(r, site, vocab) for r, (_, site) in zip(raw, digital)])
    ref = pd.concat([text_rows, parsed]).dropna(subset=["asset_type", "severity", "downtime_hours",
                                                        "estimated_cost_eur"])
    rate = ref["estimated_cost_eur"] / ref["downtime_hours"]
    bands = {k: (r.min() * (1 - RATE_BAND_MARGIN), r.max() * (1 + RATE_BAND_MARGIN))
             for k, r in rate.groupby([ref["asset_type"], ref["severity"]])}
    return {"vocab": vocab, "rate_bands": bands}


def rate_ok(fields: dict, bands: dict) -> bool | None:
    """Le coût horaire est-il dans la plage observée pour ce type et cette sévérité ?
    None si on ne peut pas le vérifier (champ manquant, combinaison jamais observée)."""
    band = bands.get((fields.get("asset_type"), fields.get("severity")))
    dt, cost = fields.get("downtime_hours"), fields.get("estimated_cost_eur")
    if band is None or not dt or cost is None:
        return None
    return bool(band[0] <= cost / dt <= band[1])


def extract_fields(text: str, site: str, ref: dict) -> tuple[dict, list[str]]:
    """Valeurs d'un rapport et avertissements (ce qu'il faut vérifier à la main)."""
    fields = clean_fields(split_fields(text), site, ref["vocab"])
    warnings = [f"{c} illisible" for c, v in fields.items() if v is None]
    if rate_ok(fields, ref["rate_bands"]) is False:
        warnings.append(f"coût/durée incohérents ({fields['estimated_cost_eur']} € pour {fields['downtime_hours']} h)")
    pdf_site = split_fields(text).get("site")
    if pdf_site and snap(pdf_site, [site]) is None:
        warnings.append(f"site du PDF '{pdf_site[:60]}' différent du CSV '{site}'")
    return fields, warnings
