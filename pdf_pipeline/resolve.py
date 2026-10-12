"""Des lectures OCR aux valeurs finales : vote, contrôle de cohérence coût/durée, combinaison
des moteurs. Rien ici ne relit les PDF : tout part des lectures mises en cache.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from .config import DECIMALS, RATE_BAND_MARGIN
from .parsing import clean_fields, clean_number, snap, split_fields, text_quality, words

NUMERIC = ("downtime_hours", "estimated_cost_eur")


def build_reference(text_rows: pd.DataFrame, digital: list[tuple[str, str]]) -> dict:
    """Référence tirée des rapports sans OCR (lignes `text` du CSV + PDF digitaux) :
    - valeurs connues de sévérité et de météo ;
    - plage de coût horaire (€/h) par (type d'installation, sévérité) ;
    - vocabulaire des descriptions, pour juger la propreté d'une lecture OCR.

    `digital` : liste de (texte du PDF, site). La référence s'enrichit si on ajoute des rapports.
    """
    raw = [split_fields(text) for text, _ in digital]
    vocab = {c: sorted(set(text_rows[c].dropna()) | {r[c] for r in raw if r.get(c)})
             for c in ("severity", "weather_condition")}
    parsed = pd.DataFrame([clean_fields(r, site, vocab) for r, (_, site) in zip(raw, digital)])
    ref = pd.concat([text_rows, parsed]).dropna(subset=["asset_type", "severity", *NUMERIC])
    rate = ref["estimated_cost_eur"] / ref["downtime_hours"]
    bands = {k: (r.min() * (1 - RATE_BAND_MARGIN), r.max() * (1 + RATE_BAND_MARGIN))
             for k, r in rate.groupby([ref["asset_type"], ref["severity"]])}
    vocabulary = {w for text in ref["text"].dropna() for w in words(text)}
    return {"vocab": vocab, "rate_bands": bands, "words": vocabulary}


def rate_ok(fields: dict, bands: dict) -> bool | None:
    """Le coût horaire est-il dans la plage observée pour ce type et cette sévérité ?
    None si on ne peut pas le vérifier (champ manquant, combinaison jamais observée)."""
    band = bands.get((fields.get("asset_type"), fields.get("severity")))
    dt, cost = fields.get("downtime_hours"), fields.get("estimated_cost_eur")
    if band is None or not dt or cost is None:
        return None
    return bool(band[0] <= cost / dt <= band[1])


def vote(readings: list[dict]) -> tuple[dict, dict[str, Counter]]:
    """Valeur majoritaire de chaque champ parmi les lectures (égalité : la première lecture gagne).
    Renvoie aussi, par champ, le nombre de lectures ayant donné chaque valeur."""
    fields, counts = {}, {}
    for col in readings[0]:
        values = [r[col] for r in readings if r.get(col) is not None]
        counts[col] = Counter(values)
        fields[col] = max(values, key=lambda v: counts[col][v]) if values else None
    return fields, counts


def resolve_reading(reading: dict, site: str, ref: dict) -> tuple[dict, list[str]]:
    """Valeurs d'un rapport à partir d'une entrée de cache (une ou plusieurs lectures).

    Renvoie (champs, notes). Les notes signalent les votes sans majorité.
    """
    readings = [clean_fields(split_fields(t), site, ref["vocab"]) for t in reading["texts"]]
    if reading.get("cells"):  # relecture des cellules numériques (Tesseract) : candidats en plus
        cells = {c: clean_number(v, DECIMALS[c]) for c, v in reading["cells"].items()}
        readings.append({c: cells.get(c) for c in readings[0]})

    fields, counts = vote(readings)
    # Description : on ne vote pas sur un paragraphe, on garde la lecture la plus propre.
    fields["text"] = max((r["text"] for r in readings if r["text"]), default=None,
                         key=lambda t: text_quality(t, ref["words"]))
    notes = [f"{c} incertain: {dict(counts[c])}" for c in fields
             if c not in (*NUMERIC, "text") and len(counts[c]) > 1
             and counts[c][fields[c]] <= sum(counts[c].values()) / 2]

    # Durée et coût : parmi les valeurs lues, la paire la plus votée dont le coût horaire est plausible.
    if rate_ok(fields, ref["rate_bands"]) is False:
        pairs = [(dt, cost) for dt in counts["downtime_hours"] for cost in counts["estimated_cost_eur"]
                 if rate_ok({**fields, "downtime_hours": dt, "estimated_cost_eur": cost}, ref["rate_bands"])]
        if pairs:
            best = max(pairs, key=lambda p: counts["downtime_hours"][p[0]] + counts["estimated_cost_eur"][p[1]])
            fields["downtime_hours"], fields["estimated_cost_eur"] = best
    return fields, notes


def needs_fallback(fields: dict, ref: dict) -> bool:
    """Un champ manque, ou le coût horaire est implausible."""
    return any(v is None for v in fields.values()) or rate_ok(fields, ref["rate_bands"]) is False


def combine(primary: dict, fallback: dict, ref: dict) -> tuple[dict, list[str]]:
    """Complète la lecture principale avec la lecture de secours. Renvoie (champs, colonnes reprises)."""
    fields, used = dict(primary), []
    for col, value in fields.items():
        if value is None and fallback.get(col) is not None:
            fields[col] = fallback[col]
            used.append(col)
    if rate_ok(fields, ref["rate_bands"]) is False and rate_ok(fallback, ref["rate_bands"]):
        for col in NUMERIC:
            fields[col] = fallback[col]
        used += [c for c in NUMERIC if c not in used]
    return fields, used


def final_warnings(fields: dict, reading: dict, site: str, ref: dict) -> list[str]:
    """Ce qu'il reste à vérifier à la main sur un rapport."""
    warnings = [f"{c} illisible" for c, v in fields.items() if v is None]
    if rate_ok(fields, ref["rate_bands"]) is False:
        warnings.append(f"coût/durée incohérents ({fields['estimated_cost_eur']} € pour "
                        f"{fields['downtime_hours']} h)")
    pdf_site = split_fields(reading["texts"][0]).get("site")
    if pdf_site and snap(pdf_site, [site]) is None:
        warnings.append(f"site du PDF '{pdf_site[:60]}' différent du CSV '{site}'")
    return warnings
