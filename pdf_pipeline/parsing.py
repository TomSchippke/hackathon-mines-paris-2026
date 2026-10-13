"""Du texte brut d'un rapport aux champs du CSV : découpage, nettoyage et validation."""

from __future__ import annotations

import difflib
import re
import unicodedata

import pandas as pd

from .config import ASSET_TYPES, DECIMALS, FIELD_LABELS, OCR_DIGIT_FIXES


def normalize(s: str) -> str:
    """Minuscules sans accents ni espaces superflus, pour comparer des libellés bruités."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip().lower()


# --------------------------------------------------------------------------------------------
# Découpage en champs
# --------------------------------------------------------------------------------------------

def match_label(line: str, label: str) -> str | None:
    """Si `line` commence par `label` (à quelques erreurs d'OCR près), renvoie le reste de la ligne."""
    line = line.strip().lstrip("|[]_ ")
    if difflib.SequenceMatcher(None, normalize(line)[: len(normalize(label))],
                               normalize(label)).ratio() < 0.85:
        return None
    # Le reste éventuel de la ligne est la valeur (l'OCR met parfois libellé et valeur sur une ligne).
    return line[len(label):].strip(" :|[]_")


def split_fields(text: str) -> dict[str, str]:
    """Découpe le texte du rapport en {colonne: valeur brute} à partir des libellés connus."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    fields: dict[str, list[str]] = {}
    current = None
    remaining = list(FIELD_LABELS.items())
    for line in lines:
        # Les libellés apparaissent dans l'ordre : on ne cherche que parmi les suivants.
        for i, (col, label) in enumerate(remaining):
            rest = match_label(line, label)
            if rest is not None:
                current = col
                fields[col] = [rest] if rest else []
                remaining = remaining[i + 1:]
                break
        else:
            if current is not None:
                fields[current].append(line)
    # La description est découpée en lignes par la mise en page : on la remet en un paragraphe.
    return {col: re.sub(r"\s+", " ", " ".join(parts)).strip() for col, parts in fields.items()}


# --------------------------------------------------------------------------------------------
# Nettoyage champ par champ (None = valeur absente ou invalide)
# --------------------------------------------------------------------------------------------

def site_code(site: str) -> str:
    """'Centrale PV Ouest-1' -> 'OUEST1', préfixe attendu des identifiants équipement/technicien."""
    m = re.search(r"(\w+)-(\d+)\s*$", site)
    return normalize(m.group(1)).upper() + m.group(2) if m else ""


def clean_number(raw: str, decimals: int) -> float | None:
    """Garde les chiffres et replace la virgule selon le format connu ('5:1' ou '51' -> 5.1)."""
    digits = re.sub(r"\D", "", raw.replace(" ", "").translate(OCR_DIGIT_FIXES))
    if len(digits) <= decimals:
        return None
    return float(f"{digits[:-decimals]}.{digits[-decimals:]}")


def clean_date(raw: str) -> str | None:
    digits = re.sub(r"\D", "", raw.translate(OCR_DIGIT_FIXES))
    if len(digits) != 8:
        return None
    date = pd.to_datetime(digits, format="%Y%m%d", errors="coerce")
    if pd.isna(date) or not 2000 <= date.year <= 2030:
        return None
    return date.strftime("%Y-%m-%d")


def clean_equipment(raw: str, code: str) -> str | None:
    """'OUESTI-INVOS' -> 'OUEST1-INV05' : préfixe du site imposé, numéro sur 2 chiffres."""
    s = re.sub(r"[\s.:]", "", raw).upper()
    m = re.search(r"-?(INV|WTG)([A-Z0-9]{2,3})$", s.replace("1NV", "INV").replace("WT6", "WTG"))
    if not m:
        return None
    num = m.group(2).translate(OCR_DIGIT_FIXES)
    if len(num) == 3 and num[0] == "0":  # 'WTGO01', 'INVOS5' : un caractère lu en double
        num = num[0] + num[2]
    return f"{code}-{m.group(1)}{num}" if re.fullmatch(r"\d\d", num) else None


def clean_technician(raw: str, code: str) -> str | None:
    """'TECH-OUEST].2' -> 'TECH-OUEST1-2' : préfixe du site imposé, numéro d'un chiffre."""
    s = re.sub(r"\s", "", raw).upper()
    if not s.startswith("TECH"):
        return None
    num = s[-1:].translate(OCR_DIGIT_FIXES)
    return f"TECH-{code}-{num}" if num.isdigit() else None


def snap(raw: str, vocabulary: list[str]) -> str | None:
    """Ramène une valeur bruitée à la valeur connue la plus proche (ou None si rien de proche)."""
    norm_vocab = {normalize(v): v for v in vocabulary}
    match = difflib.get_close_matches(normalize(raw), list(norm_vocab), n=1, cutoff=0.75)
    return norm_vocab[match[0]] if match else None


def clean_fields(raw: dict[str, str], csv_site: str, vocab: dict[str, list[str]]) -> dict:
    """Convertit les champs bruts en valeurs typées ; None pour une valeur absente ou invalide."""
    code = site_code(csv_site)
    asset = normalize(raw.get("asset_type", ""))
    return {
        "site": csv_site,  # le CSV fait foi pour le site
        "asset_type": next((v for k, v in ASSET_TYPES.items() if k in asset), None),
        "equipment_id": clean_equipment(raw.get("equipment_id", ""), code),
        "date": clean_date(raw.get("date", "")),
        "severity": snap(raw.get("severity", ""), vocab["severity"]),
        "downtime_hours": clean_number(raw.get("downtime_hours", ""), DECIMALS["downtime_hours"]),
        "estimated_cost_eur": clean_number(raw.get("estimated_cost_eur", ""),
                                           DECIMALS["estimated_cost_eur"]),
        "weather_condition": snap(raw.get("weather_condition", ""), vocab["weather_condition"]),
        "technician_id": clean_technician(raw.get("technician_id", ""), code),
        "text": raw.get("text") or None,
    }
