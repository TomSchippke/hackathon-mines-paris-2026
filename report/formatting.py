"""Typographie française des nombres, montants, pourcentages et dates."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

NBSP = " "  # espace insécable (avant « : », « € », « % », « h »)
NNBSP = " "  # espace fine insécable (séparateur de milliers, avant « ; »)
MONTHS = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]


def _is_missing(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def num(x: Any, decimals: int = 0) -> str:
    """1234567.8 -> « 1 234 568 » (milliers séparés par une espace fine, virgule décimale)."""
    if _is_missing(x):
        return "n/d"
    s = f"{float(x):,.{decimals}f}"
    return s.replace(",", NNBSP).replace(".", ",")


def eur(x: Any, decimals: int = 0) -> str:
    """Montant en euros : « 1 234 568 € »."""
    return "n/d" if _is_missing(x) else f"{num(x, decimals)}{NBSP}€"


def eur_short(x: Any) -> str:
    """Montant abrégé pour les graphiques : « 1,63 M€ », « 282 k€ »."""
    if _is_missing(x):
        return "n/d"
    x = float(x)
    if abs(x) >= 1e6:
        return f"{num(x / 1e6, 2)}{NBSP}M€"
    if abs(x) >= 1e3:
        return f"{num(x / 1e3, 0)}{NBSP}k€"
    return eur(x)


def pct(x: Any, decimals: int = 0) -> str:
    """Part (0-1) en pourcentage : « 12 % »."""
    return "n/d" if _is_missing(x) else f"{num(100 * float(x), decimals)}{NBSP}%"


def share(x: Any) -> str:
    """Part du coût : une décimale sous 1 % (« 0,4 % »), entière au-delà."""
    return pct(x, 1 if not _is_missing(x) and float(x) < 0.01 else 0)


def hours(x: Any) -> str:
    return "n/d" if _is_missing(x) else f"{num(x, 0)}{NBSP}h"


def date_fr(x: Any) -> str:
    """« 2024-12-20 » -> « 20 déc. 2024 »."""
    d = pd.to_datetime(x, errors="coerce")
    if pd.isna(d):
        return "n/d"
    return f"{d.day}{NBSP}{MONTHS[d.month - 1]}{NBSP}{d.year}"
