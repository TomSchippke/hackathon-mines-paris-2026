"""Paramètres de la pipeline : format des rapports et réglages OCR."""

# Résolution de rendu des pages scannées avant OCR.
DPI = 300

# Sous ce nombre de caractères extraits, un PDF est considéré comme un scan (pas de couche texte).
MIN_TEXT_CHARS = 50

# Seuil de binarisation des images pour Tesseract : supprime le grain du papier scanné.
BINARIZE_THRESHOLD = 140

# Libellés du rapport, dans l'ordre d'apparition, et colonne correspondante du CSV.
FIELD_LABELS = {
    "site": "Site",
    "asset_type": "Type d'installation",
    "equipment_id": "Équipement",
    "date": "Date",
    "severity": "Sévérité",
    "downtime_hours": "Durée d'arrêt (h)",
    "estimated_cost_eur": "Coût estimé (€)",
    "weather_condition": "Conditions météo",
    "technician_id": "Technicien",
    "text": "Description de l'incident",
}

ASSET_TYPES = {"photovolta": "solar", "olien": "wind"}

# Nombre de décimales des champs numériques (format constant dans tous les rapports) :
# permet de replacer le point décimal quand l'OCR l'a perdu ("27" -> 2.7).
DECIMALS = {"downtime_hours": 1, "estimated_cost_eur": 2}

# Confusions lettre -> chiffre fréquentes en OCR, appliquées uniquement là où un chiffre est attendu.
OCR_DIGIT_FIXES = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1",
                                 "|": "1", "]": "1", "!": "1", "S": "5", "s": "5", "B": "8",
                                 "Z": "2", "z": "2", "G": "6", "i": "1", "j": "1", "‘": "1",
                                 "'": "1"})

# Marge autour de la plage de coût horaire observée sur les rapports sans OCR.
RATE_BAND_MARGIN = 0.02
