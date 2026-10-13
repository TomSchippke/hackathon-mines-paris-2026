# Pipeline d'extraction des rapports PDF

Dans `data/incidents_students.csv`, les 1200 rapports au format PDF n'ont que la colonne `site`
de remplie. Cette pipeline lit les PDF et produit **`outputs/1_extraction/incidents_full.csv`** : les 1800
incidents avec toutes les colonnes remplies (équipement, date, sévérité, durée, coût, météo,
technicien, description).

## Utilisation

```bash
pip install -r requirements.txt                 # dont python-doctr[torch]
brew install tesseract tesseract-lang            # Linux : apt install tesseract-ocr tesseract-ocr-fra
python -m pdf_pipeline                           # -> outputs/1_extraction/incidents_full.csv
```

Puis dans le notebook : `df = pd.read_csv("outputs/1_extraction/incidents_full.csv")`.

Premier lancement : ~30 min (OCR des 600 scans, GPU du Mac utilisé si disponible). Lancements
suivants : quelques secondes, tout vient du cache.

**Ajouter de nouveaux rapports** : déposer les PDF dans `data/`, ajouter une ligne par rapport
dans le CSV (`incident_id`, `document_path` ; `site` et `document_format` facultatifs) et
relancer `python -m pdf_pipeline`. Seuls les nouveaux PDF sont lus.

## Ce que fait la pipeline

| Étape | Ce qui est fait | Pourquoi |
|---|---|---|
| 1. Lecture du PDF | Si le PDF contient du texte (PDF digital), il est lu directement. Sinon c'est un scan (une image) et il faut un OCR. | Le texte d'un PDF digital est exact ; l'OCR n'est utilisé que quand c'est nécessaire. |
| 2. OCR des scans | **docTR** (modèle de deep learning) lit chaque scan, en redressant les pages penchées. | Comparé à Tesseract (voir plus bas), docTR fait nettement moins d'erreurs sur les chiffres. |
| 3. Découpage | Le texte est découpé en champs grâce aux libellés du rapport (« Date », « Coût estimé (€) »...). | Un OCR renvoie du texte brut, pas un tableau. |
| 4. Nettoyage | Corrections selon le format attendu : `INVO9` → `INV09`, `27` → `2.7`, météo/sévérité ramenées à la valeur connue la plus proche. | Erreurs typiques de l'OCR : O/0, S/5, point décimal perdu. |
| 5. Contrôle de cohérence | Le coût horaire (coût / durée) doit être dans la plage observée sur les 1200 rapports sans OCR, pour le même type d'installation et la même sévérité (ex. 187–264 €/h pour du solaire « mineure »). | Détecte un chiffre mal lu dans la durée ou le coût, qui serait sinon invisible. |
| 6. Secours Tesseract | Si docTR laisse un champ vide ou un coût horaire implausible, le scan est relu par **Tesseract** (4 lectures + vote) et les champs manquants sont complétés. | Deux moteurs différents se trompent rarement au même endroit. Ici : 1 scan sur 600 concerné. |
| 7. Signalement | Ce qui reste douteux est décrit dans la colonne `extraction_warnings` (vide si tout va bien). | Ne jamais laisser passer une valeur douteuse sans le dire. |

Colonnes ajoutées au CSV :
- `extraction_method` : `csv` (déjà dans le CSV), `text_layer` (PDF digital), `ocr_doctr`,
  `ocr_doctr+tesseract` (complété par Tesseract) ;
- `extraction_warnings` : vide, ou ce qu'il faut vérifier à la main.

## Pourquoi docTR + Tesseract

Deux moteurs OCR ont été testés sur les 600 scans, puis comparés :

| | Tesseract (4 lectures + vote) | docTR (1 lecture) | **Pipeline (docTR + secours Tesseract)** |
|---|---|---|---|
| Précision sur 40 scans relus à la main | 280/280 | 280/280 | **280/280** |
| Erreurs sur les 9 scans où les moteurs divergent (vérifiés à la main) | 8 | 1 (champ vide) | **0** |
| Erreurs non signalées parmi ces 9 | 5 | 0 | **0** |
| Qualité des descriptions (mots connus) | 98,7 % | 99,2 % | **99,2 %** |
| Rapports à vérifier à la main | 3 | 1 | **0** |

- **Tesseract** est un OCR classique, rapide et léger, mais fragile sur ces scans bruités :
  sur une seule lecture, environ 1 scan sur 4 avait un champ différent d'une autre lecture.
  Il a fallu 4 lectures, un vote et le contrôle de cohérence pour le fiabiliser, et il reste
  des erreurs silencieuses (`INV09` lu `INV05` sur 6 scans).
- **docTR** est un OCR à base de réseaux de neurones, plus lent (~3 s/page sur GPU) mais bien
  plus précis : 8 des 9 désaccords sont tranchés en sa faveur. Son seul défaut (un champ vide)
  est signalé, donc rattrapable.
- **La combinaison** garde le meilleur des deux : docTR partout, Tesseract seulement en secours.

Les vérifications (formats, cohérence coût/durée) restent utiles même avec un meilleur OCR :
c'est grâce à elles que le champ manquant de docTR a été détecté et complété.

Pour refaire la comparaison :

```bash
python -m pdf_pipeline --engine doctr        # -> outputs/comparaison_ocr/incidents_full_doctr.csv
python -m pdf_pipeline --engine tesseract    # -> outputs/comparaison_ocr/incidents_full_tesseract.csv
python -m pdf_pipeline.compare               # précision, désaccords, qualité des descriptions
```

`outputs/comparaison_ocr/verite_terrain_scans.csv` contient les valeurs relues à la main sur 40 scans tirés au hasard.

## Cache

Les lectures OCR brutes sont sauvegardées dans `outputs/cache/ocr_<moteur>/<empreinte du PDF>.json`.
L'empreinte (SHA-1) est calculée sur le contenu du fichier :

- un PDF déjà lu n'est jamais relu, même renommé ou avec un autre `incident_id` ;
- un PDF modifié a une nouvelle empreinte, il est donc relu automatiquement ;
- les règles de nettoyage et de validation (étapes 3 à 7) sont refaites à chaque lancement à
  partir du cache : on peut les modifier sans relancer l'OCR.

`python -m pdf_pipeline --force` ignore le cache et relit tous les scans.

`outputs/` est ignoré par git : pour partager les résultats, envoyer `outputs/1_extraction/incidents_full.csv`
(ou le dossier `outputs/cache/` pour éviter aux autres de refaire l'OCR).

## Organisation du code

| Fichier | Rôle |
|---|---|
| `pipeline.py` | Orchestration des étapes et ligne de commande |
| `ocr_doctr.py` | Lecture d'un scan avec docTR |
| `ocr_tesseract.py` | Lecture d'un scan avec Tesseract (4 lectures + relecture des cellules numériques) |
| `parsing.py` | Découpage du texte en champs, nettoyage et validation de chaque champ |
| `resolve.py` | Vote entre lectures, contrôle coût/durée, combinaison des moteurs, avertissements |
| `cache.py` | Cache des lectures OCR indexé par le contenu des PDF |
| `config.py` | Paramètres : libellés du rapport, formats, réglages OCR |
| `compare.py` | Comparaison de plusieurs extractions |
