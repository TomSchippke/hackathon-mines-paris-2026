# Pipeline d'extraction des rapports PDF

Dans `data/incidents_students.csv`, les 1200 rapports au format PDF n'ont que la colonne `site`
de remplie. Cette pipeline lit les PDF et produit **`outputs/1_extraction/incidents_full.csv`** :
les 1800 incidents avec toutes les colonnes remplies (équipement, date, sévérité, durée, coût,
météo, technicien, description).

## Utilisation

```bash
pip install -r requirements.txt     # dont python-doctr[torch]
python main.py --steps pdf          # ou : python -m pdf_pipeline
```

Puis : `df = pd.read_csv("outputs/1_extraction/incidents_full.csv")`.

Premier lancement : ~30 min (OCR des 600 scans, GPU du Mac utilisé si disponible). Lancements
suivants : quelques secondes, tout vient du cache.

**Ajouter de nouveaux rapports** : déposer les PDF dans `data/`, ajouter une ligne par rapport
dans le CSV (`incident_id`, `document_path` ; `site` et `document_format` facultatifs) et
relancer. Seuls les nouveaux PDF sont lus.

## Ce que fait la pipeline

| Étape | Ce qui est fait | Pourquoi |
|---|---|---|
| 1. Lecture du PDF | Si le PDF contient du texte (PDF digital), il est lu directement. Sinon c'est un scan (une image) et il faut un OCR. | Le texte d'un PDF digital est exact ; l'OCR n'est utilisé que quand c'est nécessaire. |
| 2. OCR des scans | **docTR** (OCR à base de réseaux de neurones) lit chaque scan, en redressant les pages penchées. | Le plus précis des OCR testés sur ces scans bruités (voir plus bas). |
| 3. Découpage | Le texte est découpé en champs grâce aux libellés du rapport (« Date », « Coût estimé (€) »...). | Un OCR renvoie du texte brut, pas un tableau. |
| 4. Nettoyage | Corrections selon le format attendu : `INVO9` → `INV09`, `27` → `2.7`, météo/sévérité ramenées à la valeur connue la plus proche. | Erreurs typiques de l'OCR : O/0, S/5, point décimal perdu. |
| 5. Contrôle de cohérence | Le coût horaire (coût / durée) doit être dans la plage observée sur les 1200 rapports sans OCR, pour le même type d'installation et la même sévérité. | Détecte un chiffre mal lu dans la durée ou le coût. |
| 6. Signalement | Ce qui reste douteux (champ illisible, coût incohérent, site différent du CSV) est décrit dans la colonne `extraction_warnings` (vide si tout va bien). | Ne jamais laisser passer une valeur douteuse sans le dire. |

Colonnes ajoutées au CSV :
- `extraction_method` : `csv` (déjà dans le CSV), `text_layer` (PDF digital), `ocr_doctr` (scan) ;
- `extraction_warnings` : vide, ou ce qu'il faut vérifier à la main. Aujourd'hui : 1 rapport
  (incident 232, technicien illisible sur le scan).

## Pourquoi docTR

Deux OCR ont été comparés sur les 600 scans, avec 40 scans relus à la main : Tesseract (OCR
classique, 4 lectures + vote) et docTR (deep learning). Tous deux lisaient correctement les 40
scans relus, mais sur les 9 scans où ils divergeaient, docTR avait raison 8 fois ; Tesseract
faisait des erreurs silencieuses (`INV09` lu `INV05`, chiffres du coût ou de la durée). Seul docTR
est donc gardé : la pipeline est plus simple, et son unique défaut observé (un champ vide) est
signalé dans `extraction_warnings`.

## Cache

Les lectures docTR sont sauvegardées dans `outputs/cache/ocr_doctr/<empreinte du PDF>.json`.
L'empreinte (SHA-1) est calculée sur le contenu du fichier :

- un PDF déjà lu n'est jamais relu, même renommé ou avec un autre `incident_id` ;
- un PDF modifié a une nouvelle empreinte, il est donc relu automatiquement ;
- le découpage, le nettoyage et les contrôles (étapes 3 à 6) sont refaits à chaque lancement à
  partir du cache : on peut les modifier sans relancer l'OCR.

`python -m pdf_pipeline --force` ignore le cache et relit tous les scans.

## Organisation du code

| Fichier | Rôle |
|---|---|
| `pipeline.py` | Orchestration des étapes et ligne de commande |
| `ocr_doctr.py` | Lecture d'un scan avec docTR |
| `parsing.py` | Découpage du texte en champs, nettoyage de chaque champ |
| `validation.py` | Référence (valeurs connues, plages de coût horaire), contrôles, avertissements |
| `cache.py` | Cache des lectures OCR indexé par le contenu des PDF |
| `config.py` | Paramètres : libellés du rapport, formats, réglages OCR |
