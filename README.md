# Bloc 4 — AT&T Spam Detector (Deep Learning NLP)

> Projet de la **Certification Jedha — Concepteur Développeur en Science des Données (CDSD)**
> [RNCP35288 — France Compétences](https://www.francecompetences.fr/recherche/rncp/35288/)
> **Bloc 4** : Analyse prédictive de données non structurées — Deep Learning

## Objectif du projet

AT&T Inc. (le plus gros opérateur télécom américain) veut **détecter automatiquement les SMS spam** pour protéger ses utilisateurs. À partir d'un dataset de SMS labellisés (`spam` / `ham`), on construit un modèle de Deep Learning capable de classifier un SMS sur **son seul contenu textuel**.

## Approche

Le projet se construit en 3 étapes :

1. **EDA + preprocessing du texte** : tokenization, normalisation, gestion du déséquilibre de classe (typiquement ~13% de spam)
2. **Modèle baseline** : un modèle simple (TF-IDF + LogisticRegression ou LSTM léger) pour avoir une référence
3. **Transfer learning avec DistilBERT** : fine-tuning d'un modèle pré-entraîné sur SMS Spam — pour des perfs nettement supérieures avec peu de données

## Données

Le dataset (CSV de SMS labellisés) est à télécharger depuis Julie Jedha :
- <https://app.jedha.co/course/projects-deep-learning-ft/att-spam-detector-ft>
- Cliquer sur "Download the Dataset" et placer le CSV dans `data/sms_spam.csv`

## Livrable

- Notebook qui couvre les 3 étapes (preprocessing + baseline + DistilBERT)
- Performance clairement annoncée (accuracy, F1-score, confusion matrix)
- Comparaison baseline vs transfer learning

## Structure du projet

```
Bloc_4_AT&T_Spam_Detector/
├── notebooks/
│   └── 01_att_spam_detector.ipynb         (notebook complet — livrable)
├── data/
│   └── sms_spam.csv                       (à télécharger depuis Julie Jedha, gitignored)
├── requirements.txt
└── README.md
```

## Comment rejouer le projet

```bash
cd Bloc_4_AT&T_Spam_Detector
pip install -r requirements.txt
# Télécharger sms_spam.csv depuis Julie Jedha dans data/
```

⚠️ Le fine-tuning DistilBERT prend ~5-10 min sur CPU, <1 min sur GPU. Pour une démo rapide on peut limiter `num_train_epochs=1` et `max_steps=100`.

## Auteur

[Aymeric Lahonde](https://github.com/aymericlahonde-dotcom) — promotion Jedha CDSD 2026.
