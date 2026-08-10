## 4. PROJET B — Backtest Integrity Auditor
*Automne 2026 · ~65–75 h · repo : `backtest-integrity-auditor`*

### 4.1 Problem statement (EN — draft à adapter)

> Most reported backtest performance is an artifact of selection bias: researchers try many configurations and report the best, while standard metrics (Sharpe ratio, p-values) assume a single trial. Separately, backtest code itself frequently leaks future information (look-ahead bias) in ways that are invisible in the results and tedious to find by manual review. There is no lightweight tool that (i) deflates reported performance for the number of trials actually run, and (ii) audits the backtest *code* for leakage patterns. I built one — and validated it on my own strategy, whose in-sample Sharpe of 1.93 collapsed to a deflated 0.92, with net returns turning negative after transaction costs.

### 4.2 Positionnement

- **Pourquoi c'est ton projet signature** : il industrialise l'histoire DSR (1.93 → 0.92, 30 configurations) qui ancre déjà tes candidatures Squarepoint/IMC. L'outil *est* la preuve de la compétence.
- **Pourquoi c'est pertinent pour une plateforme multi-manager** : allouer du capital entre N équipes de PM à partir de leurs track records est un problème de tests multiples — exactement ce que le DSR corrige. Tu dois pouvoir dérouler cet argument en 60 secondes.
- **Différenciation dans un concours type expo** : dans une mer de chatbots, un outil qui *détruit* des résultats trop beaux est mémorable. Risque assumé : moins immédiatement lisible pour un jury mixte → la démo (§4.10) doit porter toute la charge pédagogique.

### 4.3 Architecture — 4 modules

```
[Module 1] Statistical Core        [Module 2] Code Auditor
  entrée: séries de rendements       entrée: code Python du backtest
  par essai OU stats agrégées        AST déterministe (règles R1–R10)
  + nombre d'essais N                + passe LLM sémantique
  sortie: PSR, E[max SR], DSR,       sortie: findings JSON
  PBO (CSCV), rapport de             {ligne, règle, sévérité, fix}
  déflation
            \                        /
             [Module 4] Report Generator
              "Backtest Health Report" 1 page (HTML/PDF)
             /
[Module 3] LLM-Signal Leakage Tester (optionnel, v2)
  protocole d'anonymisation (Glasserman–Lin) +
  test de coupure pré/post training cutoff du LLM
```

**UI** : Streamlit — upload (code + résultats), rapport à l'écran, export PDF.

### 4.4 Fondements statistiques — à maîtriser verbalement, formules comprises

À savoir dériver et commenter au tableau, sans notes. Unités et étapes intermédiaires verbalisées explicitement (règle personnelle : jamais mélanger les unités dans une phrase).

- **PSR — Probabilistic Sharpe Ratio** (Bailey & López de Prado, 2012). Probabilité que le vrai Sharpe dépasse un seuil SR*, en corrigeant la non-normalité :
  `PSR(SR*) = Φ( (ŜR − SR*)·√(T−1) / √(1 − γ₃·ŜR + ((γ₄−1)/4)·ŜR²) )`
  avec γ₃ skewness, γ₄ kurtosis des rendements, T le nombre d'observations.
- **Espérance du Sharpe maximal sous H₀** (N essais indépendants de vrai Sharpe nul) :
  `E[max_N SR] ≈ √V[SR] · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]`
  avec γ ≈ 0,5772 (constante d'Euler–Mascheroni) et V[SR] la variance des Sharpe estimés entre essais. Intuition à verbaliser : le maximum de N tirages gaussiens croît ≈ en √(2·ln N) — c'est pourquoi « essayer plus de configs » gonfle mécaniquement le meilleur Sharpe observé.
- **DSR — Deflated Sharpe Ratio** (Bailey & López de Prado, 2014) : `DSR = PSR(SR* = E[max_N SR])`. C'est le PSR évalué contre le seuil que le pur hasard aurait produit compte tenu du nombre d'essais.
- **PBO via CSCV** (Bailey, Borwein, López de Prado, Zhu, 2014/2017) : partition de l'historique en S blocs ; pour chaque combinaison de S/2 blocs (IS) vs le complément (OOS), on retient la config optimale IS, on mesure son rang relatif OOS, on calcule le logit λ de ce rang ; **PBO = proportion des combinaisons où λ < 0** (la config gagnante IS fait pire que la médiane OOS). Limite à connaître : le découpage en blocs suppose une dépendance temporelle raisonnablement contenue ; autocorrélation forte → blocs plus longs.
- **Repères d'ordre de grandeur** (Harvey, Liu, Zhu, 2016) : compte tenu du data mining de la littérature factorielle, un seuil de t-stat proche de 3 (et non 2) est défendable pour une « découverte ».

### 4.5 Couche IA — hybride AST + LLM

**Pourquoi hybride et pas LLM seul** (argument de défense central) : les patterns fréquents et non ambigus se détectent par règles AST déterministes — précision ~100 %, coût nul, reproductible. Le LLM n'intervient que pour la fuite *sémantique* que les règles ne capturent pas. Un LLM seul hallucine des findings, coûte cher, et n'est pas reproductible run à run.

**Règles AST déterministes (v1) — catalogue R1–R10 :**
- R1 : `shift(-k)` (k>0) appliqué à une feature → usage direct du futur.
- R2 : `rolling(..., center=True)` sur des features → fenêtre chevauchant le futur.
- R3 : scaler / PCA / sélection de variables **fit sur l'échantillon complet** avant le split train/test.
- R4 : exécution same-bar — signal calculé sur le close et exécuté au même close.
- R5 : target leakage — feature construite à partir de la variable cible.
- R6 : univers défini par les constituants **actuels** d'un indice → biais du survivant.
- R7 : fondamentaux non point-in-time (valeurs révisées utilisées à la date d'origine).
- R8 : `bfill()` / interpolation utilisant des valeurs futures dans un resampling.
- R9 : tuning d'hyperparamètres sur l'ensemble de test.
- R10 : normalisation globale par des statistiques calculées sur tout l'historique.

**Passe LLM sémantique** : `merge_asof` mal orienté, construction de labels subtilement forward-looking, fonctions custom opaques. Contraintes de sortie : JSON structuré `{rule_id|semantic, line_start, line_end, severity, explanation, suggested_fix}` ; **toute finding doit citer des numéros de ligne existants** (grounding vérifié par le code, pas par le LLM) ; température 0 ; les findings LLM sont affichés dans une section distincte « à vérifier » — jamais fusionnés silencieusement avec les findings déterministes.

**Module 3 (v2, optionnel)** - leakage des signaux générés par LLM : protocole d'anonymisation à la Glasserman–Lin (masquer entités et dates dans les textes, comparer les prédictions : si la performance s'effondre, le « signal » venait de la mémoire d'entraînement du modèle, pas du texte) + test de coupure (performance pré vs post training-cutoff du modèle). À ne construire que si le temps le permet ; à mentionner comme extension sinon.

### 4.6 Exemple de system prompt (EN — artefact à versionner dans `prompts/`)

```
You are a code auditor specialized in detecting look-ahead bias and data
leakage in Python backtesting code. You receive a numbered source file.

Rules:
1. Report ONLY leakage-related findings. Ignore style, performance, bugs
   unrelated to information leakage.
2. Every finding MUST reference existing line numbers from the input.
3. If you are not confident a pattern leaks future information, set
   "severity": "review" and say why in one sentence.
4. Output strictly valid JSON matching the provided schema. No prose.
5. Do not invent code that is not in the file.

Schema: {"findings": [{"type": "semantic", "line_start": int,
"line_end": int, "severity": "high|medium|review",
"explanation": str, "suggested_fix": str}]}
```

### 4.7 APIs & outils

- `ast` (stdlib) pour les règles déterministes ; `pandas`, `numpy`, `scipy.stats` pour le core ; `matplotlib` pour le rapport ; `weasyprint` ou `fpdf2` pour l'export PDF ; API LLM (Claude/OpenAI) ou Ollama local pour la passe sémantique. Aucune donnée externe requise.

### 4.8 Étapes de développement

| Phase | Contenu | Definition of done | Heures |
|---|---|---|---|
| B0 | Refactor du moteur existant en librairie propre (`core/`), tests unitaires sur PSR/DSR contre valeurs calculées à la main | `pytest` vert ; DSR(cas 1.93/30 configs) reproduit 0.92 | 8–10 |
| B1 | Stats core complet : E[max SR], CSCV/PBO, rapport de déflation | PBO validé sur un cas synthétique où la réponse est connue | 10–12 |
| B2 | Règles AST R1–R10 | Chaque règle a ≥2 tests (un vrai positif, un vrai négatif) | ~10 |
| B3 | Passe LLM + schéma JSON + grounding des lignes + prompt registry | 0 finding avec numéro de ligne inexistant sur 20 runs | 12–15 |
| B4 | **Benchmark à bugs injectés** : 25–30 backtests synthétiques avec fuites connues (mix R1–R10 + sémantiques) ; mesure précision/rappel AST seul, LLM seul, hybride | Tableau précision/rappel dans le README | ~10 |
| B5 | UI Streamlit + Report Generator | Démo bout-en-bout < 2 min sur ton propre backtest | ~8 |
| B6 | README, DEVLOG consolidé, brouillon vidéo 7 min | Vidéo brouillon enregistrée et auto-critiquée | ~6 |

### 4.9 Évaluation — le benchmark à bugs injectés

C'est la crédibilité scientifique du projet. Générer 25–30 scripts de backtest synthétiques : ~15 contenant chacun 1–2 fuites tirées du catalogue, ~10 propres (contrôle des faux positifs), quelques cas sémantiques hors catalogue. Métriques : précision, rappel, F1 par détecteur (AST / LLM / hybride) + taux de faux positifs sur les scripts propres. Résultat attendu et honnête à assumer : l'AST gagne en précision sur son périmètre, le LLM ajoute du rappel sur le sémantique au prix de faux positifs — c'est exactement la justification de l'architecture hybride.

### 4.10 Démo 7 minutes — storyboard

1. (45 s) Le problème en une image : distribution du max de N Sharpe sous H₀ qui monte avec N.
2. (2 min) Upload de **ton propre backtest** de M2 : le rapport affiche 1.93 → 0.92 en direct, net négatif après coûts. Phrase clé : « the tool killed my own strategy — that is the point ».
3. (2 min) Upload d'un script piégé du benchmark : findings AST + findings LLM avec lignes surlignées et fix suggéré.
4. (1 min) Tableau précision/rappel du benchmark - les limites dites franchement.
5. (1 min) Extensions (module Glasserman–Lin, intégration CI).

### 4.11 Impact & value (EN — draft)

> For any team that evaluates strategies — or allocates capital across track records - the tool converts an unverifiable claim ("Sharpe 1.9") into an auditable one (deflated Sharpe given N trials, probability of backtest overfitting, and a line-level leakage audit of the code that produced it). It shifts review time from manually hunting for leaks to verifying flagged lines, and it standardizes a discipline (trial counting, point-in-time checks) that is usually left to individual honesty. Benchmarked on 30 synthetic backtests with seeded bugs, the hybrid detector reaches [precision/recall to be measured] versus [AST-only baseline].

*(Les crochets restent des crochets tant que le benchmark n'a pas tourné — règle 4.)*

### 4.12 Reflections (EN - squelette ; **ne jamais soumettre tel quel** : à remplir avec le vécu réel du DEVLOG)

> - Key learnings: [what the seeded-bug benchmark taught about LLM false positives; what deflation did to my own past results; where AST rules were surprisingly sufficient].
> - Challenges: [grounding LLM findings to real line numbers; choosing CSCV block length under autocorrelation; keeping the report readable for non-statisticians].
> - Future enhancements: [Glasserman–Lin anonymization module for LLM-generated signals; CI/CD integration as a pre-merge check; support for multi-asset transaction-cost models].

### 4.13 Questions de défense probables (à driller à voix haute)

1. Dérive le DSR au tableau. Pourquoi E[max SR] croît-il en √(2·ln N) ?
2. Différence exacte PSR vs DSR ? Que devient le DSR si les essais sont corrélés (N effectif) ?
3. Limites du CSCV ? Pourquoi des blocs et pas un simple split ?
4. Pourquoi hybride AST+LLM plutôt que LLM seul ? Chiffres de ton benchmark à l'appui.
5. Comment garantis-tu qu'un finding LLM ne cite pas une ligne inexistante ?
6. Déroule ton cas 1.93 → 0.92 pas à pas, unités explicites à chaque étape.
7. Que détecte le test d'anonymisation - et que ne détecte-t-il pas ?

### 4.14 Références (Projet B)

- Bailey, D.H., López de Prado, M. (2012), *The Sharpe Ratio Efficient Frontier* - PSR.
- Bailey, D.H., López de Prado, M. (2014), *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality*, Journal of Portfolio Management.
- Bailey, Borwein, López de Prado, Zhu (2014/2017), *The Probability of Backtest Overfitting*, Journal of Computational Finance - CSCV/PBO.
- Harvey, C., Liu, Y. (2015), *Backtesting*, JPM ; Harvey, Liu, Zhu (2016), *…and the Cross-Section of Expected Returns* - tests multiples, seuil t≈3.
- López de Prado, M. (2018), *Advances in Financial Machine Learning*, Wiley - chapitres backtesting, CPCV.
- Glasserman, P., Lin, C. (2023), *Assessing Look-Ahead Bias in Stock Return Predictions Generated by GPT Sentiment Analysis* - protocole d'anonymisation.
- Sarkar, S. (2024, SSRN), travaux sur le lookahead bias des LLM pré-entraînés - à re-vérifier
