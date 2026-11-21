# juris-prev

**EN-US | [PT-BR](README-PTBR.md)**

**Abstract.** 

JurisPrev is an early-warning machine learning project that predicts whether a law student is at risk of failing the Bar exam (`target_risk = 1 - pass_bar`) right after finishing their first year. Using the Law School Admission Council (LSAC) dataset (22,407 students), we remove all post-Year-1 variables to prevent data leakage, split the data (`65%` train, `25%` validation, `10%` test), and apply train-only imputation, feature engineering (academic_momentum and lsat_ugpa_ratio), encoding, and scaling. Sensitive attributes (`race1` and `gender`) are kept separate from model inputs and used to audit racial fairness via Statistical Parity Difference (`SPD Gap`) and Equal Opportunity (`TPR Gap`). Early validation experiments show strong ranking performance (`ROC-AUC` up to `0.849` without sensitive attributes), setting the stage for cross-validated tuning and final test evaluation.

# Table of Contents
* [1. Introduction](#1-introduction)
* [2. Literature Review](#2-literature-review)
* [3. Methodology](#3-methodology)
    * [3.1. Exploratory Data Analysis (EDA)](#31-exploratory-data-analysis-eda)
    * [3.2. Hypothesis Testing](#32-hypothesis-testing)
    * [3.3. Data Biases](#33-data-biases)
    * [3.4. Data Processing & Pipeline Methodology](#34-data-processing--pipeline-methodology)
    * [3.5. Evaluation & Fairness Framework](#35-evaluation--fairness-framework)

----------

# 1. Introduction

* **Why this problem matters.** Failing the Bar exam delays graduation-to-career entry, creates financial stress for students, and hurts law school accreditation metrics. Usually, extra prep support is offered too late—only in the final semester or after a student has already failed.

* **Who the client is.** Law school academic coordinators, student advising teams, and institutional Deans who manage limited tutoring budgets.

* **How Machine Learning helps.** JurisPrev acts as an early-warning system at the end of the 1st year of law school. By flagging students with high failure risk early on, coordinators can proactively allocate tutoring, study workshops, and mentoring while auditing whether predictions treat demographic groups fairly.

* **Business success criteria.**

    * Catch at least `75%` to `80%` of truly at-risk students (`Recall >= 0.75` for `target_risk = 1`).

    * Reach a `ROC-AUC` above `0.85`, beating a simple majority-class baseline.

    * Keep racial disparities low when sensitive features are excluded (`|SPD Gap| < 0.15` and `|TPR Gap| < 0.10`).

# 2. Literature Review

* **Wightman (1998) — LSAC National Longitudinal Bar Passage Study.**

    * **Approach.** Collected and analyzed data from 163 U.S. law schools (1991 cohort) using statistical analysis and logistic regression to see how admission scores and law school grades predict Bar exam passage.

    * **Main findings.** First-year law school GPA (zfygpa) and LSAT scores were the strongest predictors of passing the Bar, outperforming undergraduate GPA (UGPA).

    * **Fairness limitation.** The study documented large racial gaps in pass rates (`~92%` overall for White students vs. `~78%` for Black students), but it did not apply any algorithmic fairness metrics or bias mitigation techniques.

* **Friedler et al. (2019) — Comparative Study of Fairness-Enhancing Interventions.**

    * **Approach.** Used the LSAC Bar Passage dataset as a classification benchmark to test standard models (Logistic Regression, Decision Trees, Naive Bayes) alongside pre-, in-, and post-processing debiasing methods.

    * **Main findings.** Standard classifiers achieved high overall accuracy (`~88%–90%` due to the `~90%` pass rate) but produced significant Disparate Impact across racial groups when trained only for accuracy.

    * **Takeaway for JurisPrev.** High overall accuracy can hide severe group disparities, making fairness auditing mandatory on this dataset.

# 3. Methodology

## 3.1. Exploratory Data Analysis (EDA)

The dataset comes from Kaggle (danofer/law-school-admissions-bar-passage) and contains **22,407 student records** across **39 raw columns**.

To simulate a realistic prediction at the end of Year 1, were dropped 28 columns: post-Year-1 variables that would cause data leakage (`bar`, `bar1`, `bar2`, `bar_passed`, `3rd-year GPA zgpa`, `decile3`, `grad`, `Dropout`, `dnn_bar_pass_prediction`), `IDs`, and redundant/derived columns (`gpa`, `sex`, `male`,  `parttime`, `race`, `race2`, `DOB_yr`, `racial dummies`, `index6040`, `decile1`). The target was inverted to `target_risk = 1 - pass_bar` (1 = failed / at risk, 0 = passed).

| Feature | Role | Type | Description |
| :--- | :--- | :--- | :--- |
| `lsat` | Input | Continuous | Law School Admission Test score at entry ($11\text{--}48$). |
| `ugpa` | Input | Continuous | Undergraduate GPA prior to law school (`0.0`–`4.0`). |
| `zfygpa` | Input | Continuous | Standardized 1st-year law school GPA (z-score). |
| `tier` | Input | Ordinal | Law school prestige tier (`1` = highest to `6` = lowest). |
| `cluster` | Input | Nominal | School grouping code (`1`–`6`). |
| `fulltime` | Input | Binary | Enrollment mode (`1` = full-time, `0` = part-time). |
| `fam_inc` | Input / Audit | Ordinal | Family income bracket (`1` = lowest to `5` = highest). |
| `age` | Input / Audit | Discrete | Age at law school entry (in years). |
| `race1` | Sensitive (Audit) | Nominal | Race/ethnicity (`white`, `black`, `hisp`, `asian`, `other`). Kept out of $X$. |
| `gender` | Sensitive (Audit) | Binary | Gender (`male`, `female`). Kept out of $X$. |
| `target_risk` | **Target** | Binary | Bar exam failure risk (`1` = failed / at risk, `0` = passed). |

**Key data observations:**
* **Class Imbalance.** About **90%** of students passed (`0`) and **10%** failed (`1`). Because of this imbalance, raw accuracy is misleading, so we use stratified splits, class weighting, `ROC-AUC`, and `PR-AUC`.
* **Encoding Fixes & Outliers.** In the raw CSV, `age` had negative numbers representing offsets from 1900 for the 1991 cohort (e.g., `-65` = born in 1965 = age 26), and `fulltime` was coded as `{1, 2}`. We converted `age` (`91 + age`), mapped `fulltime` to `{1, 0}`, and clipped impossible values (`age` to `[17, 85]`, `ugpa` to `[0.0, 4.0]`), while keeping low `lsat`/`zfygpa` scores since they carry real failure risk signals.
* **Missing Values & Collinearity.** Minor missingness appears in `zfygpa`, `fam_inc`, `age`, `race1`, and `gender`. Composite indices (`index6040`) and class deciles (`decile1`) were removed early to avoid multicollinearity with `lsat`, `ugpa`, and `zfygpa`. *(Exact summary statistics and plots from `pre_train.csv` are generated in the EDA notebook.)*

## 3.2. Hypothesis Testing

We test two core assumptions on `data/processed/pre_train.csv` using significance level **$\alpha = 0.05$**:

| Hypothesis | $H_0$ vs. $H_1$ | Statistical Test | Practical Meaning |
| :--- | :--- | :--- | :--- |
| **H1: 1st-Year Grades vs. Bar Risk** | **$H_0$:** `zfygpa` distribution is the same for `target_risk = 0` and `1`.<br>**$H_1$:** At-risk students (`1`) have significantly lower `zfygpa`. | **Welch's t-test & Mann-Whitney U** (one-sided, handles unequal group sizes/variances). | Confirms whether 1st-year grades give enough signal to trigger tutoring early. |
| **H2: Income & Race vs. Bar Risk** | **$H_0$:** Failure risk (`target_risk`) is independent of `fam_inc` and `race1`.<br>**$H_1$:** Failure rates differ significantly across income and racial groups. | **Chi-Square ($\chi^2$) Test of Independence** (categorical variables vs. binary target). | Quantifies structural disparities in the data and motivates fairness auditing. |

*(Exact test statistics and $p$-values will be added once the EDA statistical notebook is finalized.)*

## 3.3. Data Biases

* **Historical & Socioeconomic Bias.** Admission metrics (`LSAT`, `UGPA`) and family income (`fam_inc`) reflect unequal access to prep courses, school funding, and financial stability before and during law school.
* **Sampling & Survivorship Bias.** The dataset covers a 1991 U.S. cohort (mostly White and full-time students) and excludes students who dropped out before taking the Bar exam.
* **Sensitive Attributes & Proxy Variables.** `race1` and `gender` are excluded from the model's feature matrix $X$ (*Fairness Through Unawareness*) and saved separately in `sensitive_*.csv`. However, features like `fam_inc`, `lsat`, `tier`, and `cluster` correlate with race and income and can act as **proxies**, causing models to flag minority students at higher rates if left unchecked.
* **Real-World Impact.** Since **JurisPrev** assigns helpful tutoring rather than punishment, **False Negatives** (missing a struggling student) are worse than **False Positives** (giving extra tutoring to a student who would pass anyway).

## 3.4. Data Processing & Pipeline Methodology

The pipeline is split into modular scripts (`process_data.py`, `refine_data.py`, and `tests/notebooks/model_benchmark.ipynb`) to prevent any data leakage:

```text
[1. Raw Data] data/raw/bar_pass_prediction.csv (22,407 rows, 39 cols)
       |
       v
[2. Stateless Cleaning] process_data.py
   * Define target_risk = 1 - pass_bar; drop post-Year-1 leakage -> 10 features
   * Fix negative age offsets (91 + age) & map fulltime {1, 0}
       |
       v
[3. Stratified Split (SEED=42)] data/processed/
   * pre_train.csv (65%) | pre_val.csv (25%) | pre_test.csv (10% - locked)
       |
       v
[4. Train-Only Refinement] refine_data.py -> data/refined/
   * Clip age [17, 85] & ugpa [0.0, 4.0]
   * Fit SimpleImputer (median / mode) ONLY on pre_train
   * Save sensitive_{train,val,test}.csv separately for fairness audits
   * Engineer academic_momentum & lsat_ugpa_ratio using train mean/std
   * StandardScaler (cont/ord) + OneHotEncoder(drop='first') on cluster
       |
       v
[5. Model & Fairness Benchmark] tests/notebooks/model_benchmark.ipynb
   * Train on train.csv -> evaluate on val.csv (Normal vs. +Sensitive)
   * Compare Baseline, Logistic Regression, RF, HistGB, XGBoost, MLP
       |
       v
[6. Next Steps] Stratified 5-Fold CV Tuning & Final Evaluation on test.csv
```

**Key methodological decisions:**
1. **Preprocessing & Before/After Comparison:** Stateless cleaning runs before splitting so `pre_train.csv` stays human-readable for EDA. All statistical steps (`SimpleImputer` with median/mode, `StandardScaler`, and `OneHotEncoder`) are fitted **only on `pre_train.csv`** so regularization penalizes all features fairly and tree models do not treat `cluster` IDs as ordered numbers.

| Stage | Files | Split | Features in $X$ | Missing Values | Format |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Raw** | `data/raw/bar_pass_prediction.csv` | 100% | 38 + `pass_bar` | Yes | Raw scales, negative ages, leakage cols |
| **Processed** | `data/processed/pre_{train,val,test}.csv` | 65% / 25% / 10% | 10 | Yes | Clean units, no leakage, `target_risk` |
| **Refined** | `data/refined/{train,val,test}.csv` | 65% / 25% / 10% | 12 (Normal) / 17 (+Sens) | 0 | Scaled (`0` mean, `1` std), `{0,1}`, one-hot |

2. **Feature Engineering.** Created two domain features using only training statistics:
   * `academic_momentum`: $\text{zfygpa} - \frac{z_{\text{lsat}} + z_{\text{ugpa}}}{2}$ — measures whether a student's grades improved or dropped in Year 1 compared to their entry profile.
   * `lsat_ugpa_ratio`: $\frac{\text{lsat}}{\text{ugpa} + 0.5}$ — separates strong standardized test-takers from students with high coursework grades but weaker exam scores.
   *(A third candidate `zfygpa * tier` was discarded because multiplying a zero-centered z-score by tiers `1..6` flips the direction for above-average students.)*
3. **Train / Val / Test Split:** Stratified split (`65%` train, `25%` validation, `10%` test, `SEED = 42`). `test.csv` stays untouched until the final evaluation.
4. **Baseline Model.** `DummyClassifier(strategy="most_frequent")`, which always predicts `0` (pass), yielding `0.00` Recall/F1 on the risk class and `0.500` ROC-AUC.
5. **Preliminary Model Comparison (`train` $\rightarrow$ `val`):** Tested five classifiers across **Normal** vs. **With Sensitive (`+ race1, gender`)** scenarios:
   * **Logistic Regression (L2, `class_weight="balanced"`).** Best validation ranking (**`ROC-AUC = 0.849`** Normal / **`0.850`** +Sensitive), plus direct interpretability via Odds Ratios.
   * **MLP Neural Network (`(64, 32)`).** **`ROC-AUC = 0.844`** Normal / **`0.846`** +Sensitive.
   * **Random Forest (`max_depth=8`, `class_weight="balanced"`).** **`ROC-AUC = 0.843`** Normal / **`0.842`** +Sensitive.
   * **XGBoost (`max_depth=4`, `scale_pos_weight`):** **`ROC-AUC = 0.838`** Normal / **`0.840`** +Sensitive.
   * **HistGradientBoosting (`max_depth=5`, `class_weight="balanced"`).** **`ROC-AUC = 0.836`** Normal / **`0.840`** +Sensitive.
   * *Takeaway:* Adding `race1` and `gender` barely changed `ROC-AUC` ($\le 0.004$), supporting keeping them out of the model.
6. **Feature Selection & Upcoming Tuning.** Combines domain filtering with embedded regularization (L1/L2/Elastic Net) and feature importance. Next, we will run **Stratified 5-Fold Cross-Validation** with Grid Search on the training set to tune hyperparameters and the decision threshold.

## 3.5. Evaluation & Fairness Framework

* **Classification Metrics.** Because `target_risk = 1` is $\approx 10\%$ of the data, we evaluate **`ROC-AUC`** and **`PR-AUC`** for ranking quality alongside **`Recall`**, **`Precision`**, and **`F1-Score`** on the positive risk class. **False Negatives** (missing an at-risk student) are treated as more costly than **False Positives** (assigning extra tutoring).
* **Fairness Metrics (`black` vs. `white` groups via `sensitive_*.csv`).**
  * **Statistical Parity Difference (`SPD Gap = Selection Rate_Black - Selection Rate_White`):** Checks if both groups are flagged for tutoring at similar rates (`0.000` = parity; bounded in `[-1, 1]` to avoid division-by-zero instability).
  * **Equal Opportunity Gap (`TPR Gap = Recall_Black - Recall_White`).** Checks whether truly at-risk students are detected equally well across both racial groups (`0.000` = equal opportunity).