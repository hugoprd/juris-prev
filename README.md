# juris-prev

**EN-US | [PT-BR](README-PTBR.md)**

> **Abstract.** **JurisPrev** is an early-warning machine learning project that predicts whether a law student is at risk of failing the Bar exam (`target_risk = 1 - pass_bar`) right after finishing their first year. Using the Law School Admission Council (LSAC) dataset (`22,407` students across `39` raw variables), we remove all post-Year-1 variables to prevent data leakage, apply a stratified split (`65%` train = `14,564`, `25%` validation, `10%` test), and execute train-only imputation, feature engineering (`academic_momentum` and `lsat_ugpa_ratio`), one-hot encoding, and z-score standardization. Exploratory analysis on the training set reveals a `7.20%` baseline failure rate (`1,049` at-risk students), strong predictive signal from first-year GPA (`zfygpa`, $r = -0.355$, Welch's $p < 10^{-300}$) and `LSAT` ($r = -0.274$), and pronounced socioeconomic and racial disparities ($\chi^2$ $p < 10^{-34}$). Sensitive attributes (`race1` and `gender`) are kept separate from model inputs and used to audit racial fairness via Statistical Parity Difference (`SPD Gap`) and Equal Opportunity (`TPR Gap`). Preliminary validation experiments show strong ranking performance (`ROC-AUC` up to `0.849` without sensitive attributes), setting the stage for stratified 5-fold cross-validation and final test evaluation.

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
* **How Machine Learning helps.** **JurisPrev** acts as an early-warning system at the end of the **1st year of law school**. By flagging students with high failure risk early on, coordinators can proactively allocate tutoring, study workshops, and mentoring while auditing whether predictions treat demographic groups fairly.
* **Business success criteria.**
  * Catch at least **`75%` to `80%` of truly at-risk students** ($\text{Recall} \ge 0.75$ for `target_risk = 1`, vs. a `7.20%` random baseline).
  * Reach a **`ROC-AUC` above `0.85`**, substantially outperforming a majority-class baseline (`ROC-AUC = 0.500`).
  * Keep racial disparities low when sensitive features are excluded ($|\text{SPD Gap}| < 0.15$ and $|\text{TPR Gap}| < 0.10$).

# 2. Literature Review

* **Wightman (1998) — LSAC National Longitudinal Bar Passage Study.**
  * **Approach.** Collected and analyzed data from 163 U.S. law schools (1991 cohort) using statistical analysis and logistic regression to evaluate how admission credentials and law school grades predict Bar exam passage.
  * **Main findings.** First-year law school GPA (`zfygpa`) and `LSAT` scores were the strongest predictors of passing the Bar, outperforming undergraduate GPA (`UGPA`).
  * **Fairness limitation.** The study documented large racial gaps in pass rates ($\approx 92\%$ overall for White students vs. $\approx 78\%$ for Black students), but it did not apply algorithmic fairness metrics or bias mitigation techniques.

* **Friedler et al. (2019) — Comparative Study of Fairness-Enhancing Interventions.**
  * **Approach.** Used the LSAC Bar Passage dataset as a classification benchmark to test standard models (Logistic Regression, Decision Trees, Naive Bayes) alongside pre-, in-, and post-processing debiasing methods.
  * **Main findings.** Standard classifiers achieved high overall accuracy ($\approx 88\%\text{--}90\%$ driven by the high pass base rate) but produced significant **Disparate Impact** across racial groups when trained only for accuracy.
  * **Takeaway for JurisPrev.** High overall accuracy can hide severe group disparities, making fairness auditing mandatory on this dataset.

# 3. Methodology

## 3.1. Exploratory Data Analysis (EDA)

The dataset comes from Kaggle (`danofer/law-school-admissions-bar-passage`) and contains **22,407 student records** across **39 raw columns** (`19,096` missing cells overall).

To simulate a realistic prediction at the **end of Year 1**, we dropped 28 columns: post-Year-1 variables that would cause data leakage (`bar`, `bar1`, `bar2`, `bar_passed`, 3rd-year GPA `zgpa`, `decile3`, `grad`, `Dropout`, `dnn_bar_pass_prediction`), IDs, and redundant/derived columns (`gpa`, `sex`, `male`, `parttime`, `race`, `race2`, `DOB_yr`, racial dummies, `index6040`, `decile1`). The target was inverted to **`target_risk = 1 - pass_bar`** (`1` = failed / at risk, `0` = passed). All EDA statistics below were computed strictly on the processed training partition (`data/processed/pre_train.csv`, **$N = 14,564$**).

| Feature | Role | Type | Description |
| :--- | :--- | :--- | :--- |
| `lsat` | Input | Continuous | Law School Admission Test score at entry (`11.0`–`48.0`). |
| `ugpa` | Input | Continuous | Undergraduate GPA prior to law school (`0.0`–`4.0`). |
| `zfygpa` | Input | Continuous | Standardized 1st-year law school GPA (institutional z-score). |
| `tier` | Input | Ordinal | Law school prestige tier (`1` = highest to `6` = lowest). |
| `cluster` | Input | Nominal | School grouping code (`1`–`6`). |
| `fulltime` | Input | Binary | Enrollment mode (`1` = full-time, `0` = part-time). |
| `fam_inc` | Input / Audit | Ordinal | Family income bracket (`1` = lowest quintile to `5` = highest). |
| `age` | Input / Audit | Discrete | Age at law school entry (in years). |
| `race1` | Sensitive (Audit) | Nominal | Race/ethnicity (`white`, `black`, `hisp`, `asian`, `other`). Kept out of $X$. |
| `gender` | Sensitive (Audit) | Binary | Gender (`male`, `female`). Kept out of $X$. |
| `target_risk` | **Target** | Binary | Bar exam failure risk (`1` = failed / at risk, `0` = passed). |

### Descriptive Statistics (Numerical Variables — `pre_train.csv`)

| Variable | Count | Mean | Median | Std | Min | Q1 | Q3 | Max | IQR Outliers |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `lsat` | 14,564 | 36.737 | 37.000 | 5.424 | 11.000 | 33.000 | 41.000 | 48.000 | 159 (`1.09%`) |
| `ugpa` | 14,564 | 3.214 | 3.200 | 0.402 | 1.500 | 2.900 | 3.500 | 4.000 | 97 (`0.67%`) |
| `zfygpa` | 13,787 | 0.104 | 0.130 | 0.922 | -3.350 | -0.510 | 0.740 | 3.250 | 62 (`0.45%`) |
| `age` | 14,515 | 25.996 | 24.000 | 5.497 | 17.000 | 22.000 | 28.000 | 68.000 | 956 (`6.59%`) |

### Frequencies & Failure Rates (Categorical Variables — `pre_train.csv`)

* **Target Distribution (`target_risk`).** The classes are strongly imbalanced—**`92.80%` (`13,515` students)** passed (`target_risk = 0`), while **`7.20%` (`1,049` students)** failed (`target_risk = 1`). Raw accuracy is therefore misleading, requiring stratified splits, class weighting, `ROC-AUC`, and `PR-AUC`.
* **Institutional Tier (`tier`) & Cluster (`cluster`).** Most students attend mid-tier institutions (`tier 3`: `37.10%`, failure rate `6.62%`; `tier 4`: `23.43%`, failure rate `9.17%`), while failure rates rise sharply in lower-prestige strata (`tier 5`: `13.51%`; `tier 6`: `16.67%`; `cluster 6`: `24.37%` vs. `tier 1`: `1.53%`).
* **Enrollment Mode (`fulltime`).** **`91.75%` (`13,363`)** are full-time students (`6.59%` failure rate) vs. **`8.25%` (`1,201`)** part-time students (`13.99%` failure rate—more than double).
* **Family Income (`fam_inc`).** Concentrated in brackets `4` (`41.60%`, `5.46%` failure) and `3` (`26.68%`, `7.90%` failure), with failure risk rising monotonically as income decreases (`bracket 5`: `4.70%` $\rightarrow$ `bracket 2`: `11.98%` $\rightarrow$ `bracket 1`: `17.62%`).
* **Race (`race1`) & Gender (`gender`).** `race1` is `83.56%` `white` (`4.86%` failure), `6.06%` `black` (`26.64%` failure), `4.55%` `hisp` (`14.20%` failure), `4.27%` `asian` (`9.97%` failure), and `1.50%` `other` (`12.33%` failure). `gender` is `56.21%` `male` (`6.99%` failure) and `43.77%` `female` (`7.48%` failure).

### Encoding Anomalies, Outliers, Missing Values & Correlations

* **Encoding Anomalies & Outliers.** In the raw CSV, `age` contained negative numbers representing birth-year offsets from 1900 relative to the 1991 entering cohort ($\text{mean} = -61.34$, $\text{min} = -72.0$). Converting offsets (`91 + age`) restored realistic ages (`17.0` to `68.0`, median `24.0`). The `956` upper-tail IQR outliers in `age` ($> 37$ years) represent legitimate mature students, while the low-score outliers in `lsat` (`159`), `ugpa` (`97`), and `zfygpa` (`62`) represent genuine academic underperformance carrying strong risk signal and were preserved.
* **Missing Values Analysis.** Total missing cells in `pre_train.csv` equal `1,023` across five columns: `zfygpa` (`777` missing, `5.335%`), `fam_inc` (`184` missing, `1.263%`), `age` (`49` missing, `0.336%`), `race1` (`10` missing, `0.069%`), and `gender` (`3` missing, `0.021%`). Checking missingness across racial groups showed that `zfygpa` missingness is slightly higher among Black (`7.82%`), Other (`7.76%`), and Hispanic (`6.95%`) students than White students (`5.05%`), motivating train-only median/mode imputation rather than row deletion so minority groups are not disproportionately discarded.
* **Correlations & Multicollinearity.** The strongest correlations with `target_risk` are **`zfygpa` ($r = -0.3548$)**, **`lsat` ($r = -0.2738$)**, **`ugpa` ($r = -0.1684$)**, and **`tier` ($r = +0.1303$)**, confirming that lower first-year and admission grades strongly increase failure risk. Among predictor pairs, the highest correlations are `lsat` vs. `zfygpa` ($r = 0.4054$), `fulltime` vs. `age` ($r = -0.3738$), `lsat` vs. `tier` ($r = -0.3714$), and `lsat` vs. `ugpa` ($r = 0.3388$). Because all inter-feature correlations are well below $|r| = 0.90$ (and redundant indices like `index6040` and `decile1` were dropped upfront), there is no severe multicollinearity.

## 3.2. Hypothesis Testing

We tested our domain assumptions on `data/processed/pre_train.csv` at significance level **$\alpha = 0.05$**:

| Hypothesis | $H_0$ vs. $H_1$ | Statistical Test & Result | Practical Conclusion |
| :--- | :--- | :--- | :--- |
| **H1: 1st-Year Grades (`zfygpa`) vs. Bar Risk** | **$H_0$.** `zfygpa` is equal between passing (`0`) and at-risk (`1`) students.<br>**$H_1$:** At-risk students (`1`) have significantly lower `zfygpa`. | **Welch's t-test (one-sided).** $t = 44.3316$, $p < 10^{-300}$<br>**Mann-Whitney U:** $U = 10{,}806{,}772.5$, $p < 10^{-300}$<br>**Decision:** **Reject $H_0$** | At-risk students have a mean/median `zfygpa` of **`-1.012` / `-1.010`** vs. **`+0.193` / `+0.200`** for passing students ($>1.2\sigma$ gap), proving Year-1 grades provide strong signal for early intervention. |
| **H2a: Family Income (`fam_inc`) vs. Bar Risk** | **$H_0$.** `target_risk` is independent of `fam_inc`.<br>**$H_1$:** Failure rates vary across income brackets. | **Pearson's Chi-Square:** $\chi^2 = 169.3422$ ($\text{df} = 4$), $p = 1.45 \times 10^{-35}$<br>**Decision:** **Reject $H_0$** | Failure rates rise from **`4.70%`** in the highest income quintile (`5`) to **`17.62%`** in the lowest quintile (`1`), confirming socioeconomic vulnerability. |
| **H2b: Racial Group (`race1`) vs. Bar Risk** | **$H_0$.** `target_risk` is independent of `race1`.<br>**$H_1$:** Failure rates vary across racial groups. | **Pearson's Chi-Square:** $\chi^2 = 665.2496$ ($\text{df} = 4$), $p = 1.19 \times 10^{-142}$<br>**Decision:** **Reject $H_0$** | Failure rates reach **`26.64%`** for Black students and **`14.20%`** for Hispanic students vs. **`4.86%`** for White students, making fairness auditing essential. |

## 3.3. Data Biases

* **Historical & Socioeconomic Bias.** Admission scores (`LSAT`, `UGPA`) and family income (`fam_inc`) reflect unequal access to test preparation, educational funding, and financial stability prior to and during law school. This is empirically visible in the steep failure gradient across income brackets (`17.62%` in `fam_inc = 1` vs. `4.70%` in `fam_inc = 5`) and institutional tiers (`16.67%` in `tier 6` vs. `1.53%` in `tier 1`).
* **Sampling & Survivorship Bias.** The dataset covers a 1991 U.S. cohort heavily skewed toward White (`83.56%`) and full-time (`91.75%`) students, and excludes students who dropped out before sitting for the Bar exam. Additionally, `zfygpa` is slightly more frequently missing for Black (`7.82%`) and Hispanic (`6.95%`) students than White students (`5.05%`).
* **Sensitive Attributes & Proxy Variables.** `race1` and `gender` are excluded from the model's feature matrix $X$ (*Fairness Through Unawareness*) and saved separately in `sensitive_*.csv`. However, features like `fam_inc`, `lsat`, `tier`, and `cluster` correlate with race and income (`lsat` correlates with `tier` at $r = -0.3714$) and can act as **proxies**, causing models to flag minority students at higher rates if left unchecked.
* **Real-World Impact.** Since **JurisPrev** assigns helpful tutoring rather than punishment, **False Negatives** (missing an at-risk student) are much more costly than **False Positives** (giving extra tutoring to a student who would pass anyway).

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
   * pre_train.csv (65% = 14,564) | pre_val.csv (25% = 5,602) | pre_test.csv (10% = 2,241)
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
1. **Preprocessing & Before/After Comparison:** Stateless cleaning runs before splitting so `pre_train.csv` stays human-readable for EDA. All statistical steps (`SimpleImputer` with median/mode, `StandardScaler`, and `OneHotEncoder`) are fitted **only on `pre_train.csv`** so regularization penalizes all features fairly across scales and tree models do not treat `cluster` IDs as ordered numbers.

| Stage | Files | Split | Records | Total Columns | Missing Values | Format |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Raw** | `data/raw/bar_pass_prediction.csv` | 100% | 22,407 | 39 | 19,096 | Raw scales, negative ages, leakage cols |
| **Processed** | `data/processed/pre_{train,val,test}.csv` | 65% / 25% / 10% | 14,564 (train) | 11 | 1,023 | Clean units, no leakage, `target_risk` |
| **Refined** | `data/refined/{train,val,test}.csv` | 65% / 25% / 10% | 14,564 (train) | 13 (Normal) / 18 (+Sens) | 0 | Scaled ($\mu=0, \sigma=1$), `{0,1}`, one-hot |

2. **Feature Engineering.** Created two domain features using only training statistics:
   * `academic_momentum`: $\text{zfygpa} - \frac{z_{\text{lsat}} + z_{\text{ugpa}}}{2}$ — measures whether a student's grades improved or dropped in Year 1 compared to their entry profile.
   * `lsat_ugpa_ratio`: $\frac{\text{lsat}}{\text{ugpa} + 0.5}$ — separates strong standardized test-takers from students with high coursework grades but weaker exam scores.
   *(A third candidate `zfygpa * tier` was discarded because multiplying a zero-centered z-score by tiers `1..6` flips the direction for above-average students.)*
3. **Train / Val / Test Split:** Stratified split (`65%` train, `25%` validation, `10%` test, `SEED = 42`). `test.csv` remains completely untouched until final evaluation.
4. **Baseline Model.** `DummyClassifier(strategy="most_frequent")`, which always predicts `0` (pass), yielding `92.80%` accuracy on `pre_train`, `0.00` Recall/F1 on the risk class, and `0.500` `ROC-AUC`.
5. **Preliminary Model Comparison (`train` $\rightarrow$ `val`).** Tested five classifiers across **Normal** vs. **With Sensitive (`+ race1, gender`)** scenarios:
   * **Logistic Regression (L2, `class_weight="balanced"`).** Best validation ranking (**`ROC-AUC = 0.849`** Normal / **`0.850`** +Sensitive), plus direct interpretability via Odds Ratios.
   * **MLP Neural Network (`(64, 32)`).** **`ROC-AUC = 0.844`** Normal / **`0.846`** +Sensitive.
   * **Random Forest (`max_depth=8`, `class_weight="balanced"`).** **`ROC-AUC = 0.843`** Normal / **`0.842`** +Sensitive.
   * **XGBoost (`max_depth=4`, `scale_pos_weight`).** **`ROC-AUC = 0.838`** Normal / **`0.840`** +Sensitive.
   * **HistGradientBoosting (`max_depth=5`, `class_weight="balanced"`).** **`ROC-AUC = 0.836`** Normal / **`0.840`** +Sensitive.
   * *Takeaway.* Adding `race1` and `gender` barely changed `ROC-AUC` ($\le 0.004$), supporting keeping them out of the model.
6. **Feature Selection & Upcoming Tuning.** Combines domain filtering with embedded regularization (L1/L2/Elastic Net) and feature importance. Next, we will run **Stratified 5-Fold Cross-Validation** with Grid Search on the training set to tune hyperparameters and the probability decision threshold.

## 3.5. Evaluation & Fairness Framework

* **Classification Metrics.** Because `target_risk = 1` represents **`7.20%`** of the dataset, we evaluate **`ROC-AUC`** and **`PR-AUC`** for ranking quality alongside **`Recall`**, **`Precision`**, and **`F1-Score`** on the positive risk class. **False Negatives** (missing an at-risk student) are treated as more costly than **False Positives** (assigning extra tutoring).
* **Fairness Metrics (`black` unprivileged group vs. `white` privileged reference group via `sensitive_*.csv`).**
  * **Statistical Parity Difference (`SPD Gap = Selection Rate_Black - Selection Rate_White`).** Checks if both groups are flagged for tutoring at similar rates (`0.000` = parity; bounded in `[-1, 1]` to avoid division-by-zero instability).
  * **Equal Opportunity Gap (`TPR Gap = Recall_Black - Recall_White`).** Checks whether truly at-risk students are detected equally well across both racial groups (`0.000` = equal opportunity).