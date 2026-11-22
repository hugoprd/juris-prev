# juris-prev

**[EN-US](README.md) | PT-BR**

> **Resumo.** O **JurisPrev** é um projeto de aprendizado de máquina voltado para alerta precoce que prevê se um estudante de Direito corre risco de reprovar no exame da ordem (`target_risk = 1 - pass_bar`) logo após concluir o primeiro ano de curso. Utilizando a base do *Law School Admission Council* (LSAC, com `22.407` estudantes e `39` variáveis brutas), removemos todas as variáveis posteriores ao 1º ano para evitar vazamento de dados (*data leakage*), aplicamos uma divisão estratificada (`65%` treino = `14.564`, `25%` validação e `10%` teste) e executamos imputação, engenharia de atributos (`academic_momentum` e `lsat_ugpa_ratio`), codificação *one-hot* e padronização *z-score* ajustadas exclusivamente no treino. A análise exploratória no conjunto de treino revela uma taxa base de reprovação de `7.20%` (`1.049` estudantes em risco), forte sinal preditivo da média do 1º ano (`zfygpa`, $r = -0.355$, $p < 10^{-300}$ no teste de Welch) e do `LSAT` ($r = -0.274$), além de disparidades socioeconômicas e raciais acentuadas ($\chi^2$ com $p < 10^{-34}$). Os atributos sensíveis (`race1` e `gender`) são mantidos separados das entradas do modelo e utilizados para auditar a justiça algorítmica (*fairness*) por meio da Diferença de Paridade Demográfica (`SPD Gap`) e da Igualdade de Oportunidade (`TPR Gap`). Experimentos preliminares na validação demonstram forte capacidade de ranqueamento (`ROC-AUC` de até `0.849` sem atributos sensíveis), preparando a base para a validação cruzada estratificada de 5 *folds* e a avaliação final no conjunto de teste.

# Sumário
* [1. Introdução](#1-introdução)
* [2. Revisão da Literatura](#2-revisão-da-literatura)
* [3. Metodologia](#3-metodologia)
  * [3.1. Análise Exploratória da Base de Dados (EDA)](#31-análise-exploratória-da-base-de-dados-eda)
  * [3.2. Testes de Hipótese](#32-testes-de-hipótese)
  * [3.3. Vieses nos Dados](#33-vieses-nos-dados)
  * [3.4. Processamento de Dados e Metodologia do Pipeline](#34-processamento-de-dados-e-metodologia-do-pipeline)
  * [3.5. Avaliação e Estrutura de Fairness](#35-avaliação-e-estrutura-de-fairness)

----------

# 1. Introdução

* **Por que esse problema importa.** Reprovar no exame da ordem atrasa o ingresso do graduado no mercado de trabalho, gera desgaste financeiro para o estudante e prejudica os indicadores institucionais das faculdades de Direito. Normalmente, o suporte preparatório extra é oferecido tarde demais — apenas no último semestre ou depois que o aluno já reprovou.
* **Quem é o contratante.** Coordenações acadêmicas de cursos de Direito, equipes de orientação estudantil e direções institucionais que gerenciam orçamentos limitados de tutoria e monitoria.
* **Como o Machine Learning ajuda.** O **JurisPrev** atua como um sistema de alerta precoce ao final do **1º ano de Direito**. Ao sinalizar antecipadamente estudantes com alto risco de reprovação, a coordenação pode alocar de forma preventiva tutorias individuais, oficinas de estudo e mentoria, enquanto audita se as predições tratam os grupos demográficos de maneira justa.
* **Critérios de sucesso do negócio.**
  * Identificar pelo menos **`75%` a `80%` dos estudantes realmente em risco** ($\text{Recall} \ge 0.75$ para `target_risk = 1`, frente a uma taxa base aleatória de `7.20%`).
  * Alcançar um **`ROC-AUC` acima de `0.85`**, superando substancialmente um modelo de referência (*baseline*) da classe majoritária (`ROC-AUC = 0.500`).
  * Manter baixas disparidades raciais quando atributos sensíveis forem excluídos ($|\text{SPD Gap}| < 0.15$ e $|\text{TPR Gap}| < 0.10$).

# 2. Revisão da Literatura

* **Wightman (1998) — LSAC National Longitudinal Bar Passage Study.**
  * **Abordagem.** Coletou e analisou dados de 163 faculdades de Direito dos EUA (coorte de 1991) usando análise estatística e regressão logística para avaliar como credenciais de admissão e notas na graduação preveem a aprovação no exame da ordem.
  * **Principais resultados.** A nota padronizada do 1º ano de Direito (`zfygpa`) e a pontuação no `LSAT` foram os preditores mais fortes de aprovação, superando a média da graduação anterior (`UGPA`).
  * **Limitação quanto a fairness.** O estudo documentou grandes disparidades raciais nas taxas de aprovação ($\approx 92\%$ no geral para estudantes brancos contra $\approx 78\%$ para estudantes negros), mas não aplicou métricas de justiça algorítmica nem técnicas de mitigação de viés.

* **Friedler et al. (2019) — Comparative Study of Fairness-Enhancing Interventions.**
  * **Abordagem.** Utilizou a base *LSAC Bar Passage* como *benchmark* de classificação para testar modelos padrão (Regressão Logística, Árvores de Decisão e Naive Bayes) combinados com métodos de mitigação de viés no pré-processamento, processamento e pós-processamento.
  * **Principais resultados.** Classificadores padrão atingiram alta acurácia global ($\approx 88\%$ a $90\%$, impulsionada pela alta taxa base de aprovação), mas geraram forte **Impacto Desproporcional (*Disparate Impact*)** entre grupos raciais quando treinados apenas para maximizar acurácia.
  * **Lição para o JurisPrev.** Uma acurácia geral alta pode mascarar disparidades severas entre grupos, tornando obrigatória a auditoria de *fairness* nesta base.

# 3. Metodologia

## 3.1. Análise Exploratória da Base de Dados (EDA)

A base de dados foi obtida no Kaggle (`danofer/law-school-admissions-bar-passage`) e contém **22.407 registros de estudantes** distribuídos em **39 colunas brutas** (`19.096` células faltantes no total).

Para simular uma predição realista ao **final do 1º ano**, descartamos 28 colunas: variáveis posteriores ao 1º ano que causariam vazamento de dados (`bar`, `bar1`, `bar2`, `bar_passed`, nota acumulada do 3º ano `zgpa`, `decile3`, `grad`, `Dropout`, `dnn_bar_pass_prediction`), identificadores (`ID`) e colunas redundantes ou derivadas (`gpa`, `sex`, `male`, `parttime`, `race`, `race2`, `DOB_yr`, *dummies* raciais, `index6040`, `decile1`). A variável-alvo foi invertida para **`target_risk = 1 - pass_bar`** (`1` = reprovado / em risco, `0` = aprovado). Todas as estatísticas da EDA abaixo foram calculadas estritamente sobre a partição processada de treino (`data/processed/pre_train.csv`, **$N = 14.564$**).

| Atributo | Papel | Tipo | Descrição |
| :--- | :--- | :--- | :--- |
| `lsat` | Entrada | Contínuo | Pontuação no exame de admissão *Law School Admission Test* (`11.0` a `48.0`). |
| `ugpa` | Entrada | Contínuo | Média de notas da graduação anterior ao Direito (`0.0` a `4.0`). |
| `zfygpa` | Entrada | Contínuo | Média padronizada do 1º ano de Direito (*z-score* institucional). |
| `tier` | Entrada | Ordinal | Nível de prestígio da instituição (`1` = mais alto até `6` = mais baixo). |
| `cluster` | Entrada | Nominal | Código de agrupamento institucional (`1` a `6`). |
| `fulltime` | Entrada | Binário | Regime de estudo (`1` = tempo integral, `0` = meio período). |
| `fam_inc` | Entrada / Auditoria | Ordinal | Faixa de renda familiar (`1` = menor quintil até `5` = maior quintil). |
| `age` | Entrada / Auditoria | Discreto | Idade no ingresso da faculdade de Direito (em anos). |
| `race1` | Sensível (Auditoria) | Nominal | Raça/etnia (`white`, `black`, `hisp`, `asian`, `other`). Mantida fora de $X$. |
| `gender` | Sensível (Auditoria) | Binário | Gênero (`male`, `female`). Mantido fora de $X$. |
| `target_risk` | **Alvo** | Binário | Risco de reprovação no exame da ordem (`1` = reprovado / em risco, `0` = aprovado). |

### Estatísticas Descritivas (Variáveis Numéricas — `pre_train.csv`)

| Variável | Contagem | Média | Mediana | Desvio Padrão | Mínimo | Q1 | Q3 | Máximo | Outliers (IIQ) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `lsat` | 14.564 | 36.737 | 37.000 | 5.424 | 11.000 | 33.000 | 41.000 | 48.000 | 159 (`1.09%`) |
| `ugpa` | 14.564 | 3.214 | 3.200 | 0.402 | 1.500 | 2.900 | 3.500 | 4.000 | 97 (`0.67%`) |
| `zfygpa` | 13.787 | 0.104 | 0.130 | 0.922 | -3.350 | -0.510 | 0.740 | 3.250 | 62 (`0.45%`) |
| `age` | 14.515 | 25.996 | 24.000 | 5.497 | 17.000 | 22.000 | 28.000 | 68.000 | 956 (`6.59%`) |

### Frequências e Taxas de Reprovação (Variáveis Categóricas — `pre_train.csv`)

* **Distribuição do Alvo (`target_risk`).** As classes são fortemente desbalanceadas — **`92.80%` (`13.515` estudantes)** foram aprovados (`target_risk = 0`), enquanto **`7.20%` (`1.049` estudantes)** reprovaram (`target_risk = 1`). Por isso, a acurácia bruta é enganosa, exigindo divisão estratificada, ponderação de classes (*class weight*), `ROC-AUC` e `PR-AUC`.
* **Nível Institucional (`tier`) e Agrupamento (`cluster`).** A maioria dos estudantes frequenta instituições intermediárias (`tier 3`: `37.10%`, reprovação de `6.62%`; `tier 4`: `23.43%`, reprovação de `9.17%`), enquanto as taxas de reprovação sobem acentuadamente nos estratos de menor prestígio (`tier 5`: `13.51%`; `tier 6`: `16.67%`; `cluster 6`: `24.37%` contra `tier 1`: `1.53%`).
* **Regime de Estudo (`fulltime`).** **`91.75%` (`13.363`)** estudam em tempo integral (`6.59%` de reprovação) contra **`8.25%` (`1.201`)** em meio período (`13.99%` de reprovação — mais que o dobro).
* **Renda Familiar (`fam_inc`).** Concentrada nas faixas `4` (`41.60%`, `5.46%` de reprovação) e `3` (`26.68%`, `7.90%` de reprovação), com o risco subindo monotonicamente conforme a renda diminui (`faixa 5`: `4.70%` $\rightarrow$ `faixa 2`: `11.98%` $\rightarrow$ `faixa 1`: `17.62%`).
* **Raça (`race1`) e Gênero (`gender`).** `race1` distribui-se em `83.56%` `white` (`4.86%` de reprovação), `6.06%` `black` (`26.64%` de reprovação), `4.55%` `hisp` (`14.20%` de reprovação), `4.27%` `asian` (`9.97%` de reprovação) e `1.50%` `other` (`12.33%` de reprovação). `gender` divide-se em `56.21%` `male` (`6.99%` de reprovação) e `43.77%` `female` (`7.48%` de reprovação).

### Anomalias de Codificação, Outliers, Valores Faltantes e Correlações

* **Anomalias de Codificação e Outliers.** No CSV bruto, `age` continha números negativos representando deslocamentos do ano de nascimento em relação a 1900 para a coorte de ingresso de 1991 ($\text{média} = -61.34$, $\text{mínimo} = -72.0$). A conversão (`91 + age`) restaurou idades realistas (`17.0` a `68.0`, mediana `24.0`). Os `956` *outliers* superiores pelo critério IIQ em `age` ($> 37$ anos) representam estudantes maduros legítimos, enquanto os *outliers* inferiores em `lsat` (`159`), `ugpa` (`97`) e `zfygpa` (`62`) refletem baixo desempenho acadêmico real com forte sinal de risco e foram mantidos.
* **Análise de Valores Faltantes.** O total de células ausentes em `pre_train.csv` é de `1.023` em cinco colunas: `zfygpa` (`777` faltantes, `5.335%`), `fam_inc` (`184` faltantes, `1.263%`), `age` (`49` faltantes, `0.336%`), `race1` (`10` faltantes, `0.069%`) e `gender` (`3` faltantes, `0.021%`). A verificação por grupo racial mostrou que a ausência de `zfygpa` é ligeiramente maior entre estudantes negros (`7.82%`), outros (`7.76%`) e hispânicos (`6.95%`) do que entre brancos (`5.05%`), justificando a imputação por mediana/moda ajustada no treino em vez da exclusão de linhas, evitando descartar grupos minoritários desproporcionalmente.
* **Correlações e Multicolinearidade.** As maiores correlações com `target_risk` são **`zfygpa` ($r = -0.3548$)**, **`lsat` ($r = -0.2738$)**, **`ugpa` ($r = -0.1684$)** e **`tier` ($r = +0.1303$)**, confirmando que notas menores no 1º ano e na admissão elevam o risco de reprovação. Entre pares de preditores, as correlações mais altas ocorrem entre `lsat` e `zfygpa` ($r = 0.4054$), `fulltime` e `age` ($r = -0.3738$), `lsat` e `tier` ($r = -0.3714$) e `lsat` e `ugpa` ($r = 0.3388$). Como todas as correlações entre atributos estão bem abaixo de $|r| = 0.90$ (e índices redundantes como `index6040` e `decile1` foram descartados na extração), não há multicolinearidade severa.

## 3.2. Testes de Hipótese

Testamos nossas premissas de domínio sobre `data/processed/pre_train.csv` ao nível de significância **$\alpha = 0.05$**:

| Hipótese | $H_0$ vs. $H_1$ | Teste Estatístico e Resultado | Conclusão Prática |
| :--- | :--- | :--- | :--- |
| **H1: Notas do 1º Ano (`zfygpa`) vs. Risco de Reprovação** | **$H_0$.** `zfygpa` é igual entre estudantes aprovados (`0`) e em risco (`1`).<br>**$H_1$:** Estudantes em risco (`1`) possuem `zfygpa` significativamente menor. | **Teste t de Welch (unilateral):** $t = 44.3316$, $p < 10^{-300}$<br>**Mann-Whitney U:** $U = 10.806.772{,}5$, $p < 10^{-300}$<br>**Decisão:** **Rejeita $H_0$** | Estudantes em risco têm média/mediana de `zfygpa` de **`-1.012` / `-1.010`** contra **`+0.193` / `+0.200`** dos aprovados (diferença $> 1.2\sigma$), comprovando que o desempenho no 1º ano fornece sinal forte para intervenção precoce. |
| **H2a: Renda Familiar (`fam_inc`) vs. Risco de Reprovação** | **$H_0$.** `target_risk` é independente de `fam_inc`.<br>**$H_1$:** As taxas de reprovação variam entre faixas de renda. | **Qui-Quadrado de Pearson:** $\chi^2 = 169.3422$ ($\text{gl} = 4$), $p = 1.45 \times 10^{-35}$<br>**Decisão:** **Rejeita $H_0$** | A reprovação sobe de **`4.70%`** no quintil de renda mais alto (`5`) para **`17.62%`** no quintil mais baixo (`1`), confirmando vulnerabilidade socioeconômica. |
| **H2b: Grupo Racial (`race1`) vs. Risco de Reprovação** | **$H_0$.** `target_risk` é independente de `race1`.<br>**$H_1$:** As taxas de reprovação variam entre grupos raciais. | **Qui-Quadrado de Pearson:** $\chi^2 = 665.2496$ ($\text{gl} = 4$), $p = 1.19 \times 10^{-142}$<br>**Decisão:** **Rejeita $H_0$** | A reprovação atinge **`26.64%`** entre estudantes negros e **`14.20%`** entre hispânicos contra **`4.86%`** entre brancos, tornando essencial a auditoria de *fairness*. |

## 3.3. Vieses nos Dados

* **Viés Histórico e Socioeconômico.** Notas de admissão (`LSAT`, `UGPA`) e renda familiar (`fam_inc`) refletem desigualdades prévias de acesso a cursos preparatórios, financiamento educacional e estabilidade financeira antes e durante a graduação. Isso aparece empiricamente no gradiente acentuado de reprovação entre faixas de renda (`17.62%` em `fam_inc = 1` contra `4.70%` em `fam_inc = 5`) e níveis institucionais (`16.67%` em `tier 6` contra `1.53%` em `tier 1`).
* **Viés de Amostragem e Sobrevivência.** A base retrata uma coorte norte-americana de 1991 fortemente concentrada em estudantes brancos (`83.56%`) e em tempo integral (`91.75%`), além de excluir alunos que evadiram antes de prestar o exame da ordem. Adicionalmente, `zfygpa` apresenta taxa de ausência ligeiramente maior entre estudantes negros (`7.82%`) e hispânicos (`6.95%`) do que entre brancos (`5.05%`).
* **Atributos Sensíveis e Variáveis Proxy.** `race1` e `gender` são removidos da matriz de entrada $X$ do modelo (*Fairness Through Unawareness*) e salvos separadamente em `sensitive_*.csv`. Contudo, atributos como `fam_inc`, `lsat`, `tier` e `cluster` correlacionam-se com raça e renda (`lsat` correlaciona-se com `tier` em $r = -0.3714$) e podem atuar como **proxies**, fazendo com que modelos sinalizem estudantes minoritários em taxas mais altas caso não sejam auditados.
* **Impacto no Mundo Real.** Como o **JurisPrev** aciona uma intervenção de apoio (tutoria preventiva) e não uma punição, **Falsos Negativos** (deixar de identificar um estudante em risco) são muito mais custosos do que **Falsos Positivos** (ofertar reforço extra a um estudante que passaria de qualquer forma).

## 3.4. Processamento de Dados e Metodologia do Pipeline

O pipeline é dividido em scripts modulares (`process_data.py`, `refine_data.py` e `tests/notebooks/model_benchmark.ipynb`) para evitar qualquer vazamento de dados:

```text
[1. Dados Brutos] data/raw/bar_pass_prediction.csv (22.407 linhas, 39 colunas)
       |
       v
[2. Limpeza Sem Estado] process_data.py
   * Define target_risk = 1 - pass_bar; remove vazamento pós-1º ano -> 10 atributos
   * Corrige deslocamentos negativos de idade (91 + age) e mapeia fulltime {1, 0}
       |
       v
[3. Divisão Estratificada (SEED=42)] data/processed/
   * pre_train.csv (65% = 14.564) | pre_val.csv (25% = 5.602) | pre_test.csv (10% = 2.241)
       |
       v
[4. Refinamento Ajustado Apenas no Treino] refine_data.py -> data/refined/
   * Limita age [17, 85] e ugpa [0.0, 4.0]
   * Ajusta SimpleImputer (mediana / moda) SOMENTE em pre_train
   * Salva sensitive_{train,val,test}.csv separadamente para auditoria de fairness
   * Cria academic_momentum e lsat_ugpa_ratio usando média/desvio do treino
   * StandardScaler (contínuas/ordinais) + OneHotEncoder(drop='first') em cluster
       |
       v
[5. Benchmark de Modelos e Fairness] tests/notebooks/model_benchmark.ipynb
   * Treina em train.csv -> avalia em val.csv (Normal vs. +Sensíveis)
   * Compara Baseline, Regressão Logística, RF, HistGB, XGBoost e MLP
       |
       v
[6. Próximos Passos] Validação Cruzada Estratificada (5-Fold) e Avaliação Final em test.csv
```

**Principais decisões metodológicas:**
1. **Pré-processamento e Comparação Antes/Depois:** A limpeza sem estado ocorre antes da divisão para que `pre_train.csv` permaneça legível na análise exploratória. Todas as etapas estatísticas (`SimpleImputer` com mediana/moda, `StandardScaler` e `OneHotEncoder`) são ajustadas **apenas em `pre_train.csv`**, garantindo penalização justa na regularização entre escalas e impedindo que modelos de árvore tratem códigos nominais de `cluster` como valores ordenados.

| Etapa | Arquivos | Divisão | Registros | Colunas Totais | Valores Faltantes | Formato |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Bruto** | `data/raw/bar_pass_prediction.csv` | 100% | 22.407 | 39 | 19.096 | Escalas brutas, idades negativas, colunas com leakage |
| **Processado** | `data/processed/pre_{train,val,test}.csv` | 65% / 25% / 10% | 14.564 (treino) | 11 | 1.023 | Unidades limpas, sem leakage, `target_risk` |
| **Refinado** | `data/refined/{train,val,test}.csv` | 65% / 25% / 10% | 14.564 (treino) | 13 (Normal) / 18 (+Sens) | 0 | Padronizado ($\mu=0, \sigma=1$), `{0,1}`, *one-hot* |

2. **Engenharia de Atributos.** Foram criadas duas variáveis de domínio usando apenas estatísticas do conjunto de treino:
   * `academic_momentum`: $\text{zfygpa} - \frac{z_{\text{lsat}} + z_{\text{ugpa}}}{2}$ — mede se o desempenho do estudante subiu ou caiu no 1º ano em comparação ao seu histórico de ingresso.
   * `lsat_ugpa_ratio`: $\frac{\text{lsat}}{\text{ugpa} + 0.5}$ — diferencia estudantes com bom desempenho em provas padronizadas de alta pressão (`LSAT`) daqueles com boas notas contínuas na graduação, mas menor rendimento em exames.
   *(Uma terceira candidata, `zfygpa * tier`, foi descartada porque multiplicar um z-score centrado em zero pelos níveis `1..6` inverte o sentido para alunos acima da média.)*
3. **Divisão Treino / Validação / Teste:** Divisão estratificada (`65%` treino, `25%` validação e `10%` teste, `SEED = 42`). O arquivo `test.csv` permanece intocado até a avaliação final.
4. **Modelo de Referência (Baseline).** `DummyClassifier(strategy="most_frequent")`, que prevê sempre `0` (aprovado), resultando em `92.80%` de acurácia em `pre_train`, `0.00` de Recall/F1 na classe de risco e `0.500` de `ROC-AUC`.
5. **Comparação Preliminar de Modelos (`train` $\rightarrow$ `val`).** Cinco classificadores foram comparados nos cenários **Normal** vs. **Com Sensíveis (`+ race1, gender`)**:
   * **Regressão Logística (L2, `class_weight="balanced"`).** Melhor capacidade de ranqueamento na validação (**`ROC-AUC = 0.849`** Normal / **`0.850`** +Sensíveis), além de interpretabilidade direta via Razões de Chances (*Odds Ratios*).
   * **MLP Rede Neural (`(64, 32)`).** **`ROC-AUC = 0.844`** Normal / **`0.846`** +Sensíveis.
   * **Random Forest (`max_depth=8`, `class_weight="balanced"`).** **`ROC-AUC = 0.843`** Normal / **`0.842`** +Sensíveis.
   * **XGBoost (`max_depth=4`, `scale_pos_weight`):** **`ROC-AUC = 0.838`** Normal / **`0.840`** +Sensíveis.
   * **HistGradientBoosting (`max_depth=5`, `class_weight="balanced"`).** **`ROC-AUC = 0.836`** Normal / **`0.840`** +Sensíveis.
   * *Conclusão preliminar.* A inclusão direta de `race1` e `gender` praticamente não alterou o `ROC-AUC` ($\le 0.004$), reforçando a decisão de mantê-los fora do modelo.
6. **Seleção de Atributos e Próximos Ajustes.** Combina filtragem guiada pelo domínio com regularização embutida (L1/L2/Elastic Net) e importância de atributos. Na próxima etapa, será executada **Validação Cruzada Estratificada em 5 Folds** com *Grid Search* no conjunto de treino para ajustar hiperparâmetros e o limiar de decisão probabilístico.

## 3.5. Avaliação e Estrutura de Fairness

* **Métricas de Classificação.** Como `target_risk = 1` representa **`7.20%`** da base, avaliamos **`ROC-AUC`** e **`PR-AUC`** para qualidade de ranqueamento, juntamente com **`Recall`**, **`Precision`** e **`F1-Score`** na classe positiva de risco. **Falsos Negativos** (deixar de detectar um aluno em risco) são tratados como mais custosos do que **Falsos Positivos** (alocar tutoria preventiva).
* **Métricas de Fairness (grupo não privilegiado `black` vs. grupo de referência privilegiado `white` via `sensitive_*.csv`).**
  * **Diferença de Paridade Demográfica (`SPD Gap = Selection Rate_Black - Selection Rate_White`).** Verifica se ambos os grupos são encaminhados para tutoria em proporções semelhantes (`0.000` = paridade; limitada em `[-1, 1]` para evitar instabilidade de divisão por zero).
  * **Diferença de Igualdade de Oportunidade (`TPR Gap = Recall_Black - Recall_White`).** Verifica se estudantes realmente em risco são detectados na mesma proporção nos dois grupos raciais (`0.000` = igualdade de oportunidade).