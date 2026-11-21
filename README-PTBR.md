# juris-prev

**[EN-US](README.md) | PT-BR**

**Resumo.**

O **JurisPrev** é um projeto de aprendizado de máquina voltado para alerta precoce que prevê se um estudante de Direito corre risco de reprovar no exame da ordem (`target_risk = 1 - pass_bar`) logo após concluir o primeiro ano de curso. Utilizando a base do *Law School Admission Council* (LSAC, com 22.407 estudantes), removemos todas as variáveis posteriores ao 1º ano para evitar vazamento de dados (*data leakage*), dividimos o conjunto (`65%` treino, `25%` validação e `10%` teste) e aplicamos imputação, engenharia de atributos (`academic_momentum` e `lsat_ugpa_ratio`), codificação e padronização ajustadas exclusivamente no treino. Os atributos sensíveis (`race1` e `gender`) são mantidos separados das entradas do modelo e utilizados para auditar a justiça algorítmica (*fairness*) por meio da Diferença de Paridade Demográfica (`SPD Gap`) e da Igualdade de Oportunidade (`TPR Gap`). Experimentos iniciais na validação mostram forte capacidade de ranqueamento (`ROC-AUC` de até `0.849` sem atributos sensíveis), preparando a base para o ajuste via validação cruzada e a avaliação final no conjunto de teste.

# Sumário
- [1. Introdução](#1-introdução)
- [2. Revisão da Literatura](#2-revisão-da-literatura)
- [3. Metodologia](#3-metodologia)
  - [3.1. Análise Exploratória da Base de Dados (EDA)](#31-análise-exploratória-da-base-de-dados-eda)
  - [3.2. Testes de Hipótese](#32-testes-de-hipótese)
  - [3.3. Vieses nos Dados](#33-vieses-nos-dados)
  - [3.4. Processamento de Dados e Metodologia do Pipeline](#34-processamento-de-dados-e-metodologia-do-pipeline)
  - [3.5. Avaliação e Estrutura de Fairness](#35-avaliação-e-estrutura-de-fairness)

----------

# 1. Introdução

- **Por que esse problema importa.** Reprovar no exame da ordem atrasa o ingresso do graduado no mercado de trabalho, gera desgaste financeiro para o estudante e prejudica os indicadores institucionais das faculdades de Direito. Normalmente, o suporte preparatório extra é oferecido tarde demais — apenas no último semestre ou depois que o aluno já reprovou.

- **Quem é o contratante.** Coordenações acadêmicas de cursos de Direito, equipes de orientação estudantil e direções institucionais que gerenciam orçamentos limitados de tutoria e monitoria.

- **Como o Machine Learning ajuda.** O **JurisPrev** atua como um sistema de alerta precoce ao final do **1º ano de Direito**. Ao sinalizar antecipadamente estudantes com alto risco de reprovação, a coordenação pode alocar de forma preventiva tutorias individuais, oficinas de estudo e mentoria, enquanto audita se as predições tratam os grupos demográficos de maneira justa.

- **Critérios de sucesso do negócio.**
  - Identificar pelo menos `75%` a `80%` dos estudantes realmente em risco ($\text{Recall} \ge 0.75$ para `target_risk = 1`).
  - Alcançar um `ROC-AUC` acima de `0.85`, superando um modelo de referência (*baseline*) da classe majoritária.
  - Manter baixas disparidades raciais quando atributos sensíveis forem excluídos ($|\text{SPD Gap}| < 0.15$ e $|\text{TPR Gap}| < 0.10$).

# 2. Revisão da Literatura

- **Wightman (1998) — LSAC National Longitudinal Bar Passage Study.**
  - **Abordagem.** Coletou e analisou dados de 163 faculdades de Direito dos EUA (coorte de 1991) usando análise estatística e regressão logística para avaliar como notas de ingresso e desempenho na graduação preveem a aprovação no exame da ordem.
  - **Principais resultados.** A nota padronizada do 1º ano de Direito (`zfygpa`) e a pontuação no `LSAT` foram os preditores mais fortes de aprovação, superando a média da graduação anterior (`UGPA`).
  - **Limitação quanto a fairness.** O estudo documentou grandes disparidades raciais nas taxas de aprovação ($\approx 92\%$ no geral para estudantes brancos contra $\approx 78\%$ para estudantes negros), mas não aplicou métricas de justiça algorítmica nem técnicas de mitigação de viés.

- **Friedler et al. (2019) — Comparative Study of Fairness-Enhancing Interventions.**
  - **Abordagem.** Utilizou a base *LSAC Bar Passage* como *benchmark* de classificação para testar modelos padrão (Regressão Logística, Árvores de Decisão e Naive Bayes) combinados com métodos de mitigação de viés no pré-processamento, processamento e pós-processamento.
  - **Principais resultados.** Classificadores padrão atingiram alta acurácia global ($\approx 88\%\text{--}90\%$, impulsionada pela taxa base de $\approx 90\%$ de aprovação), mas geraram forte Impacto Desproporcional (*Disparate Impact*) entre grupos raciais quando treinados apenas para maximizar acurácia.
  - **Lição para o JurisPrev.** Uma acurácia geral alta pode mascarar disparidades severas entre grupos, tornando obrigatória a auditoria de *fairness* nesta base.

# 3. Metodologia

## 3.1. Análise Exploratória da Base de Dados (EDA)

A base de dados foi obtida no Kaggle (`danofer/law-school-admissions-bar-passage`) e contém **22.407 registros de estudantes** distribuídos em **39 colunas brutas**.

Para simular uma predição realista ao **final do 1º ano**, descartamos 28 colunas: variáveis posteriores ao 1º ano que causariam vazamento de dados (`bar`, `bar1`, `bar2`, `bar_passed`, nota acumulada do 3º ano `zgpa`, `decile3`, `grad`, `Dropout`, `dnn_bar_pass_prediction`), identificadores (`ID`) e colunas redundantes ou derivadas (`gpa`, `sex`, `male`, `parttime`, `race`, `race2`, `DOB_yr`, *dummies* raciais, `index6040`, `decile1`). A variável-alvo foi invertida para **`target_risk = 1 - pass_bar`** (`1` = reprovado / em risco, `0` = aprovado).

| Atributo | Papel | Tipo | Descrição |
| :--- | :--- | :--- | :--- |
| `lsat` | Entrada | Contínuo | Pontuação no exame de admissão *Law School Admission Test* ($11\text{--}48$). |
| `ugpa` | Entrada | Contínuo | Média de notas da graduação anterior ao Direito (`0.0`–`4.0`). |
| `zfygpa` | Entrada | Contínuo | Média padronizada do 1º ano de Direito (*z-score* institucional). |
| `tier` | Entrada | Ordinal | Nível de prestígio da instituição (`1` = mais alto até `6` = mais baixo). |
| `cluster` | Entrada | Nominal | Código de agrupamento institucional (`1`–`6`). |
| `fulltime` | Entrada | Binário | Regime de estudo (`1` = tempo integral, `0` = meio período). |
| `fam_inc` | Entrada / Auditoria | Ordinal | Faixa de renda familiar (`1` = menor quintil até `5` = maior quintil). |
| `age` | Entrada / Auditoria | Discreto | Idade no ingresso da faculdade de Direito (em anos). |
| `race1` | Sensível (Auditoria) | Nominal | Raça/etnia (`white`, `black`, `hisp`, `asian`, `other`). Mantida fora de $X$. |
| `gender` | Sensível (Auditoria) | Binário | Gênero (`male`, `female`). Mantido fora de $X$. |
| `target_risk` | **Alvo** | Binário | Risco de reprovação no exame da ordem (`1` = reprovado / em risco, `0` = aprovado). |

**Principais observações dos dados:**
- **Desbalanceamento de Classes.** Cerca de **90%** dos estudantes foram aprovados (`0`) e **10%** reprovaram (`1`). Por conta desse desbalanceamento, a acurácia bruta é enganosa, exigindo divisão estratificada, ponderação de classes (*class weight*) e métricas como `ROC-AUC` e `PR-AUC`.
- **Correções de Codificação e Outliers.** No CSV bruto, `age` continha valores negativos representando deslocamentos em relação a 1900 para a coorte de 1991 (ex.: `-65` = nascido em 1965 = 26 anos) e `fulltime` estava codificado como `{1, 2}`. Convertemos `age` (`91 + age`), mapeamos `fulltime` para `{1, 0}` e limitamos valores fisicamente impossíveis (`age` em `[17, 85]` e `ugpa` em `[0.0, 4.0]`), mantendo notas baixas reais de `lsat` e `zfygpa` por carregarem sinal legítimo de risco.
- **Valores Faltantes e Colinearidade.** Pequenas proporções de dados ausentes ocorrem em `zfygpa`, `fam_inc`, `age`, `race1` e `gender`. Índices compostos (`index6040`) e decis de turma (`decile1`) foram removidos inicialmente para evitar multicolinearidade com `lsat`, `ugpa` e `zfygpa`. *(As estatísticas descritivas exatas e gráficos gerados a partir de `pre_train.csv` compõem o notebook de EDA.)*

## 3.2. Testes de Hipótese

Testamos duas hipóteses centrais sobre `data/processed/pre_train.csv` adotando nível de significância **$\alpha = 0.05$**:

| Hipótese | $H_0$ vs. $H_1$ | Teste Estatístico | Significado Prático |
| :--- | :--- | :--- | :--- |
| **H1: Notas do 1º Ano vs. Risco de Reprovação** | **$H_0$:** A distribuição de `zfygpa` é igual para `target_risk = 0` e `1`.<br>**$H_1$:** Estudantes em risco (`1`) possuem `zfygpa` significativamente menor. | **Teste t de Welch e Mann-Whitney U** (unilateral, lida com variâncias e tamanhos desiguais). | Confirma se o desempenho no 1º ano fornece sinal suficiente para acionar tutoria precoce. |
| **H2: Renda e Raça vs. Risco de Reprovação** | **$H_0$:** O risco de reprovação (`target_risk`) é independente de `fam_inc` e `race1`.<br>**$H_1$:** As taxas de reprovação diferem significativamente entre faixas de renda e grupos raciais. | **Teste Qui-Quadrado ($\chi^2$) de Independência** (variáveis categóricas vs. alvo binário). | Quantifica disparidades estruturais na base e fundamenta a auditoria de *fairness*. |

*(As estatísticas exatas dos testes e os $p$-valores serão adicionados após a finalização do notebook estatístico de EDA.)*

## 3.3. Vieses nos Dados

- **Viés Histórico e Socioeconômico.** Notas de admissão (`LSAT`, `UGPA`) e renda familiar (`fam_inc`) refletem desigualdades prévias de acesso a cursos preparatórios, financiamento educacional e estabilidade financeira antes e durante a graduação.
- **Viés de Amostragem e Sobrevivência.** A base retrata uma coorte norte-americana de 1991 (majoritariamente composta por estudantes brancos e em tempo integral) e exclui alunos que evadiram antes de prestar o exame da ordem.
- **Atributos Sensíveis e Variáveis Proxy.** `race1` e `gender` são removidos da matriz de entrada $X$ do modelo (*Fairness Through Unawareness*) e salvos separadamente em `sensitive_*.csv`. Contudo, atributos como `fam_inc`, `lsat`, `tier` e `cluster` correlacionam-se com raça e renda e podem atuar como **proxies**, fazendo com que modelos sinalizem estudantes minoritários em taxas mais altas caso não sejam auditados.
- **Impacto no Mundo Real.** Como o **JurisPrev** aciona uma intervenção de apoio (tutoria preventiva) e não uma punição, **Falsos Negativos** (deixar de identificar um estudante que precisa de ajuda) são mais custosos do que **Falsos Positivos** (ofertar reforço extra a um estudante que passaria de qualquer forma).

## 3.4. Processamento de Dados e Metodologia do Pipeline

O pipeline é dividido em scripts modulares (`process_data.py`, `refine_data.py` e `tests/notebooks/model_benchmark.ipynb`) para evitar qualquer vazamento de dados:

```text
[1. Dados Brutos] data/raw/bar_pass_prediction.csv (22.407 linhas, 39 colunas)
       |
       v
[2. Limpeza Sem Estado] process_data.py
- Define target_risk = 1 - pass_bar; remove vazamento pós-1º ano -> 10 atributos
- Corrige deslocamentos negativos de idade (91 + age) e mapeia fulltime {1, 0}
       |
       v
[3. Divisão Estratificada (SEED=42)] data/processed/
- pre_train.csv (65%) | pre_val.csv (25%) | pre_test.csv (10% - bloqueado)
       |
       v
[4. Refinamento Ajustado Apenas no Treino] refine_data.py -> data/refined/
- Limita age [17, 85] e ugpa [0.0, 4.0]
- Ajusta SimpleImputer (mediana / moda) SOMENTE em pre_train
- Salva sensitive_{train,val,test}.csv separadamente para auditoria de fairness
- Cria academic_momentum e lsat_ugpa_ratio usando média/desvio do treino
- StandardScaler (contínuas/ordinais) + OneHotEncoder(drop='first') em cluster
       |
       v
[5. Benchmark de Modelos e Fairness] tests/notebooks/model_benchmark.ipynb
- Treina em train.csv -> avalia em val.csv (Normal vs. +Sensíveis)
- Compara Baseline, Regressão Logística, RF, HistGB, XGBoost e MLP
       |
       v
[6. Próximos Passos] Validação Cruzada Estratificada (5-Fold) e Avaliação Final em test.csv
```

**Principais decisões metodológicas:**
1. **Pré-processamento e Comparação Antes/Depois.** A limpeza sem estado ocorre antes da divisão para que `pre_train.csv` permaneça legível na análise exploratória. Todas as etapas estatísticas (`SimpleImputer` com mediana/moda, `StandardScaler` e `OneHotEncoder`) são ajustadas **apenas em `pre_train.csv`**, garantindo penalização justa na regularização e impedindo que modelos de árvore tratem códigos nominais de `cluster` como valores ordenados.

| Etapa | Arquivos | Divisão | Atributos em $X$ | Valores Faltantes | Formato |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Bruto** | `data/raw/bar_pass_prediction.csv` | 100% | 38 + `pass_bar` | Sim | Escalas brutas, idades negativas, colunas com leakage |
| **Processado** | `data/processed/pre_{train,val,test}.csv` | 65% / 25% / 10% | 10 | Sim | Unidades limpas, sem leakage, `target_risk` |
| **Refinado** | `data/refined/{train,val,test}.csv` | 65% / 25% / 10% | 12 (Normal) / 17 (+Sens) | 0 | Padronizado (`média=0, dp=1`), `{0,1}`, *one-hot* |

2. **Engenharia de Atributos.** Foram criadas duas variáveis de domínio usando apenas estatísticas do conjunto de treino:
   - `academic_momentum`: $\text{zfygpa} - \frac{z_{\text{lsat}} + z_{\text{ugpa}}}{2}$ — mede se o desempenho do estudante subiu ou caiu no 1º ano em comparação ao seu histórico de ingresso.
   - `lsat_ugpa_ratio`: $\frac{\text{lsat}}{\text{ugpa} + 0.5}$ — diferencia estudantes com bom desempenho em provas padronizadas de alta pressão (`LSAT`) daqueles com boas notas contínuas na graduação, mas menor rendimento em exames.
   *(Uma terceira candidata, `zfygpa * tier`, foi descartada porque multiplicar um z-score centrado em zero pelos níveis `1..6` inverte o sentido para alunos acima da média.)*
3. **Divisão Treino / Validação / Teste.** Divisão estratificada (`65%` treino, `25%` validação e `10%` teste, `SEED = 42`). O arquivo `test.csv` permanece intocado até a avaliação final.
4. **Modelo de Referência (Baseline).** `DummyClassifier(strategy="most_frequent")`, que prevê sempre `0` (aprovado), resultando em `0.00` de Recall/F1 na classe de risco e `0.500` de ROC-AUC.
5. **Comparação Preliminar de Modelos (`train` $\rightarrow$ `val`).** Cinco classificadores foram comparados nos cenários **Normal** vs. **Com Sensíveis (`+ race1, gender`)**:
   - **Regressão Logística (L2, `class_weight="balanced"`).** Melhor capacidade de ranqueamento na validação (**`ROC-AUC = 0.849`** Normal / **`0.850`** +Sensíveis), além de interpretabilidade direta via Razões de Chances (*Odds Ratios*).
   - **MLP Rede Neural (`(64, 32)`).** **`ROC-AUC = 0.844`** Normal / **`0.846`** +Sensíveis.
   - **Random Forest (`max_depth=8`, `class_weight="balanced"`).** **`ROC-AUC = 0.843`** Normal / **`0.842`** +Sensíveis.
   - **XGBoost (`max_depth=4`, `scale_pos_weight`).** **`ROC-AUC = 0.838`** Normal / **`0.840`** +Sensíveis.
   - **HistGradientBoosting (`max_depth=5`, `class_weight="balanced"`).** **`ROC-AUC = 0.836`** Normal / **`0.840`** +Sensíveis.
   - *Conclusão preliminar:* A inclusão direta de `race1` e `gender` praticamente não alterou o `ROC-AUC` ($\le 0.004$), reforçando a decisão de mantê-los fora do modelo.
6. **Seleção de Atributos e Próximos Ajustes.** Combina filtragem guiada pelo domínio com regularização embutida (L1/L2/Elastic Net) e importância de atributos. Na próxima etapa, será executada **Validação Cruzada Estratificada em 5 Folds** com *Grid Search* no conjunto de treino para ajustar hiperparâmetros e o limiar de decisão.

## 3.5. Avaliação e Estrutura de Fairness

- **Métricas de Classificação.** Como `target_risk = 1` representa $\approx 10\%$ da base, avaliamos **`ROC-AUC`** e **`PR-AUC`** para qualidade de ranqueamento, juntamente com **`Recall`**, **`Precision`** e **`F1-Score`** na classe positiva de risco. **Falsos Negativos** (deixar de detectar um aluno em risco) têm custo maior do que **Falsos Positivos** (alocar tutoria preventiva).
- **Métricas de Fairness (grupos `black` vs. `white` via `sensitive_*.csv`).**
  - **Diferença de Paridade Demográfica (`SPD Gap = Selection Rate_Black - Selection Rate_White`).** Verifica se ambos os grupos são encaminhados para tutoria em proporções semelhantes (`0.000` = paridade; limitada em `[-1, 1]` para evitar instabilidade de divisão por zero).
  - **Diferença de Igualdade de Oportunidade (`TPR Gap = Recall_Black - Recall_White`).** Verifica se estudantes realmente em risco são detectados na mesma proporção nos dois grupos raciais (`0.000` = igualdade de oportunidade).