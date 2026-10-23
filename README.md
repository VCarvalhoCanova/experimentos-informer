# Previsão de séries temporais — Informer vs. ARIMA

Este repositório é um laboratório de previsão de séries temporais. Ele já continha um
pipeline clássico (ARIMA/`pmdarima`, janelas, métricas, persistência em `.pkl`) e notebooks
de análise. Neste trabalho foi adicionado:

1. uma **implementação do Informer** (Zhou et al., 2021 — `docs/2012.07436v3.pdf`) em
   PyTorch, em `src/model/informer.py`;
2. uma **camada de experimentos** (`src/experiment/benchmark.py`) com o registro dos oito
   datasets de `data/raw`, execução do ARIMA e do Informer e montagem da tabela de métricas;
3. um **notebook de demonstração** comparando os dois modelos nas oito séries
   (`notebooks_examples/informer_vs_arima.ipynb`);
4. um **runner de linha de comando** equivalente ao notebook (`scripts/run_informer_vs_arima.py`);
5. `requirements.txt` e este README.

Nenhum arquivo existente foi alterado: o pipeline antigo (`src/model/arima.py`,
`src/input/input.py`, `src/metrics/metrics.py`, `src/config.py`) e os notebooks de
`notebooks_examples/` continuam exatamente como estavam.

---

## 1. Estrutura do projeto

```
.
├── data/
│   └── raw/                       # 8 séries: ETTh1/ETTh2/ETTm1/ETTm2 (.csv) e
│                                  # coloradoRiver/sunspot/taylor/temperature (.txt)
├── docs/
│   ├── 2012.07436v3.pdf           # artigo do Informer (referência da implementação)
│   └── UKSim2014_IEEE.pdf
├── notebooks_examples/
│   ├── arima_exec.ipynb           # (existente) execução do ARIMA
│   ├── calculate_metrics.ipynb    # (existente) tabela de métricas + testes estatísticos
│   ├── plot_test_set.ipynb        # (existente) gráficos real vs. previsto
│   ├── informer_vs_arima.ipynb    # NOVO: demonstração Informer x ARIMA nas 8 séries
│   ├── informer_vs_arima_optimized.ipynb  # NOVO: mesma comparação + busca de hiperparâmetros
│   └── informer_vs_arima_paper_ETT.ipynb  # NOVO: ETT multivariado no protocolo do artigo
├── scripts/
│   └── run_informer_vs_arima.py   # NOVO: mesma demonstração sem Jupyter (CLI)
├── src/
│   ├── config.py                  # (existente) caminhos e metadados das séries antigas
│   ├── input/input.py             # (existente) janelamento, normalização, KPSS, filtros
│   ├── metrics/metrics.py         # (existente) 9 métricas + leitura de resultados .pkl
│   ├── model/
│   │   ├── arima.py               # (existente) wrapper do pmdarima.auto_arima
│   │   ├── generics.py            # (existente) utilitários de fit/save
│   │   └── informer.py            # NOVO: rede Informer + forecasters uni/multivariado
│   └── experiment/
│       ├── benchmark.py           # NOVO: registry dos datasets, execução e métricas
│       └── ett_paper.py           # NOVO: protocolo ETT do artigo (split, horizontes, ARIMA)
├── requirements.txt               # NOVO
├── setup.py                       # raiz do projeto (usada por src/config.py)
└── README.md                      # NOVO (este arquivo)
```

---

## 2. O modelo Informer (`src/model/informer.py`)

A implementação segue o artigo seção por seção. O mapeamento entre o texto e as classes:

| Artigo | Implementação |
| --- | --- |
| Entrada unificada (Apêndice B, Eq. 7-8): projeção escalar por `Conv1d` + embedding posicional fixo + embeddings de calendário | `DataEmbedding`, `PositionalEmbedding`, `TemporalEmbedding`, `time_features` |
| **ProbSparse self-attention** (Eq. 2-4): mede `M(q_i, K) = max_j(q_i k_jᵀ/√d) − mean_j(...)` amostrando `U_part = c·ln L_K` pares e mantém as `u = c·ln L_Q` queries dominantes | `ProbSparseAttention` (`_prob_qk` faz a medição, o `scatter_` devolve as queries ao lugar) |
| **Self-attention distilling** (Eq. 5): `MaxPool(ELU(Conv1d(x)))` dobra a redução da dimensão temporal a cada camada | `ConvLayer` |
| **Pirâmide de stacks** do encoder (Fig. 3): réplicas com entrada pela metade e uma camada de distilling a menos, saídas concatenadas | `Informer.forward` (bloco `stack_outputs`) |
| **Decoder generativo** (Eq. 6): `Concat(start token, placeholder zero)` e todo o horizonte em uma passada | `Decoder`, `Informer.forward`, `InformerForecaster.predict` |
| Máscara causal no self-attention mascarado do decoder | `torch.triu(...)` em `Informer.forward` |

Hiperparâmetros do modelo (`Informer` e `InformerForecaster`): `seq_len`, `label_len`,
`pred_len`, `d_model`, `n_heads`, `e_layers`, `d_layers`, `d_ff`, `factor` (o `c` do artigo),
`dropout`, `distil` e `n_encoder_stacks`.

Atenção: `e_layers` conta as camadas do **stack principal** do encoder, não o total. Cada réplica
adicional tem uma camada a menos (mínimo 1), então `e_layers=2` com `n_encoder_stacks=2`
instancia **3 camadas de encoder** (2 + 1) e `d_layers=1` instancia 1 camada de decoder. Para a
arquitetura do artigo (`3+1` camadas no encoder, `1/4` de entrada na réplica e 2 camadas no
decoder), use `e_layers=3`, `n_encoder_stacks=2`, `replica_layers=1`,
`replica_input_fraction=0.25` e `d_layers=2`.

### 2.1 `InformerForecaster` (API no estilo scikit-learn)

O módulo `Informer` é só a rede. Ele não normaliza, não janela, não treina. Essa parte fica no
wrapper:

```python
from model.informer import InformerForecaster

forecaster = InformerForecaster(
    seq_len=96, label_len=48, pred_len=24,   # janelas
    d_model=48, n_heads=4, d_ff=96,
    e_layers=2,                       # stack principal do encoder (a réplica soma +1)
    n_encoder_stacks=2,               # 3 camadas de encoder no total
    d_layers=1,                       # 1 camada de decoder
    epochs=6, batch_size=64, lr=1e-3, patience=2,
)
forecaster.fit(train_values, train_timestamps)        # normaliza por z-score e treina
forecast = forecaster.predict(context, context_ts)     # 24 passos em uma passada
rolling  = forecaster.rolling_forecast(history, test, timestamps)  # blocos de 24
```

Detalhes do que o wrapper faz:

* **normalização** z-score da série de treino (o artigo normaliza cada dataset com média zero);
* **janelas** `enc_x = série[o : o+L]`, `dec_x = [últimos label_len valores reais, zeros]`,
  `target = série[o+L : o+L+pred_len]`;
* **treino** com `MSELoss`, Adam, `clip_grad_norm_` e decaimento do learning rate de 0.5 por
  época (mesma receita do artigo: `lr` começa em 1e-3);
* **early stopping** sobre os últimos 10% das janelas de treino, restaurando o melhor estado;
* **`time_exec`** com tempos de treino e de inferência, no mesmo formato usado no restante do
  projeto (`{'training': ..., 'testing': ...}`);
* **funções temporal**: se o dataset não tem timestamps, `time_features` gera um índice
  sintético; os `.txt` do repositório recebem `DatetimeIndex` na frequência declarada no
  registry (ver seção 4).

---

## 3. Camada de experimentos (`src/experiment/benchmark.py`)

* `DATASETS` / `DatasetSpec`: metadados dos oito arquivos (alvo, frequência, sazonalidade,
  horizonte, `seq_len`, `label_len`, `max_points`, se o ARIMA usa componente sazonal);
* `load_series(name)`: devolve `pd.Series` com `DatetimeIndex`;
* `split_series(series, test_size=0.2)`: split cronológico;
* `split_validation(train, val_size=0.15)`: separa a cauda do treino como validação — usado
  pela busca de hiperparâmetros para nunca tocar no conjunto de teste;
* `run_arima(...)`: usa o wrapper existente `model.arima.Arima` (portanto
  `pmdarima.auto_arima`), prevê o bloco de `horizon` passos e depois atualiza o estado com os
  valores observados;
* `run_informer(..., params=None)`: usa `InformerForecaster`, prevê o bloco com uma passada
  generativa; `params` aceita overrides de hiperparâmetros (chaves em `INFORMER_PARAM_KEYS`
  mais `seq_len`/`label_len`, truncados ao tamanho da janela de treino). `pred_len` não pode
  ser sobrescrito: é sempre o bloco do protocolo. `params=None` reproduz exatamente a
  configuração original;
* `run_comparison(..., informer_params=None)`: roda os dois em cada dataset e devolve
  `(df_metrics, results)`, onde `df_metrics` tem índice `(ts, model)` e as colunas
  `MSE, RMSE, MAPE, SMAPE, MAE, theil, ARV, IA, POCID, time_training, time_testing`
  — **o mesmo layout** de `metrics.open_fold_result` usado em `calculate_metrics.ipynb`;
* `comparison_table(df, metric)`: pivot com `arima`, `informer`, `best` e `informer_gain_%`;
* `plot_series_overview`, `plot_forecasts`, `plot_metric_comparison`: gráficos do notebook;
* `compute_metrics`: delega para `metrics.gerenerate_metric_results` (mesmas nove métricas).

As métricas são exatamente as do projeto, então os números são comparáveis com os
experimentos anteriores do repositório.

### 3.1 Protocolo de avaliação

* split cronológico **80% treino / 20% teste** (uma única partição, sem validação cruzada);
* todo dataset tem um `horizon` (24 pontos para ETT horária, 48 para ETT de 15 min e
  `taylor`, 11-12 para as séries anuais/mensais), reduzido automaticamente se a janela de
  teste for curta (`effective_horizon`);
* previsão **em blocos**: a cada origem o modelo escreve `horizon` passos; depois a origem
  avança `horizon` e o ARIMA recebe os valores observados do bloco (rolling origin). Nada de
  *one-step-ahead* escondido: os dois modelos são avaliados em previsão multi-step;
* o último bloco pode ser parcial (as previsões são alinhadas ao fim da janela de teste);
* MAPE fica `inf` quando o alvo tem zeros (comportamento de `metrics.py`); nesse caso use
  RMSE/MAE/SMAPE/IA.

---

## 4. Datasets usados na demonstração

`src/config.py` registra apenas as séries antigas do projeto (vento, irradiação, etc.) e não
conhece os arquivos ETT. Por isso o registry novo é independente:

| dataset | grupo | alvo | frequência | sazonalidade | horizonte | `max_points` | ARIMA sazonal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ETTh1.csv` | ETT | `OT` | horária | 24 | 24 | 1200 | não |
| `ETTh2.csv` | ETT | `OT` | horária | 24 | 24 | 1200 | não |
| `ETTm1.csv` | ETT | `OT` | 15 min | 96 | 48 | 1200 | não |
| `ETTm2.csv` | ETT | `OT` | 15 min | 96 | 48 | 1200 | não |
| `coloradoRiver.txt` | clássica | `y` | mensal | 12 | 12 | todos (744) | sim (m=12) |
| `sunspot.txt` | clássica | `y` | anual | 11 | 11 | todos (288) | sim (m=11) |
| `taylor.txt` | clássica | `y` | 30 min | 48 | 48 | 1200 | não |
| `temperature.txt` | clássica | `y` | mensal | 12 | 12 | todos (240) | sim (m=12) |

Notas:

* nas ETT usamos a coluna `OT` (temperatura do óleo), que é o alvo do artigo;
* `max_points` mantém os pontos **mais recentes** e existe para o `auto_arima` não inviabilizar
  a execução em CPU (o ETTm1 tem ~69 mil linhas). Aumente/remova em `benchmark.DATASETS` se
  quiser o experimento completo;
* o ARIMA fica **não sazonal** nas ETT de propósito: `SARIMA(m=96)` sobre centenas de pontos
  levaria horas. Nas séries mensais/anuais a sazonalidade é estimada.

---

## 5. Como rodar

### 5.1 Dependências

```bash
pip install -r requirements.txt
# PyTorch em CPU (recomendado para este experimento):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

O Informer depende apenas de `numpy`, `pandas` e `torch`. O ARIMA do projeto depende de
`pmdarima`, e o `src/input/input.py` (importado por `metrics`/`generics`) depende de `sktime`
e `PyWavelets` — por isso essas três também aparecem no `requirements.txt`. Sem elas o
notebook não roda, mesmo que só a parte do Informer seja usada, porque as métricas vêm de
`src/metrics/metrics.py`.

### 5.2 Notebooks

```bash
jupyter notebook notebooks_examples/informer_vs_arima.ipynb
```

`informer_vs_arima.ipynb` está dividido em: preparação do ambiente → visão geral dos 8
datasets → sanity check do Informer → protocolo → execução (célula longa) → tabela de métricas
no formato do projeto → métricas normalizadas → gráficos → persistência em `data/result/` →
discussão. Ele não foi executado neste ambiente (ver seção 8), portanto está salvo **sem
outputs**.

### 5.3 Linha de comando

```bash
python scripts/run_informer_vs_arima.py --list          # metadados dos datasets
python scripts/run_informer_vs_arima.py --quick         # rede pequena, 3 épocas
python scripts/run_informer_vs_arima.py --epochs 8 \
    --datasets ETTh1.csv ETTm1.csv sunspot.txt temperature.txt
```

A CLI salva a tabela em `data/result/informer_vs_arima_metrics.csv` e, com `--plots`, também
renderiza os gráficos no backend `Agg`.

### 5.4 Tempo de execução

Em uma CPU comum, com `QUICK=False`, a execução completa (8 datasets, 6 épocas) leva algo
entre **10 e 25 minutos**; a maior parte é `auto_arima` + as ~700 janelas de treino de cada
série longa. `--quick` reduz para poucos minutos, com previsões piores.

---

## 6. O que esperar dos resultados

A hipótese do experimento:

* **ETT horária e de 15 minutos** — séries longas, com ciclo diário forte e não estacionárias.
  É o cenário do artigo: o Informer escreve o horizonte inteiro de uma vez e tende a ter
  RMSE/MAE menores; o ARIMA, aqui sem componente sazonal, costuma degradar conforme o
  horizonte cresce.
* **`sunspot`, `temperature`, `coloradoRiver`** — poucos pontos, baixa complexidade e forte
  estrutura autorregressiva. É onde o ARIMA pode empatar ou ganhar: um modelo com poucos
  parâmetros estimados sobre a própria série costuma ser imbatível nesse regime, e o Informer
  (rede pequena, poucas épocas, pouca massa de dados) tende a não justificar o custo extra.
* **`taylor`** (30 min) — meio-termo: os dois capturam o ciclo diário e a diferença aparece na
  cauda da distribuição do erro.

Para comparar datasets de escalas diferentes, o notebook calcula `nRMSE = RMSE / std(teste)`
além da tabela bruta.

---

## 7. Limitações (importantes para não superinterpretar)

1. **Uma única partição** treino/teste por dataset, sem validação cruzada nem repetições.
2. **Rede pequena e poucas épocas** (`d_model=48`, `d_ff=96`, 3 camadas de encoder em 2 stacks +
   1 de decoder, ~106 mil parâmetros, 6 épocas) treinada em CPU — não é o setup do artigo
   (`d_model=512`, grid search, GPU V100).
3. **Séries truncadas** (`max_points=1200`) nas ETT/taylor para o `auto_arima` ser viável.
4. **Sem tuning**: nem `auto_arima` nem o Informer recebem busca de hiperparâmetros. O
   notebook `informer_vs_arima_optimized.ipynb` cobre parcialmente essa lacuna, mas com uma
   única janela de validação por dataset (2 a 6 blocos) e semente única por trial — o ranking
   de 40 trials tem ruído considerável.
5. **ARIMA não sazonal** nas ETT — uma vantagem real do ARIMA sazonal não foi explorada.
6. Métricas de erro percentual (`MAPE`) podem ser `inf` quando há zeros no alvo.

O objetivo é a **demonstração do comportamento**, não um benchmark publicado.


---

## 8. Referências

* Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., Zhang, W. *Informer: Beyond
  Efficient Transformer for Long Sequence Time-Series Forecasting*. AAAI 2021.
  (`docs/2012.07436v3.pdf`; código oficial: https://github.com/zhouhaoyi/Informer2020)
* Dataset ETT: https://github.com/zhouhaoyi/ETDataset
