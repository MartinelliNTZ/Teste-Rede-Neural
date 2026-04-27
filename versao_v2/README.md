# Classificador Raster com Redes Neurais — v4

Sistema de classificação supervisionada de imagens raster (GeoTIFF) utilizando redes neurais densas (Multilayer Perceptron — MLP). O pipeline extrai amostras espectrais a partir de shapefiles, treina um modelo de deep learning e classifica a imagem inteira pixel a pixel, exportando o resultado como um novo GeoTIFF.

---

## 1. Visão Geral

Este projeto realiza a **classificação de uso/cobertura do solo** a partir de imagens multiespectrais (rasters) e amostras de campo (shapefiles). Cada pixel da imagem é representado por um vetor de valores espectrais (uma banda por dimensão), que alimenta uma rede neural densa. O modelo aprende a mapear assinaturas espectrais para classes predefinidas (ex: solo, vegetação, palhada) e, em seguida, classifica toda a cena raster.

### Principais características

- **Predição em chunks (blocos)**: a imagem é processada em faixas horizontais para não estourar a memória RAM, permitindo classificar imagens gigantes mesmo em máquinas modestas.
- **Sem normalização de dados**: os pixels são mantidos em sua escala original (ex: 0–255). Isso evita o bug de divisão por zero que ocorre quando bandas constantes (como alpha = 255) têm desvio padrão zero.
- **Suporte a GPU/CPU**: detecta automaticamente GPUs NVIDIA e configura o TensorFlow para uso de memória dinâmica.
- **Classificação binária ou multiclasse**: adapta automaticamente a camada de saída e a função de perda conforme o número de classes.
- **Máscara opcional**: permite filtrar pixels válidos pela última banda (ex: canal alpha/máscara).
- **Múltiplas formas de configuração**: via dicionário Python embutido, argumentos de linha de comando (CLI) ou arquivo JSON externo.
- **Relatório de performance**: exibe tempo por etapa, pixels/minuto, consumo de RAM e outros indicadores ao final da execução.

---

## 2. Estrutura do Projeto

```
.
├── main.py              # Código-fonte principal (pipeline completo)
├── config.json          # Configuração externa opcional (sobrescreve o embutido)
├── requirements.txt     # Dependências Python
├── dados/               # Dados de entrada (imagens + shapefiles)
│   ├── imagem.tif           # Imagem pequena para classificar (teste)
│   ├── imagemFull.tif       # Imagem grande para extrair amostras (treino)
│   ├── solo.shp / solo2.shp # Amostras da classe "solo"
│   ├── palhada.shp          # Amostras da classe "palhada"
│   └── vegetacao.shp        # Amostras da classe "vegetação"
└── resultado/           # Diretório de saída (criado automaticamente)
    ├── mapa_classificado.tif    # GeoTIFF com a classificação final
    ├── modelo.keras             # Modelo treinado (opcional)
    ├── graficos_treinamento.png # Curvas de loss e accuracy
    └── matriz_confusao.png      # Matriz de confusão do teste
```

---

## 3. Pré-requisitos e Instalação

### 3.1 Ambiente

- Python 3.8+
- (Opcional) GPU NVIDIA com drivers e CUDA/cuDNN compatíveis para aceleração via TensorFlow-GPU

### 3.2 Instalação das dependências

```bash
pip install -r requirements.txt
```

**Dependências principais:**

| Pacote | Função no projeto |
|--------|-------------------|
| `tensorflow` | Construção e treinamento da rede neural |
| `numpy` / `pandas` | Manipulação matricial e tabular dos dados |
| `geopandas` / `shapely` | Leitura e manipulação de shapefiles |
| `rasterio` | Leitura/escrita de imagens raster (GeoTIFF) |
| `scikit-learn` | Divisão treino/teste e métricas de avaliação |
| `matplotlib` / `seaborn` | Geração de gráficos e matriz de confusão |
| `psutil` | Monitoramento de RAM para cálculo de chunks |

---

## 4. Funcionamento do Sistema

O pipeline é executado em **9 etapas sequenciais**, todas encapsuladas na classe `ClassificadorRaster`.

### 4.1 Etapa 1 — Carregar Imagem de Treino

Abre o GeoTIFF de referência (`path_img`) com `rasterio`. Extrai metadados essenciais: dimensões (altura × largura), número de bandas, sistema de coordenadas (CRS) e tamanho em disco. Essa imagem serve apenas para extrair os valores espectrais nas coordenadas das amostras.

### 4.2 Etapa 2 — Carregar Amostras (Shapefiles)

Lê cada shapefile informado na configuração. Cada shapefile está associado a um `id_classe` (inteiro). Se o CRS do shapefile for diferente do CRS da imagem raster, o sistema reprojeta o shapefile automaticamente para o mesmo CRS. Os GeoDataFrames de todas as classes são concatenados em uma única tabela.

**Exemplo de shapefiles:**
```json
"shapefiles": [
    ["dados/solo.shp", 0],
    ["dados/vegetacao.shp", 1]
]
```

### 4.3 Etapa 3 — Extração de Valores Espectrais

Para cada geometria (ponto) do shapefile concatenado, o sistema amostra o valor do pixel correspondente no raster de treino. O resultado é uma matriz `X` de dimensão `(N_amostras, N_bandas)` com as assinaturas espectrais, e um vetor `Y` de dimensão `(N_amostras, 1)` com os IDs das classes.

Se `usar_mascara = true`, a última banda é descartada de `X` e tratada como canal alpha/máscara.

### 4.4 Etapa 4 — Preparação dos Dados

Os dados são divididos em conjuntos de **treino** e **teste** via `train_test_split` (padrão: 70% treino / 30% teste). Não há normalização ou padronização — os pixels permanecem na escala original da imagem (ex: 0–255), garantindo compatibilidade total com o treino e evitando instabilidade numérica em bandas constantes.

### 4.5 Etapa 5 — Construção do Modelo

É montada uma rede neural densa sequencial (`tf.keras.Sequential`) com:

- Uma camada de entrada (`Dense`) com o número de neurônios da primeira camada oculta.
- Camadas ocultas adicionais conforme a lista `camadas_ocultas`.
- Uma camada de saída adaptativa:
  - **2 classes**: 1 neurônio com ativação `sigmoid` + perda `binary_crossentropy`.
  - **>2 classes**: `N` neurônios com ativação `softmax` + perda `sparse_categorical_crossentropy`.

**Arquitetura padrão:** `[256, 128, 64, 8]` com ativação `relu`.

Otimizador: **Adam**.

### 4.6 Etapa 6 — Treinamento

O modelo é treinado com os dados de treino. A validação interna (validation split) acompanha a evolução da loss e da accuracy. Ao final, se `salvar_modelo = true`, o modelo é persistido em disco no formato `.keras`.

### 4.7 Etapa 7 — Avaliação

São gerados automaticamente:

- **Gráficos de treinamento** (`graficos_treinamento.png`): curvas de loss e accuracy para treino e validação.
- **Classification report**: precisão, recall e F1-score por classe no conjunto de teste.
- **Matriz de confusão** (`matriz_confusao.png`): heatmap das predições vs. valores reais.
- **Métricas finais**: loss e accuracy no conjunto de teste.

### 4.8 Etapa 8 — Predição e Exportação (Chunked)

Esta é a etapa mais crítica para a eficiência de memória. A imagem a ser classificada (`path_img_teste`) é processada em **chunks de linhas**:

1. Calcula-se quantas linhas cabem na RAM disponível (respeitando `ram_limite_pct`).
2. A imagem é lida bloco a bloco via `rasterio.windows.Window`.
3. Cada bloco é remodelado de `(Bandas, Altura, Largura)` para `(Pixels, Bandas)`.
4. O modelo prediz a classe de cada pixel válido em lotes (`batch_size_predicao`).
5. O resultado é escrito diretamente no GeoTIFF de saída, preservando a georreferenciação original.

Pixels inválidos (fora da máscara, se houver) recebem `nodata = NaN`.

### 4.9 Etapa 9 — Relatório Final

Exibe um relatório consolidado no terminal com:

- Duração de cada etapa (início, fim, tempo decorrido, percentual do total).
- Tempo total de execução.
- Throughput: pixels/minuto, MB/minuto, ms/pixel.
- Informações de hardware: dispositivo (GPU/CPU), threads, RAM total/usada.

---

## 5. Configuração

O sistema aceita três níveis de configuração, em ordem de prioridade (maior prioridade primeiro):

1. **Arquivo JSON externo** (`--config meu_config.json`)
2. **Argumentos de linha de comando** (CLI)
3. **Dicionário `CONFIG` embutido** no topo do `main.py`

### 5.1 Parâmetros disponíveis

| Parâmetro | Tipo | Descrição | Padrão |
|-----------|------|-----------|--------|
| `path_img` | string | Imagem raster para extrair amostras de treino | `dados/imagemFull.tif` |
| `path_img_teste` | string | Imagem a ser classificada | `dados/imagem.tif` |
| `path_saida` | string | Caminho do GeoTIFF classificado de saída | `resultado/mapa_classificado.tif` |
| `shapefiles` | lista | Lista de `[caminho_shapefile, id_classe]` | `[["dados/solo.shp",0],["dados/vegetacao.shp",1]]` |
| `usar_mascara` | booleano | Se `true`, usa a última banda como máscara/alpha | `false` |
| `valor_valido` | inteiro | Valor da máscara que indica pixel válido | `255` |
| `ram_limite_pct` | float | Percentual máximo de RAM disponível para chunks | `70.0` |
| `test_size` | float | Fração dos dados reservada para teste (0.0–1.0) | `0.3` |
| `random_state` | inteiro | Semente aleatória para reprodutibilidade | `42` |
| `epochs` | inteiro | Número máximo de épocas de treinamento | `100` |
| `batch_size_treino` | inteiro | Tamanho do lote durante o treino | `250` |
| `batch_size_predicao` | inteiro | Tamanho do lote durante a predição (pode ser maior) | `4096` |
| `camadas_ocultas` | lista | Neurônios em cada camada oculta | `[256, 128, 64, 8]` |
| `activation` | string | Função de ativação (`relu`, `elu`, `tanh`) | `"relu"` |
| `salvar_modelo` | booleano | Se `true`, salva o modelo treinado em disco | `false` |
| `path_modelo` | string | Caminho para salvar/carregar o modelo | `"resultado/modelo.keras"` |

### 5.2 Exemplos de uso

**Executar com configuração embutida (edite o dicionário `CONFIG` no `main.py`):**
```bash
python main.py
```

**Usar configuração externa JSON:**
```bash
python main.py --config config.json
```

**Sobrescrever via CLI:**
```bash
python main.py \
    --imagem dados/imagemFull.tif \
    --shapefile dados/solo.shp,0 \
    --shapefile dados/vegetacao.shp,1 \
    --teste dados/imagem.tif \
    --saida resultado/mapa.tif \
    --epochs 150 \
    --batch-size 512
```

---

## 6. Detalhes da Arquitetura

### 6.1 Rede Neural

- **Tipo**: Multilayer Perceptron (MLP) totalmente conectado.
- **Entrada**: vetor de dimensão `(N_bandas,)` — cada pixel é uma amostra independente.
- **Camadas ocultas**: configuráveis via `camadas_ocultas`.
- **Saída**:
  - Binária: 1 neurônio (`sigmoid`).
  - Multiclasse: `C` neurônios (`softmax`).

### 6.2 Por que não normalizar?

Em versões anteriores (v2/v3), a normalização padronizava cada banda. Quando uma banda possui valor constante (ex: canal alpha = 255 em todos os pixels), seu desvio padrão é zero, causando divisão por zero e produzindo `NaN`/`Inf` que destruíam o treinamento. A v4 retornou ao comportamento da v1: **sem normalização**. Para dados espectrais em faixa limitada (0–255), a rede neural aprende adequadamente sem padronização.

### 6.3 Predição em Chunks

Para uma imagem de 10.000 × 10.000 pixels com 4 bandas em `float64`:
- Carregamento total: ~3.2 GB de RAM.
- Com chunks de 500 linhas: ~160 MB por vez.

O sistema calcula automaticamente o número de linhas por chunk com base na RAM livre e no `ram_limite_pct`.

---

## 7. Saídas Geradas

Após a execução bem-sucedida, os seguintes arquivos são produzidos:

| Arquivo | Descrição |
|---------|-----------|
| `resultado/mapa_classificado.tif` | GeoTIFF com a classificação final (1 banda, float64, compressão LZW) |
| `resultado/modelo.keras` *(opcional)* | Modelo treinado no formato Keras v3 |
| `graficos_treinamento.png` | Curvas de loss e accuracy durante o treino |
| `matriz_confusao.png` | Heatmap da matriz de confusão no conjunto de teste |

---

## 8. Dicas de Performance

- **Aumente `batch_size_predicao`** se tiver RAM e GPU disponíveis (ex: 8192 ou 16384) para acelerar a predição.
- **Diminua `batch_size_treino`** apenas se houver falta de VRAM durante o treino.
- **Use GPU**: o tempo de predição cai drasticamente com uma GPU NVIDIA compatível.
- **Imagens muito grandes**: se a imagem for maior que a RAM total, o chunking automático ainda permite processá-la, mas o tempo de execução aumenta proporcionalmente.
- **Ajuste `camadas_ocultas`**: redes menores (`[64, 32]`) treinam mais rápido e consomem menos memória, mas podem ter menor capacidade de aprendizado.

---

## 9. Troubleshooting

| Problema | Causa provável | Solução |
|----------|---------------|---------|
| `FileNotFoundError` na imagem ou shapefile | Caminho incorreto no `CONFIG` ou JSON | Verifique os caminhos relativos ao diretório de execução |
| `NaN` no resultado classificado | Divisão por zero em normalização (versões antigas) | Use a v4 atual (sem normalização) |
| RAM cheia durante a predição | `ram_limite_pct` muito alto ou imagem muito grande | Reduza `ram_limite_pct` ou aumente a memória virtual |
| CRS incompatível | Shapefile e raster em sistemas de coordenadas diferentes | O sistema reprojeta automaticamente, mas verifique se os CRS são válidos |
| Modelo não salva | `salvar_modelo = false` | Altere para `true` na configuração |

---

## 10. Licença e Autoria

Desenvolvido como ferramenta de classificação de imagens de sensoriamento remoto utilizando TensorFlow/Keras. O código é modular e pode ser adaptado para outros domínios de classificação pixel-based.

---

**Versão atual:** v4  
**Linguagem:** Python 3  
**Framework de ML:** TensorFlow / Keras  
**Formato de dados:** GeoTIFF (via Rasterio) + Shapefile (via GeoPandas)

