# Módulo Classificador de Imagens Raster

Script Python puro (`classificador.py`) para classificação supervisionada de imagens raster (GeoTIFF) utilizando redes neurais densas (Keras/TensorFlow). Pode ser executado em qualquer máquina local com Python 3.8+ — **não requer Google Colab**.

---

## Estrutura da pasta

```
modulo_classificador/
├── classificador.py      # Script principal
├── requirements.txt      # Dependências
└── README.md             # Este arquivo
```

---

## Instalação das dependências

```bash
pip install -r requirements.txt
```

> **Recomendação:** use um ambiente virtual (`venv` ou `conda`) para evitar conflitos de pacotes.

---

## Preparação dos dados

1. Coloque sua **imagem raster** (`.tif`) em uma pasta, ex: `dados/`.
2. Coloque os **shapefiles** de amostra (`.shp` + `.shx` + `.dbf` + `.prj` + `.cpg`) na mesma pasta ou em outra.
3. Os shapefiles devem conter pontos (ou polígonos) representativos de cada classe.

Exemplo de estrutura:

```
projeto/
├── modulo_classificador/
│   ├── classificador.py
│   └── requirements.txt
├── dados/
│   ├── imagem.tif
│   ├── solo.shp
│   ├── solo.shx
│   ├── solo.dbf
│   ├── vegetacao.shp
│   ├── vegetacao.shx
│   └── vegetacao.dbf
└── resultado/
```

---

## Configuração

Abra `classificador.py` e edite o dicionário `config` na função `main()`:

```python
config = {
    "path_img": "./dados/imagem.tif",                # sua imagem raster
    "shapefiles": [
        ("./dados/solo.shp", 0),                      # (caminho, id_classe)
        ("./dados/vegetacao.shp", 1),
    ],
    "path_img_teste": "./dados/imagem.tif",          # imagem a classificar
    "path_saida": "./resultado/mapa_classificado.tif",
    "usar_mascara": False,                           # True se houver banda alpha
    "epochs": 100,
    "batch_size": 250,
}
```

- **Adicione mais classes** incluindo novas tuplas em `shapefiles`.
- Se a imagem tiver banda alfa (máscara), defina `usar_mascara=True`.
- Se você tiver **mais de 2 classes**, o script ajusta automaticamente a camada de saída para `softmax` e usa `sparse_categorical_crossentropy`.

---

## Execução

```bash
python classificador.py
```

O script realiza automaticamente:
1. Carga da imagem raster
2. Carga e união dos shapefiles
3. Extração dos valores espectrais nas coordenadas das amostras
4. Divisão treino / teste
5. Construção da rede neural
6. Treinamento
7. Avaliação (gráficos + métricas)
8. Predição na imagem completa
9. Exportação do resultado como GeoTIFF

---

## Saídas geradas

- **`mapa_classificado.tif`** — imagem classificada (pode abrir no QGIS/ArcGIS)
- **`graficos_treinamento.png`** — evolução de loss e accuracy
- **`matriz_confusao.png`** — matriz de confusão

---

## Adaptações comuns

### Usar outra imagem para predição

Altere `path_img_teste` para apontar para outro GeoTIFF. Certifique-se de que ele tenha as mesmas bandas (e na mesma ordem) da imagem usada no treino.

### Adicionar mais classes

Basta adicionar mais tuplas no `shapefiles` e garantir que cada uma tenha um `id` único:

```python
"shapefiles": [
    ("./dados/solo.shp", 0),
    ("./dados/vegetacao.shp", 1),
    ("./dados/agua.shp", 2),
],
```

O modelo ajusta a camada de saída automaticamente para `softmax` quando `num_classes > 2`.

### Ajustar a arquitetura da rede

Chame `construir_modelo(camadas_ocultas=[128, 64, 32])` antes de treinar para usar menos neurônios (útil em datasets pequenos).

---

## Requisitos de hardware

- **CPU:** qualquer processador moderno (treinamento em CPU é aceitável para datasets pequenos)
- **GPU:** opcional, mas acelera drasticamente o treinamento (certifique-se de ter o `tensorflow-gpu` compatível instalado)
- **RAM:** 8 GB+ recomendados para imagens grandes
- **Disco:** espaço suficiente para a imagem raster + shapefiles + resultado

---

## Licença e autoria

Este módulo foi gerado como parte de um pipeline de classificação de imagens de sensoriamento remoto com aprendizado de máquina.

