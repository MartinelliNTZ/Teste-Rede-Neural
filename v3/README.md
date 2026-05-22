# Vetorizador Raster (GPKG + SHP)

Scripts Python para transformar raster classificado (ex.: mapas temáticos) em vetores (polígonos) com filtro de suavização (majority 3x3), remoção de pequenas regiões e pós-processamento (polygonize e dissolve por classe). 

## Arquivos no repositório

- **`main.py`** — Vetorizador “inteligente” v1 (CPU): majority 3x3 + remove áreas pequenas + polygonize + dissolve.
- **`main2.py`** — Vetorizador híbrido GPU+CPU v3: tenta usar **GPU (CuPy/CUDA)** no filtro e faz pipeline igual ao `main.py`.
- **`testando3.py`** — Script de teste: extrai contornos via `skimage.measure.find_contours`, cria polígonos e salva em `saida.gpkg`.

> Todos os scripts assumem rasters **.tif/.tiff** na **mesma pasta** do script (exceto `testando3.py`, que usa `raster2.tif`).

---

## Pré-requisitos

Python 3.x.

### Dependências

Instale as bibliotecas necessárias:

```bash
pip install rasterio geopandas shapely scipy scikit-image fiona pyogrio
```

Para habilitar GPU no `main2.py` (opcional), instale CuPy compatível com sua versão CUDA:

```bash
pip install cupy-cuda12x
```

---

## Como executar

### 1) `main.py` (CPU v1)

```bash
python main.py
```

**Entrada esperada**: arquivo `*.tif` ou `*.tiff` na mesma pasta. O script usa o primeiro raster encontrado.

**Saídas** (na mesma pasta):

- `resultado_vetorizado.gpkg`
- `resultado_vetorizado.shp`

---

### 2) `main2.py` (GPU+CPU v3)

```bash
python main2.py
```

O script:
- Detecta GPU via CuPy.
- Se GPU estiver disponível, tenta aplicar o filtro 3x3 usando GPU (aproximação via `median_filter`).
- Caso contrário, cai para o modo CPU.

**Entrada esperada**: arquivo `*.tif` ou `*.tiff` na mesma pasta. O script usa o primeiro raster encontrado.

**Saídas** (na mesma pasta):

- `resultado_vetorizado.gpkg`
- `resultado_vetorizado.shp`

---

### 3) `testando3.py` (teste de contornos)

```bash
python testando3.py
```

**Entrada esperada (fixa)**: `raster2.tif` na mesma pasta.

**Saída**:

- `saida.gpkg`

---

## Pipeline detalhado

### `main.py`

1. **Busca raster**: procura `*.tif` e `*.tiff` na pasta.
2. **Filtro Majority 3x3**:
   - Para cada célula, escolhe a classe mais frequente na janela 3x3.
   - Ignora `nodata`.
3. **Remove regiões pequenas**:
   - Para cada classe (exceto `nodata`), faz rotulagem por conectividade e remove componentes cuja área estimada seja menor que `MIN_AREA_M2`.
4. **Polygonize (shapes)**:
   - Converte regiões classificadas em polígonos.
5. **Dissolve por classe**:
   - Une polígonos da mesma classe e calcula `area_m2`.
6. **Salva**:
   - `GPKG` e `SHP`.

### `main2.py`

Segue o mesmo pipeline do `main.py`, com a diferença principal no **filtro 3x3**:

- CPU: majority via `generic_filter`.
- GPU (quando disponível): aproxima o efeito usando um `median_filter` no CuPy.

---

### `testando3.py`

1. Abre `raster2.tif`.
2. Para cada classe (exceto `0`):
   - Cria máscara (`img == classe`).
   - Obtém contornos com `measure.find_contours`.
   - Converte coordenadas de pixel para coordenadas do mundo usando `rasterio.transform.xy`.
   - Cria `Polygon`, simplifica (`simplify(1.0)`) e valida (`is_valid`).
3. Salva todas as geometrias em `saida.gpkg`.

---

## Configurações (principalmente `main.py`/`main2.py`)

Em `main.py`:

- `MIN_AREA_M2 = 25` — remove componentes menores que esta área.
- `USE_8_CONN = True` — usa conectividade 8 (via `label(..., connectivity=2)`).
- `NODATA_DEFAULT = 255` — valor padrão para `nodata` (o script tenta sobrescrever com `src.nodata`).
- `OUTPUT_NAME = "resultado_vetorizado"`.

Em `main2.py`:

- `MIN_AREA_M2 = 25`
- `NODATA = 255` (mas pode ser sobrescrito por `src.nodata`)
- `OUTPUT_NAME = "resultado_vetorizado"`

---

## Notas técnicas

- **Nodata**: se o raster tiver `nodata` declarado, o script tenta respeitar.
- **Conectividade**: influencia como componentes são rotuladas (8-vizinhos vs 4-vizinhos).
- **Performance**: o `polygonize` usa `rasterio.features.shapes` e pode gerar muitos polígonos; para rasters grandes, o tempo/memória pode ser relevante.
- **`testando3.py`**: gera polígonos diretamente a partir de contornos; pode produzir geometrias complexas dependendo da resolução e do ruído do raster.

---

## Como usar em projetos maiores

- Recomenda-se criar um diretório de entrada/saída ou renomear o `OUTPUT_NAME` conforme necessário.
- Considere encapsular execução em uma CLI (`argparse`) caso pretenda automatizar.

---

## Licença

Sem licença definida neste repositório.

