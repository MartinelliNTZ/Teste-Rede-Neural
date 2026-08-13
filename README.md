# Ferramentas de Processamento de Nuvens de Pontos (.LAS/.LAZ)

Este repositório contém scripts Python para:
1) **Analisar** nuvens de pontos LAZ/LAS e listar estatísticas simples por atributo.
2) **Dividir** uma nuvem LAZ/LAS entre múltiplas **trajetórias de drone** usando o arquivo de posição (**.pos**) e o `gps_time` dos pontos.

---

## 1) `analyze_pointclouds.py` — Análise de atributos da nuvem

### O que faz
Para cada arquivo `.laz` (e apenas `.laz`), o script:
- Abre o arquivo com `laspy`.
- Lê a quantidade total de pontos (`header.point_count`).
- Percorre os pontos em **chunks** para não estourar a RAM.
- Descobre, a partir do primeiro chunk, quais dimensões (dim) existem:
  - **Numéricas** (ex.: coordenadas, intensidades, etc.)
  - **“Text/Flags”**: algumas flags típicas de LAS são tratadas como dimensões “não-numéricas” para fins do relatório (o script não calcula média de verdade, apenas registra que a dimensão existe).
- Para cada dimensão numérica:
  - Soma os valores válidos (ignorando `NaN`/infinito)
  - Conta quantos valores válidos existem
  - Calcula a **média**.
- Para as dimensões de flags/text:
  - Avisa que a dimensão existe
  - Reporta `média = 0.0` (o script explicitamente faz isso; serve mais para “detectar presença”).

### Saída
Um relatório em tabela, com colunas:
- `Atributo`
- `Média`
- `Tipo` (`numeric` ou `text`)

### O que não faz
- **Não altera nem salva** nenhum arquivo.

### Dependências
```bash
pip install laspy[lazrs] numpy
```

### Uso
```bash
python analyze_pointclouds.py
```

---

## 2) `split_by_trajectory.py` (no arquivo `A.PY`) — Divisão da nuvem por trajetórias

> Obs.: no seu diretório atual, esse script está presente como `A.PY`, mas o conteúdo é de um arquivo chamado `split_by_trajectory.py` (consta no cabeçalho do próprio código).

### O que faz
Esse script divide **uma** nuvem `.laz`/`.las` em **vários arquivos de saída**, um por trajetória, usando:
- Os arquivos `*.pos` na pasta `trajetorias/`
- O `gps_time` de cada ponto do `.laz/.las`
- A **posição interpolada** da trajetória (x,y,z) ao longo do tempo

Em resumo:
1. Lê todas as trajetórias `trajetorias/*.pos`.
2. Para cada `.pos`, extrai do nome do arquivo um intervalo `t_start` e `t_end`.
3. A partir das linhas do `.pos`, faz interpolação linear para obter `x(t)`, `y(t)`, `z(t)`.
4. Percorre a nuvem em chunks e, para cada ponto (pelo `gps_time`):
   - só considera trajetórias cujo intervalo de tempo cobre o ponto (com folga `TIME_MARGIN`)
   - calcula a distância 3D entre o ponto e a posição interpolada da trajetória no mesmo tempo
   - atribui o ponto à trajetória com **menor distância**
5. Os pontos que não encaixarem em nenhuma trajetória vão para um arquivo `__orphans.laz`.

### Correções importantes incluídas no script
O código indica (e faz) algumas correções para evitar problemas comuns no Windows e com ferramentas GIS:
- Não duplica trajetórias no Windows (problema de `glob` case-insensitive).
- Copia o **header completo**/compatível do LAZ original (VLRs, projeção, offsets, escalas).
- Produz arquivos que “abrem corretamente” em softwares como LiDAR 360.

### Dependências
```bash
pip install laspy[lazrs] numpy scipy
```

### Estrutura esperada de arquivos
- Coloque **o script** na mesma pasta do seu `.laz`/`.las`.
- Crie uma subpasta:
  - `trajetorias/`
- Coloque nessa pasta arquivos:
  - `*.pos`

### Uso
```bash
python split_by_trajectory.py
```

(na prática: execute o arquivo `A.PY`, pois é onde o script está no repositório atual)

### Arquivos de saída
Para um arquivo de entrada `minhaNuvem.laz`, o script grava:
- `minhaNuvem__<nome_da_trajetoria>.laz` (um por trajetória)
- `minhaNuvem__orphans.laz` (pontos sem atribuição)

---

## Nota sobre classes
Os dois scripts presentes neste repositório são implementados majoritariamente com **funções** (não há uma estrutura grande de classes no código que foi lido). Se você quiser, posso refatorar para uma arquitetura com classes (ex.: `Trajectory`, `LazAnalyzer`, `LazSplitter`) e atualizar este README com a visão orientada a objetos.

