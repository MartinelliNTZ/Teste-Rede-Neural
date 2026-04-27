# Manual de Instruções - Classificador de Imagens com Redes Neurais

Este manual foi criado para guiar você, usuário final, no uso do notebook `Minicurso_IA9_intro_1.ipynb`. Aqui você encontrará as instruções passo a passo para testar o classificador com **outras imagens** e **outros vetores** (shapefiles) de sua escolha.

---

## 1. O que você precisa antes de começar

Antes de abrir o notebook, certifique-se de ter:

- Uma conta no **Google** (para usar o Google Colab)
- Acesso ao **Google Drive** (para armazenar seus arquivos)
- Uma **imagem raster** no formato `.tif` (GeoTIFF) — por exemplo, uma imagem de satélite ou drone
- Um **shapefile de pontos** (ou polígonos) indicando as áreas de amostra das classes que você deseja mapear (ex: solo, vegetação, água, etc.)

> **Dica:** O shapefile deve conter uma coluna que identifique a classe de cada ponto. Se não tiver, você precisará adicionar essa informação manualmente.

---

## 2. Preparando seus dados

### 2.1 Organizando a pasta no Google Drive

1. Acesse seu [Google Drive](https://drive.google.com)
2. Crie uma pasta para o projeto, por exemplo: `Meu_Classificador_IA`
3. Dentro dessa pasta, crie uma subpasta chamada `Dados`
4. Coloque seus arquivos dentro de `Dados`:
   - A imagem raster (ex: `minha_imagem.tif`)
   - Os shapefiles de amostra (ex: `classe_solo.shp`, `classe_vegetacao.shp`)

> **Atenção:** Shapefiles são compostos por vários arquivos (`.shp`, `.shx`, `.dbf`, `.prj`, `.cpg`). **Envie todos eles** para o Drive, não apenas o `.shp`.

### 2.2 Requisitos da imagem raster

- **Formato:** GeoTIFF (`.tif`)
- **Bandas:** Preferencialmente 3 bandas (Vermelho, Verde, Azul — RGB). O modelo atual foi treinado com 3 bandas, mas pode ser adaptado.
- **Resolução:** Quanto maior a resolução, melhor o detalhe, mas maior o tempo de processamento.
- **Tamanho:** Se a imagem for muito grande, o processamento pode demorar. Para testes iniciais, use imagens de até algumas centenas de megabytes.

### 2.3 Requisitos dos shapefiles

- **Formato:** Shapefile de pontos (ou polígonos)
- **Conteúdo:** Cada ponto representa uma amostra de campo de uma classe específica
- **Coluna de classe:** É ideal que o shapefile tenha uma coluna indicando a classe (ex: `classe`, `tipo`, `id`). Se não tiver, você precisará criar essa coluna.
- **Sistema de Coordenadas (CRS):** O shapefile deve estar no mesmo sistema de coordenadas da imagem raster. Se não estiver, o notebook fará a conversão automaticamente, mas é melhor já deixar tudo alinhado.

---

## 3. Abrindo e configurando o notebook

### 3.1 Abrindo no Google Colab

1. Faça upload do arquivo `Minicurso_IA9_intro_1.ipynb` para o seu Google Drive (na mesma pasta do projeto)
2. Clique com o botão direito no arquivo e selecione **"Abrir com" > "Google Colaboratory"**
3. O notebook será aberto em uma nova aba do navegador

### 3.2 Conectando o Google Drive

No notebook, a primeira célula de código instala a biblioteca `rasterio`. A segunda célula conecta seu Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Ao executar essa célula, um link será exibido. Clique no link, faça login com sua conta Google, copie o código de autorização e cole no campo indicado no Colab.

### 3.3 Alterando os caminhos dos arquivos

Localize no notebook a célula onde são definidos os caminhos:

```python
path_img = '/content/drive/MyDrive/Datasets/Cana_solo_veg/AOI.tif'
path_classe1 = '/content/drive/MyDrive/Datasets/Cana_solo_veg/Solo.shp'
path_classe2 = '/content/drive/MyDrive/Datasets/Cana_solo_veg/Veg.shp'
```

Altere esses caminhos para apontar para **seus arquivos** no Google Drive. Por exemplo:

```python
path_img = '/content/drive/MyDrive/Meu_Classificador_IA/Dados/minha_imagem.tif'
path_classe1 = '/content/drive/MyDrive/Meu_Classificador_IA/Dados/classe_solo.shp'
path_classe2 = '/content/drive/MyDrive/Meu_Classificador_IA/Dados/classe_vegetacao.shp'
```

> **Dica:** Para descobrir o caminho exato, vá até o arquivo no Google Drive, clique com o botão direito e escolha **"Copiar caminho"**.

---

## 4. Executando o notebook passo a passo

### 4.1 Instalando dependências

Execute a primeira célula de código (com `!pip install rasterio`). Isso instala a biblioteca necessária para ler imagens raster.

### 4.2 Importando bibliotecas

Execute a célula de importações. Não é necessário alterar nada aqui, a menos que você queira adicionar outras bibliotecas.

### 4.3 Carregando e visualizando os dados

Execute as células seguintes até a visualização. Se tudo estiver correto, você verá:
- A imagem raster exibida
- Os pontos do shapefile sobrepostos à imagem (em vermelho e verde)

Se os pontos não aparecerem no lugar certo, verifique se o sistema de coordenadas (CRS) do shapefile está correto.

### 4.4 Definindo as classes

No notebook, as classes são numeradas:
- `id = 0` → Classe 1 (ex: Solo)
- `id = 1` → Classe 2 (ex: Vegetação)

Se você tiver **mais de duas classes**, adicione novas linhas:

```python
gdf3 = gpd.read_file(path_classe3)
gdf3['id'] = 2  # Terceira classe
```

E inclua `gdf3` na concatenação:

```python
gdf = pd.concat([gdf1, gdf2, gdf3], axis=0)
```

> **Importante:** Se você tiver mais de 2 classes, a camada de saída da rede neural precisará ser alterada. Isso será explicado na seção 5.

### 4.5 Treinando o modelo

Execute as células de treinamento. O modelo será treinado por 100 épocas (iterações). Você pode acompanhar a evolução pelo gráfico de `loss` e `accuracy`.

Se o modelo estiver com **overfitting** (quando a linha de validação sobe enquanto a de treino desce), você pode:
- Reduzir o número de épocas
- Diminuir a quantidade de neurônios nas camadas
- Adicionar mais dados de amostra

### 4.6 Aplicando à imagem e salvando o resultado

Ao final, o modelo será aplicado à imagem completa e o resultado será salvo como um novo arquivo GeoTIFF (`mapa.tif`).

Você pode baixar esse arquivo para seu computador ou visualizá-lo diretamente no Google Drive.

---

## 5. Adaptando para outras imagens e vetores

### 5.1 Usando uma nova imagem raster

1. Substitua o arquivo `AOI.tif` pela sua nova imagem
2. Atualize o caminho em `path_img`
3. **Verifique se a nova imagem tem o mesmo número de bandas** (o modelo espera 3 bandas RGB)
4. Se a imagem tiver uma banda alfa/máscara (como no exemplo), mantenha o tratamento da máscara. Se não tiver, remova o filtro da máscara.

### 5.2 Usando novos shapefiles

1. Prepare seus shapefiles com amostras representativas de cada classe
2. Certifique-se de que cada shapefile tenha pontos suficientes (quanto mais amostras, melhor o treinamento)
3. Atualize os caminhos `path_classe1`, `path_classe2`, etc.
4. Se necessário, adicione mais classes seguindo o exemplo da seção 4.4

### 5.3 Alterando o número de classes (mais de 2)

Se você tem 3 ou mais classes, precisa ajustar a **camada de saída** da rede neural:

**Original (2 classes — classificação binária):**
```python
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', ...)
```

**Para 3 ou mais classes (classificação multiclasse):**
```python
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', ...)
```

E também converter os labels para formato **one-hot encoding**:
```python
from tensorflow.keras.utils import to_categorical
Y_train = to_categorical(Y_train, num_classes=num_classes)
Y_test = to_categorical(Y_test, num_classes=num_classes)
```

> **Nota:** `num_classes` é a quantidade total de classes diferentes que você possui.

---

## 6. Interpretando os resultados

### 6.1 Gráficos de treinamento

- **Loss (Perda):** Deve diminuir com o tempo. Se a linha de validação subir, é sinal de overfitting.
- **Accuracy (Acurácia):** Deve aumentar. Valores acima de 0.8 (80%) geralmente indicam um bom modelo.

### 6.2 Classification Report

- **Precision:** Das vezes que o modelo previu uma classe, quantas estavam corretas?
- **Recall:** De todas as amostras reais de uma classe, quantas o modelo encontrou?
- **F1-Score:** Equilíbrio entre Precision e Recall. Quanto mais próximo de 1.0, melhor.

### 6.3 Matriz de Confusão

Mostra quantas amostras de cada classe foram classificadas corretamente e quantas foram confundidas com outras classes.

### 6.4 Mapa final (`mapa.tif`)

- Cada pixel da imagem recebeu um valor: 0, 1, 2, etc., correspondente à classe prevista
- Você pode abrir esse arquivo no **QGIS** ou **ArcGIS** para visualizar o mapa classificado
- As cores podem ser ajustadas no software GIS de sua preferência

---

## 7. Dicas e solução de problemas

| Problema | Possível causa | Solução |
|----------|---------------|---------|
| Erro ao abrir shapefile | Faltando arquivos auxiliares (.shx, .dbf) | Envie todos os arquivos do shapefile para o Drive |
| Pontos não aparecem na imagem | CRS diferente entre imagem e shapefile | Converta o shapefile para o mesmo CRS da imagem |
| Acurácia muito baixa | Poucas amostras ou amostras ruins | Colete mais pontos de amostra representativos |
| Overfitting | Modelo muito complexo para poucos dados | Reduza o número de neurônios ou épocas |
| Erro "input shape" | Número de bandas da nova imagem é diferente | Verifique se a imagem tem 3 bandas (RGB) |
| Memória insuficiente | Imagem muito grande | Recorte a imagem em partes menores antes de processar |

---

## 8. Fluxo resumido para testar com seus dados

1. **Prepare seus dados:** imagem `.tif` + shapefiles de amostra
2. **Organize no Drive:** crie uma pasta e envie todos os arquivos
3. **Abra o notebook** no Google Colab
4. **Conecte seu Drive** e atualize os caminhos dos arquivos
5. **Execute as células** uma a uma, verificando os resultados
6. **Ajuste as classes** se necessário (adicione mais se tiver)
7. **Treine o modelo** e observe os gráficos
8. **Gere o mapa final** e baixe o arquivo `.tif`
9. **Visualize no QGIS/ArcGIS** e analise os resultados

---

## 9. Contato e suporte

Se você encontrar dificuldades ou tiver dúvidas sobre adaptações mais avançadas (mais bandas, mais classes, outros tipos de modelo), consulte a documentação das bibliotecas utilizadas (Rasterio, GeoPandas, Keras/TensorFlow) ou procure suporte especializado em processamento de imagens e aprendizado de máquina.

---

**Bom trabalho e bons mapeamentos!**

