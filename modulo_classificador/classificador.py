#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Classificação de Imagens Raster com Redes Neurais
============================================================
Executável em máquina local (sem Google Colab).
Pipeline completo: carrega imagem + amostras (shapefiles), treina uma
rede neural densa e classifica a imagem inteira, exportando um GeoTIFF.

Uso rápido:
-----------
1. Edite a seção CONFIGuração no final deste arquivo.
2. Execute: python classificador.py
"""

import os
import sys
import json
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.plot import show
from rasterio.windows import Window
from shapely.geometry import box

import matplotlib
matplotlib.use("Agg")  # backend headless (não abre janelas)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class ClassificadorRaster:
    """
    Encapsula todo o fluxo de classificação supervisionada de imagens raster.
    """

    def __init__(self,
                 path_img: str,
                 shapefiles: List[Tuple[str, int]],
                 path_img_teste: Optional[str] = None,
                 path_saida: str = "mapa_classificado.tif",
                 usar_mascara: bool = True,
                 coluna_mascara: str = "Mask",
                 valor_valido: int = 255):
        """
        Parâmetros
        ----------
        path_img : str
            Caminho da imagem raster (GeoTIFF) de entrada.
        shapefiles : list[tuple[str, int]]
            Lista de tuplas (caminho_shapefile, id_classe).
            Exemplo: [("solo.shp", 0), ("vegetacao.shp", 1)]
        path_img_teste : str, opcional
            Imagem a ser classificada (pode ser a mesma de `path_img`).
            Se None, usa `path_img`.
        path_saida : str
            Caminho do GeoTIFF de saída.
        usar_mascara : bool
            Se True, remove pixels onde a banda alpha != valor_valido.
        coluna_mascara : str
            Nome da coluna no DataFrame que contém a banda alfa.
        valor_valido : int
            Valor considerado "pixel válido" na máscara.
        """
        self.path_img = path_img
        self.shapefiles = shapefiles
        self.path_img_teste = path_img_teste or path_img
        self.path_saida = path_saida
        self.usar_mascara = usar_mascara
        self.coluna_mascara = coluna_mascara
        self.valor_valido = valor_valido

        # atributos preenchidos ao longo do pipeline
        self.src = None                 # objeto rasterio da imagem de treino
        self.gdf = None                 # GeoDataFrame combinado das amostras
        self.X = None                   # features (valores espectrais)
        self.Y = None                   # labels
        self.X_train = self.X_test = None
        self.Y_train = self.Y_test = None
        self.model = None               # modelo Keras
        self.history = None             # histórico de treinamento
        self.img_size = None            # (altura, largura) da imagem de teste
        self.export_image = None        # array 3D resultado (1 banda)

    # ------------------------------------------------------------------
    # 1. Carga de dados
    # ------------------------------------------------------------------
    def carregar_imagem(self):
        """Abre a imagem raster e armazena o objeto rasterio."""
        if not os.path.isfile(self.path_img):
            raise FileNotFoundError(f"Imagem não encontrada: {self.path_img}")
        self.src = rasterio.open(self.path_img)
        print(f"[OK] Imagem carregada: {self.path_img}")
        print(f"     Formato: {self.src.shape} | CRS: {self.src.crs}")

    def carregar_amostras(self):
        """Lê todos os shapefiles, converte CRS, atribui IDs e concatena."""
        lista_gdf = []
        for shp_path, classe_id in self.shapefiles:
            if not os.path.isfile(shp_path):
                raise FileNotFoundError(f"Shapefile não encontrado: {shp_path}")
            gdf = gpd.read_file(shp_path)
            # garante mesmo CRS da imagem
            if self.src is not None and gdf.crs != self.src.crs:
                gdf = gdf.to_crs(self.src.crs.to_dict())
            gdf["id"] = classe_id
            lista_gdf.append(gdf)
            print(f"[OK] Shapefile '{shp_path}' → classe {classe_id} ({len(gdf)} pts)")

        self.gdf = pd.concat(lista_gdf, axis=0, ignore_index=True)
        print(f"[OK] Total de amostras combinadas: {len(self.gdf)}")

    # ------------------------------------------------------------------
    # 2. Extração de valores espectrais
    # ------------------------------------------------------------------
    def extrair_valores(self):
        """Extrai os valores de pixel para cada coordenada de amostra."""
        if self.src is None or self.gdf is None:
            raise RuntimeError("Carregue a imagem e as amostras primeiro.")

        coord_list = [(x, y) for x, y in zip(self.gdf.geometry.x,
                                              self.gdf.geometry.y)]
        valores = []
        for val in self.src.sample(coord_list):
            valores.append(val)

        valores = np.array(valores)
        # assume que a última coluna pode ser alpha/máscara — ajuste se necessário
        # por padrão guardamos tudo; o recorte correto é feito no preparar_dados
        self.X = valores.copy()
        self.Y = self.gdf["id"].values[:, np.newaxis]
        print(f"[OK] Extraídos {self.X.shape[0]} vetores de {self.X.shape[1]} bandas.")

    # ------------------------------------------------------------------
    # 3. Preparação / divisão treino-teste
    # ------------------------------------------------------------------
    def preparar_dados(self, test_size: float = 0.3, random_state: int = 42):
        """Separa treino/teste e calcula dimensões do modelo."""
        if self.X is None or self.Y is None:
            raise RuntimeError("Extraia os valores antes de preparar os dados.")

        # Se houver banda alpha, você pode descartá-la aqui:
        # ex: self.X = self.X[:, :-1]   # remove última banda
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=random_state
        )
        self.input_shape = (self.X_train.shape[1],)
        self.num_classes = len(np.unique(self.gdf["id"].values))
        print(f"[OK] Dados preparados: {len(self.X_train)} treino | {len(self.X_test)} teste")
        print(f"     Input shape: {self.input_shape} | Classes: {self.num_classes}")

    # ------------------------------------------------------------------
    # 4. Modelo
    # ------------------------------------------------------------------
    def construir_modelo(self,
                         camadas_ocultas: List[int] = None,
                         activation: str = "relu"):
        """Monta a rede neural densa."""
        if camadas_ocultas is None:
            camadas_ocultas = [256, 128, 64, 8]

        model = Sequential()
        # primeira camada
        model.add(Dense(camadas_ocultas[0],
                        input_shape=self.input_shape,
                        activation=activation))
        # camadas ocultas restantes
        for unidades in camadas_ocultas[1:]:
            model.add(Dense(unidades, activation=activation))

        # camada de saída
        if self.num_classes == 2:
            model.add(Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
        else:
            model.add(Dense(self.num_classes, activation="softmax"))
            loss = "sparse_categorical_crossentropy"  # labels inteiros

        model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])
        self.model = model
        print("[OK] Modelo construído.")
        self.model.summary()

    # ------------------------------------------------------------------
    # 5. Treinamento
    # ------------------------------------------------------------------
    def treinar(self, epochs: int = 100, batch_size: int = 250,
                validation_split: float = 0.25, verbose: int = 1):
        """Executa o fit do modelo."""
        if self.model is None:
            raise RuntimeError("Construa o modelo antes de treinar.")
        self.history = self.model.fit(
            self.X_train, self.Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        print("[OK] Treinamento concluído.")

    # ------------------------------------------------------------------
    # 6. Avaliação
    # ------------------------------------------------------------------
    def avaliar(self):
        """Gera gráficos de loss/accuracy, classification report e confusion matrix."""
        if self.history is None:
            raise RuntimeError("Treine o modelo antes de avaliar.")

        # gráficos
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].plot(self.history.history["loss"], color="b", label="Training loss")
        ax[0].plot(self.history.history["val_loss"], color="r", label="Validation loss")
        ax[0].legend(loc="best", shadow=True)
        ax[0].set_title("Loss")

        ax[1].plot(self.history.history["accuracy"], color="b", label="Training accuracy")
        ax[1].plot(self.history.history["val_accuracy"], color="r", label="Validation accuracy")
        ax[1].legend(loc="best", shadow=True)
        ax[1].set_title("Accuracy")
        plt.tight_layout()
        plt.savefig("graficos_treinamento.png", dpi=150)
        print("[OK] Gráficos salvos em: graficos_treinamento.png")

        # métricas no conjunto de teste
        score = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        print(f"\nTest loss: {score[0]:.4f} | Test accuracy: {score[1]:.4f}")

        y_pred = self.model.predict(self.X_test)
        if self.num_classes == 2:
            y_pred = np.round(y_pred).astype(int)
        else:
            y_pred = np.argmax(y_pred, axis=1)

        print("\nClassification Report:")
        print(classification_report(self.Y_test, y_pred))

        cm = confusion_matrix(self.Y_test, y_pred)
        nomes = [f"Classe {c}" for c in sorted(self.gdf["id"].unique())]
        df_cm = pd.DataFrame(cm, index=nomes, columns=nomes)

        plt.figure(figsize=(8, 8))
        sns.heatmap(df_cm, annot=True, annot_kws={"size": 18},
                    fmt="d", cmap="Blues", cbar=False)
        plt.ylabel("True")
        plt.xlabel("Predict")
        plt.tight_layout()
        plt.savefig("matriz_confusao.png", dpi=150)
        print("[OK] Matriz de confusão salva em: matriz_confusao.png")

    # ------------------------------------------------------------------
    # 7. Predição na imagem completa
    # ------------------------------------------------------------------
    def prever_imagem(self):
        """Aplica o modelo treinado a toda a imagem de teste."""
        print(f"[INFO] Abrindo imagem de teste: {self.path_img_teste}")
        with rasterio.open(self.path_img_teste) as src_test:
            img = src_test.read()                       # (bandas, altura, largura)
            img = img.transpose([1, 2, 0])              # (altura, largura, bandas)
            self.img_size = (img.shape[0], img.shape[1])
            img = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

        # DataFrame com colunas genéricas B0, B1, B2 ...
        cols = [f"B{i}" for i in range(img.shape[1])]
        df = pd.DataFrame(img, columns=cols)

        # se existir banda alpha/máscara, supomos que é a última banda
        if self.usar_mascara and img.shape[1] > self.input_shape[0]:
            # a última coluna vira máscara
            df.rename(columns={cols[-1]: self.coluna_mascara}, inplace=True)
            df_pred = df[df[self.coluna_mascara] == self.valor_valido].copy()
            valores = df_pred.values[:, :self.input_shape[0]]
        else:
            df_pred = df.copy()
            valores = df_pred.values

        # predição
        print(f"[INFO] Classificando {len(valores)} pixels válidos...")
        pred = self.model.predict(valores)
        if self.num_classes == 2:
            pred = np.round(pred).astype(int).flatten()
        else:
            pred = np.argmax(pred, axis=1)

        # mescla de volta ao DataFrame completo
        df_pred["pred"] = pred
        df = pd.merge(df, df_pred, how="left", left_index=True, right_index=True)

        # reorganiza para (1, altura, largura)
        classify = df["pred"].values.reshape(self.img_size)
        self.export_image = classify[np.newaxis, :, :].astype("float64")
        print("[OK] Predição na imagem completa concluída.")

    # ------------------------------------------------------------------
    # 8. Exportação
    # ------------------------------------------------------------------
    def salvar_resultado(self):
        """Salva o mapa classificado como GeoTIFF."""
        with rasterio.open(self.path_img) as src:
            out_meta = src.meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "height": self.export_image.shape[1],
            "width": self.export_image.shape[2],
            "compress": "lzw",
            "nodata": np.nan,
            "dtype": "float64",
            "count": 1,
        })

        with rasterio.open(self.path_saida, "w", **out_meta) as dest:
            dest.write(self.export_image)
        print(f"[OK] Mapa salvo em: {self.path_saida}")


# ---------------------------------------------------------------------------
# Ponto de entrada (exemplo de uso)
# ---------------------------------------------------------------------------

def main():
    # ===================================================================
    # CONFIGURAÇÃO — ALTERE AQUI PARA SEUS DADOS
    # ===================================================================
    config = {
        "path_img": "./dados/imagem.tif",                       # imagem raster
        "shapefiles": [
            ("./dados/solo.shp", 0),                             # (arquivo, id_classe)
            ("./dados/vegetacao.shp", 1),
        ],
        "path_img_teste": "./dados/imagem.tif",                  # imagem a classificar (pode ser a mesma)
        "path_saida": "./resultado/mapa_classificado.tif",       # arquivo de saída
        "usar_mascara": False,                                   # True se houver banda alpha
        "test_size": 0.3,
        "random_state": 42,
        "epochs": 100,
        "batch_size": 250,
    }
    # ===================================================================

    os.makedirs(os.path.dirname(config["path_saida"]), exist_ok=True)

    # inicializa pipeline
    clf = ClassificadorRaster(
        path_img=config["path_img"],
        shapefiles=config["shapefiles"],
        path_img_teste=config["path_img_teste"],
        path_saida=config["path_saida"],
        usar_mascara=config["usar_mascara"]
    )

    # execução sequencial
    clf.carregar_imagem()
    clf.carregar_amostras()
    clf.extrair_valores()
    clf.preparar_dados(test_size=config["test_size"], random_state=config["random_state"])
    clf.construir_modelo()
    clf.treinar(epochs=config["epochs"], batch_size=config["batch_size"])
    clf.avaliar()
    clf.prever_imagem()
    clf.salvar_resultado()

    print("\n=== PIPELINE FINALIZADO COM SUCESSO ===")


if __name__ == "__main__":
    main()

