# -*- coding: utf-8 -*-
#CRIADO PELO CHAGPT
"""
CLASSIFICADOR RASTER NN v6
=========================================================
Reformulação completa solicitada:
- Compressão GeoTIFF LZW + predictor + tiled + BIGTIFF
- Overviews internos: 2,4,8,16,32,64
- Script modularizado
- EarlyStopping + ReduceLROnPlateau
- Checkpoint automático
- Limpeza de memória
- Logs melhores
- Robustez maior
- Correções gerais
- Saída otimizada para QGIS / ArcGIS / GDAL

Dependências:
pip install rasterio geopandas tensorflow scikit-learn pandas numpy matplotlib seaborn psutil
"""

import os
import gc
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import psutil
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# =========================================================
# SILENCIAR TF
# =========================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# =========================================================
# BASE DIR
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# =========================================================
# CONFIG
# =========================================================

CONFIG = {

    # IMAGENS
    "path_img": "dados/imagemTreino.tif",
    "path_img_teste": "dados/imagemCompleta.tif",
    "path_saida": "resultado/mapa_classificado_v6_CHATGPT.tif",

    # SHAPEFILES
    "shapefiles": [
        ["dados/solo.shp", 0],
        ["dados/floresta.shp", 1],
        ["dados/palhada.shp", 2],
        ["dados/daninhas.shp", 3],
    ],

    # MASCARA
    "usar_mascara": True,
    "valor_valido": 255,

    # RAM
    "ram_limite_pct": 70,

    # TREINO
    "test_size": 0.30,
    "random_state": 42,
    "epochs": 300,
    "batch_size_treino": 128,

    # PREDICAO
    "batch_size_predicao": 8192,

    # REDE
    "camadas_ocultas": [128, 64, 32, 16, 8],
    "activation": "relu",
    "dropout_rate": 0.15,

    # MODELO
    "salvar_modelo": True,
    "path_modelo": "resultado/modelo_v6.keras",
}

# =========================================================
# UTIL
# =========================================================

def sep(n=70):
    print("=" * n)


def fmt_tempo(seg):
    h = int(seg // 3600)
    m = int((seg % 3600) // 60)
    s = seg % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def caminho(p):
    p = Path(p)
    if p.is_absolute():
        return p
    return BASE_DIR / p


# =========================================================
# HARDWARE
# =========================================================

def setup_hardware(ram_pct):

    mem = psutil.virtual_memory()

    info = {
        "ram_total": mem.total,
        "ram_total_gb": mem.total / 1024**3,
        "ram_limite": mem.total * ram_pct / 100
    }

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        info["device"] = "GPU"
        info["gpu_count"] = len(gpus)
    else:
        info["device"] = "CPU"
        info["gpu_count"] = 0

    cpu = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(cpu)
    tf.config.threading.set_inter_op_parallelism_threads(cpu)

    info["cpu_threads"] = cpu

    sep()
    print("HARDWARE")
    sep()
    print("RAM TOTAL :", round(info["ram_total_gb"], 2), "GB")
    print("RAM LIMITE:", round(info["ram_limite"] / 1024**3, 2), "GB")
    print("DEVICE    :", info["device"])
    print("THREADS   :", info["cpu_threads"])
    sep()

    return info


# =========================================================
# CLASSIFICADOR
# =========================================================

class RasterNN:

    def __init__(self, cfg, hw):

        self.cfg = cfg
        self.hw = hw

        self.path_img = caminho(cfg["path_img"])
        self.path_img_teste = caminho(cfg["path_img_teste"])
        self.path_saida = caminho(cfg["path_saida"])
        self.path_modelo = caminho(cfg["path_modelo"])

        self.shapefiles = [(caminho(a), b) for a, b in cfg["shapefiles"]]

        self.usar_mascara = cfg["usar_mascara"]
        self.valor_valido = cfg["valor_valido"]

        self.src = None
        self.model = None

        self.X = None
        self.Y = None

        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

        self.num_classes = 0
        self.n_features = 0

        self.nomes = {}

    # -----------------------------------------------------

    def abrir_treino(self):
        self.src = rasterio.open(self.path_img)

    # -----------------------------------------------------

    def carregar_amostras(self):

        lista = []

        for shp, cls in self.shapefiles:

            gdf = gpd.read_file(shp)

            if gdf.crs != self.src.crs:
                gdf = gdf.to_crs(self.src.crs)

            gdf["id"] = cls
            lista.append(gdf)

            self.nomes[cls] = shp.stem

        self.gdf = pd.concat(lista, ignore_index=True)

    # -----------------------------------------------------

    def extrair(self):

        coords = [(g.x, g.y) for g in self.gdf.geometry]

        valores = np.array(list(self.src.sample(coords)))

        if self.usar_mascara:
            self.n_features = self.src.count - 1
            self.X = valores[:, :self.n_features]
        else:
            self.n_features = self.src.count
            self.X = valores

        self.Y = self.gdf["id"].values

    # -----------------------------------------------------

    def split(self):

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X,
            self.Y,
            test_size=self.cfg["test_size"],
            random_state=self.cfg["random_state"],
            stratify=self.Y
        )

        self.num_classes = len(np.unique(self.Y))

    # -----------------------------------------------------

    def construir(self):

        model = Sequential()
        model.add(Input(shape=(self.n_features,)))

        for n in self.cfg["camadas_ocultas"]:
            model.add(Dense(n, activation=self.cfg["activation"]))
            model.add(Dropout(self.cfg["dropout_rate"]))

        if self.num_classes == 2:
            model.add(Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
        else:
            model.add(Dense(self.num_classes, activation="softmax"))
            loss = "sparse_categorical_crossentropy"

        model.compile(
            optimizer="adam",
            loss=loss,
            metrics=["accuracy"]
        )

        self.model = model

    # -----------------------------------------------------

    def treinar(self):

        self.path_modelo.parent.mkdir(parents=True, exist_ok=True)

        callbacks = [

            EarlyStopping(
                monitor="val_loss",
                patience=20,
                restore_best_weights=True
            ),

            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=8,
                verbose=1
            ),

            ModelCheckpoint(
                filepath=str(self.path_modelo),
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            )
        ]

        self.history = self.model.fit(
            self.X_train,
            self.Y_train,
            validation_split=0.25,
            epochs=self.cfg["epochs"],
            batch_size=self.cfg["batch_size_treino"],
            verbose=1,
            callbacks=callbacks
        )

    # -----------------------------------------------------

    def avaliar(self):

        pred = self.model.predict(self.X_test, verbose=0)

        if self.num_classes == 2:
            y_pred = (pred > 0.5).astype(int).flatten()
        else:
            y_pred = np.argmax(pred, axis=1)

        acc = accuracy_score(self.Y_test, y_pred)

        sep()
        print("ACURACIA:", round(acc * 100, 2), "%")
        sep()

        nomes = [self.nomes[i] for i in sorted(self.nomes)]

        print(classification_report(
            self.Y_test,
            y_pred,
            target_names=nomes,
            digits=4
        ))

        cm = confusion_matrix(self.Y_test, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=nomes,
            yticklabels=nomes
        )
        plt.title("Matriz de Confusão")
        plt.tight_layout()
        plt.savefig(BASE_DIR / "resultado/matriz_confusao.png", dpi=180)
        plt.close()

    # -----------------------------------------------------

    def linhas_chunk(self, largura):

        livre = psutil.virtual_memory().available
        usar = min(livre, self.hw["ram_limite"])

        bytes_linha = largura * self.n_features * 8

        n = int((usar * 0.35) / bytes_linha)

        return max(1, n)

    # -----------------------------------------------------

    def classificar(self):

        self.path_saida.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(self.path_img_teste) as src:

            h = src.height
            w = src.width
            bands = src.count

            out_meta = src.meta.copy()

            out_meta.update({
                "driver": "GTiff",
                "count": 1,
                "dtype": "uint8",
                "compress": "lzw",
                "predictor": 2,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "BIGTIFF": "IF_SAFER",
                "nodata": 255
            })

            linhas = self.linhas_chunk(w)
            total_chunks = math.ceil(h / linhas)

            with rasterio.open(self.path_saida, "w", **out_meta) as dst:

                t0 = time.time()

                for i in range(total_chunks):

                    li = i * linhas
                    lf = min(li + linhas, h)
                    nl = lf - li

                    win = Window(0, li, w, nl)

                    chunk = src.read(window=win)

                    if self.usar_mascara:
                        mask = chunk[-1]
                        feat = chunk[:self.n_features]
                    else:
                        mask = None
                        feat = chunk

                    X = feat.transpose(1, 2, 0).reshape(-1, self.n_features)

                    saida = np.full(X.shape[0], 255, dtype=np.uint8)

                    if mask is not None:
                        valid = mask.reshape(-1) == self.valor_valido
                    else:
                        valid = np.ones(X.shape[0], dtype=bool)

                    if np.any(valid):

                        pred = self.model.predict(
                            X[valid],
                            batch_size=self.cfg["batch_size_predicao"],
                            verbose=0
                        )

                        if self.num_classes == 2:
                            cls = (pred > 0.5).astype(np.uint8).flatten()
                        else:
                            cls = np.argmax(pred, axis=1).astype(np.uint8)

                        saida[valid] = cls

                    saida = saida.reshape(nl, w)

                    dst.write(saida, 1, window=win)

                    pct = (i + 1) / total_chunks * 100
                    dec = time.time() - t0
                    eta = (dec / (i + 1)) * (total_chunks - i - 1)

                    print(
                        f"Chunk {i+1}/{total_chunks} "
                        f"{pct:5.1f}% ETA {fmt_tempo(eta)}",
                        end="\r"
                    )

                    del chunk, feat, X, saida
                    gc.collect()

            print()

        self.criar_overviews()

    # -----------------------------------------------------

    def criar_overviews(self):

        print("Criando overviews internos...")

        fatores = [2, 4, 8, 16, 32, 64]

        with rasterio.open(
            self.path_saida,
            "r+"
        ) as dst:

            dst.build_overviews(
                fatores,
                Resampling.nearest
            )

            dst.update_tags(
                ns="rio_overview",
                resampling="nearest"
            )

        print("Overviews OK ->", fatores)

    # -----------------------------------------------------

    def fechar(self):

        if self.src:
            self.src.close()


# =========================================================
# MAIN
# =========================================================

def main():

    t0 = time.time()

    hw = setup_hardware(CONFIG["ram_limite_pct"])

    clf = RasterNN(CONFIG, hw)

    sep()
    print("INICIANDO")
    sep()

    clf.abrir_treino()
    clf.carregar_amostras()
    clf.extrair()
    clf.split()
    clf.construir()
    clf.treinar()
    clf.avaliar()
    clf.classificar()
    clf.fechar()

    tf.keras.backend.clear_session()
    gc.collect()

    sep()
    print("FINALIZADO")
    print("Tempo total:", fmt_tempo(time.time() - t0))
    print("Saída:", clf.path_saida)
    sep()


if __name__ == "__main__":
    main()