#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLASSIFICADOR RASTER COM GPU (PYTORCH CUDA)
Reescrito 100% para usar:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Substitui TensorFlow por PyTorch.
"""

import os
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ==========================================================
# CONFIG
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

CONFIG = {
    "path_img": "dados/imagemTreino.tif",
    "path_img_teste": "dados/imagemTreino.tif",
    "path_saida": "resultado/mapa_classificado_gpu15.tif",

    "shapefiles": [
        ["dados/solo.shp", 0],
        ["dados/floresta.shp", 1],
    ],

    "usar_mascara": True,
    "valor_minimo_alpha": 250,

    "epochs": 100,
    "batch_size": 4096,
    "lr": 0.001,

    "hidden": [256, 128, 64],
    "dropout": 0.15,

    "chunk_linhas": 512,
}


# ==========================================================
# GPU
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("DISPOSITIVO:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
print("=" * 60)


# ==========================================================
# REDE
# ==========================================================

class Rede(nn.Module):
    def __init__(self, n_in, n_classes, hidden, dropout):
        super().__init__()

        layers = []
        prev = n_in

        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h

        if n_classes == 2:
            layers.append(nn.Linear(prev, 1))
        else:
            layers.append(nn.Linear(prev, n_classes))

        self.net = nn.Sequential(*layers)
        self.n_classes = n_classes

    def forward(self, x):
        return self.net(x)


# ==========================================================
# FUNÇÕES
# ==========================================================

def resolver(p):
    return BASE_DIR / p


def carregar_amostras(src):
    lista = []
    nomes = {}

    for shp, classe in CONFIG["shapefiles"]:
        shp = resolver(shp)

        gdf = gpd.read_file(shp)

        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        gdf["id"] = classe
        nomes[classe] = shp.stem
        lista.append(gdf)

    gdf = pd.concat(lista, ignore_index=True)
    return gdf, nomes


def extrair_pixels(src, gdf):
    coords = [(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
    vals = np.array(list(src.sample(coords)))

    if CONFIG["usar_mascara"]:
        X = vals[:, :-1]
    else:
        X = vals

    y = gdf["id"].values.astype(np.int64)

    return X.astype(np.float32), y


# ==========================================================
# TREINO
# ==========================================================

def treinar_modelo(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    n_in = X.shape[1]
    n_classes = len(np.unique(y))

    model = Rede(
        n_in=n_in,
        n_classes=n_classes,
        hidden=CONFIG["hidden"],
        dropout=CONFIG["dropout"]
    ).to(DEVICE)

    ds = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train)
    )

    dl = DataLoader(
        ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True
    )

    if n_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    print("\nTREINANDO...\n")

    for epoch in range(CONFIG["epochs"]):

        model.train()
        loss_total = 0

        for xb, yb in dl:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            out = model(xb)

            if n_classes == 2:
                loss = criterion(out.squeeze(), yb.float())
            else:
                loss = criterion(out, yb)

            loss.backward()
            optimizer.step()

            loss_total += loss.item()

        print(
            f"Epoch {epoch+1:03d}/{CONFIG['epochs']} "
            f"Loss={loss_total:.4f}"
        )

    # TESTE
    model.eval()

    xt = torch.tensor(X_test).to(DEVICE)

    with torch.no_grad():

        pred = model(xt)

        if n_classes == 2:
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).int().cpu().numpy().flatten()
        else:
            pred = torch.argmax(pred, dim=1).cpu().numpy()

    acc = accuracy_score(y_test, pred)

    print("\nACURÁCIA:", round(acc * 100, 2), "%")
    print()

    print(classification_report(y_test, pred))

    cm = confusion_matrix(y_test, pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusão")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "resultado/matriz_gpu.png", dpi=150)

    return model, n_classes, n_in


# ==========================================================
# CLASSIFICAR IMAGEM
# ==========================================================

def classificar_imagem(model, n_classes, n_feat):

    path = resolver(CONFIG["path_img_teste"])
    saida = resolver(CONFIG["path_saida"])

    with rasterio.open(path) as src:

        meta = src.meta.copy()

        h = src.height
        w = src.width
        bandas = src.count

        meta.update({
            "count": 1,
            "dtype": "uint8",
            "compress": "lzw",
            "nodata": 255
        })

        saida.parent.mkdir(exist_ok=True)

        with rasterio.open(saida, "w", **meta) as dst:

            chunk = CONFIG["chunk_linhas"]

            total = math.ceil(h / chunk)

            for i in range(total):

                y0 = i * chunk
                y1 = min(y0 + chunk, h)

                win = Window(0, y0, w, y1 - y0)

                img = src.read(window=win)

                if CONFIG["usar_mascara"]:
                    alpha = img[-1]
                    feat = img[:-1]
                else:
                    alpha = None
                    feat = img

                feat = feat.transpose(1, 2, 0)
                feat = feat.reshape(-1, n_feat)
                feat = feat.astype(np.float32)

                valid = np.ones(len(feat), dtype=bool)

                if alpha is not None:
                    valid = alpha.reshape(-1) >= CONFIG["valor_minimo_alpha"]

                out = np.full(len(feat), 255, dtype=np.uint8)

                if valid.sum() > 0:

                    x = torch.tensor(feat[valid]).to(DEVICE)

                    with torch.no_grad():

                        pred = model(x)

                        if n_classes == 2:
                            pred = torch.sigmoid(pred)
                            pred = (pred > 0.5).int().cpu().numpy().flatten()
                        else:
                            pred = torch.argmax(pred, dim=1).cpu().numpy()

                    out[valid] = pred.astype(np.uint8)

                out = out.reshape(y1 - y0, w)

                dst.write(out, 1, window=win)

                print(
                    f"Chunk {i+1}/{total} concluído"
                )

    print("\nGeoTIFF salvo em:", saida)


# ==========================================================
# MAIN
# ==========================================================

def main():

    img = resolver(CONFIG["path_img"])

    with rasterio.open(img) as src:

        gdf, nomes = carregar_amostras(src)

        X, y = extrair_pixels(src, gdf)

    model, n_classes, n_feat = treinar_modelo(X, y)

    classificar_imagem(model, n_classes, n_feat)

    print("\nFINALIZADO COM GPU:", DEVICE)


if __name__ == "__main__":
    main()