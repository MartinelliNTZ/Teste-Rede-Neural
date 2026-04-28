#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classificador Raster com Redes Neurais — v4
============================================
Combina a CORRETUDE da v1 (sem normalização, que já funcionava) com
a eficiência de memória da v2/v3 (predição em chunks, sem Pandas).

Por que sem normalização?
  A v1 classificava corretamente sem normalizar. A tentativa de
  normalizar na v2/v3 introduziu um bug grave: a banda com desvio
  padrão zero (ex: banda alpha constante = 255) gera divisão por ~0,
  produzindo NaN/Inf e destruindo a classificação.
  Para dados espectrais em faixa 0-255, a rede aprende bem sem
  normalização — como provado pela v1.

Uso (edite CONFIG abaixo e execute):
    python classificador_v4.py

Ou via CLI:
    python classificador_v4.py --imagem dados/img.tif \
        --shapefile dados/solo.shp,0 --shapefile dados/veg.shp,1

Ou via JSON externo (sobrescreve CONFIG embutido):
    python classificador_v4.py --config meu_config.json
"""

import os
import sys
import json
import time
import math
import argparse
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
import psutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG EMBUTIDO — edite aqui e execute sem argumentos
# ═══════════════════════════════════════════════════════════════════════════

CONFIG = {
    # Imagem usada para EXTRAIR AMOSTRAS (treino)
    "path_img": "dados/imagemFull.tif",

    # Imagem a ser CLASSIFICADA (pode ser a mesma que path_img)
    "path_img_teste": "dados/imagemFull.tif",

    # Onde salvar o GeoTIFF classificado
    "path_saida": "resultado/mapa_classificado.tif",

    # Amostras: [caminho_shapefile, id_classe]
    "shapefiles": [
        ["dados/solo.shp", 0],  
        ["dados/vegetacao.shp", 1],
        ["dados/palhada.shp", 2]
    ],

    # Máscara: true = filtra pixels pela última banda (canal alpha)
    "usar_mascara": False,
    # Valor da máscara que indica pixel válido
    "valor_valido": 255,

    # Percentual máximo de RAM disponível para cada chunk
    "ram_limite_pct": 70,

    # Divisão treino/teste
    "test_size": 0.3,
    "random_state": 42,

    # Treinamento
    "epochs": 100,
    "batch_size_treino": 250,

    # Predição (pode ser maior que o batch de treino)
    "batch_size_predicao": 4096,

    # Arquitetura da rede: neurônios por camada oculta
    "camadas_ocultas": [256, 128, 64, 8],

    # Função de ativação: "relu", "elu" ou "tanh"
    "activation": "relu",

    # Salvar modelo treinado em disco para reutilização futura
    "salvar_modelo": False,
    "path_modelo": "resultado/modelo.keras",
}


# ═══════════════════════════════════════════════════════════════════════════
# HARDWARE
# ═══════════════════════════════════════════════════════════════════════════

def configurar_hardware(ram_limite_pct: float = 70.0) -> dict:
    info = {}
    mem = psutil.virtual_memory()
    info["ram_total_gb"]     = mem.total / (1024 ** 3)
    info["ram_limite_gb"]    = info["ram_total_gb"] * (ram_limite_pct / 100)
    info["ram_limite_bytes"] = int(info["ram_limite_gb"] * (1024 ** 3))

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            info["device"]    = "GPU"
            info["gpu_count"] = len(gpus)
            info["gpu_nomes"] = [g.name for g in gpus]
        except RuntimeError as e:
            info["device"]    = "CPU"
            info["gpu_erro"]  = str(e)
            info["gpu_count"] = 0
    else:
        info["device"]    = "CPU"
        info["gpu_count"] = 0

    cpu_count = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(cpu_count)
    tf.config.threading.set_inter_op_parallelism_threads(cpu_count)
    info["cpu_threads"] = cpu_count

    _sep()
    print(" CONFIGURACAO DE HARDWARE")
    _sep()
    print(f"  RAM total          : {info['ram_total_gb']:.1f} GB")
    print(f"  RAM limite ({ram_limite_pct:.0f}%)   : {info['ram_limite_gb']:.1f} GB")
    print(f"  Dispositivo TF     : {info['device']}")
    if info["device"] == "GPU":
        print(f"  GPUs               : {info['gpu_count']} -> {info['gpu_nomes']}")
    print(f"  Threads CPU (TF)   : {info['cpu_threads']}")
    _sep()
    return info


def calcular_chunk_linhas(largura: int, n_bandas: int,
                           ram_livre_bytes: int, fator: float = 0.35) -> int:
    """Quantas linhas cabem num chunk usando `fator` da RAM disponível."""
    bytes_por_linha = largura * n_bandas * 8  # float64 = 8 bytes (mesma dtype da v1)
    return max(1, int(ram_livre_bytes * fator / bytes_por_linha))


# ═══════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ═══════════════════════════════════════════════════════════════════════════

def _sep(char="=", n=68):
    print(char * n)


def formatar_tempo(segundos: float) -> str:
    h = int(segundos // 3600)
    m = int((segundos % 3600) // 60)
    s = segundos % 60
    return f"{h:02d}h {m:02d}m {s:05.2f}s"


class Timer:
    def __init__(self, nome: str, relatorio: dict):
        self.nome      = nome
        self.relatorio = relatorio

    def __enter__(self):
        self._t0  = time.perf_counter()
        self._ts0 = time.time()
        print(f"\n+-- INICIO  {self.nome}")
        print(f"|   {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return self

    def __exit__(self, *_):
        t1  = time.perf_counter()
        ts1 = time.time()
        dur = t1 - self._t0
        self.relatorio[self.nome] = {
            "duracao"    : dur,
            "inicio_str" : time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._ts0)),
            "fim_str"    : time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts1)),
        }
        print(f"+-- FIM     {self.nome}  --  {formatar_tempo(dur)}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# CLASSE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

class ClassificadorRaster:
    def __init__(self,
                 path_img        : str,
                 shapefiles      : List[Tuple[str, int]],
                 hw_info         : dict,
                 path_img_teste  : Optional[str] = None,
                 path_saida      : str            = "mapa_classificado.tif",
                 usar_mascara    : bool           = False,
                 valor_valido    : int            = 255,
                 camadas_ocultas : List[int]      = None,
                 activation      : str            = "relu",
                 salvar_modelo   : bool           = False,
                 path_modelo     : str            = "resultado/modelo.keras"):

        self.path_img        = path_img
        self.shapefiles      = shapefiles
        self.hw              = hw_info
        self.path_img_teste  = path_img_teste or path_img
        self.path_saida      = path_saida
        self.usar_mascara    = usar_mascara
        self.valor_valido    = valor_valido
        self.camadas_ocultas = camadas_ocultas or [256, 128, 64, 8]
        self.activation      = activation
        self.salvar_modelo   = salvar_modelo
        self.path_modelo     = path_modelo

        self.src              = None
        self.gdf              = None
        self.X = self.Y       = None
        self.X_train          = self.X_test  = None
        self.Y_train          = self.Y_test  = None
        self.model            = None
        self.history          = None
        self.input_shape      = None
        self.num_classes      = None
        self.n_bandas_feature = None

        self.tamanho_mb   = 0.0
        self.total_pixels = 0

    # ── Exibe config ────────────────────────────────────────────────────
    def exibir_config(self):
        _sep()
        print(" CONFIGURACAO DO PIPELINE")
        _sep()
        print(f"  Imagem de treino       : {self.path_img}")
        print(f"  Imagem a classificar   : {self.path_img_teste}")
        print(f"  Saida                  : {self.path_saida}")
        print(f"  Shapefiles / classes   :")
        for shp, cls in self.shapefiles:
            print(f"    [Classe {cls}] {shp}")
        print(f"  Usar mascara           : {self.usar_mascara}  (valor valido = {self.valor_valido})")
        print(f"  Camadas ocultas        : {self.camadas_ocultas}")
        print(f"  Ativacao               : {self.activation}")
        print(f"  Normalizacao           : DESATIVADA (comportamento identico a v1)")
        print(f"  Salvar modelo          : {self.salvar_modelo}")
        if self.salvar_modelo:
            print(f"  Path modelo            : {self.path_modelo}")
        _sep()

    # ── 1. Carga de imagem ───────────────────────────────────────────────
    def carregar_imagem(self):
        if not os.path.isfile(self.path_img):
            raise FileNotFoundError(f"Imagem nao encontrada: {self.path_img}")
        self.src = rasterio.open(self.path_img)
        self.tamanho_mb = os.path.getsize(self.path_img) / (1024 * 1024)
        print(f"  Arquivo  : {self.path_img}")
        print(f"  Dimensoes: {self.src.height} x {self.src.width} px  |  {self.src.count} bandas")
        print(f"  CRS      : {self.src.crs}")
        print(f"  Disco    : {self.tamanho_mb:.2f} MB")

    # ── 2. Amostras ──────────────────────────────────────────────────────
    def carregar_amostras(self):
        lista = []
        for shp_path, classe_id in self.shapefiles:
            if not os.path.isfile(shp_path):
                raise FileNotFoundError(f"Shapefile nao encontrado: {shp_path}")
            gdf = gpd.read_file(shp_path)
            if self.src is not None and gdf.crs != self.src.crs:
                gdf = gdf.to_crs(self.src.crs.to_dict())
            gdf["id"] = classe_id
            lista.append(gdf)
            print(f"  [Classe {classe_id}] {shp_path}  ->  {len(gdf)} pontos")
        self.gdf = pd.concat(lista, axis=0, ignore_index=True)
        print(f"  Total de amostras: {len(self.gdf)}")

    # ── 3. Extração espectral ────────────────────────────────────────────
    def extrair_valores(self):
        coords  = [(x, y) for x, y in zip(self.gdf.geometry.x, self.gdf.geometry.y)]
        valores = np.array(list(self.src.sample(coords)))

        if self.usar_mascara and self.src.count > 1:
            self.n_bandas_feature = self.src.count - 1
            self.X = valores[:, :self.n_bandas_feature]
        else:
            self.n_bandas_feature = self.src.count
            self.X = valores

        # Igual à v1: Y com shape (N, 1) para compatibilidade com binary_crossentropy
        self.Y = self.gdf["id"].values[:, np.newaxis]
        print(f"  Vetores extraidos: {self.X.shape[0]} x {self.X.shape[1]} bandas")

    # ── 4. Preparação — SEM normalização (igual v1) ──────────────────────
    def preparar_dados(self, test_size: float = 0.3, random_state: int = 42):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=random_state
        )
        self.input_shape = (self.X_train.shape[1],)
        self.num_classes = len(np.unique(self.gdf["id"].values))
        print(f"  Treino: {len(self.X_train)}  |  Teste: {len(self.X_test)}  |  Classes: {self.num_classes}")
        print(f"  Input shape : {self.input_shape}")

    # ── 5. Modelo — igual v1 ────────────────────────────────────────────
    def construir_modelo(self):
        model = Sequential()
        model.add(Dense(self.camadas_ocultas[0],
                        input_shape=self.input_shape,
                        activation=self.activation))
        for unidades in self.camadas_ocultas[1:]:
            model.add(Dense(unidades, activation=self.activation))

        if self.num_classes == 2:
            model.add(Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
        else:
            model.add(Dense(self.num_classes, activation="softmax"))
            loss = "sparse_categorical_crossentropy"

        model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])
        self.model = model
        self.model.summary()

    # ── 6. Treinamento — igual v1 ────────────────────────────────────────
    def treinar(self, epochs: int = 100, batch_size: int = 250,
                validation_split: float = 0.25, verbose: int = 1):
        self.history = self.model.fit(
            self.X_train, self.Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
        )
        if self.salvar_modelo:
            os.makedirs(os.path.dirname(self.path_modelo) or ".", exist_ok=True)
            self.model.save(self.path_modelo)
            print(f"  Modelo salvo em: {self.path_modelo}")

    # ── 7. Avaliação — igual v1 ──────────────────────────────────────────
    def avaliar(self):
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].plot(self.history.history["loss"],     color="b", label="Training loss")
        ax[0].plot(self.history.history["val_loss"], color="r", label="Validation loss")
        ax[0].legend(loc="best", shadow=True)
        ax[0].set_title("Loss")
        ax[1].plot(self.history.history["accuracy"],     color="b", label="Training accuracy")
        ax[1].plot(self.history.history["val_accuracy"], color="r", label="Validation accuracy")
        ax[1].legend(loc="best", shadow=True)
        ax[1].set_title("Accuracy")
        plt.tight_layout()
        plt.savefig("graficos_treinamento.png", dpi=150)
        print("  Graficos -> graficos_treinamento.png")

        score  = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        print(f"  Test loss: {score[0]:.4f}  |  Test accuracy: {score[1]:.4f}")

        y_pred = self.model.predict(self.X_test)
        if self.num_classes == 2:
            y_pred = np.round(y_pred).astype(int)
        else:
            y_pred = np.argmax(y_pred, axis=1)

        print("\n  Classification Report:")
        print(classification_report(self.Y_test, y_pred))

        cm    = confusion_matrix(self.Y_test, y_pred)
        nomes = [f"Classe {c}" for c in sorted(self.gdf["id"].unique())]
        plt.figure(figsize=(8, 8))
        sns.heatmap(pd.DataFrame(cm, index=nomes, columns=nomes),
                    annot=True, annot_kws={"size": 18}, fmt="d",
                    cmap="Blues", cbar=False)
        plt.ylabel("True"); plt.xlabel("Predict")
        plt.tight_layout()
        plt.savefig("matriz_confusao.png", dpi=150)
        print("  Matriz de confusao -> matriz_confusao.png")

    # ── 8. Predição chunked — memória eficiente, lógica idêntica à v1 ───
    def prever_imagem(self, batch_size_pred: int = 4096):
        """
        Classifica a imagem em chunks de linhas para não estourar a RAM.
        NÃO normaliza os pixels — mesma escala que entrou no treino (v1).
        """
        print(f"  Imagem: {self.path_img_teste}")

        with rasterio.open(self.path_img_teste) as src:
            altura         = src.height
            largura        = src.width
            n_bandas_total = src.count
            out_meta       = src.meta.copy()

        self.total_pixels = altura * largura

        ram_livre  = psutil.virtual_memory().available
        ram_usar   = min(ram_livre, self.hw["ram_limite_bytes"])
        chunk_lin  = calcular_chunk_linhas(largura, self.n_bandas_feature, ram_usar)
        n_chunks   = math.ceil(altura / chunk_lin)

        print(f"  Dimensoes     : {altura} x {largura} px  |  {n_bandas_total} bandas")
        print(f"  Total pixels  : {self.total_pixels:,}")
        print(f"  RAM disponivel: {ram_livre / 1e9:.1f} GB  |  Limite: {self.hw['ram_limite_gb']:.1f} GB")
        print(f"  Chunk         : {chunk_lin} linhas  ->  {n_chunks} chunks no total")

        # dtype float64 igual à v1 (model.predict recebe float64)
        out_meta.update({
            "driver"  : "GTiff",
            "count"   : 1,
            "dtype"   : "float64",
            "compress": "lzw",
            "nodata"  : np.nan,
            "height"  : altura,
            "width"   : largura,
        })
        os.makedirs(os.path.dirname(self.path_saida) or ".", exist_ok=True)

        t_inicio = time.perf_counter()

        with rasterio.open(self.path_img_teste) as src, \
             rasterio.open(self.path_saida, "w", **out_meta) as dst:

            for i in range(n_chunks):
                lin_ini = i * chunk_lin
                lin_fim = min(lin_ini + chunk_lin, altura)
                n_lin   = lin_fim - lin_ini
                window  = Window(0, lin_ini, largura, n_lin)

                chunk = src.read(window=window)  # (B, H, W)

                if self.usar_mascara and n_bandas_total > self.n_bandas_feature:
                    mascara_2d = chunk[-1]
                    features   = chunk[:self.n_bandas_feature]
                else:
                    mascara_2d = None
                    features   = chunk

                # (H*W, bandas) — float64 igual à v1, SEM normalização
                feat_2d = (features.transpose(1, 2, 0)
                                   .reshape(-1, self.n_bandas_feature)
                                   .astype(np.float64))

                resultado = np.full(n_lin * largura, np.nan)

                if mascara_2d is not None:
                    validos = mascara_2d.reshape(-1) == self.valor_valido
                else:
                    validos = np.ones(feat_2d.shape[0], dtype=bool)

                pixels_validos = feat_2d[validos]

                if len(pixels_validos) > 0:
                    pred_raw = self.model.predict(
                        pixels_validos, batch_size=batch_size_pred, verbose=0
                    )
                    if self.num_classes == 2:
                        pred_cls = np.round(pred_raw).astype(float).flatten()
                    else:
                        pred_cls = np.argmax(pred_raw, axis=1).astype(float)
                    resultado[validos] = pred_cls

                dst.write(resultado.reshape(1, n_lin, largura), window=window)

                # Progresso
                pct     = (i + 1) / n_chunks * 100
                ram_pct = psutil.virtual_memory().percent
                t_decor = time.perf_counter() - t_inicio
                eta     = (t_decor / (i + 1)) * (n_chunks - i - 1)
                barra   = "#" * int(pct / 5)
                print(f"  Chunk {i+1:4d}/{n_chunks}  [{barra:<20}] {pct:5.1f}%  "
                      f"RAM {ram_pct:4.1f}%  ETA {formatar_tempo(eta)}",
                      end="\r")

                del chunk, features, feat_2d, resultado, pixels_validos
                if mascara_2d is not None:
                    del mascara_2d

        print(f"\n  Concluido. GeoTIFF salvo em: {self.path_saida}")

    def salvar_resultado(self):
        print(f"  Resultado ja salvo: {self.path_saida}")


# ═══════════════════════════════════════════════════════════════════════════
# RELATÓRIO FINAL
# ═══════════════════════════════════════════════════════════════════════════

def exibir_relatorio_final(relatorio: dict, hw_info: dict,
                           tamanho_mb: float, total_pixels: int):
    tempo_total = sum(v["duracao"] for v in relatorio.values())
    tempo_min   = tempo_total / 60 if tempo_total > 0 else 1e-9

    print("\n")
    _sep()
    print("  RELATORIO FINAL DE EXECUCAO")
    _sep()

    cN, cI, cF, cD, cP = 35, 19, 19, 16, 7
    print(f"\n{'Etapa':<{cN}} {'Inicio':<{cI}} {'Fim':<{cF}} {'Duracao':>{cD}} {'%':>{cP}}")
    print("-" * (cN + cI + cF + cD + cP + 4))

    for nome, d in relatorio.items():
        pct = d["duracao"] / tempo_total * 100 if tempo_total else 0
        print(f"{nome:<{cN}} {d['inicio_str']:<{cI}} {d['fim_str']:<{cF}}"
              f" {formatar_tempo(d['duracao']):>{cD}} {pct:>{cP - 1}.1f}%")

    print("-" * (cN + cI + cF + cD + cP + 4))
    print(f"{'TEMPO TOTAL':<{cN}} {'':<{cI}} {'':<{cF}} {formatar_tempo(tempo_total):>{cD}}")

    print()
    _sep("-")
    print("  DESEMPENHO")
    _sep("-")
    print(f"  Imagem processada      : {tamanho_mb:>10,.2f} MB")
    print(f"  Total de pixels        : {total_pixels:>10,}")
    print(f"  Tempo total            : {formatar_tempo(tempo_total):>16}")
    print()
    print(f"  Pixels / minuto        : {total_pixels / tempo_min:>15,.0f}")
    print(f"  MB / minuto            : {tamanho_mb / tempo_min:>15.2f}")
    if total_pixels > 0:
        print(f"  ms / pixel             : {(tempo_total / total_pixels) * 1000:>15.6f}")

    print()
    _sep("-")
    print("  HARDWARE")
    _sep("-")
    print(f"  Dispositivo TF         : {hw_info['device']}")
    print(f"  Threads CPU            : {hw_info['cpu_threads']}")
    print(f"  RAM total              : {hw_info['ram_total_gb']:.1f} GB")
    print(f"  RAM limite configurado : {hw_info['ram_limite_gb']:.1f} GB")
    ram_f = psutil.virtual_memory()
    print(f"  RAM em uso ao final    : {ram_f.used / 1e9:.1f} GB  ({ram_f.percent:.1f}%)")
    _sep()
    print()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_shapefile(arg: str) -> Tuple[str, int]:
    p = arg.rsplit(",", 1)
    if len(p) != 2:
        raise argparse.ArgumentTypeError("Formato: caminho_shapefile,id_classe")
    return p[0], int(p[1])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classificador raster v4. Edite CONFIG no topo ou use --config / args CLI."
    )
    parser.add_argument("--config",      "-c", default=None)
    parser.add_argument("--imagem",      "-i", default=None)
    parser.add_argument("--shapefile",   "-s", action="append", type=parse_shapefile)
    parser.add_argument("--teste",       "-t", default=None)
    parser.add_argument("--saida",       "-o", default=None)
    parser.add_argument("--epochs",            type=int,   default=None)
    parser.add_argument("--batch-size",        type=int,   default=None)
    parser.add_argument("--test-size",         type=float, default=None)
    parser.add_argument("--random-state",      type=int,   default=None)
    parser.add_argument("--usar-mascara",      action="store_true")
    parser.add_argument("--ram-limite",        type=float, default=None)
    return parser.parse_args()


def montar_config(args) -> dict:
    """Prioridade: JSON externo > args CLI > CONFIG embutido."""
    cfg = CONFIG.copy()

    if args.config and os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            ext = json.load(f)
        ext.pop("_comentarios", None)
        cfg.update(ext)
        cfg["shapefiles"] = [tuple(x) for x in cfg.get("shapefiles", [])]
        return cfg

    if args.imagem:       cfg["path_img"]          = args.imagem
    if args.shapefile:    cfg["shapefiles"]         = args.shapefile
    if args.teste:        cfg["path_img_teste"]     = args.teste
    if args.saida:        cfg["path_saida"]         = args.saida
    if args.epochs        is not None: cfg["epochs"]            = args.epochs
    if args.batch_size    is not None: cfg["batch_size_treino"] = args.batch_size
    if args.test_size     is not None: cfg["test_size"]         = args.test_size
    if args.random_state  is not None: cfg["random_state"]      = args.random_state
    if args.usar_mascara:              cfg["usar_mascara"]       = True
    if args.ram_limite    is not None: cfg["ram_limite_pct"]     = args.ram_limite

    cfg["shapefiles"] = [tuple(x) for x in cfg["shapefiles"]]
    return cfg


# ═══════════════════════════════════════════════════════════════════════════
# PONTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args      = parse_args()
    cfg       = montar_config(args)
    hw_info   = configurar_hardware(ram_limite_pct=cfg.get("ram_limite_pct", 70.0))
    relatorio = {}

    clf = ClassificadorRaster(
        path_img        = cfg["path_img"],
        shapefiles      = cfg["shapefiles"],
        hw_info         = hw_info,
        path_img_teste  = cfg.get("path_img_teste", cfg["path_img"]),
        path_saida      = cfg["path_saida"],
        usar_mascara    = cfg.get("usar_mascara", False),
        valor_valido    = cfg.get("valor_valido", 255),
        camadas_ocultas = cfg.get("camadas_ocultas", [256, 128, 64, 8]),
        activation      = cfg.get("activation", "relu"),
        salvar_modelo   = cfg.get("salvar_modelo", False),
        path_modelo     = cfg.get("path_modelo", "resultado/modelo.keras"),
    )

    clf.exibir_config()

    with Timer("1. Carregar imagem", relatorio):
        clf.carregar_imagem()

    with Timer("2. Carregar amostras", relatorio):
        clf.carregar_amostras()

    with Timer("3. Extrair valores espectrais", relatorio):
        clf.extrair_valores()

    with Timer("4. Preparar dados", relatorio):
        clf.preparar_dados(
            test_size    = cfg.get("test_size", 0.3),
            random_state = cfg.get("random_state", 42),
        )

    with Timer("5. Construir modelo", relatorio):
        clf.construir_modelo()

    with Timer("6. Treinar modelo", relatorio):
        clf.treinar(
            epochs     = cfg.get("epochs", 100),
            batch_size = cfg.get("batch_size_treino", 250),
        )

    with Timer("7. Avaliar modelo", relatorio):
        clf.avaliar()

    with Timer("8. Predicao + exportacao (chunked)", relatorio):
        clf.prever_imagem(
            batch_size_pred = cfg.get("batch_size_predicao", 4096),
        )

    with Timer("9. Finalizar", relatorio):
        clf.salvar_resultado()

    exibir_relatorio_final(relatorio, hw_info, clf.tamanho_mb, clf.total_pixels)
    print("=== PIPELINE FINALIZADO COM SUCESSO ===\n")


if __name__ == "__main__":
    main()