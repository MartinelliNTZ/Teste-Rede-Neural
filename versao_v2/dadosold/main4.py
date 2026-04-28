#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classificador Raster com Redes Neurais — v5
============================================
Melhorias em relação à v4:
  - Config embutido completo (incluindo dropout_rate), sem necessidade de JSON externo
  - Suporte real a DUAS imagens distintas: uma pequena para treino e outra completa para classificação
  - Exibição detalhada de pontos por shapefile e precisão por classe
  - Dropout nas camadas ocultas para regularização
  - Relatório final expandido com métricas por classe

Estrutura de pastas esperada (relativa ao script):
    classificador_v5.py
    dados/
        imagemFull.tif         ← imagem completa a classificar
        imagemTreino.tif       ← imagem pequena para extração de amostras
        solo.shp (+ .dbf .shx .prj)
        vegetacao.shp
        palhada.shp
        daninhas.shp
    resultado/
        (criado automaticamente)
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path
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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ─── Diretório base = pasta onde este script está ───────────────────────────
BASE_DIR = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG EMBUTIDO
# Edite aqui. Todos os caminhos são RELATIVOS à pasta do script.
# ═══════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Imagens ──────────────────────────────────────────────────────────
    # Imagem usada para EXTRAIR AMOSTRAS (pode ser um recorte menor)
    "path_img": "dados/imagemTreino.tif",

    # Imagem a ser CLASSIFICADA (pode ser a mesma ou a versão completa)
    "path_img_teste": "dados/imagemCompleta.tif",

    # Onde salvar o GeoTIFF classificado
    "path_saida": "resultado/mapa_classificado_CLAUDECODE.tif",

    # ── Amostras ─────────────────────────────────────────────────────────
    # Cada entrada: [caminho_shapefile, id_classe]
    "shapefiles": [
        ["dados/solo.shp",       0],
        ["dados/floresta.shp",  1],
        ["dados/palhada.shp",    2],
        ["dados/daninhas.shp",   3],
    ],

    # ── Máscara ───────────────────────────────────────────────────────────
    # true = usa a ÚLTIMA banda como máscara (canal alpha)
    "usar_mascara": True,
    # Valor que indica pixel válido na máscara
    "valor_valido": 255,

    # ── Memória ───────────────────────────────────────────────────────────
    # Percentual máximo de RAM disponível para cada chunk de predição
    "ram_limite_pct": 70,

    # ── Divisão treino/teste ─────────────────────────────────────────────
    "test_size":    0.3,
    "random_state": 42,

    # ── Treinamento ───────────────────────────────────────────────────────
    "epochs":           150,
    "batch_size_treino": 64,

    # ── Predição ─────────────────────────────────────────────────────────
    "batch_size_predicao": 4096,

    # ── Arquitetura ───────────────────────────────────────────────────────
    # Neurônios por camada oculta
    "camadas_ocultas": [128, 64, 32],

    # Função de ativação das camadas ocultas: "relu", "elu" ou "tanh"
    "activation": "relu",

    # Dropout entre camadas ocultas (0 = desativado)
    "dropout_rate": 0.1,

    # ── Persistência do modelo ────────────────────────────────────────────
    "salvar_modelo": True,
    "path_modelo":   "resultado/modelo2.keras",
}


# ═══════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ═══════════════════════════════════════════════════════════════════════════

def _sep(char: str = "=", n: int = 72):
    print(char * n)


def formatar_tempo(segundos: float) -> str:
    h = int(segundos // 3600)
    m = int((segundos % 3600) // 60)
    s = segundos % 60
    return f"{h:02d}h {m:02d}m {s:05.2f}s"


def resolver_caminho(p: str) -> Path:
    """Resolve caminho relativo em relação ao diretório do script."""
    path = Path(p)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


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
        dur = time.perf_counter() - self._t0
        ts1 = time.time()
        self.relatorio[self.nome] = {
            "duracao"    : dur,
            "inicio_str" : time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._ts0)),
            "fim_str"    : time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts1)),
        }
        print(f"+-- FIM     {self.nome}  --  {formatar_tempo(dur)}")
        return False


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
    bytes_por_linha = largura * n_bandas * 8
    return max(1, int(ram_livre_bytes * fator / bytes_por_linha))


# ═══════════════════════════════════════════════════════════════════════════
# CLASSE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

class ClassificadorRaster:
    def __init__(self, cfg: dict, hw_info: dict):
        # ── Caminhos resolvidos ──────────────────────────────────────────
        self.path_img       = resolver_caminho(cfg["path_img"])
        self.path_img_teste = resolver_caminho(cfg["path_img_teste"])
        self.path_saida     = resolver_caminho(cfg["path_saida"])
        self.path_modelo    = resolver_caminho(cfg["path_modelo"])

        self.shapefiles      = [(resolver_caminho(p), int(c)) for p, c in cfg["shapefiles"]]
        self.hw              = hw_info
        self.usar_mascara    = cfg.get("usar_mascara", False)
        self.valor_valido    = cfg.get("valor_valido", 255)
        self.camadas_ocultas = cfg.get("camadas_ocultas", [128, 64, 32])
        self.activation      = cfg.get("activation", "relu")
        self.dropout_rate    = cfg.get("dropout_rate", 0.0)
        self.salvar_modelo   = cfg.get("salvar_modelo", False)

        # Estado interno
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
        self.nomes_classes    = {}

        self.tamanho_mb_treino    = 0.0
        self.tamanho_mb_classific = 0.0
        self.total_pixels         = 0

    # ── Exibe config ────────────────────────────────────────────────────
    def exibir_config(self):
        _sep()
        print(" CONFIGURACAO DO PIPELINE")
        _sep()

        treino_igual_classific = (self.path_img.resolve() == self.path_img_teste.resolve())

        print(f"  Imagem de treino       : {self.path_img}")
        print(f"  Imagem a classificar   : {self.path_img_teste}")
        if treino_igual_classific:
            print("  [ATENCAO] Treino e classificacao usam a MESMA imagem.")
        else:
            print("  [OK] Treino e classificacao usam imagens DIFERENTES.")
        print(f"  Saida                  : {self.path_saida}")
        print(f"  Shapefiles / classes   :")
        for shp, cls in self.shapefiles:
            print(f"    [Classe {cls}] {shp}")
        print(f"  Usar mascara           : {self.usar_mascara}  (valor valido = {self.valor_valido})")
        print(f"  Camadas ocultas        : {self.camadas_ocultas}")
        print(f"  Ativacao               : {self.activation}")
        print(f"  Dropout                : {self.dropout_rate}")
        print(f"  Normalizacao           : DESATIVADA")
        print(f"  Salvar modelo          : {self.salvar_modelo}")
        if self.salvar_modelo:
            print(f"  Path modelo            : {self.path_modelo}")
        _sep()

    # ── Valida se as imagens existem e loga informações ─────────────────
    def validar_imagens(self):
        _sep("-")
        print("  VALIDACAO DAS IMAGENS")
        _sep("-")

        for label, path in [("Treino (amostras)", self.path_img),
                             ("Classificacao"    , self.path_img_teste)]:
            if not path.is_file():
                raise FileNotFoundError(f"Imagem nao encontrada: {path}")
            with rasterio.open(path) as src:
                tam_mb = path.stat().st_size / (1024 * 1024)
                print(f"  [{label}]")
                print(f"    Arquivo   : {path.name}")
                print(f"    Dimensoes : {src.height} x {src.width} px | {src.count} bandas")
                print(f"    CRS       : {src.crs}")
                print(f"    Disco     : {tam_mb:.2f} MB")
                if label.startswith("Treino"):
                    self.tamanho_mb_treino = tam_mb
                else:
                    self.tamanho_mb_classific = tam_mb

        iguais = (self.path_img.resolve() == self.path_img_teste.resolve())
        print(f"\n  Imagens identicas: {'SIM' if iguais else 'NAO'}")
        _sep("-")

    # ── 1. Carga de imagem de treino ─────────────────────────────────────
    def carregar_imagem(self):
        if not self.path_img.is_file():
            raise FileNotFoundError(f"Imagem nao encontrada: {self.path_img}")
        self.src = rasterio.open(str(self.path_img))
        print(f"  Arquivo  : {self.path_img.name}")
        print(f"  Dimensoes: {self.src.height} x {self.src.width} px  |  {self.src.count} bandas")
        print(f"  CRS      : {self.src.crs}")

    # ── 2. Amostras — exibe pontos por shapefile ─────────────────────────
    def carregar_amostras(self):
        _sep("-")
        print("  AMOSTRAS POR SHAPEFILE")
        _sep("-")
        print(f"  {'Classe':^7}  {'Shapefile':<40}  {'Pontos':>8}")
        print(f"  {'-'*7}  {'-'*40}  {'-'*8}")

        lista = []
        for shp_path, classe_id in self.shapefiles:
            if not shp_path.is_file():
                raise FileNotFoundError(f"Shapefile nao encontrado: {shp_path}")
            gdf = gpd.read_file(str(shp_path))
            if self.src is not None and gdf.crs != self.src.crs:
                gdf = gdf.to_crs(self.src.crs.to_dict())
            gdf["id"] = classe_id
            nome_cls  = shp_path.stem
            self.nomes_classes[classe_id] = nome_cls
            print(f"  {classe_id:^7}  {shp_path.name:<40}  {len(gdf):>8,}")
            lista.append(gdf)

        self.gdf = pd.concat(lista, axis=0, ignore_index=True)
        print(f"  {'-'*7}  {'-'*40}  {'-'*8}")
        print(f"  {'TOTAL':^7}  {'':<40}  {len(self.gdf):>8,}")
        _sep("-")

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

        self.Y = self.gdf["id"].values[:, np.newaxis]
        print(f"  Vetores extraidos: {self.X.shape[0]:,} x {self.X.shape[1]} bandas")

    # ── 4. Preparação ────────────────────────────────────────────────────
    def preparar_dados(self, test_size: float = 0.3, random_state: int = 42):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=random_state,
            stratify=self.Y  # garante distribuição proporcional de classes
        )
        self.input_shape = (self.X_train.shape[1],)
        self.num_classes = len(np.unique(self.gdf["id"].values))

        _sep("-")
        print("  DISTRIBUICAO DOS DADOS")
        _sep("-")
        print(f"  Total amostras : {len(self.X):,}")
        print(f"  Treino         : {len(self.X_train):,}  ({100*(1-test_size):.0f}%)")
        print(f"  Teste          : {len(self.X_test):,}  ({100*test_size:.0f}%)")
        print(f"  Classes        : {self.num_classes}")
        print(f"  Input shape    : {self.input_shape}")

        print(f"\n  {'Classe':<8}  {'Nome':<20}  {'Total':>8}  {'Treino':>8}  {'Teste':>8}")
        print(f"  {'-'*8}  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}")
        for cls in sorted(self.nomes_classes.keys()):
            t = int(np.sum(self.Y        == cls))
            r = int(np.sum(self.Y_train  == cls))
            e = int(np.sum(self.Y_test   == cls))
            print(f"  {cls:<8}  {self.nomes_classes[cls]:<20}  {t:>8,}  {r:>8,}  {e:>8,}")
        _sep("-")

    # ── 5. Modelo com Dropout ────────────────────────────────────────────
    def construir_modelo(self):
        model = Sequential()
        first = True
        for unidades in self.camadas_ocultas:
            if first:
                model.add(Dense(unidades, input_shape=self.input_shape,
                                activation=self.activation))
                first = False
            else:
                model.add(Dense(unidades, activation=self.activation))
            if self.dropout_rate > 0:
                model.add(Dropout(self.dropout_rate))

        if self.num_classes == 2:
            model.add(Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
        else:
            model.add(Dense(self.num_classes, activation="softmax"))
            loss = "sparse_categorical_crossentropy"

        model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])
        self.model = model
        self.model.summary()

    # ── 6. Treinamento ───────────────────────────────────────────────────
    def treinar(self, epochs: int = 150, batch_size: int = 64,
                validation_split: float = 0.25, verbose: int = 1):
        self.history = self.model.fit(
            self.X_train, self.Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
        )
        if self.salvar_modelo:
            self.path_modelo.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.path_modelo))
            print(f"  Modelo salvo em: {self.path_modelo}")

    # ── 7. Avaliação detalhada por classe ────────────────────────────────
    def avaliar(self):
        # Gráficos de treinamento
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].plot(self.history.history["loss"],     color="b", label="Training loss")
        ax[0].plot(self.history.history["val_loss"], color="r", label="Validation loss")
        ax[0].legend(loc="best", shadow=True); ax[0].set_title("Loss")
        ax[1].plot(self.history.history["accuracy"],     color="b", label="Training accuracy")
        ax[1].plot(self.history.history["val_accuracy"], color="r", label="Validation accuracy")
        ax[1].legend(loc="best", shadow=True); ax[1].set_title("Accuracy")
        plt.tight_layout()
        graf_path = BASE_DIR / "resultado" / "graficos_treinamento.png"
        graf_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(graf_path), dpi=150)
        print(f"  Graficos -> {graf_path}")

        # Avaliação geral
        score  = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        print(f"\n  Test loss     : {score[0]:.4f}")
        print(f"  Test accuracy : {score[1]:.4f}  ({score[1]*100:.2f}%)")

        # Predições
        y_pred_raw = self.model.predict(self.X_test, verbose=0)
        if self.num_classes == 2:
            y_pred = np.round(y_pred_raw).astype(int).flatten()
        else:
            y_pred = np.argmax(y_pred_raw, axis=1)

        y_true = self.Y_test.flatten()

        # ── Precisão global ─────────────────────────────────────────────
        acc_global = accuracy_score(y_true, y_pred)
        _sep("-")
        print("  METRICAS POR CLASSE (conjunto de teste)")
        _sep("-")

        nomes_lista = [self.nomes_classes.get(c, f"Classe {c}")
                       for c in sorted(self.nomes_classes.keys())]
        report = classification_report(
            y_true, y_pred,
            target_names=nomes_lista,
            digits=4,
            output_dict=True
        )

        # Tabela legível
        print(f"\n  {'Classe':<20}  {'Precisao':>10}  {'Recall':>10}  {'F1-Score':>10}  {'Suporte':>10}")
        print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
        for nome in nomes_lista:
            r = report[nome]
            print(f"  {nome:<20}  {r['precision']:>10.4f}  {r['recall']:>10.4f}"
                  f"  {r['f1-score']:>10.4f}  {int(r['support']):>10,}")
        print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
        mac = report["macro avg"]
        wei = report["weighted avg"]
        print(f"  {'macro avg':<20}  {mac['precision']:>10.4f}  {mac['recall']:>10.4f}"
              f"  {mac['f1-score']:>10.4f}")
        print(f"  {'weighted avg':<20}  {wei['precision']:>10.4f}  {wei['recall']:>10.4f}"
              f"  {wei['f1-score']:>10.4f}")
        print(f"\n  Acuracia global : {acc_global:.4f}  ({acc_global*100:.2f}%)")
        _sep("-")

        # Classification report completo no terminal
        print("\n  Classification Report (scikit-learn):\n")
        print(classification_report(y_true, y_pred, target_names=nomes_lista, digits=4))

        # Matriz de confusão
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(max(6, self.num_classes * 2), max(5, self.num_classes * 2)))
        sns.heatmap(
            pd.DataFrame(cm, index=nomes_lista, columns=nomes_lista),
            annot=True, annot_kws={"size": 14}, fmt="d",
            cmap="Blues", cbar=False
        )
        plt.ylabel("Real"); plt.xlabel("Predito")
        plt.title("Matriz de Confusao")
        plt.tight_layout()
        cm_path = BASE_DIR / "resultado" / "matriz_confusao.png"
        plt.savefig(str(cm_path), dpi=150)
        print(f"  Matriz de confusao -> {cm_path}")

        # Salva métricas em texto
        self._salvar_relatorio_metricas(report, acc_global, nomes_lista)

    def _salvar_relatorio_metricas(self, report: dict, acc: float, nomes: list):
        txt_path = BASE_DIR / "resultado" / "relatorio_metricas.txt"
        with open(str(txt_path), "w", encoding="utf-8") as f:
            f.write("RELATORIO DE METRICAS — Classificador Raster v5\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Imagem de treino       : {self.path_img}\n")
            f.write(f"Imagem classificada    : {self.path_img_teste}\n")
            f.write(f"Acuracia global        : {acc:.4f} ({acc*100:.2f}%)\n\n")
            f.write(f"{'Classe':<22} {'Precisao':>10} {'Recall':>10} {'F1':>10} {'Suporte':>10}\n")
            f.write("-" * 62 + "\n")
            for nome in nomes:
                r = report[nome]
                f.write(f"{nome:<22} {r['precision']:>10.4f} {r['recall']:>10.4f}"
                        f" {r['f1-score']:>10.4f} {int(r['support']):>10}\n")
        print(f"  Metricas salvas -> {txt_path}")

    # ── 8. Predição chunked ──────────────────────────────────────────────
    # ── 8. Predição chunked ──────────────────────────────────────────────────
    def prever_imagem(self, batch_size_pred: int = 4096):
        """
        Classifica a imagem (path_img_teste) em chunks de linhas.
        - dtype de saída : uint8  (valores 0-255, suficiente para N classes)
        - nodata         : 255    (reservado; pixels sem máscara ficam 255)
        - compressão     : LZW
        - overviews      : 2 4 8 16 32 64  (pirâmides internas)
        - feedback       : barra de progresso a cada chunk + linha de status
        """
        NODATA_OUT = 255          # valor reservado para "sem dado"
        DTYPE_OUT  = "uint8"

        path_src = self.path_img_teste
        print(f"\n  Imagem a classificar : {path_src}")

        with rasterio.open(str(path_src)) as src:
            altura         = src.height
            largura        = src.width
            n_bandas_total = src.count
            out_meta       = src.meta.copy()
            self.total_pixels = altura * largura

        # ── verifica compatibilidade de bandas ────────────────────────────
        n_feat_classif = (n_bandas_total - 1) if self.usar_mascara else n_bandas_total
        if n_feat_classif != self.n_bandas_feature:
            raise ValueError(
                f"Incompatibilidade de bandas: treino usou {self.n_bandas_feature} "
                f"bandas, mas a imagem de classificação tem {n_feat_classif} bandas de feature."
            )

        # ── tamanho do chunk ──────────────────────────────────────────────
        ram_livre  = psutil.virtual_memory().available
        ram_usar   = min(ram_livre, self.hw["ram_limite_bytes"])
        chunk_lin  = calcular_chunk_linhas(largura, self.n_bandas_feature, ram_usar)
        n_chunks   = math.ceil(altura / chunk_lin)

        print(f"  Dimensões            : {altura} x {largura} px | {n_bandas_total} bandas")
        print(f"  Total de pixels      : {self.total_pixels:,}")
        print(f"  RAM disponível       : {ram_livre/1e9:.1f} GB  |  Limite: {self.hw['ram_limite_gb']:.1f} GB")
        print(f"  Chunk                : {chunk_lin:,} linhas  →  {n_chunks} chunks")
        print(f"  dtype saída          : {DTYPE_OUT}  |  nodata: {NODATA_OUT}  |  compressão: LZW")
        print(f"  Overviews internos   : 2 4 8 16 32 64\n")

        # ── metadados de saída ────────────────────────────────────────────
        out_meta.update({
            "driver"  : "GTiff",
            "count"   : 1,
            "dtype"   : DTYPE_OUT,
            "compress": "lzw",
            "nodata"  : NODATA_OUT,
            "height"  : altura,
            "width"   : largura,
            # tiled para facilitar overviews e leitura eficiente
            "tiled"   : True,
            "blockxsize": 512,
            "blockysize": 512,
        })

        self.path_saida.parent.mkdir(parents=True, exist_ok=True)

        # ── helpers de progresso ──────────────────────────────────────────
        BAR_WIDTH  = 30
        t_inicio   = time.perf_counter()
        pixels_ok  = 0
        pixels_nan = 0

        def _barra(pct: float) -> str:
            filled = int(BAR_WIDTH * pct / 100)
            return "█" * filled + "░" * (BAR_WIDTH - filled)

        def _eta_str(elapsed: float, pct: float) -> str:
            if pct <= 0:
                return "--:--:--"
            eta = elapsed / (pct / 100) - elapsed
            return formatar_tempo(eta)

        def _imprimir_progresso(i_chunk: int, pct: float, ram_pct: float,
                                 elapsed: float, pixels_validos_chunk: int):
            barra  = _barra(pct)
            eta    = _eta_str(elapsed, pct)
            vel    = pixels_ok / elapsed if elapsed > 0 else 0
            print(
                f"\r  Chunk {i_chunk+1:>{len(str(n_chunks))}}/{n_chunks}  "
                f"[{barra}] {pct:5.1f}%  "
                f"RAM {ram_pct:4.1f}%  "
                f"ETA {eta}  "
                f"vel {vel/1e6:6.2f} Mpx/s",
                end="", flush=True
            )

        # ── loop principal ────────────────────────────────────────────────
        with rasterio.open(str(path_src)) as src, \
             rasterio.open(str(self.path_saida), "w", **out_meta) as dst:

            for i in range(n_chunks):
                lin_ini = i * chunk_lin
                lin_fim = min(lin_ini + chunk_lin, altura)
                n_lin   = lin_fim - lin_ini
                window  = Window(0, lin_ini, largura, n_lin)

                # lê chunk
                chunk = src.read(window=window)          # (B, H, W)  uint8

                # separa features e máscara
                if self.usar_mascara and n_bandas_total > self.n_bandas_feature:
                    mascara_2d = chunk[-1]               # última banda = alpha
                    features   = chunk[:self.n_bandas_feature]
                else:
                    mascara_2d = None
                    features   = chunk

                # reshape para (N_pixels, N_features)
                feat_2d = (features
                           .transpose(1, 2, 0)
                           .reshape(-1, self.n_bandas_feature)
                           .astype(np.float64))

                # array de saída — começa todo como nodata
                resultado = np.full(n_lin * largura, NODATA_OUT, dtype=np.uint8)

                # máscara booleana de pixels válidos
                if mascara_2d is not None:
                    validos = mascara_2d.reshape(-1) == self.valor_valido
                else:
                    validos = np.ones(feat_2d.shape[0], dtype=bool)

                pixels_validos_chunk = int(validos.sum())

                if pixels_validos_chunk > 0:
                    pred_raw = self.model.predict(
                        feat_2d[validos],
                        batch_size=batch_size_pred,
                        verbose=0
                    )
                    if self.num_classes == 2:
                        pred_cls = np.round(pred_raw).astype(np.uint8).flatten()
                    else:
                        pred_cls = np.argmax(pred_raw, axis=1).astype(np.uint8)

                    resultado[validos] = pred_cls

                pixels_ok  += pixels_validos_chunk
                pixels_nan += int((~validos).sum())

                # grava chunk
                dst.write(resultado.reshape(1, n_lin, largura), window=window)

                # ── feedback ──────────────────────────────────────────────
                elapsed = time.perf_counter() - t_inicio
                pct     = (i + 1) / n_chunks * 100
                ram_pct = psutil.virtual_memory().percent
                _imprimir_progresso(i, pct, ram_pct, elapsed, pixels_validos_chunk)

                # libera memória
                del chunk, features, feat_2d, resultado, pred_raw, pred_cls
                if mascara_2d is not None:
                    del mascara_2d

        # linha em branco após a barra
        print()

        # ── overviews internos ────────────────────────────────────────────
        niveis_ov = [2, 4, 8, 16, 32, 64]
        print(f"\n  Calculando overviews {niveis_ov} ... ", end="", flush=True)
        t_ov = time.perf_counter()
        with rasterio.open(str(self.path_saida), "r+") as dst:
            dst.build_overviews(niveis_ov, rasterio.enums.Resampling.nearest)
            dst.update_tags(ns="rio_overview", resampling="nearest")
        print(f"OK  ({formatar_tempo(time.perf_counter() - t_ov)})")

        # ── resumo final ──────────────────────────────────────────────────
        elapsed_total = time.perf_counter() - t_inicio
        tam_saida_mb  = self.path_saida.stat().st_size / (1024 * 1024)
        pct_validos   = pixels_ok / self.total_pixels * 100 if self.total_pixels else 0

        print(f"\n  ┌─ Resumo da classificação ──────────────────────────────")
        print(f"  │  GeoTIFF salvo em  : {self.path_saida}")
        print(f"  │  Tamanho em disco  : {tam_saida_mb:.2f} MB")
        print(f"  │  Pixels válidos    : {pixels_ok:,}  ({pct_validos:.2f}%)")
        print(f"  │  Pixels nodata     : {pixels_nan:,}  ({100-pct_validos:.2f}%)")
        print(f"  │  Tempo total       : {formatar_tempo(elapsed_total)}")
        print(f"  │  Velocidade        : {pixels_ok/elapsed_total/1e6:.2f} Mpx/s")
        print(f"  └───────────────────────────────────────────────────────────")

    def finalizar(self):
        if self.src and not self.src.closed:
            self.src.close()
        print(f"  Resultado: {self.path_saida}")


# ═══════════════════════════════════════════════════════════════════════════
# RELATÓRIO FINAL
# ═══════════════════════════════════════════════════════════════════════════

def exibir_relatorio_final(relatorio: dict, hw_info: dict,
                           clf: ClassificadorRaster):
    tempo_total = sum(v["duracao"] for v in relatorio.values())
    tempo_min   = tempo_total / 60 if tempo_total > 0 else 1e-9

    print("\n")
    _sep()
    print("  RELATORIO FINAL DE EXECUCAO")
    _sep()

    cN, cI, cF, cD, cP = 38, 20, 20, 18, 7
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
    print(f"  Imagem treino (tam)    : {clf.tamanho_mb_treino:>10,.2f} MB")
    print(f"  Imagem classif. (tam)  : {clf.tamanho_mb_classific:>10,.2f} MB")
    print(f"  Total de pixels        : {clf.total_pixels:>10,}")
    print(f"  Tempo total            : {formatar_tempo(tempo_total):>18}")
    print()
    print(f"  Pixels / minuto        : {clf.total_pixels / tempo_min:>15,.0f}")
    print(f"  MB classif. / minuto   : {clf.tamanho_mb_classific / tempo_min:>15.2f}")
    if clf.total_pixels > 0:
        print(f"  ms / pixel             : {(tempo_total / clf.total_pixels) * 1000:>15.6f}")

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
# PONTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════

def main():
    cfg       = CONFIG.copy()
    hw_info   = configurar_hardware(ram_limite_pct=cfg.get("ram_limite_pct", 70.0))
    relatorio = {}

    clf = ClassificadorRaster(cfg=cfg, hw_info=hw_info)
    clf.exibir_config()

    # Valida e exibe informações de ambas as imagens antes de iniciar
    clf.validar_imagens()

    with Timer("1. Carregar imagem de treino", relatorio):
        clf.carregar_imagem()

    with Timer("2. Carregar amostras (shapefiles)", relatorio):
        clf.carregar_amostras()

    with Timer("3. Extrair valores espectrais", relatorio):
        clf.extrair_valores()

    with Timer("4. Preparar dados (split)", relatorio):
        clf.preparar_dados(
            test_size    = cfg.get("test_size", 0.3),
            random_state = cfg.get("random_state", 42),
        )

    with Timer("5. Construir modelo", relatorio):
        clf.construir_modelo()

    with Timer("6. Treinar modelo", relatorio):
        clf.treinar(
            epochs     = cfg.get("epochs", 150),
            batch_size = cfg.get("batch_size_treino", 64),
        )

    with Timer("7. Avaliar modelo (metricas por classe)", relatorio):
        clf.avaliar()

    with Timer("8. Predicao + exportacao GeoTIFF (chunked)", relatorio):
        clf.prever_imagem(
            batch_size_pred = cfg.get("batch_size_predicao", 4096),
        )

    with Timer("9. Finalizar", relatorio):
        clf.finalizar()

    exibir_relatorio_final(relatorio, hw_info, clf)
    print("=== PIPELINE FINALIZADO COM SUCESSO ===\n")


if __name__ == "__main__":
    main()