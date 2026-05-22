#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Classificação de Imagens Raster com Redes Neurais
============================================================
Executável em máquina local (sem Google Colab).
Pipeline completo: carrega imagem + amostras (shapefiles), treina uma
rede neural densa e classifica a imagem inteira, exportando um GeoTIFF.

Uso via linha de comando:
-------------------------
    python classificador.py \
        --imagem dados/imagem.tif \
        --shapefile dados/solo.shp,0 \
        --shapefile dados/vegetacao.shp,1 \
        --saida resultado/mapa_classificado.tif \
        --epochs 100 \
        --batch-size 250

Ou usando um arquivo JSON de configuração:
------------------------------------------
    python classificador.py --config config.json
"""

import os
import sys
import json
import time
import argparse
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
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
# Utilitários de tempo e formatação
# ---------------------------------------------------------------------------

def formatar_tempo(segundos: float) -> str:
    """Converte segundos para string formatada hh:mm:ss.ss."""
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    secs = segundos % 60
    return f"{horas:02d}h {minutos:02d}m {secs:05.2f}s"


class Timer:
    """Context manager simples para medir tempo de um bloco de código."""
    def __init__(self, nome: str, relatorio: dict):
        self.nome = nome
        self.relatorio = relatorio
        self.t0 = None

    def __enter__(self):
        self.t0 = time.time()
        inicio = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.t0))
        print(f"\n[INÍCIO] {self.nome}  —  {inicio}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        t1 = time.time()
        duracao = t1 - self.t0
        fim = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1))
        self.relatorio[self.nome] = {
            "inicio": self.t0,
            "fim": t1,
            "duracao": duracao,
            "inicio_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.t0)),
            "fim_str": fim,
        }
        print(f"[FIM]   {self.nome}  —  {fim}  (durou {formatar_tempo(duracao)})")
        return False


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
        self.path_img = path_img
        self.shapefiles = shapefiles
        self.path_img_teste = path_img_teste or path_img
        self.path_saida = path_saida
        self.usar_mascara = usar_mascara
        self.coluna_mascara = coluna_mascara
        self.valor_valido = valor_valido

        # atributos preenchidos ao longo do pipeline
        self.src = None
        self.gdf = None
        self.X = None
        self.Y = None
        self.X_train = self.X_test = None
        self.Y_train = self.Y_test = None
        self.model = None
        self.history = None
        self.img_size = None
        self.export_image = None

        # métricas de desempenho e arquivo
        self.tamanho_mb = 0.0
        self.total_pixels = 0

    # ------------------------------------------------------------------
    # 1. Carga de dados
    # ------------------------------------------------------------------
    def carregar_imagem(self):
        """Abre a imagem raster e armazena o objeto rasterio."""
        if not os.path.isfile(self.path_img):
            raise FileNotFoundError(f"Imagem não encontrada: {self.path_img}")
        self.src = rasterio.open(self.path_img)
        self.tamanho_mb = os.path.getsize(self.path_img) / (1024 * 1024)
        print(f"[OK] Imagem carregada: {self.path_img}")
        print(f"     Formato: {self.src.shape} | CRS: {self.src.crs}")
        print(f"     Tamanho: {self.tamanho_mb:.2f} MB")

    def carregar_amostras(self):
        """Lê todos os shapefiles, converte CRS, atribui IDs e concatena."""
        lista_gdf = []
        for shp_path, classe_id in self.shapefiles:
            if not os.path.isfile(shp_path):
                raise FileNotFoundError(f"Shapefile não encontrado: {shp_path}")
            gdf = gpd.read_file(shp_path)
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
        valores = list(self.src.sample(coord_list))
        valores = np.array(valores)
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
        model.add(Dense(camadas_ocultas[0],
                        input_shape=self.input_shape,
                        activation=activation))
        for unidades in camadas_ocultas[1:]:
            model.add(Dense(unidades, activation=activation))

        if self.num_classes == 2:
            model.add(Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
        else:
            model.add(Dense(self.num_classes, activation="softmax"))
            loss = "sparse_categorical_crossentropy"

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
            img = src_test.read()
            img = img.transpose([1, 2, 0])
            self.img_size = (img.shape[0], img.shape[1])
            self.total_pixels = img.shape[0] * img.shape[1]
            img = img.reshape(self.total_pixels, img.shape[2])

        cols = [f"B{i}" for i in range(img.shape[1])]
        df = pd.DataFrame(img, columns=cols)

        if self.usar_mascara and img.shape[1] > self.input_shape[0]:
            df.rename(columns={cols[-1]: self.coluna_mascara}, inplace=True)
            df_pred = df[df[self.coluna_mascara] == self.valor_valido].copy()
            valores = df_pred.values[:, :self.input_shape[0]]
        else:
            df_pred = df.copy()
            valores = df_pred.values

        print(f"[INFO] Classificando {len(valores)} pixels válidos...")
        pred = self.model.predict(valores)
        if self.num_classes == 2:
            pred = np.round(pred).astype(int).flatten()
        else:
            pred = np.argmax(pred, axis=1)

        df_pred["pred"] = pred
        df = pd.merge(df, df_pred, how="left", left_index=True, right_index=True)
        print(f"Reorganizando resultado para formato de imagem...")
        # reorganiza para (1, altura, largura)
        classify = df["pred"].values.reshape(self.img_size)
        print(f"Mescla concluída. Exportando resultado...")
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
# Relatório final
# ---------------------------------------------------------------------------

def exibir_relatorio_final(relatorio: dict, tamanho_mb: float, total_pixels: int):
    """Exibe o resumo de tempos e métricas de performance no terminal."""
    print("\n" + "=" * 70)
    print(" RELATÓRIO FINAL DE EXECUÇÃO")
    print("=" * 70)

    tempos = [v["duracao"] for v in relatorio.values()]
    tempo_total = sum(tempos)

    print(f"\n{'Etapa':<35} {'Início':<20} {'Fim':<20} {'Duração':<15}")
    print("-" * 90)
    for nome, dados in relatorio.items():
        print(f"{nome:<35} {dados['inicio_str']:<20} {dados['fim_str']:<20} {formatar_tempo(dados['duracao']):<15}")

    print("-" * 90)
    print(f"{'TEMPO TOTAL':<35} {'':<20} {'':<20} {formatar_tempo(tempo_total):<15}")
    print("=" * 70)

    print(f"\n📁  Tamanho da imagem lida : {tamanho_mb:,.2f} MB")
    print(f"🖼️   Quantidade de pixels   : {total_pixels:,}")

    if total_pixels > 0:
        tempo_por_pixel = tempo_total / total_pixels
        print(f"⏱️   Tempo / pixel          : {tempo_por_pixel:.6f} s  ({tempo_por_pixel * 1000:.4f} ms)")

    if tamanho_mb > 0:
        tempo_por_mb = tempo_total / tamanho_mb
        print(f"⏱️   Tempo / MB             : {tempo_por_mb:.4f} s")

    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI / Argparse
# ---------------------------------------------------------------------------

def parse_shapefile(arg: str) -> Tuple[str, int]:
    """Converte string 'caminho,id' em tupla (caminho, int)."""
    partes = arg.rsplit(",", 1)
    if len(partes) != 2:
        raise argparse.ArgumentTypeError(
            "Formato inválido. Use: caminho_do_shapefile,id_classe"
        )
    return partes[0], int(partes[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classificador de imagens raster com redes neurais (execução local)."
    )
    parser.add_argument(
        "--imagem", "-i", required=True, help="Caminho da imagem raster (GeoTIFF)."
    )
    parser.add_argument(
        "--shapefile", "-s", action="append", type=parse_shapefile,
        required=True, help="Shapefile de amostra no formato 'caminho,id_classe'. Pode ser usado múltiplas vezes."
    )
    parser.add_argument(
        "--teste", "-t", default=None,
        help="Imagem a ser classificada (padrão: mesma que --imagem)."
    )
    parser.add_argument(
        "--saida", "-o", default="mapa_classificado.tif",
        help="Caminho do GeoTIFF de saída (padrão: mapa_classificado.tif)."
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Número de épocas de treinamento (padrão: 100)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=250, help="Tamanho do batch (padrão: 250)."
    )
    parser.add_argument(
        "--test-size", type=float, default=0.3, help="Proporção do conjunto de teste (padrão: 0.3)."
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Semente aleatória para reprodutibilidade (padrão: 42)."
    )
    parser.add_argument(
        "--usar-mascara", action="store_true",
        help="Se ativado, filtra pixels pela banda alpha/máscara."
    )
    parser.add_argument(
        "--config", "-c", default=None,
        help="Caminho para um arquivo JSON de configuração (sobrescreve os demais argumentos)."
    )
    return parser.parse_args()


def montar_config(args: argparse.Namespace) -> dict:
    """Monta o dicionário de configuração a partir dos argumentos CLI."""
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "path_img": args.imagem,
        "shapefiles": args.shapefile,
        "path_img_teste": args.teste or args.imagem,
        "path_saida": args.saida,
        "usar_mascara": args.usar_mascara,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    config = montar_config(args)

    # garante diretório de saída
    os.makedirs(os.path.dirname(config["path_saida"]) or ".", exist_ok=True)

    # dicionário para acumular tempos
    relatorio = {}

    # inicializa classificador
    clf = ClassificadorRaster(
        path_img=config["path_img"],
        shapefiles=config["shapefiles"],
        path_img_teste=config["path_img_teste"],
        path_saida=config["path_saida"],
        usar_mascara=config["usar_mascara"]
    )

    # pipeline com medição de tempo por etapa
    with Timer("1. Carregar imagem", relatorio):
        clf.carregar_imagem()

    with Timer("2. Carregar amostras", relatorio):
        clf.carregar_amostras()

    with Timer("3. Extrair valores espectrais", relatorio):
        clf.extrair_valores()

    with Timer("4. Preparar dados", relatorio):
        clf.preparar_dados(test_size=config["test_size"], random_state=config["random_state"])

    with Timer("5. Construir modelo", relatorio):
        clf.construir_modelo()

    with Timer("6. Treinar modelo", relatorio):
        clf.treinar(epochs=config["epochs"], batch_size=config["batch_size"])

    with Timer("7. Avaliar modelo", relatorio):
        clf.avaliar()

    with Timer("8. Predição na imagem", relatorio):
        clf.prever_imagem()

    with Timer("9. Salvar resultado", relatorio):
        clf.salvar_resultado()

    # relatório final
    exibir_relatorio_final(relatorio, clf.tamanho_mb, clf.total_pixels)
    print("\n=== PIPELINE FINALIZADO COM SUCESSO ===")


if __name__ == "__main__":
    main()

