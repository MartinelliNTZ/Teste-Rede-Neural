# -*- coding: utf-8 -*-
"""
Pipeline: Random Forest + SAM 2 para Classificação de Uso do Solo
==================================================================
Requisitos:
  pip install rasterio fiona shapely pyproj scikit-learn scipy joblib numpy scikit-image torch torchvision transformers pillow

CORREÇÕES v7 (em relação ao v6):
  [BUG-1] dst.write shape 4D → squeeze explícito antes de write
  [BUG-2] np.bincount com vals contendo CLASSE_OUTROS_ID=99 → reindexação segura
  [BUG-3] Tiles SAM menores que SAM_TILE_SZ → reshape e alinhamento de máscara ao tile real
  [BUG-4] count_map não fazia votação por maioria → acumulador de votos por classe
  [BUG-5] reproject_geom com CRS que invertem eixos → uso de Transformer correto com always_xy
  [BUG-6] uniform_filter em float32**2 → cast para float64 antes de operar, volta para float32
  [BUG-7] MIN_AREA_M2 comparado com graus² em CRS geográfico → aviso e skip
  [BUG-8] seg_mask shape incompatível com tile real → verificação e resize
  [BUG-9] Leitura de bandas com canal alpha → sempre lê exatamente 3 bandas RGB
  [BUG-10] tmp PNG sem verificar shape → assert antes de salvar
"""

import os
import sys
import glob
import json
import warnings
import tempfile
import numpy as np
import rasterio
from rasterio.features import rasterize, shapes as rio_shapes
from rasterio.windows import Window
import fiona
from pyproj import CRS, Transformer
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import torch
from PIL import Image

warnings.filterwarnings("ignore")

# ──────────────────────────────────────
# CONFIGURAÇÕES CENTRALIZADAS
# ──────────────────────────────────────
ROOT      = os.path.abspath(".")
DATA_DIR  = os.path.join(ROOT, "dados")
OUT_DIR   = os.path.join(ROOT, "saida_v9")
MODEL_DIR = os.path.join(ROOT, "modelos")

# Classes (id: {shapefile, nome})
CLASSES = {
    1: {"shp": os.path.join(DATA_DIR, "palhada.shp"),  "nome": "palhada"},
    2: {"shp": os.path.join(DATA_DIR, "solo.shp"),     "nome": "solo"},
    3: {"shp": os.path.join(DATA_DIR, "floresta.shp"), "nome": "floresta"},
}
# IMPORTANTE: CLASSE_OUTROS_ID deve ser PEQUENO para não explodir np.bincount.
# Usamos 4 internamente e remapeamos para 99 apenas na saída final.
CLASSE_OUTROS_ID_INTERNO = 4
CLASSE_OUTROS_ID_SAIDA   = 99
CLASSE_OUTROS_NOME        = "outros"

# Amostras e treino RF
SAMPLES_PER_CLASS = 60_000
VALIDATION_SPLIT  = 0.2
RF_N_TREES        = 200
RF_JOBS           = -1

# Predição RF
TILE_SZ_RF    = 2048
CONF_THRESHOLD = 0.45

# SAM 2
SAM_TILE_SZ = 1024   # deve ser múltiplo de 16
SAM_OVERLAP  = 128   # sobreposição entre tiles
MIN_AREA_M2  = 3.0   # área mínima de segmento/polígono (m²)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH_RF = os.path.join(OUT_DIR, "model_rf.joblib")

# Nomes para log/relatório (usa IDs internos)
CLASS_NAMES = {cid: cfg["nome"] for cid, cfg in CLASSES.items()}
CLASS_NAMES[CLASSE_OUTROS_ID_INTERNO] = CLASSE_OUTROS_NOME
# Para saída de vetor, ainda usamos 99
CLASS_NAMES_SAIDA = dict(CLASS_NAMES)
CLASS_NAMES_SAIDA[CLASSE_OUTROS_ID_SAIDA] = CLASSE_OUTROS_NOME


# ──────────────────────────────────────
# UTILITÁRIOS
# ──────────────────────────────────────
def find_tiff():
    tiffs = (glob.glob(os.path.join(DATA_DIR, "*.tif")) +
             glob.glob(os.path.join(DATA_DIR, "*.tiff")))
    if not tiffs:
        raise FileNotFoundError(f"Nenhum TIFF em {DATA_DIR}")
    return sorted(tiffs)[0]


def tiff_info(path):
    with rasterio.open(path) as src:
        return src.meta.copy(), src.crs, src.transform, src.height, src.width


def reproject_geom(geom, src_crs, dst_crs):
    """
    [BUG-5] Reprojeção correta usando Transformer com always_xy=True.
    always_xy garante que a ordem de eixos é sempre (longitude, latitude) / (x, y),
    independente da definição do CRS, evitando inversão silenciosa de coordenadas.
    """
    t = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shp_transform(t.transform, geom)


def load_geoms(shp_path, tiff_crs):
    geoms = []
    with fiona.open(shp_path) as f:
        shp_crs = CRS.from_user_input(f.crs)
        reproj = not shp_crs.equals(tiff_crs)
        for feat in f:
            raw = feat.get("geometry")
            if raw is None:
                continue
            try:
                g = shape(raw).buffer(0)
            except Exception:
                continue
            if g.is_empty or not g.is_valid:
                continue
            if reproj:
                g = reproject_geom(g, shp_crs, tiff_crs)
            geoms.append(g)
    return geoms


def crs_is_geographic(crs):
    """Retorna True se o CRS for geográfico (graus), False se projetado (metros)."""
    return CRS.from_user_input(crs).is_geographic


# ──────────────────────────────────────
# FEATURES ESPECTRAIS
# ──────────────────────────────────────
FEAT_NAMES = [
    "R", "G", "B", "ExG", "ExR",
    "VARI", "NGRDI", "GLI",
    "brilho", "R/bri", "G/bri", "Sat", "Value",
    "stdR", "stdG", "stdB"
]


def compute_features(rgb):
    """
    [BUG-6] local_std: cast para float64 antes de elevar ao quadrado para evitar
    overflow/NaN em pixels próximos de 1.0 no formato float32.
    Resultado é convertido de volta para float32 ao final.
    """
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    eps = 1e-6

    ExG   = 2 * G - R - B
    ExR   = 1.4 * R - G
    VARI  = np.where(np.abs(G + R - B) > 0.02, (G - R) / (G + R - B + eps), 0.0)
    NGRDI = np.where((G + R) > 0.02, (G - R) / (G + R + eps), 0.0)
    GLI   = np.where((2 * G + R + B) > 0.02, (2 * G - R - B) / (2 * G + R + B + eps), 0.0)

    bri  = (R + G + B) / 3.0 + eps
    r_r  = R / bri
    r_g  = G / bri
    Cmax = np.maximum.reduce([R, G, B])
    Cmin = np.minimum.reduce([R, G, B])
    S    = np.where(Cmax > 0.02, (Cmax - Cmin) / (Cmax + eps), 0.0)

    def local_std(ch):
        # [BUG-6] float64 evita overflow ao calcular ch**2 quando ch ≈ 1.0
        c = ch.astype(np.float64)
        m1 = uniform_filter(c,      size=5)
        m2 = uniform_filter(c ** 2, size=5)
        return np.sqrt(np.maximum(m2 - m1 ** 2, 0)).astype(np.float32)

    return np.stack([
        R, G, B, ExG, ExR,
        VARI, NGRDI, GLI,
        bri - eps, r_r, r_g, S, Cmax,
        local_std(R), local_std(G), local_std(B)
    ], axis=-1).astype(np.float32)


# ──────────────────────────────────────
# ETAPA 1: TREINO RF
# ──────────────────────────────────────
def train(image_path):
    print("─" * 60)
    print("ETAPA 1: EXTRAÇÃO DE AMOSTRAS DE TREINO")
    print("─" * 60)

    meta, tiff_crs, transform, H, W = tiff_info(image_path)
    res_m = abs(transform.a)
    print(f"  TIFF : {H:,} × {W:,} px  |  resolução ≈ {res_m * 100:.1f} cm/px")

    if crs_is_geographic(tiff_crs):
        print("  [AVISO] CRS geográfico detectado. Resolução em graus/px — "
              "cálculos de área em metros podem ser imprecisos.")

    label_raster = np.zeros((H, W), dtype=np.uint8)
    for cid, cfg in CLASSES.items():
        if not os.path.exists(cfg["shp"]):
            print(f"  [AVISO] {cfg['shp']} não encontrado.")
            continue
        geoms = load_geoms(cfg["shp"], tiff_crs)
        if not geoms:
            continue
        burned = rasterize(
            [(g, cid) for g in geoms],
            out_shape=(H, W),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
        label_raster = np.where(burned > 0, burned, label_raster)
        print(f"  {cfg['nome']}: {(burned > 0).sum():,} px disponíveis")

    X_all, y_all = [], []
    with rasterio.open(image_path) as src:
        n_bands = src.count
        # [BUG-9] Garante sempre leitura de exatamente 3 bandas RGB
        rgb_bands = [1, 2, 3]
        if n_bands < 3:
            raise ValueError(f"Imagem tem apenas {n_bands} banda(s); necessário >= 3.")

        for cid, cfg in CLASSES.items():
            ys, xs = np.where(label_raster == cid)
            if len(ys) == 0:
                continue
            n = min(SAMPLES_PER_CLASS, len(ys))
            idx = np.random.default_rng(42).choice(len(ys), n, replace=False)
            ys_s, xs_s = ys[idx], xs[idx]
            tile_y = ys_s // TILE_SZ_RF
            tile_x = xs_s // TILE_SZ_RF

            feats_list = []
            for ty in np.unique(tile_y):
                for tx in np.unique(tile_x[tile_y == ty]):
                    sel = (tile_y == ty) & (tile_x == tx)
                    y0 = int(ty * TILE_SZ_RF)
                    x0 = int(tx * TILE_SZ_RF)
                    h  = min(TILE_SZ_RF, H - y0)
                    w  = min(TILE_SZ_RF, W - x0)
                    win = Window(x0, y0, w, h)
                    raw = src.read(rgb_bands, window=win).astype(np.float32) / 255.0
                    rgb = np.moveaxis(raw, 0, -1)
                    ft  = compute_features(rgb)
                    ly  = np.clip(ys_s[sel] - y0, 0, h - 1)
                    lx  = np.clip(xs_s[sel] - x0, 0, w - 1)
                    feats_list.append(ft[ly, lx])

            if feats_list:
                total_samples = sum(len(f) for f in feats_list)
                X_all.append(np.vstack(feats_list))
                y_all.append(np.full(total_samples, cid, dtype=np.int32))
                print(f"  {cfg['nome']}: {total_samples:,} amostras ✓")

    if not X_all:
        raise RuntimeError("Nenhuma amostra extraída. Verifique os shapefiles e o TIFF.")

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    print(f"\n  Total: {len(X):,} amostras | {len(np.unique(y))} classes")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, stratify=y, random_state=42
    )
    print(f"  Split: {len(X_tr):,} treino / {len(X_val):,} validação")

    print("\n" + "─" * 60)
    print("ETAPA 2: TREINO RANDOM FOREST")
    print("─" * 60)
    clf = RandomForestClassifier(
        n_estimators=RF_N_TREES,
        max_features="sqrt",
        min_samples_leaf=5,
        class_weight="balanced",
        oob_score=True,
        n_jobs=RF_JOBS,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)

    print(f"  OOB Score: {clf.oob_score_ * 100:.2f}%")
    y_pred_val = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred_val)
    print(f"  Acurácia validação: {acc * 100:.2f}%")
    print(classification_report(
        y_val, y_pred_val,
        target_names=[CLASS_NAMES[c] for c in sorted(np.unique(y_val))]
    ))

    imp = clf.feature_importances_
    top = np.argsort(imp)[::-1][:5]
    print("  Top-5 features:")
    for i in top:
        print(f"    {FEAT_NAMES[i]:8s}: {imp[i]:.3f}")

    joblib.dump(clf, MODEL_PATH_RF)
    print(f"  Modelo salvo em: {MODEL_PATH_RF}")
    return clf


# ──────────────────────────────────────
# ETAPA 2: PREDIÇÃO RF (classificado_rf.tif)
# ──────────────────────────────────────
def predict_rf(clf, image_path):
    print("\n" + "─" * 60)
    print("ETAPA 3: PREDIÇÃO RF (TILES)")
    print("─" * 60)

    meta, _, transform, H, W = tiff_info(image_path)
    out_meta = {k: v for k, v in meta.items() if k not in ("count", "dtype", "nodata")}
    out_meta.update({"count": 1, "dtype": "uint8", "nodata": 0})

    pred_path = os.path.join(OUT_DIR, "classificado_rf.tif")
    n_ty = (H + TILE_SZ_RF - 1) // TILE_SZ_RF
    n_tx = (W + TILE_SZ_RF - 1) // TILE_SZ_RF
    total = n_ty * n_tx
    done  = 0

    with rasterio.open(image_path) as src:
        n_bands  = src.count
        has_alpha = n_bands >= 4
        rgb_bands = [1, 2, 3]

        with rasterio.open(pred_path, "w", **out_meta) as dst:
            for ty in range(n_ty):
                y0 = ty * TILE_SZ_RF
                th = min(TILE_SZ_RF, H - y0)
                for tx in range(n_tx):
                    x0 = tx * TILE_SZ_RF
                    tw = min(TILE_SZ_RF, W - x0)
                    win = Window(x0, y0, tw, th)

                    raw  = src.read(rgb_bands, window=win).astype(np.float32) / 255.0
                    rgb  = np.moveaxis(raw, 0, -1)
                    # [BUG-9] Canal alpha lido separadamente e apenas se existir
                    if has_alpha:
                        alpha = src.read(4, window=win) > 0
                    else:
                        alpha = np.ones((th, tw), dtype=bool)

                    ft        = compute_features(rgb)
                    flat      = ft.reshape(-1, ft.shape[-1])
                    mask      = alpha.ravel()
                    pred_flat = np.zeros(th * tw, dtype=np.uint8)

                    if mask.sum() > 0:
                        proba    = clf.predict_proba(flat[mask])
                        max_prob = proba.max(axis=1)
                        pred_cls = clf.classes_[proba.argmax(axis=1)].astype(np.uint8)
                        # [BUG-2] Usa ID interno pequeno para "outros"
                        pred_cls[max_prob < CONF_THRESHOLD] = CLASSE_OUTROS_ID_INTERNO
                        pred_flat[mask] = pred_cls

                    # [BUG-1] Escrita explícita com shape (1, th, tw) — sem np.newaxis em 2D extra
                    tile_out = pred_flat.reshape(th, tw)
                    dst.write(tile_out[np.newaxis, :, :], window=win)

                    done += 1
                    pct = 100 * done / total
                    bar = "█" * int(pct / 5)
                    print(f"\r  [{bar:░<20}] {pct:5.1f}%", end="", flush=True)

    print(f"\n  RF classificado salvo: {pred_path}")
    return pred_path


# ──────────────────────────────────────
# ETAPA 3: REFINAMENTO COM SAM 2
# ──────────────────────────────────────
def load_sam():
    try:
        from transformers import pipeline as hf_pipeline
        device    = "cuda" if torch.cuda.is_available() else "cpu"
        device_id = 0 if device == "cuda" else -1
        print(f"  Carregando SAM2 via Transformers (device={device})...")
        gen = hf_pipeline(
            "mask-generation",
            model="facebook/sam2-hiera-large",
            device=device_id,
        )
        print("  SAM2 carregado com sucesso")
        return gen
    except Exception as e:
        print(f"  [AVISO] Falha ao carregar SAM2: {e}")
        print("  Usando apenas RF (sem refinamento SAM2)")
        return None


def _dominant_class(rf_tile, seg_mask):
    """
    [BUG-2] Calcula a classe dominante dentro de seg_mask usando apenas IDs
    internos pequenos (máx = CLASSE_OUTROS_ID_INTERNO = 4).
    np.bincount falha/fica lento com IDs grandes como 99.
    """
    vals = rf_tile[seg_mask]
    vals = vals[vals > 0]           # ignora fundo
    if len(vals) == 0:
        return 0
    # IDs válidos: 1,2,3,4 — bincount seguro
    counts = np.bincount(vals, minlength=CLASSE_OUTROS_ID_INTERNO + 1)
    dominant = int(counts[1:].argmax()) + 1  # +1 porque ignoramos índice 0
    return dominant


def refine_with_sam(image_path, rf_pred_path):
    print("\n" + "─" * 60)
    print("ETAPA 4: REFINAMENTO COM SAM 2")
    print("─" * 60)

    mask_gen = load_sam()

    with rasterio.open(image_path) as src_img, \
         rasterio.open(rf_pred_path) as src_rf:

        meta_rf   = src_rf.meta.copy()
        H, W      = src_img.height, src_img.width
        transform = src_img.transform
        crs       = src_img.crs
        res_m     = abs(transform.a)
        rgb_bands = [1, 2, 3]
        n_bands   = src_img.count

        # [BUG-4] vote_map acumula votos por classe; ao final, pixel recebe a classe com mais votos
        n_classes = CLASSE_OUTROS_ID_INTERNO + 1  # índices 0..4
        vote_map  = np.zeros((H, W, n_classes), dtype=np.uint16)
        refined   = np.zeros((H, W), dtype=np.uint8)

        refined_path = os.path.join(OUT_DIR, "classificado_sam2.tif")

        stride = SAM_TILE_SZ - SAM_OVERLAP
        n_ty   = max(1, int(np.ceil((H - SAM_OVERLAP) / stride)))
        n_tx   = max(1, int(np.ceil((W - SAM_OVERLAP) / stride)))
        total  = n_ty * n_tx
        done   = 0

        for ty in range(n_ty):
            y0 = min(ty * stride, max(0, H - SAM_TILE_SZ))
            th = min(SAM_TILE_SZ, H - y0)
            for tx in range(n_tx):
                x0 = min(tx * stride, max(0, W - SAM_TILE_SZ))
                tw = min(SAM_TILE_SZ, W - x0)

                win     = Window(x0, y0, tw, th)
                y_slice = slice(y0, y0 + th)
                x_slice = slice(x0, x0 + tw)

                # [BUG-9] Lê sempre exatamente 3 bandas RGB
                img_tile = src_img.read(rgb_bands, window=win)
                img_tile = np.moveaxis(img_tile, 0, -1)
                if img_tile.max() <= 1.0:
                    img_tile = (img_tile * 255).astype(np.uint8)
                else:
                    img_tile = img_tile.astype(np.uint8)

                # [BUG-10] Verifica shape antes de salvar PNG
                assert img_tile.ndim == 3 and img_tile.shape[2] == 3, \
                    f"img_tile shape inesperado: {img_tile.shape}"

                rf_tile = src_rf.read(1, window=win)  # shape (th, tw)

                if mask_gen is not None:
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                            tmp_path = tmp.name
                        Image.fromarray(img_tile).save(tmp_path)

                        result = mask_gen(tmp_path, points_per_batch=64)
                        masks_list = result if isinstance(result, list) else []

                        # Processa máscaras SAM e acumula votos
                        masks_added = 0
                        for mask_dict in masks_list:
                            mask_array = mask_dict["mask"] if isinstance(mask_dict, dict) else mask_dict
                            if hasattr(mask_array, "numpy"):
                                mask_array = mask_array.numpy()
                            else:
                                mask_array = np.asarray(mask_array)
                            # Força 2D
                            if mask_array.ndim == 3:
                                mask_array = mask_array.squeeze()
                            if mask_array.ndim != 2:
                                continue
                            # Alinha ao tamanho do tile
                            if mask_array.shape != (th, tw):
                                if mask_array.shape[0] >= th and mask_array.shape[1] >= tw:
                                    mask_array = mask_array[:th, :tw]
                                else:
                                    continue  # máscara menor que o tile
                            seg_mask = mask_array > 0.5
                            if seg_mask.sum() < 10:
                                continue
                            dominant = _dominant_class(rf_tile, seg_mask)
                            if dominant == 0:
                                continue
                            vote_map[y_slice, x_slice][seg_mask, dominant] += 1
                            masks_added += 1

                        # FALLBACK: pixels do tile que não foram cobertos por nenhuma máscara SAM
                        # recebem voto da classificação RF original.
                        # Vamos computar quais pixels já têm pelo menos um voto em qualquer classe.
                        tile_votes = vote_map[y_slice, x_slice].sum(axis=2)  # (th, tw) soma dos votos
                        uncovered = tile_votes == 0
                        if uncovered.any():
                            # atribui a classe RF como um voto único
                            rf_classes = rf_tile[uncovered]  # valores 0..4, sendo 0 fundo
                            # só atribui se classe != 0
                            valid = rf_classes > 0
                            rows, cols = np.where(uncovered)
                            rows_valid = rows[valid]
                            cols_valid = cols[valid]
                            classes_valid = rf_classes[valid]
                            # Adiciona 1 voto na classe correspondente
                            vote_map[y_slice, x_slice][rows_valid, cols_valid, classes_valid] += 1

                    except Exception as e:
                        print(f"\n  [ERRO] Falha no tile ({ty},{tx}): {e}")
                        # fallback de segurança: usa RF para o tile inteiro
                        for cid in range(1, n_classes):
                            m = rf_tile == cid
                            vote_map[y_slice, x_slice][m, cid] += 1
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            try:
                                os.unlink(tmp_path)
                            except OSError:
                                pass
                else:
                    # Sem SAM2: voto RF puro
                    for cid in range(1, n_classes):
                        m = rf_tile == cid
                        vote_map[y_slice, x_slice][m, cid] += 1

                done += 1
                pct = 100 * done / total
                bar = "█" * int(pct / 5)
                print(f"\r  [{bar:░<20}] {pct:5.1f}%", end="", flush=True)

        # [BUG-4] Resolve votos: cada pixel recebe a classe com mais votos
        print("\n  Resolvendo votação por maioria...")
        total_votes = vote_map.sum(axis=2)  # (H, W)
        voted_class = vote_map[:, :, 1:].argmax(axis=2) + 1  # ignora índice 0
        # Pixels sem nenhum voto ficam com 0 (fundo)
        refined = np.where(total_votes > 0, voted_class, 0).astype(np.uint8)

        # Verifica shape final
        assert refined.ndim == 2 and refined.shape == (H, W), \
            f"Shape final inesperado: {refined.shape}"
        print(f"  Shape corrigido: {refined.shape}")

        # [BUG-1] Escrita correta: shape explícito (1, H, W)
        with rasterio.open(refined_path, "w", **meta_rf) as dst:
            dst.write(refined[np.newaxis, :, :])   # shape = (1, H, W) — sempre correto

    print(f"\n  Raster refinado SAM2 salvo: {refined_path}")
    return refined_path, res_m, transform, crs


# ──────────────────────────────────────
# ETAPA 4: REMAPEAMENTO DE IDs INTERNOS → IDs DE SAÍDA
# ──────────────────────────────────────
def remap_to_output_ids(refined_path):
    """
    Converte CLASSE_OUTROS_ID_INTERNO (4) → CLASSE_OUTROS_ID_SAIDA (99) no raster final,
    garantindo que o GeoTIFF de saída use os IDs documentados externamente.
    """
    remapped_path = os.path.join(OUT_DIR, "classificado_final.tif")
    with rasterio.open(refined_path) as src:
        meta  = src.meta.copy()
        data  = src.read(1)
    data[data == CLASSE_OUTROS_ID_INTERNO] = CLASSE_OUTROS_ID_SAIDA
    with rasterio.open(remapped_path, "w", **meta) as dst:
        dst.write(data[np.newaxis, :, :])
    print(f"  IDs remapeados (4→99) salvo em: {remapped_path}")
    return remapped_path


# ──────────────────────────────────────
# ETAPA 5: VETORIZAÇÃO FINAL
# ──────────────────────────────────────
from scipy.ndimage import label, find_objects

def vectorize_final(final_raster_path, res_m, transform, crs):
    """
    [BUG-7] Verificação de CRS geográfico.
    [OTIMIZAÇÃO] Pré-filtragem de manchas < min_px usando sieve.
    """
    print("\n" + "─" * 60)
    print("ETAPA 5: VETORIZAÇÃO FINAL")
    print("─" * 60)

    vec_dir = os.path.join(OUT_DIR, "vetores")
    os.makedirs(vec_dir, exist_ok=True)

    geo_crs = crs_is_geographic(crs)
    if geo_crs:
        print("  [AVISO] CRS geográfico — shapely.area retorna graus², não m².")
        print("          MIN_AREA_M2 não pode ser aplicado com precisão.")
        print("          Considere reprojetar o TIFF para um CRS projetado (ex: UTM).")
        min_area = 0.0
    else:
        min_area = MIN_AREA_M2

    with rasterio.open(final_raster_path) as src:
        data = src.read(1)

    print(f"  Resolução: {res_m * 100:.1f} cm/px")
    if not geo_crs:
        min_px = max(1, int(min_area / (res_m ** 2)))
        print(f"  Área mínima: {min_area} m² = {min_px} px")
    else:
        min_px = 0

    schema = {
        "geometry": "Polygon",
        "properties": {
            "class_id":   "int",
            "class_name": "str",
            "area_m2":    "float",
        },
    }

    all_ids = list(CLASSES.keys()) + [CLASSE_OUTROS_ID_SAIDA]  # 1,2,3,99
    for cid in all_ids:
        cname = CLASS_NAMES_SAIDA.get(cid, str(cid))
        mask_bool = (data == cid)                                # booleano
        if not mask_bool.any():
            print(f"  {cname}: sem pixels, pulando.")
            continue

        # --- OTIMIZAÇÃO: peneirar blobs menores que min_px ---
        if min_px > 1:
            labeled, nfeats = label(mask_bool)                  # rótulos inteiros
            sizes = np.bincount(labeled.ravel())[1:]            # tamanhos de blob (índice 0 = fundo)
            remove_mask = np.isin(labeled, np.where(sizes < min_px)[0] + 1)
            mask_bool[remove_mask] = 0
            del labeled, sizes, remove_mask

        mask = mask_bool.astype(np.uint8)
        # -------------------------------------------------------

        polys = []
        for geom_dict, val in rio_shapes(mask, mask=mask, transform=transform):
            if int(val) != 1:
                continue
            p = shape(geom_dict)
            if p.area < min_area:                               # redundante agora, mas seguro
                continue
            polys.append(p)

        if not polys:
            print(f"  {cname}: sem polígonos >= {min_area} m²")
            continue

        tol   = res_m * 0.5
        feats = []
        for p in polys:
            ps = p.simplify(tol, preserve_topology=True)
            if ps.is_empty:
                continue
            feats.append({
                "geometry": mapping(ps),
                "properties": {
                    "class_id":   cid,
                    "class_name": cname,
                    "area_m2":    round(p.area, 3),
                },
            })

        if not feats:
            continue

        # GeoJSON
        geojson_path = os.path.join(vec_dir, f"classe_{cname}.geojson")
        with open(geojson_path, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "type": "FeatureCollection",
                    "crs": {
                        "type": "name",
                        "properties": {"name": CRS.from_user_input(crs).to_string()},
                    },
                    "features": [{"type": "Feature", **f} for f in feats],
                },
                fp,
                ensure_ascii=False,
            )

        # Shapefile
        shp_path = os.path.join(vec_dir, f"classe_{cname}.shp")
        with fiona.open(
            shp_path, "w",
            driver="ESRI Shapefile",
            crs=CRS.from_user_input(crs).to_wkt(),
            schema=schema,
        ) as dst:
            dst.writerecords(feats)

        total_ha = sum(f["properties"]["area_m2"] for f in feats) / 10_000
        print(f"  {cname:12s}: {len(feats):>6,} polígonos | {total_ha:.2f} ha")

    print(f"\n  Vetores salvos em: {vec_dir}")

# ──────────────────────────────────────
# MAIN
# ──────────────────────────────────────
if __name__ == "__main__":
    image_path    = find_tiff()
    force_retrain = "--retrain" in sys.argv or not os.path.exists(MODEL_PATH_RF)

    print("=" * 60)
    print("  CLASSIFICADOR RF + SAM 2  (v7)")
    print("=" * 60)
    print(f"  Imagem   : {image_path}")
    print(f"  Saída    : {OUT_DIR}")
    print(f"  Modelo RF: {'[treinar]' if force_retrain else '[reutilizar]'}")
    print("=" * 60)

    # 1) Treino RF (se necessário)
    if force_retrain:
        clf = train(image_path)
    else:
        print(f"\n  Carregando modelo RF: {MODEL_PATH_RF}")
        clf = joblib.load(MODEL_PATH_RF)

    # 2) Predição RF
    rf_pred_path = predict_rf(clf, image_path)

    # 3) Refinamento SAM2
    refined_path, res_m, transform, crs = refine_with_sam(image_path, rf_pred_path)

    # 4) Remapeamento de IDs internos → externos (4 → 99)
    final_raster_path = remap_to_output_ids(refined_path)

    # 5) Vetorização final
    vectorize_final(final_raster_path, res_m, transform, crs)

    print("\n" + "=" * 60)
    print("  PIPELINE CONCLUÍDO")
    print("=" * 60)
    print(f"  Classificado RF      : {rf_pred_path}")
    print(f"  Classificado SAM2    : {refined_path}")
    print(f"  Classificado final   : {final_raster_path}")
    print(f"  Shapefiles/GeoJSON   : {os.path.join(OUT_DIR, 'vetores/')}")
    print("=" * 60)