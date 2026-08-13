# -*- coding: utf-8 -*-
"""
Pipeline de Classificação v11: Random Forest + Vetorização + Log com Timestamps
================================================================================
Entrada : 1 TIFF RGB+Alpha  +  3 shapefiles de polígonos de treino
Saída   : TIFF classificado + GeoJSON/Shapefile por classe (QGIS-ready)
          + Arquivo de log (.txt) com timestamps, hashes e métricas

Classes geradas
  1 = palhada   2 = solo   3 = floresta   4 = outros
  (outros = pixels válidos com baixa confiança nas 3 classes acima)

Novidades da v11:
  • Todas as mensagens de print com timestamp [HH:MM:SS]
  • Arquivo de log .txt automático a cada execução
  • Hash SHA-256 dos arquivos de entrada e saída no log
  • Métricas de qualidade e resumo completo no log

Uso
  python ClassificarRandom_v11_funcional.py           # treina + prediz + vetoriza
  python ClassificarRandom_v11_funcional.py --retrain # força re-treino do modelo

Dependências
  pip install rasterio fiona shapely pyproj scikit-learn scipy joblib numpy
"""

import os
import sys
import glob
import json
import hashlib
import datetime
import warnings
import numpy as np
import rasterio
from rasterio.features import rasterize, shapes as rio_shapes
from rasterio.windows import Window
import fiona
from pyproj import CRS, Transformer
from shapely.geometry import shape, mapping, Polygon
from shapely.ops import transform as shp_transform
from scipy.ndimage import binary_opening, binary_closing, uniform_filter
from sklearn.ensemble import RandomForestClassifier
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES  ←  edite aqui
# ─────────────────────────────────────────────────────────────────────────────

ROOT     = os.path.abspath(".")
DATA_DIR = os.path.join(ROOT, "imaru2")
OUT_DIR  = os.path.join(ROOT, "saida_v11_imaru4")

SHAPES = {
    1: os.path.join(DATA_DIR, "solo.shp"),
    2: os.path.join(DATA_DIR, "floresta.shp"),
}

CLASS_NAMES  = {1: "solo", 2: "floresta", 3: "outros"}

# Treino
SAMPLES_PER_CLASS = 60_000   # pixels de treino por classe (balanceado)
RF_N_TREES        = 200      # árvores — mais = mais lento, mas mais preciso
RF_JOBS           = -1       # -1 = usa todos os núcleos da CPU

# Predição
TILE_SZ           = 2048     # pixels por tile (RAM ↔ velocidade)
CONF_THRESHOLD    = 0.45     # confiança mínima para 1/2/3; abaixo → "outros"

# Vetorização
MIN_AREA_M2       = 5.0      # ÁREA MÍNIMA: 5 m² - polígonos e buracos menores que isso são eliminados
SMOOTH_ITER       = 2        # iterações de abertura/fechamento morfológico
HOLE_AREA_M2      = 5.0      # Buracos internos menores que 5 m² são preenchidos

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES INTERNAS
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(OUT_DIR, "model_rf.joblib")

os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SISTEMA DE LOG COM TIMESTAMP
# ─────────────────────────────────────────────────────────────────────────────

# Gera nome do arquivo de log com data/hora
LOG_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = os.path.join(OUT_DIR, f"log_{LOG_TIMESTAMP}.txt")

_log_file = None

def open_log():
    """Abre o arquivo de log para escrita."""
    global _log_file
    _log_file = open(LOG_PATH, "w", encoding="utf-8")
    # Cabeçalho do log
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _log_file.write("=" * 70 + "\n")
    _log_file.write(f"  CLASSIFICADOR v11 — LOG DE EXECUÇÃO\n")
    _log_file.write(f"  Início: {now_str}\n")
    _log_file.write("=" * 70 + "\n\n")

def close_log():
    """Fecha o arquivo de log."""
    global _log_file
    if _log_file:
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _log_file.write(f"\n  Fim: {now_str}\n")
        _log_file.write("=" * 70 + "\n")
        _log_file.close()
        _log_file = None

def get_timestamp():
    """Retorna timestamp formatado [HH:MM:SS]."""
    return datetime.datetime.now().strftime("[%H:%M:%S]")

def log(msg, end="\n", to_log=True):
    """
    Imprime mensagem com timestamp [HH:MM:SS] e opcionalmente grava no arquivo de log.

    Args:
        msg: Mensagem a ser exibida
        end: Caractere de final de linha (default: "\\n")
        to_log: Se True, grava também no arquivo de log (default: True)
    """
    timestamp = get_timestamp()
    line = f"{timestamp} {msg}"
    print(line, end=end, flush=True)
    if _log_file and to_log:
        _log_file.write(line + ("\n" if end == "\n" else ""))
        _log_file.flush()

def log_separator(char="-", width=60):
    """Imprime linha separadora."""
    line = char * width
    log(line)

def log_header(title, char="=", width=60):
    """Imprime cabeçalho centralizado."""
    log(char * width)
    log(f"  {title}")
    log(char * width)

def log_value(label, value):
    """Imprime par chave-valor formatado."""
    log(f"  {label}: {value}")

# ─────────────────────────────────────────────────────────────────────────────
# UTILITÁRIOS DE HASH
# ─────────────────────────────────────────────────────────────────────────────

def compute_file_hash(filepath):
    """
    Calcula o hash SHA-256 de um arquivo.
    Retorna string hexadecimal ou 'N/D' se arquivo não existir.
    """
    if not os.path.exists(filepath):
        return "N/D (arquivo não encontrado)"
    
    try:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        return f"ERRO: {e}"

def compute_tiff_hash(filepath):
    """
    Calcula hash SHA-256 baseado no conteúdo dos pixels de um TIFF.
    (útil para verificar se o mesmo TIFF foi usado)
    """
    if not os.path.exists(filepath):
        return "N/D"
    
    try:
        sha256 = hashlib.sha256()
        with rasterio.open(filepath) as src:
            # Lê todas as bandas e calcula hash dos dados
            data = src.read()
            sha256.update(data.tobytes())
            # Inclui metadados relevantes no hash
            meta_str = json.dumps({
                "height": src.height,
                "width": src.width,
                "count": src.count,
                "crs": str(src.crs),
                "transform": list(src.transform),
            }, sort_keys=True)
            sha256.update(meta_str.encode())
        return sha256.hexdigest()
    except Exception as e:
        return f"ERRO: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# UTILITÁRIOS GERAIS
# ─────────────────────────────────────────────────────────────────────────────

def find_tiff():
    tiffs = (glob.glob(os.path.join(DATA_DIR, "*.tif")) +
             glob.glob(os.path.join(DATA_DIR, "*.tiff")))
    if not tiffs:
        raise FileNotFoundError(f"Nenhum TIFF encontrado em: {DATA_DIR}")
    return tiffs[0]

def tiff_info(path):
    with rasterio.open(path) as src:
        return src.meta.copy(), src.crs, src.transform, src.height, src.width

def reproject_geom(geom, src_crs, dst_crs):
    t = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shp_transform(t.transform, geom)

def load_geoms(shp_path, tiff_crs):
    geoms = []
    with fiona.open(shp_path) as f:
        shp_crs = CRS.from_user_input(f.crs)
        reproj  = not shp_crs.equals(tiff_crs, ignore_axis_order=True)
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

def remove_small_holes(polygon, min_area_m2):
    """
    Remove buracos internos (interior rings) menores que min_area_m2
    e retorna o polígono sem esses buracos.
    """
    if not hasattr(polygon, 'interiors') or len(polygon.interiors) == 0:
        return polygon
    
    # Lista para armazenar apenas os buracos grandes
    large_holes = []
    
    for hole in polygon.interiors:
        hole_polygon = Polygon(hole)
        if hole_polygon.area >= min_area_m2:
            large_holes.append(hole)
    
    # Reconstrói o polígono apenas com os buracos grandes
    return Polygon(polygon.exterior, large_holes)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES ESPECTRAIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(rgb_tile):
    """
    Entrada : float32 [H, W, 3]  valores 0–1
    Saída   : float32 [H, W, 15] — features espectrais + textura

    Features
    ─────────────────────────────────────────────────────────────────────────────
    0–2  : R, G, B normalizados
    3    : ExG  — Excess Green   = 2G − R − B  (vegetação viva)
    4    : ExR  — Excess Red     = 1.4R − G    (solo exposto)
    5    : VARI — (G−R)/(G+R−B)               (dossel verde)
    6    : Brilho médio  (R+G+B)/3
    7    : Razão R/brilho  (tons avermelhados)
    8    : Razão G/brilho  (tons esverdeados)
    9    : Saturação  (Cmax−Cmin)/Cmax
    10   : Value  Cmax
    11–13: Desvio local R, G, B  (janela 5×5)  — textura
    14   : NGRDI  (G−R)/(G+R)                  (separador palhada×floresta)
    """
    R, G, B = rgb_tile[..., 0], rgb_tile[..., 1], rgb_tile[..., 2]
    eps = 1e-6

    ExG   = 2*G - R - B
    ExR   = 1.4*R - G
    denom = G + R - B
    VARI  = np.where(np.abs(denom) > 0.02, (G - R) / (denom + eps), 0.0)
    NGRDI = np.where((G + R) > 0.02, (G - R) / (G + R + eps), 0.0)
    bri   = (R + G + B) / 3.0 + eps
    r_r   = R / bri
    r_g   = G / bri
    Cmax  = np.maximum.reduce([R, G, B])
    Cmin  = np.minimum.reduce([R, G, B])
    S     = np.where(Cmax > 0.02, (Cmax - Cmin) / (Cmax + eps), 0.0)

    def local_std(ch):
        m1 = uniform_filter(ch.astype(np.float32), size=5)
        m2 = uniform_filter(ch.astype(np.float32)**2, size=5)
        return np.sqrt(np.maximum(m2 - m1**2, 0.0))

    feats = np.stack([
        R, G, B,
        ExG, ExR, VARI, bri - eps, r_r, r_g,
        S, Cmax,
        local_std(R), local_std(G), local_std(B),
        NGRDI,
    ], axis=-1).astype(np.float32)

    return feats  # [H, W, 15]

# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 1 — TREINO
# ─────────────────────────────────────────────────────────────────────────────

def train(image_path):
    log_separator()
    log_header("ETAPA 1: EXTRAÇÃO DE AMOSTRAS DE TREINO")

    meta, tiff_crs, transform, H, W = tiff_info(image_path)
    res_m = abs(transform.a)
    log_value(f"TIFF", f"{H:,} × {W:,} px  |  resolução ≈ {res_m*100:.1f} cm/px")

    # ── Rasteriza todos os shapefiles em um único raster de labels ──────────
    label_raster = np.zeros((H, W), dtype=np.uint8)
    for cid, shp_path in SHAPES.items():
        cname = CLASS_NAMES[cid]
        if not os.path.exists(shp_path):
            log(f"  [AVISO] {shp_path} não encontrado, pulando.")
            continue
        geoms = load_geoms(shp_path, tiff_crs)
        if not geoms:
            log(f"  [AVISO] Nenhuma geometria válida em {cname}.")
            continue
        burned = rasterize(
            [(g, cid) for g in geoms],
            out_shape=(H, W),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
        label_raster = np.where(burned > 0, burned, label_raster)
        px = int((burned > 0).sum())
        log_value(f"  {cname}", f"{px:,} px de treino disponíveis  ({100*px/(H*W):.2f}% da imagem)")

    # ── Coleta coordenadas por classe ───────────────────────────────────────
    X_all, y_all = [], []

    with rasterio.open(image_path) as src:
        has_alpha = (src.count >= 4)

        for cid in list(SHAPES.keys()):
            cname = CLASS_NAMES[cid]
            ys_c, xs_c = np.where(label_raster == cid)
            if len(ys_c) == 0:
                log(f"  [AVISO] {cname}: sem pixels após rasterização.")
                continue

            n = min(SAMPLES_PER_CLASS, len(ys_c))
            idx = np.random.default_rng(42).choice(len(ys_c), n, replace=False)
            ys_s, xs_s = ys_c[idx], xs_c[idx]

            # Agrupa por tile para leitura eficiente
            tile_y = ys_s // TILE_SZ
            tile_x = xs_s // TILE_SZ

            feats_list = []
            for ty in np.unique(tile_y):
                for tx in np.unique(tile_x[(tile_y == ty)]):
                    sel = (tile_y == ty) & (tile_x == tx)
                    y0 = int(ty * TILE_SZ)
                    x0 = int(tx * TILE_SZ)
                    h  = min(TILE_SZ, H - y0)
                    w  = min(TILE_SZ, W - x0)
                    win = Window(x0, y0, w, h)
                    raw = src.read([1, 2, 3], window=win).astype(np.float32) / 255.0
                    rgb = np.moveaxis(raw, 0, -1)
                    ft  = compute_features(rgb)
                    ly  = np.clip(ys_s[sel] - y0, 0, h - 1)
                    lx  = np.clip(xs_s[sel] - x0, 0, w - 1)
                    feats_list.append(ft[ly, lx])

            if not feats_list:
                continue
            X_c = np.vstack(feats_list)
            y_c = np.full(len(X_c), cid, dtype=np.int32)
            X_all.append(X_c)
            y_all.append(y_c)
            log(f"  {cname}: {len(X_c):,} amostras extraídas ✓")

    if not X_all:
        raise RuntimeError("Nenhuma amostra de treino extraída. Verifique os shapefiles.")

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    log(f"\n  Total: {len(X):,} amostras | {len(np.unique(y))} classes")

    log_separator()
    log_header("ETAPA 2: TREINO DO RANDOM FOREST")

    clf = RandomForestClassifier(
        n_estimators=RF_N_TREES,
        max_features="sqrt",
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=RF_JOBS,
        verbose=0,
        random_state=42,
    )
    log(f"  Treinando {RF_N_TREES} árvores com {len(X):,} amostras …")
    clf.fit(X, y)

    # Top features
    imp = clf.feature_importances_
    feat_names = ["R","G","B","ExG","ExR","VARI","brilho","R/bri","G/bri",
                  "Sat","Value","stdR","stdG","stdB","NGRDI"]
    top = np.argsort(imp)[::-1][:5]
    log("  Top-5 features:")
    for i in top:
        log(f"    {feat_names[i]:8s}: {imp[i]:.3f}")

    joblib.dump(clf, MODEL_PATH)
    log(f"\n  Modelo salvo: {MODEL_PATH}")
    return clf

# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 2 — PREDIÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def predict(clf, image_path):
    log_separator()
    log_header("ETAPA 3: PREDIÇÃO EM TODA A IMAGEM")

    meta, _, transform, H, W = tiff_info(image_path)
    res_m = abs(transform.a)

    out_meta = {k: v for k, v in meta.items()
                if k not in ("count", "dtype", "nodata")}
    out_meta.update({"count": 1, "dtype": "uint8", "nodata": 0})

    pred_path = os.path.join(OUT_DIR, "classificado.tif")

    n_ty = (H + TILE_SZ - 1) // TILE_SZ
    n_tx = (W + TILE_SZ - 1) // TILE_SZ
    total_tiles = n_ty * n_tx
    done = 0

    with rasterio.open(image_path) as src:
        has_alpha = src.count >= 4
        with rasterio.open(pred_path, "w", **out_meta) as dst:
            for ty in range(n_ty):
                y0 = ty * TILE_SZ
                th = min(TILE_SZ, H - y0)
                for tx in range(n_tx):
                    x0 = tx * TILE_SZ
                    tw = min(TILE_SZ, W - x0)
                    win = Window(x0, y0, tw, th)

                    raw = src.read([1, 2, 3], window=win).astype(np.float32) / 255.0
                    rgb = np.moveaxis(raw, 0, -1)

                    alpha_valid = (src.read(4, window=win) > 0
                                   if has_alpha
                                   else np.ones((th, tw), dtype=bool))

                    ft   = compute_features(rgb)          # [H, W, 15]
                    flat = ft.reshape(-1, ft.shape[-1])
                    mask = alpha_valid.ravel()

                    pred_flat = np.zeros(th * tw, dtype=np.uint8)
                    if mask.sum() > 0:
                        proba     = clf.predict_proba(flat[mask])       # [N, n_classes]
                        max_prob  = proba.max(axis=1)
                        pred_cls  = clf.classes_[proba.argmax(axis=1)].astype(np.uint8)
                        # Pixels com confiança baixa → classe 4 "outros"
                        pred_cls[max_prob < CONF_THRESHOLD] = 4
                        pred_flat[mask] = pred_cls

                    dst.write(pred_flat.reshape(th, tw)[np.newaxis], window=win)

                    done += 1
                    pct = 100 * done / total_tiles
                    bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
                    # Usa print direto para a barra de progresso (mesma linha)
                    msg = f"  [{bar}] {pct:5.1f}%  ({done}/{total_tiles} tiles)"
                    log(msg, end="\r", to_log=False)
                    # Se for o último tile, quebra linha
                    if done == total_tiles:
                        log("")

    log(f"  Salvo: {pred_path}")
    return pred_path

# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 3 — VETORIZAÇÃO COM LIMPEZA DE PEQUENOS POLÍGONOS E BURACOS
# ─────────────────────────────────────────────────────────────────────────────

def vectorize(pred_path):
    log_separator()
    log_header("ETAPA 4: VETORIZAÇÃO + LIMPEZA MORFOLÓGICA")

    vec_dir = os.path.join(OUT_DIR, "vetores")
    os.makedirs(vec_dir, exist_ok=True)

    with rasterio.open(pred_path) as src:
        data      = src.read(1)
        transform = src.transform
        crs       = src.crs
        res_m     = abs(transform.a)

    px_area  = res_m ** 2
    min_px   = max(1, int(MIN_AREA_M2 / px_area))
    min_hole_px = max(1, int(HOLE_AREA_M2 / px_area))

    log(f"  Resolução: {res_m*100:.1f} cm/px")
    log(f"  Área mínima de polígonos: {MIN_AREA_M2} m²  =  {min_px} px")
    log(f"  Área mínima de buracos: {HOLE_AREA_M2} m²  =  {min_hole_px} px\n")
    log("  Distribuição das classes preditas:")
    total_px = int((data > 0).sum())
    for cid, cname in CLASS_NAMES.items():
        n = int((data == cid).sum())
        pct = 100 * n / max(total_px, 1)
        log(f"    {cname:10s}: {n:>12,} px  ({pct:5.1f}%)")
    log("")

    struct = np.ones((3, 3), dtype=bool)

    schema = {
        "geometry": "Polygon",
        "properties": {"class_id": "int", "class_name": "str", "area_m2": "float"},
    }

    total_poligonos = 0
    total_area = 0

    for cid, cname in CLASS_NAMES.items():
        mask = (data == cid).astype(np.uint8)
        if mask.sum() == 0:
            log(f"  {cname}: sem pixels, pulando.")
            continue

        # Limpeza morfológica
        mb = mask.astype(bool)
        mb = binary_opening(mb, structure=struct, iterations=SMOOTH_ITER)
        mb = binary_closing(mb, structure=struct, iterations=SMOOTH_ITER)
        mask = mb.astype(np.uint8)

        if mask.sum() == 0:
            log(f"  {cname}: sem pixels após limpeza, pulando.")
            continue

        # Vetoriza
        raw_polys = []
        for geom_dict, val in rio_shapes(mask, mask=mask, transform=transform):
            if int(val) != 1:
                continue
            p = shape(geom_dict)
            raw_polys.append(p)

        if not raw_polys:
            log(f"  {cname}: sem polígonos, pulando.")
            continue

        # Processamento de polígonos: remove pequenos e preenche buracos
        polys = []
        holes_removed = 0
        small_removed = 0
        
        for p in raw_polys:
            # Verifica se é um polígono principal ou um buraco
            if not p.is_valid:
                p = p.buffer(0)  # Corrige geometria inválida
            
            # Pula polígonos muito pequenos
            if p.area < MIN_AREA_M2:
                small_removed += 1
                continue
            
            # Remove buracos pequenos
            if p.interiors:
                p_cleaned = remove_small_holes(p, HOLE_AREA_M2)
                holes_removed += len(p.interiors) - len(p_cleaned.interiors) if hasattr(p_cleaned, 'interiors') else 0
                polys.append(p_cleaned)
            else:
                polys.append(p)

        if not polys:
            log(f"  {cname}: sem polígonos acima de {MIN_AREA_M2} m², pulando.")
            continue

        # Simplifica geometria (metade do pixel → suaviza sem perder detalhe)
        tol = res_m * 0.5
        feats = []
        for p in polys:
            ps = p.simplify(tol, preserve_topology=True)
            if ps.is_empty:
                continue
            
            # Garante que o polígono simplificado ainda é válido
            if not ps.is_valid:
                ps = ps.buffer(0)
            
            feats.append({
                "geometry": mapping(ps),
                "properties": {
                    "class_id":  cid,
                    "class_name": cname,
                    "area_m2":   round(p.area, 3),
                },
            })

        if not feats:
            continue

        # GeoJSON ─────────────────────────────────────────────────────────────
        geojson_path = os.path.join(vec_dir, f"classe_{cname}.geojson")
        with open(geojson_path, "w", encoding="utf-8") as fp:
            json.dump({
                "type": "FeatureCollection",
                "crs": {"type": "name",
                        "properties": {"name": crs.to_string()}},
                "features": [{"type": "Feature", **f} for f in feats],
            }, fp, ensure_ascii=False)

        # Shapefile ───────────────────────────────────────────────────────────
        shp_path = os.path.join(vec_dir, f"classe_{cname}.shp")
        with fiona.open(shp_path, "w",
                        driver="ESRI Shapefile",
                        crs=crs.to_wkt(),
                        schema=schema) as dst:
            dst.writerecords(feats)

        class_ha = sum(f["properties"]["area_m2"] for f in feats) / 10_000
        total_poligonos += len(feats)
        total_area += class_ha
        
        log(f"  {cname}: {len(feats):>6,} polígonos  |  {class_ha:.2f} ha")
        if small_removed > 0:
            log(f"    → {small_removed} polígonos pequenos (< {MIN_AREA_M2} m²) removidos")
        if holes_removed > 0:
            log(f"    → {holes_removed} buracos pequenos (< {HOLE_AREA_M2} m²) preenchidos")
        log(f"    → {shp_path}")

    log(f"\n  RESUMO DA VETORIZAÇÃO:")
    log(f"  • Total de polígonos: {total_poligonos:,}")
    log(f"  • Área total mapeada: {total_area:.2f} ha")
    log(f"\n  Todos os vetores em: {vec_dir}")
    
    # Retorna métricas para o log final
    return {
        "total_poligonos": total_poligonos,
        "total_area_ha": total_area,
        "vec_dir": vec_dir,
    }

# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÃO DE LOG FINAL — MÉTRICAS, HASHES E QUALIDADE
# ─────────────────────────────────────────────────────────────────────────────

def log_final_summary(image_path, pred_path, force_retrain, vec_metrics):
    """
    Registra no arquivo de log um resumo completo com:
      - Datas e horários
      - Caminhos de arquivos de entrada e saída
      - Hash SHA-256 dos arquivos
      - Métricas de qualidade da classificação
      - Configurações aplicadas
    """
    log("")
    log_separator(char="=")
    log_header("RESUMO FINAL — MÉTRICAS E QUALIDADE", char="=")
    
    # ── Informações de data/hora ────────────────────────────────────────────
    now = datetime.datetime.now()
    log("  INFORMAÇÕES DE DATA/HORA:")
    log(f"    Início da execução : {LOG_TIMESTAMP[:4]}-{LOG_TIMESTAMP[4:6]}-{LOG_TIMESTAMP[6:8]} "
        f"{LOG_TIMESTAMP[9:11]}:{LOG_TIMESTAMP[11:13]}:{LOG_TIMESTAMP[13:15]}")
    log(f"    Término            : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"    Duração aproximada : {(now - datetime.datetime.strptime(LOG_TIMESTAMP, '%Y%m%d_%H%M%S')).total_seconds():.1f} s")
    log("")
    
    # ── Arquivos de entrada ──────────────────────────────────────────────────
    log("  ARQUIVOS DE ENTRADA:")
    log(f"    Imagem TIFF:")
    log(f"      Caminho : {image_path}")
    log(f"      Tamanho : {os.path.getsize(image_path):,} bytes")
    log(f"      SHA-256 : {compute_tiff_hash(image_path)}")
    
    for cid, shp_path in SHAPES.items():
        cname = CLASS_NAMES[cid]
        if os.path.exists(shp_path):
            log(f"    Shapefile '{cname}':")
            log(f"      Caminho : {shp_path}")
            log(f"      Tamanho : {os.path.getsize(shp_path):,} bytes")
            log(f"      SHA-256 : {compute_file_hash(shp_path)}")
        else:
            log(f"    Shapefile '{cname}': NÃO ENCONTRADO ({shp_path})")
    
    log(f"    Modelo Random Forest:")
    if os.path.exists(MODEL_PATH):
        log(f"      Caminho : {MODEL_PATH}")
        log(f"      Tamanho : {os.path.getsize(MODEL_PATH):,} bytes")
        log(f"      SHA-256 : {compute_file_hash(MODEL_PATH)}")
    else:
        log(f"      Caminho : {MODEL_PATH} (não salvo ainda)")
    log("")
    
    # ── Arquivos de saída ───────────────────────────────────────────────────
    log("  ARQUIVOS DE SAÍDA:")
    log(f"    TIFF Classificado:")
    log(f"      Caminho : {pred_path}")
    if os.path.exists(pred_path):
        log(f"      Tamanho : {os.path.getsize(pred_path):,} bytes")
        log(f"      SHA-256 : {compute_tiff_hash(pred_path)}")
    log(f"    Arquivo de Log:")
    log(f"      Caminho : {LOG_PATH}")
    log(f"      Tamanho : {os.path.getsize(LOG_PATH):,} bytes")
    log("")
    
    # ── Métricas de qualidade da classificação ──────────────────────────────
    log("  MÉTRICAS DE QUALIDADE DA CLASSIFICAÇÃO:")
    
    with rasterio.open(pred_path) as src:
        data = src.read(1)
        transform = src.transform
        res_m = abs(transform.a)
        px_area = res_m ** 2
    
    total_pixels_validos = int((data > 0).sum())
    total_area_m2 = total_pixels_validos * px_area
    total_area_ha = total_area_m2 / 10000
    
    log(f"    Dimensões da imagem: {data.shape[1]:,} × {data.shape[0]:,} px")
    log(f"    Resolução: {res_m*100:.1f} cm/px")
    log(f"    Total de pixels válidos: {total_pixels_validos:,}")
    log(f"    Área total processada: {total_area_m2:.2f} m²  ({total_area_ha:.2f} ha)")
    
    # Distribuição por classe
    for cid, cname in CLASS_NAMES.items():
        n_px = int((data == cid).sum())
        if n_px == 0:
            continue
        area_m2 = n_px * px_area
        area_ha = area_m2 / 10000
        pct = 100 * n_px / max(total_pixels_validos, 1)
        log(f"    • {cname:10s}: {n_px:>12,} px  |  {area_m2:>12.2f} m²  |  {area_ha:>8.2f} ha  |  {pct:>5.1f}%")
    
    log("")
    
    # ── Métricas de vetorização ─────────────────────────────────────────────
    log("  MÉTRICAS DE VETORIZAÇÃO:")
    log(f"    Total de polígonos gerados : {vec_metrics['total_poligonos']:,}")
    log(f"    Área total vetorizada     : {vec_metrics['total_area_ha']:.2f} ha")
    log(f"    Diretório dos vetores     : {vec_metrics['vec_dir']}")
    log("")
    
    # ── Configurações aplicadas ─────────────────────────────────────────────
    log("  CONFIGURAÇÕES APLICADAS:")
    log(f"    • Amostras por classe : {SAMPLES_PER_CLASS:,}")
    log(f"    • Árvores RF          : {RF_N_TREES}")
    log(f"    • Limiar de confiança : {CONF_THRESHOLD}")
    log(f"    • Tamanho do tile     : {TILE_SZ} px")
    log(f"    • Área min. polígono  : {MIN_AREA_M2} m²")
    log(f"    • Área min. buraco    : {HOLE_AREA_M2} m²")
    log(f"    • Iterações morfológicas: {SMOOTH_ITER}")
    log(f"    • Modo do modelo      : {'NOVO TREINO' if force_retrain else 'REUTILIZADO'}")
    log("")
    
    # ── Hash de todos os shapefiles de saída ────────────────────────────────
    log("  HASH DOS ARQUIVOS GERADOS:")
    log(f"    {pred_path}")
    log(f"      SHA-256: {compute_tiff_hash(pred_path)}")
    
    vec_dir = vec_metrics['vec_dir']
    if os.path.exists(vec_dir):
        for fname in sorted(os.listdir(vec_dir)):
            fpath = os.path.join(vec_dir, fname)
            if os.path.isfile(fpath):
                log(f"    {fname}")
                log(f"      SHA-256: {compute_file_hash(fpath)}")
    
    log("")
    log_separator(char="=")
    log("  LOG FINALIZADO")
    log_separator(char="=")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Inicializa o sistema de log ─────────────────────────────────────────
    open_log()

    image_path    = find_tiff()
    force_retrain = "--retrain" in sys.argv or not os.path.exists(MODEL_PATH)

    log_separator(char="=")
    log_header("CLASSIFICADOR DE USO DO SOLO v11 — Random Forest")
    log_separator(char="=")
    log(f"  Imagem : {image_path}")
    log(f"  Saída  : {OUT_DIR}")
    log(f"  Log    : {LOG_PATH}")
    log(f"  Modelo : {'[NOVO TREINO]' if force_retrain else '[REUTILIZANDO MODELO SALVO]'}")
    log_separator(char="=")

    if force_retrain:
        clf = train(image_path)
    else:
        log(f"\n  Carregando modelo: {MODEL_PATH}")
        log("  (use --retrain para forçar novo treino)")
        clf = joblib.load(MODEL_PATH)

    pred_path   = predict(clf, image_path)
    vec_metrics = vectorize(pred_path)

    # ── Log final com resumo, hashes e qualidade ────────────────────────────
    log_final_summary(image_path, pred_path, force_retrain, vec_metrics)

    # ── Mensagem final no terminal ──────────────────────────────────────────
    log_separator(char="=")
    log_header("PIPELINE CONCLUÍDO")
    log_separator(char="=")
    log(f"  TIFF classificado  →  saida/classificado.tif")
    log(f"  Vetores por classe →  saida/vetores/classe_<nome>.shp")
    log(f"                        saida/vetores/classe_<nome>.geojson")
    log(f"  Log completo       →  {LOG_PATH}")
    log("")
    log("  Configurações aplicadas:")
    log(f"  • Polígonos menores que {MIN_AREA_M2} m² → REMOVIDOS")
    log(f"  • Buracos internos menores que {HOLE_AREA_M2} m² → PREENCHIDOS")
    log("")
    log("  Dicas para o QGIS:")
    log("  • Abra o .tif com 'Estilo → Valores únicos' para visualizar")
    log("  • Os .shp já têm campo area_m2 para filtrar por tamanho")
    log("  • Para re-treinar: python ClassificarRandom_v11_funcional.py --retrain")
    log_separator(char="=")

    # ── Fecha o arquivo de log ──────────────────────────────────────────────
    close_log()