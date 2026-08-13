# -*- coding: utf-8 -*-
"""
Pipeline de Classificação: Random Forest + Vetorização
======================================================
Entrada : 1 TIFF RGB+Alpha  +  3 shapefiles de polígonos de treino
Saída   : TIFF classificado + GeoJSON/Shapefile por classe (QGIS-ready)

Classes geradas
  1 = palhada   2 = solo   3 = floresta   4 = outros
  (outros = pixels válidos com baixa confiança nas 3 classes acima)

Uso
  python classificador_rf.py           # treina + prediz + vetoriza
  python classificador_rf.py --retrain # força re-treino do modelo

Dependências
  pip install rasterio fiona shapely pyproj scikit-learn scipy joblib numpy
"""

import os
import sys
import glob
import json
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
DATA_DIR = os.path.join(ROOT, "1-AETHERIS_CLASSIFIER_")
OUT_DIR  = os.path.join(ROOT, "1-AETHERIS_CLASSIFIER_output")

SHAPES = {
    #1: os.path.join(DATA_DIR, "palhada.shp"),
    1: os.path.join(DATA_DIR, "solo.shp"),
    2: os.path.join(DATA_DIR, "palhada.shp"),
    3: os.path.join(DATA_DIR, "vegetacao.shp"),
    # Para adicionar a classe "outros" com shapefile próprio:
    # 4: os.path.join(DATA_DIR, "outros.shp"),
}

CLASS_NAMES  = {1: "solo", 2: "palhada", 3: "vegetacao", 4: "outros"}

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
BUFFER_M        = 0.1
# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES INTERNAS
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(OUT_DIR, "model_rf.joblib")

os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITÁRIOS
# ─────────────────────────────────────────────────────────────────────────────

def find_tiff():
    """
    Encontra o TIFF mais apropriado:
    1. Prefere imagemFull.tif (contém RGB+Alpha)
    2. Senão, procura qualquer *.tif com múltiplas bandas
    """
    # Prioritário: procura imagemFull.tif
    full_path = os.path.join(DATA_DIR, "imagemFull.tif")
    if os.path.exists(full_path):
        with rasterio.open(full_path) as src:
            if src.count >= 3:  # RGB ou RGB+Alpha
                return full_path
    
    # Fallback: procura qualquer TIFF com 3+ bandas
    tiffs = (glob.glob(os.path.join(DATA_DIR, "*.tif")) +
             glob.glob(os.path.join(DATA_DIR, "*.tiff")))
    
    for tiff_path in sorted(tiffs):
        try:
            with rasterio.open(tiff_path) as src:
                if src.count >= 3:
                    return tiff_path
        except:
            continue
    
    raise FileNotFoundError(f"Nenhum TIFF RGB encontrado em: {DATA_DIR}")

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
        
        # Detecta tipo de geometria para aplicar buffer se for Point
        is_point = f.schema['geometry'] == 'Point'
        
        for feat in f:
            raw = feat.get("geometry")
            if raw is None:
                continue
            try:
                g = shape(raw)
                
                # Se é ponto, aplica buffer ANTES de limpar (converte para polígono circular)
                if is_point:
                    g = g.buffer(BUFFER_M)
                
                # Depois limpa a geometria
                g = g.buffer(0)
                    
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
    print("─" * 60)
    print("ETAPA 1: EXTRAÇÃO DE AMOSTRAS DE TREINO")
    print("─" * 60)

    meta, tiff_crs, transform, H, W = tiff_info(image_path)
    res_m = abs(transform.a)
    print(f"  TIFF : {H:,} × {W:,} px  |  resolução ≈ {res_m*100:.1f} cm/px")

    # Verifica número de bandas
    with rasterio.open(image_path) as src:
        bands_available = src.count
    
    if bands_available < 3:
        print(f"  [ERRO] Esperado mínimo 3 bandas (RGB), mas encontrado apenas {bands_available}.")
        print(f"  Verifique: a imagem tem α (alpha/máscara)?")
        raise ValueError(f"Imagem deve ter pelo menos 3 bandas (RGB), tem {bands_available}")

    # ── Rasteriza todos os shapefiles em um único raster de labels ──────────
    label_raster = np.zeros((H, W), dtype=np.uint8)
    for cid, shp_path in SHAPES.items():
        cname = CLASS_NAMES[cid]
        if not os.path.exists(shp_path):
            print(f"  [AVISO] {shp_path} não encontrado, pulando.")
            continue
        
        # Verifica tipo de geometria
        with fiona.open(shp_path) as f:
            geom_type = f.schema['geometry']
        
        geoms = load_geoms(shp_path, tiff_crs)
        if not geoms:
            print(f"  [AVISO] Nenhuma geometria válida em {cname}.")
            continue
        
        # Informa se buffer foi aplicado
        if geom_type == 'Point':
            print(f"  {cname}: geometria PONTO detectada → aplicando buffer de {BUFFER_M}m")
        
        burned = rasterize(
            [(g, cid) for g in geoms],
            out_shape=(H, W),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
        label_raster = np.where(burned > 0, burned, label_raster)
        px = int((burned > 0).sum())
        print(f"  {cname}: {px:,} px de treino disponíveis  ({100*px/(H*W):.2f}% da imagem)")

    # ── Coleta coordenadas por classe ───────────────────────────────────────
    X_all, y_all = [], []

    with rasterio.open(image_path) as src:
        has_alpha = (src.count >= 4)

        for cid in list(SHAPES.keys()):
            cname = CLASS_NAMES[cid]
            ys_c, xs_c = np.where(label_raster == cid)
            if len(ys_c) == 0:
                print(f"  [AVISO] {cname}: sem pixels após rasterização.")
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
            print(f"  {cname}: {len(X_c):,} amostras extraídas ✓")

    if not X_all:
        raise RuntimeError("Nenhuma amostra de treino extraída. Verifique os shapefiles.")

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    print(f"\n  Total: {len(X):,} amostras | {len(np.unique(y))} classes")

    print("\n─" * 60)
    print("ETAPA 2: TREINO DO RANDOM FOREST")
    print("─" * 60)

    clf = RandomForestClassifier(
        n_estimators=RF_N_TREES,
        max_features="sqrt",
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=RF_JOBS,
        verbose=0,
        random_state=42,
    )
    print(f"  Treinando {RF_N_TREES} árvores com {len(X):,} amostras …")
    clf.fit(X, y)

    # Top features
    imp = clf.feature_importances_
    feat_names = ["R","G","B","ExG","ExR","VARI","brilho","R/bri","G/bri",
                  "Sat","Value","stdR","stdG","stdB","NGRDI"]
    top = np.argsort(imp)[::-1][:5]
    print("  Top-5 features:")
    for i in top:
        print(f"    {feat_names[i]:8s}: {imp[i]:.3f}")

    joblib.dump(clf, MODEL_PATH)
    print(f"\n  Modelo salvo: {MODEL_PATH}")
    return clf

# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 2 — PREDIÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def predict(clf, image_path):
    print("\n─" * 60)
    print("ETAPA 3: PREDIÇÃO EM TODA A IMAGEM")
    print("─" * 60)

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
                    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                    print(f"\r  [{bar}] {pct:5.1f}%  ({done}/{total_tiles} tiles)",
                          end="", flush=True)

    print(f"\n  Salvo: {pred_path}")
    return pred_path

# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 3 — VETORIZAÇÃO COM LIMPEZA DE PEQUENOS POLÍGONOS E BURACOS
# ─────────────────────────────────────────────────────────────────────────────

def vectorize(pred_path):
    print("\n─" * 60)
    print("ETAPA 4: VETORIZAÇÃO + LIMPEZA MORFOLÓGICA")
    print("─" * 60)

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

    print(f"  Resolução: {res_m*100:.1f} cm/px")
    print(f"  Área mínima de polígonos: {MIN_AREA_M2} m²  =  {min_px} px")
    print(f"  Área mínima de buracos: {HOLE_AREA_M2} m²  =  {min_hole_px} px\n")
    print("  Distribuição das classes preditas:")
    total_px = int((data > 0).sum())
    for cid, cname in CLASS_NAMES.items():
        n = int((data == cid).sum())
        pct = 100 * n / max(total_px, 1)
        print(f"    {cname:10s}: {n:>12,} px  ({pct:5.1f}%)")
    print()

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
            print(f"  {cname}: sem pixels, pulando.")
            continue

        # Limpeza morfológica
        mb = mask.astype(bool)
        mb = binary_opening(mb, structure=struct, iterations=SMOOTH_ITER)
        mb = binary_closing(mb, structure=struct, iterations=SMOOTH_ITER)
        mask = mb.astype(np.uint8)

        if mask.sum() == 0:
            print(f"  {cname}: sem pixels após limpeza, pulando.")
            continue

        # Vetoriza
        raw_polys = []
        for geom_dict, val in rio_shapes(mask, mask=mask, transform=transform):
            if int(val) != 1:
                continue
            p = shape(geom_dict)
            raw_polys.append(p)

        if not raw_polys:
            print(f"  {cname}: sem polígonos, pulando.")
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
            print(f"  {cname}: sem polígonos acima de {MIN_AREA_M2} m², pulando.")
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
        
        print(f"  {cname}: {len(feats):>6,} polígonos  |  {class_ha:.2f} ha")
        if small_removed > 0:
            print(f"    → {small_removed} polígonos pequenos (< {MIN_AREA_M2} m²) removidos")
        if holes_removed > 0:
            print(f"    → {holes_removed} buracos pequenos (< {HOLE_AREA_M2} m²) preenchidos")
        print(f"    → {shp_path}")

    print(f"\n  RESUMO DA VETORIZAÇÃO:")
    print(f"  • Total de polígonos: {total_poligonos:,}")
    print(f"  • Área total mapeada: {total_area:.2f} ha")
    print(f"\n  Todos os vetores em: {vec_dir}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    image_path   = find_tiff()
    force_retrain = "--retrain" in sys.argv or not os.path.exists(MODEL_PATH)

    print("=" * 60)
    print("  CLASSIFICADOR DE USO DO SOLO — Random Forest")
    print("=" * 60)
    print(f"  Imagem : {image_path}")
    print(f"  Saída  : {OUT_DIR}")
    print(f"  Modelo : {'[novo treino]' if force_retrain else '[reutilizando modelo salvo]'}")
    print("=" * 60)

    if force_retrain:
        clf = train(image_path)
    else:
        print(f"\n  Carregando modelo: {MODEL_PATH}")
        print("  (use --retrain para forçar novo treino)")
        clf = joblib.load(MODEL_PATH)

    pred_path = predict(clf, image_path)
    vectorize(pred_path)

    print("\n" + "=" * 60)
    print("  PIPELINE CONCLUÍDO")
    print("=" * 60)
    print(f"  TIFF classificado  →  saida/classificado.tif")
    print(f"  Vetores por classe →  saida/vetores/classe_<nome>.shp")
    print(f"                        saida/vetores/classe_<nome>.geojson")
    print()
    print("  Configurações aplicadas:")
    print(f"  • Polígonos menores que {MIN_AREA_M2} m² → REMOVIDOS")
    print(f"  • Buracos internos menores que {HOLE_AREA_M2} m² → PREENCHIDOS")
    print()
    print("  Dicas para o QGIS:")
    print("  • Abra o .tif com 'Estilo → Valores únicos' para visualizar")
    print("  • Os .shp já têm campo area_m2 para filtrar por tamanho")
    print("  • Para re-treinar: python classificador_rf.py --retrain")
    print("=" * 60)
