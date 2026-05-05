# -*- coding: utf-8 -*-
"""
Pipeline: Random Forest + SAM 2 para Classificação de Uso do Solo
==================================================================
Requisitos:
  pip install rasterio fiona shapely pyproj scikit-learn scipy joblib numpy scikit-image torch torchvision
  pip install git+https://github.com/facebookresearch/sam2.git

Modelo SAM 2:
  Baixar sam2.1_hiera_large.pt e colocar em ./modelos/
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
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import torch

warnings.filterwarnings("ignore")

# ──────────────────────────────────────
# CONFIGURAÇÕES CENTRALIZADAS
# ──────────────────────────────────────
ROOT     = os.path.abspath(".")
DATA_DIR = os.path.join(ROOT, "dados")
OUT_DIR  = os.path.join(ROOT, "saida")
MODEL_DIR = os.path.join(ROOT, "modelos")          # onde o modelo SAM2 será salvo
SAM_CHECKPOINT = os.path.join(MODEL_DIR, "sam2.1_hiera_large.pt")

# Classes (id: {shapefile, nome})
CLASSES = {
    1: {"shp": os.path.join(DATA_DIR, "palhada.shp"), "nome": "palhada"},
    2: {"shp": os.path.join(DATA_DIR, "solo.shp"),    "nome": "solo"},
    3: {"shp": os.path.join(DATA_DIR, "floresta.shp"),"nome": "floresta"},
}
CLASSE_OUTROS_ID   = 99
CLASSE_OUTROS_NOME = "outros"

# Amostras e treino RF
SAMPLES_PER_CLASS = 60_000
VALIDATION_SPLIT  = 0.2
RF_N_TREES        = 200
RF_JOBS           = -1

# Predição RF
TILE_SZ_RF        = 2048
CONF_THRESHOLD    = 0.45

# SAM 2
SAM_TILE_SZ       = 1024       # tamanho do tile para SAM (múltiplo de 16)
SAM_OVERLAP       = 128        # sobreposição entre tiles
MIN_AREA_M2       = 3.0        # área mínima de segmento/polígono (m²)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH_RF = os.path.join(OUT_DIR, "model_rf.joblib")
CLASS_NAMES = {cid: cfg["nome"] for cid, cfg in CLASSES.items()}
CLASS_NAMES[CLASSE_OUTROS_ID] = CLASSE_OUTROS_NOME


# ──────────────────────────────────────
# UTILITÁRIOS
# ──────────────────────────────────────
def find_tiff():
    tiffs = (glob.glob(os.path.join(DATA_DIR, "*.tif")) +
             glob.glob(os.path.join(DATA_DIR, "*.tiff")))
    if not tiffs:
        raise FileNotFoundError(f"Nenhum TIFF em {DATA_DIR}")
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


# ──────────────────────────────────────
# FEATURES ESPECTRAIS
# ──────────────────────────────────────
FEAT_NAMES = [
    "R","G","B","ExG","ExR",
    "VARI","NGRDI","GLI",
    "brilho","R/bri","G/bri","Sat","Value",
    "stdR","stdG","stdB"
]

def compute_features(rgb):
    R, G, B = rgb[...,0], rgb[...,1], rgb[...,2]
    eps = 1e-6
    ExG = 2*G - R - B
    ExR = 1.4*R - G
    VARI = np.where(np.abs(G+R-B)>0.02, (G-R)/(G+R-B+eps), 0)
    NGRDI = np.where((G+R)>0.02, (G-R)/(G+R+eps), 0)
    GLI = np.where((2*G+R+B)>0.02, (2*G-R-B)/(2*G+R+B+eps), 0)
    bri = (R+G+B)/3.0 + eps
    r_r = R/bri
    r_g = G/bri
    Cmax = np.maximum.reduce([R,G,B])
    Cmin = np.minimum.reduce([R,G,B])
    S = np.where(Cmax>0.02, (Cmax-Cmin)/(Cmax+eps), 0)
    def local_std(ch):
        m1 = uniform_filter(ch.astype(np.float32), size=5)
        m2 = uniform_filter(ch.astype(np.float32)**2, size=5)
        return np.sqrt(np.maximum(m2 - m1**2, 0))
    return np.stack([
        R,G,B, ExG,ExR,
        VARI,NGRDI,GLI,
        bri-eps, r_r, r_g, S, Cmax,
        local_std(R), local_std(G), local_std(B)
    ], axis=-1).astype(np.float32)


# ──────────────────────────────────────
# ETAPA 1: TREINO RF
# ──────────────────────────────────────
def train(image_path):
    print("─"*60)
    print("ETAPA 1: EXTRAÇÃO DE AMOSTRAS DE TREINO")
    print("─"*60)
    meta, tiff_crs, transform, H, W = tiff_info(image_path)
    res_m = abs(transform.a)
    print(f"  TIFF : {H:,} × {W:,} px  |  resolução ≈ {res_m*100:.1f} cm/px")

    label_raster = np.zeros((H,W), dtype=np.uint8)
    for cid, cfg in CLASSES.items():
        if not os.path.exists(cfg["shp"]):
            print(f"  [AVISO] {cfg['shp']} não encontrado.")
            continue
        geoms = load_geoms(cfg["shp"], tiff_crs)
        if not geoms:
            continue
        burned = rasterize([(g,cid) for g in geoms], out_shape=(H,W),
                           transform=transform, fill=0, dtype=np.uint8)
        label_raster = np.where(burned>0, burned, label_raster)
        print(f"  {cfg['nome']}: {(burned>0).sum():,} px disponíveis")

    X_all, y_all = [], []
    with rasterio.open(image_path) as src:
        for cid, cfg in CLASSES.items():
            ys, xs = np.where(label_raster == cid)
            if len(ys) == 0:
                continue
            n = min(SAMPLES_PER_CLASS, len(ys))
            idx = np.random.default_rng(42).choice(len(ys), n, replace=False)
            ys_s, xs_s = ys[idx], xs[idx]
            tile_y, tile_x = ys_s//TILE_SZ_RF, xs_s//TILE_SZ_RF
            feats_list = []
            for ty in np.unique(tile_y):
                for tx in np.unique(tile_x[tile_y==ty]):
                    sel = (tile_y==ty)&(tile_x==tx)
                    y0 = int(ty*TILE_SZ_RF); x0 = int(tx*TILE_SZ_RF)
                    h = min(TILE_SZ_RF, H-y0); w = min(TILE_SZ_RF, W-x0)
                    win = Window(x0, y0, w, h)
                    raw = src.read([1,2,3], window=win).astype(np.float32)/255.0
                    rgb = np.moveaxis(raw, 0, -1)
                    ft  = compute_features(rgb)
                    ly  = np.clip(ys_s[sel]-y0, 0, h-1)
                    lx  = np.clip(xs_s[sel]-x0, 0, w-1)
                    feats_list.append(ft[ly, lx])
            if feats_list:
                total_samples = sum(len(f) for f in feats_list)
                X_all.append(np.vstack(feats_list))
                y_all.append(np.full(total_samples, cid, dtype=np.int32))
                print(f"  {cfg['nome']}: {total_samples:,} amostras ✓")

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    print(f"\n  Total: {len(X):,} amostras | {len(np.unique(y))} classes")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, stratify=y, random_state=42)
    print(f"  Split: {len(X_tr):,} treino / {len(X_val):,} validação")

    print("\n"+"─"*60)
    print("ETAPA 2: TREINO RANDOM FOREST")
    print("─"*60)
    clf = RandomForestClassifier(
        n_estimators=RF_N_TREES, max_features="sqrt",
        min_samples_leaf=5, class_weight="balanced",
        oob_score=True, n_jobs=RF_JOBS, random_state=42)
    clf.fit(X_tr, y_tr)

    print(f"  OOB Score: {clf.oob_score_*100:.2f}%")
    y_pred_val = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred_val)
    print(f"  Acurácia validação: {acc*100:.2f}%")
    print(classification_report(y_val, y_pred_val,
                                target_names=[CLASS_NAMES[c] for c in sorted(np.unique(y_val))]))

    imp = clf.feature_importances_
    top = np.argsort(imp)[::-1][:5]
    print("  Top-5 features:")
    for i in top:
        print(f"    {FEAT_NAMES[i]:8s}: {imp[i]:.3f}")

    joblib.dump(clf, MODEL_PATH_RF)
    print(f"  Modelo salvo em: {MODEL_PATH_RF}")
    return clf


# ──────────────────────────────────────
# ETAPA 2: PREDIÇÃO RF (classificado.tif)
# ──────────────────────────────────────
def predict_rf(clf, image_path):
    print("\n"+"─"*60)
    print("ETAPA 3: PREDIÇÃO RF (TILES)")
    print("─"*60)
    meta, _, transform, H, W = tiff_info(image_path)
    out_meta = {k:v for k,v in meta.items() if k not in ("count","dtype","nodata")}
    out_meta.update({"count":1, "dtype":"uint8", "nodata":0})
    pred_path = os.path.join(OUT_DIR, "classificado_rf.tif")
    n_ty = (H+TILE_SZ_RF-1)//TILE_SZ_RF
    n_tx = (W+TILE_SZ_RF-1)//TILE_SZ_RF
    total = n_ty*n_tx
    done = 0

    with rasterio.open(image_path) as src:
        has_alpha = src.count >= 4
        with rasterio.open(pred_path, "w", **out_meta) as dst:
            for ty in range(n_ty):
                y0 = ty*TILE_SZ_RF; th = min(TILE_SZ_RF, H-y0)
                for tx in range(n_tx):
                    x0 = tx*TILE_SZ_RF; tw = min(TILE_SZ_RF, W-x0)
                    win = Window(x0, y0, tw, th)
                    raw = src.read([1,2,3], window=win).astype(np.float32)/255.0
                    rgb = np.moveaxis(raw, 0, -1)
                    alpha = src.read(4, window=win)>0 if has_alpha else np.ones((th,tw), dtype=bool)
                    ft = compute_features(rgb)
                    flat = ft.reshape(-1, ft.shape[-1])
                    mask = alpha.ravel()
                    pred_flat = np.zeros(th*tw, dtype=np.uint8)
                    if mask.sum()>0:
                        proba = clf.predict_proba(flat[mask])
                        max_prob = proba.max(axis=1)
                        pred_cls = clf.classes_[proba.argmax(axis=1)].astype(np.uint8)
                        pred_cls[max_prob < CONF_THRESHOLD] = CLASSE_OUTROS_ID
                        pred_flat[mask] = pred_cls
                    dst.write(pred_flat.reshape(th,tw)[np.newaxis], window=win)
                    done+=1
                    pct=100*done/total
                    print(f"\r  [{('█'*int(pct/5)):{'░'}<20}] {pct:5.1f}%", end="", flush=True)
    print(f"\n  RF classificado salvo: {pred_path}")
    return pred_path


# ──────────────────────────────────────
# ETAPA 3: REFINAMENTO COM SAM 2
# ──────────────────────────────────────
def load_sam():
    try:
        from transformers import pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_id = 0 if device == "cuda" else -1
        print(f"  Carregando SAM2 via Transformers (device={device})...")
        gen = pipeline("mask-generation", model="facebook/sam2-hiera-large", device=device_id)
        print(f"  SAM2 carregado com sucesso")
        return gen
    except Exception as e:
        print(f"  [AVISO] Falha ao carregar SAM2: {e}")
        print(f"  Usando apenas RF (sem refinamento SAM2)")
        return None

def refine_with_sam(image_path, rf_pred_path):
    print("\n"+"─"*60)
    print("ETAPA 4: REFINAMENTO COM SAM 2")
    print("─"*60)
    
    # Carrega SAM2 (pode retornar None se falhar)
    mask_gen = load_sam()

    # Abre imagem original e raster classificado
    with rasterio.open(image_path) as src_img, \
         rasterio.open(rf_pred_path) as src_rf:
        meta_img = src_img.meta
        meta_rf  = src_rf.meta
        H, W = src_img.height, src_img.width
        transform = src_img.transform
        crs = src_img.crs
        res_m = abs(transform.a)

        # Criar raster de saída refinado
        refined = np.zeros((H,W), dtype=np.uint8)
        out_meta = meta_rf.copy()
        refined_path = os.path.join(OUT_DIR, "classificado_sam2.tif")

        # Tiling com sobreposição
        stride = SAM_TILE_SZ - SAM_OVERLAP
        n_ty = max(1, int(np.ceil((H - SAM_TILE_SZ) / stride) + 1))
        n_tx = max(1, int(np.ceil((W - SAM_TILE_SZ) / stride) + 1))
        total = n_ty * n_tx
        done = 0

        # Mapa de contagem para segmentos SAM
        count_map = np.zeros((H,W), dtype=np.uint8)

        for ty in range(n_ty):
            y0 = ty * stride
            y0 = min(y0, H - SAM_TILE_SZ)
            th = min(SAM_TILE_SZ, H - y0)
            for tx in range(n_tx):
                x0 = tx * stride
                x0 = min(x0, W - SAM_TILE_SZ)
                tw = min(SAM_TILE_SZ, W - x0)

                # Lê tile da imagem original (RGB 0-255 para SAM)
                img_tile = src_img.read([1,2,3], window=Window(x0,y0,tw,th))
                img_tile = np.moveaxis(img_tile, 0, -1)  # (H,W,3)

                # Lê tile do classificado RF
                rf_tile = src_rf.read(1, window=Window(x0,y0,tw,th))

                y_slice = slice(y0, y0+th)
                x_slice = slice(x0, x0+tw)

                # Se SAM2 disponível, gera máscaras
                if mask_gen is not None:
                    try:
                        result = mask_gen(img_tile, points_per_batch=64)
                        masks_pred = result.get("masks", [])
                        
                        # Para cada máscara, voto majoritário no RF
                        for mask in masks_pred:
                            seg_mask = mask > 0.5  # threshold
                            if seg_mask.sum() < 10:  # muito pequeno
                                continue
                            # Pega classes do RF dentro da máscara
                            vals = rf_tile[seg_mask]
                            vals = vals[vals > 0]  # exclui nodata
                            if len(vals) == 0:
                                continue
                            dominant = np.bincount(vals).argmax()
                            # Atribui ao raster refinado
                            refined[y_slice, x_slice][seg_mask] = dominant
                            count_map[y_slice, x_slice][seg_mask] += 1
                    except Exception as e:
                        print(f"\n  [AVISO] Erro em SAM2 tile ({ty},{tx}): {e}")
                        print(f"  Usando RF diretamente para este tile")

                # Pixels não cobertos por SAM ou sem SAM: usa RF original
                if mask_gen is None:
                    # Sem SAM2: copia RF diretamente
                    refined[y_slice, x_slice] = rf_tile
                else:
                    # Com SAM2: preenche furos com RF
                    uncovered = count_map[y_slice, x_slice] == 0
                    rf_tile_2d = rf_tile  # shape (th, tw)
                    assign_mask = uncovered & (rf_tile_2d > 0)
                    refined[y_slice, x_slice][assign_mask] = rf_tile_2d[assign_mask]

                done += 1
                pct = 100 * done / total
                print(f"\r  [{('█'*int(pct/5)):{'░':<20}}] {pct:5.1f}%", end="", flush=True)

        # Salva raster refinado
        with rasterio.open(refined_path, "w", **out_meta) as dst:
            dst.write(refined[np.newaxis], 1)

    if mask_gen is None:
        print(f"\n  Usando apenas RF (sem SAM2): {refined_path}")
    else:
        print(f"\n  Raster refinado SAM2 salvo: {refined_path}")
    return refined_path, res_m, transform, crs


# ──────────────────────────────────────
# ETAPA 4: VETORIZAÇÃO FINAL
# ──────────────────────────────────────
def vectorize_final(refined_path, res_m, transform, crs):
    print("\n"+"─"*60)
    print("ETAPA 5: VETORIZAÇÃO FINAL")
    print("─"*60)
    vec_dir = os.path.join(OUT_DIR, "vetores")
    os.makedirs(vec_dir, exist_ok=True)

    with rasterio.open(refined_path) as src:
        data = src.read(1)

    min_px = max(1, int(MIN_AREA_M2 / (res_m**2)))
    print(f"  Resolução: {res_m*100:.1f} cm/px")
    print(f"  Área mínima: {MIN_AREA_M2} m² = {min_px} px")

    schema = {
        "geometry": "Polygon",
        "properties": {"class_id":"int", "class_name":"str", "area_m2":"float"}
    }

    all_ids = list(CLASSES.keys()) + [CLASSE_OUTROS_ID]
    for cid in all_ids:
        cname = CLASS_NAMES[cid]
        mask = (data == cid).astype(np.uint8)
        if mask.sum() == 0:
            print(f"  {cname}: sem pixels, pulando.")
            continue

        polys = []
        for geom_dict, val in rio_shapes(mask, mask=mask, transform=transform):
            if int(val) != 1:
                continue
            p = shape(geom_dict)
            if p.area < MIN_AREA_M2:
                continue
            polys.append(p)

        if not polys:
            print(f"  {cname}: sem polígonos >= {MIN_AREA_M2} m²")
            continue

        tol = res_m * 0.5
        feats = []
        for p in polys:
            ps = p.simplify(tol, preserve_topology=True)
            if ps.is_empty:
                continue
            feats.append({
                "geometry": mapping(ps),
                "properties": {
                    "class_id": cid,
                    "class_name": cname,
                    "area_m2": round(p.area, 3)
                }
            })

        # GeoJSON
        geojson_path = os.path.join(vec_dir, f"classe_{cname}.geojson")
        with open(geojson_path, "w", encoding="utf-8") as fp:
            json.dump({
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": crs.to_string()}},
                "features": [{"type":"Feature", **f} for f in feats]
            }, fp, ensure_ascii=False)

        # Shapefile
        shp_path = os.path.join(vec_dir, f"classe_{cname}.shp")
        with fiona.open(shp_path, "w", driver="ESRI Shapefile",
                        crs=crs.to_wkt(), schema=schema) as dst:
            dst.writerecords(feats)

        total_ha = sum(f["properties"]["area_m2"] for f in feats)/10000
        print(f"  {cname:12s}: {len(feats):>6,} polígonos | {total_ha:.2f} ha")

    print(f"\n  Vetores salvos em: {vec_dir}")


# ──────────────────────────────────────
# MAIN
# ──────────────────────────────────────
if __name__ == "__main__":
    image_path = find_tiff()
    force_retrain = "--retrain" in sys.argv or not os.path.exists(MODEL_PATH_RF)

    print("="*60)
    print("  CLASSIFICADOR RF + SAM 2")
    print("="*60)
    print(f"  Imagem : {image_path}")
    print(f"  Saída  : {OUT_DIR}")
    print(f"  Modelo RF: {'[treinar]' if force_retrain else '[reutilizar]'}")
    print("="*60)

    # 1) RF treino (se necessário)
    if force_retrain:
        clf = train(image_path)
    else:
        print(f"\n  Carregando modelo RF: {MODEL_PATH_RF}")
        clf = joblib.load(MODEL_PATH_RF)

    # 2) Predição RF
    rf_pred_path = predict_rf(clf, image_path)

    # 3) Refinamento SAM2
    refined_path, res_m, transform, crs = refine_with_sam(image_path, rf_pred_path)

    # 4) Vetorização final
    vectorize_final(refined_path, res_m, transform, crs)

    print("\n"+"="*60)
    print("  PIPELINE CONCLUÍDO")
    print("="*60)
    print(f"  Classificado RF      : {rf_pred_path}")
    print(f"  Classificado SAM2    : {refined_path}")
    print(f"  Shapefiles/GeoJSON   : {os.path.join(OUT_DIR, 'vetores/')}")
    print("="*60)