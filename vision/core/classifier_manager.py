# -*- coding: utf-8 -*-
"""Lógica de classificação Random Forest — agnóstica de UI, com callbacks de progresso."""

import os
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


class ClassifierConfig:
    """Configuração do classificador — todos os parâmetros editáveis."""

    def __init__(self, **kwargs):
        # Arquivos de entrada/saída
        self.input_tiff = kwargs.get("input_tiff", "")
        self.output_tiff = kwargs.get("output_tiff", "")

        # Lista de shapes: [{"name": "solo", "path": "solo.shp"}, ...]
        self.shapes_list = kwargs.get("shapes_list", [])

        # Treino
        self.samples_per_class = int(kwargs.get("samples_per_class", 60000))
        self.rf_n_trees = int(kwargs.get("rf_n_trees", 200))
        self.rf_jobs = int(kwargs.get("rf_jobs", -1))

        # Predição
        self.tile_sz = int(kwargs.get("tile_sz", 2048))
        self.conf_threshold = float(kwargs.get("conf_threshold", 0.45))

        # Vetorização
        self.min_area_m2 = float(kwargs.get("min_area_m2", 5.0))
        self.smooth_iter = int(kwargs.get("smooth_iter", 2))
        self.hole_area_m2 = float(kwargs.get("hole_area_m2", 5.0))
        self.buffer_m = float(kwargs.get("buffer_m", 0.1))

        # Comportamento
        self.force_retrain = bool(kwargs.get("force_retrain", False))

    @property
    def out_dir(self):
        """Pasta onde o arquivo de saída está — demais arquivos vão para lá."""
        if self.output_tiff:
            return os.path.dirname(os.path.abspath(self.output_tiff))
        return os.path.abspath(".")

    @property
    def shapes(self):
        """Dicionário de shapefiles por classe (IDs sequenciais 1..N)."""
        result = {}
        for i, item in enumerate(self.shapes_list, start=1):
            name = item.get("name", f"classe_{i}").strip() or f"classe_{i}"
            path = item.get("path", "").strip()
            if path:
                result[i] = path
        return result

    @property
    def class_names(self):
        """Nomes das classes (1..N)."""
        names = {}
        for i, item in enumerate(self.shapes_list, start=1):
            name = item.get("name", f"classe_{i}").strip() or f"classe_{i}"
            names[i] = name
        return names

    @property
    def model_path(self):
        return os.path.join(self.out_dir, "model_rf.joblib")


class ClassifierManager:
    """Pipeline de classificação: treino → predição → vetorização."""

    def __init__(self, config: ClassifierConfig, log_callback=None, progress_callback=None):
        self.config = config
        self._log = log_callback or (lambda msg: print(msg, end=""))
        self._progress = progress_callback or (lambda pct, msg: None)
        self._stop_requested = False

    # ── Helpers ────────────────────────────────────────────────────────────

    def _print(self, message: str):
        self._log(str(message) + "\n")

    def request_stop(self):
        self._stop_requested = True

    def _check_stop(self):
        if self._stop_requested:
            raise InterruptedError("Processamento interrompido pelo usuário.")

    def _tiff_info(self, path):
        with rasterio.open(path) as src:
            return src.meta.copy(), src.crs, src.transform, src.height, src.width

    def _reproject_geom(self, geom, src_crs, dst_crs):
        t = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        return shp_transform(t.transform, geom)

    def _load_geoms(self, shp_path, tiff_crs):
        geoms = []
        with fiona.open(shp_path) as f:
            shp_crs = CRS.from_user_input(f.crs)
            reproj = not shp_crs.equals(tiff_crs, ignore_axis_order=True)
            is_point = f.schema['geometry'] == 'Point'

            for feat in f:
                raw = feat.get("geometry")
                if raw is None:
                    continue
                try:
                    g = shape(raw)
                    if is_point:
                        g = g.buffer(self.config.buffer_m)
                    g = g.buffer(0)
                except Exception:
                    continue
                if g.is_empty or not g.is_valid:
                    continue
                if reproj:
                    g = self._reproject_geom(g, shp_crs, tiff_crs)
                geoms.append(g)
        return geoms

    def _remove_small_holes(self, polygon, min_area_m2):
        if not hasattr(polygon, 'interiors') or len(polygon.interiors) == 0:
            return polygon
        large_holes = []
        for hole in polygon.interiors:
            hole_polygon = Polygon(hole)
            if hole_polygon.area >= min_area_m2:
                large_holes.append(hole)
        return Polygon(polygon.exterior, large_holes)

    # ── Features espectrais ────────────────────────────────────────────────

    def _compute_features(self, rgb_tile):
        R, G, B = rgb_tile[..., 0], rgb_tile[..., 1], rgb_tile[..., 2]
        eps = 1e-6

        ExG = 2 * G - R - B
        ExR = 1.4 * R - G
        denom = G + R - B
        VARI = np.where(np.abs(denom) > 0.02, (G - R) / (denom + eps), 0.0)
        NGRDI = np.where((G + R) > 0.02, (G - R) / (G + R + eps), 0.0)
        bri = (R + G + B) / 3.0 + eps
        r_r = R / bri
        r_g = G / bri
        Cmax = np.maximum.reduce([R, G, B])
        Cmin = np.minimum.reduce([R, G, B])
        S = np.where(Cmax > 0.02, (Cmax - Cmin) / (Cmax + eps), 0.0)

        def local_std(ch):
            m1 = uniform_filter(ch.astype(np.float32), size=5)
            m2 = uniform_filter(ch.astype(np.float32) ** 2, size=5)
            return np.sqrt(np.maximum(m2 - m1 ** 2, 0.0))

        feats = np.stack([
            R, G, B,
            ExG, ExR, VARI, bri - eps, r_r, r_g,
            S, Cmax,
            local_std(R), local_std(G), local_std(B),
            NGRDI,
        ], axis=-1).astype(np.float32)

        return feats

    # ── Etapa 1: Treino ────────────────────────────────────────────────────

    def train(self, image_path):
        self._print("─" * 60)
        self._print("ETAPA 1: EXTRAÇÃO DE AMOSTRAS DE TREINO")
        self._print("─" * 60)

        meta, tiff_crs, transform, H, W = self._tiff_info(image_path)
        res_m = abs(transform.a)
        self._print(f"  TIFF : {H:,} × {W:,} px  |  resolução ≈ {res_m*100:.1f} cm/px")

        with rasterio.open(image_path) as src:
            bands_available = src.count

        if bands_available < 3:
            raise ValueError(f"Imagem deve ter pelo menos 3 bandas (RGB), tem {bands_available}")

        # Rasteriza shapefiles
        label_raster = np.zeros((H, W), dtype=np.uint8)
        shapes = self.config.shapes
        class_names = self.config.class_names

        for cid, shp_path in shapes.items():
            cname = class_names[cid]
            if not os.path.exists(shp_path):
                self._print(f"  [AVISO] {shp_path} não encontrado, pulando.")
                continue

            with fiona.open(shp_path) as f:
                geom_type = f.schema['geometry']

            geoms = self._load_geoms(shp_path, tiff_crs)
            if not geoms:
                self._print(f"  [AVISO] Nenhuma geometria válida em {cname}.")
                continue

            if geom_type == 'Point':
                self._print(f"  {cname}: geometria PONTO detectada → aplicando buffer de {self.config.buffer_m}m")

            burned = rasterize(
                [(g, cid) for g in geoms],
                out_shape=(H, W),
                transform=transform,
                fill=0,
                dtype=np.uint8,
            )
            label_raster = np.where(burned > 0, burned, label_raster)
            px = int((burned > 0).sum())
            self._print(f"  {cname}: {px:,} px de treino disponíveis  ({100*px/(H*W):.2f}% da imagem)")

        # Coleta amostras
        X_all, y_all = [], []
        tile_sz = self.config.tile_sz

        with rasterio.open(image_path) as src:
            for cid in list(shapes.keys()):
                cname = class_names[cid]
                ys_c, xs_c = np.where(label_raster == cid)
                if len(ys_c) == 0:
                    self._print(f"  [AVISO] {cname}: sem pixels após rasterização.")
                    continue

                n = min(self.config.samples_per_class, len(ys_c))
                idx = np.random.default_rng(42).choice(len(ys_c), n, replace=False)
                ys_s, xs_s = ys_c[idx], xs_c[idx]

                tile_y = ys_s // tile_sz
                tile_x = xs_s // tile_sz

                feats_list = []
                for ty in np.unique(tile_y):
                    for tx in np.unique(tile_x[(tile_y == ty)]):
                        sel = (tile_y == ty) & (tile_x == tx)
                        y0 = int(ty * tile_sz)
                        x0 = int(tx * tile_sz)
                        h = min(tile_sz, H - y0)
                        w = min(tile_sz, W - x0)
                        win = Window(x0, y0, w, h)
                        raw = src.read([1, 2, 3], window=win).astype(np.float32) / 255.0
                        rgb = np.moveaxis(raw, 0, -1)
                        ft = self._compute_features(rgb)
                        ly = np.clip(ys_s[sel] - y0, 0, h - 1)
                        lx = np.clip(xs_s[sel] - x0, 0, w - 1)
                        feats_list.append(ft[ly, lx])

                if not feats_list:
                    continue
                X_c = np.vstack(feats_list)
                y_c = np.full(len(X_c), cid, dtype=np.int32)
                X_all.append(X_c)
                y_all.append(y_c)
                self._print(f"  {cname}: {len(X_c):,} amostras extraídas ✓")

        if not X_all:
            raise RuntimeError("Nenhuma amostra de treino extraída. Verifique os shapefiles.")

        X = np.vstack(X_all)
        y = np.concatenate(y_all)
        self._print(f"\n  Total: {len(X):,} amostras | {len(np.unique(y))} classes")

        self._print("\n─" * 60)
        self._print("ETAPA 2: TREINO DO RANDOM FOREST")
        self._print("─" * 60)

        clf = RandomForestClassifier(
            n_estimators=self.config.rf_n_trees,
            max_features="sqrt",
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=self.config.rf_jobs,
            verbose=0,
            random_state=42,
        )
        self._print(f"  Treinando {self.config.rf_n_trees} árvores com {len(X):,} amostras …")
        clf.fit(X, y)

        imp = clf.feature_importances_
        feat_names = ["R", "G", "B", "ExG", "ExR", "VARI", "brilho", "R/bri", "G/bri",
                      "Sat", "Value", "stdR", "stdG", "stdB", "NGRDI"]
        top = np.argsort(imp)[::-1][:5]
        self._print("  Top-5 features:")
        for i in top:
            self._print(f"    {feat_names[i]:8s}: {imp[i]:.3f}")

        os.makedirs(self.config.out_dir, exist_ok=True)
        joblib.dump(clf, self.config.model_path)
        self._print(f"\n  Modelo salvo: {self.config.model_path}")
        return clf

    # ── Etapa 2: Predição ──────────────────────────────────────────────────

    def predict(self, clf, image_path):
        self._print("\n─" * 60)
        self._print("ETAPA 3: PREDIÇÃO EM TODA A IMAGEM")
        self._print("─" * 60)

        meta, _, transform, H, W = self._tiff_info(image_path)
        res_m = abs(transform.a)

        out_meta = {k: v for k, v in meta.items()
                    if k not in ("count", "dtype", "nodata")}
        out_meta.update({"count": 1, "dtype": "uint8", "nodata": 0})

        os.makedirs(self.config.out_dir, exist_ok=True)
        pred_path = self.config.output_tiff

        tile_sz = self.config.tile_sz
        n_ty = (H + tile_sz - 1) // tile_sz
        n_tx = (W + tile_sz - 1) // tile_sz
        total_tiles = n_ty * n_tx
        done = 0

        with rasterio.open(image_path) as src:
            has_alpha = src.count >= 4
            with rasterio.open(pred_path, "w", **out_meta) as dst:
                for ty in range(n_ty):
                    y0 = ty * tile_sz
                    th = min(tile_sz, H - y0)
                    for tx in range(n_tx):
                        self._check_stop()
                        x0 = tx * tile_sz
                        tw = min(tile_sz, W - x0)
                        win = Window(x0, y0, tw, th)

                        raw = src.read([1, 2, 3], window=win).astype(np.float32) / 255.0
                        rgb = np.moveaxis(raw, 0, -1)

                        alpha_valid = (src.read(4, window=win) > 0
                                       if has_alpha
                                       else np.ones((th, tw), dtype=bool))

                        ft = self._compute_features(rgb)
                        flat = ft.reshape(-1, ft.shape[-1])
                        mask = alpha_valid.ravel()

                        pred_flat = np.zeros(th * tw, dtype=np.uint8)
                        if mask.sum() > 0:
                            proba = clf.predict_proba(flat[mask])
                            max_prob = proba.max(axis=1)
                            pred_cls = clf.classes_[proba.argmax(axis=1)].astype(np.uint8)
                            pred_cls[max_prob < self.config.conf_threshold] = 0
                            pred_flat[mask] = pred_cls

                        dst.write(pred_flat.reshape(th, tw)[np.newaxis], window=win)

                        done += 1
                        pct = 100 * done / total_tiles
                        self._progress(pct, f"Predição: {done}/{total_tiles} tiles")
                        if done == total_tiles or done % max(1, total_tiles // 20) == 0:
                            self._print(f"  [{pct:5.1f}%] ({done}/{total_tiles} tiles)")

        self._print(f"\n  Salvo: {pred_path}")
        return pred_path

    # ── Etapa 3: Vetorização ───────────────────────────────────────────────

    def vectorize(self, pred_path):
        self._print("\n─" * 60)
        self._print("ETAPA 4: VETORIZAÇÃO + LIMPEZA MORFOLÓGICA")
        self._print("─" * 60)

        vec_dir = os.path.join(self.config.out_dir, "vetores")
        os.makedirs(vec_dir, exist_ok=True)

        with rasterio.open(pred_path) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs
            res_m = abs(transform.a)

        px_area = res_m ** 2
        min_px = max(1, int(self.config.min_area_m2 / px_area))
        min_hole_px = max(1, int(self.config.hole_area_m2 / px_area))

        self._print(f"  Resolução: {res_m*100:.1f} cm/px")
        self._print(f"  Área mínima de polígonos: {self.config.min_area_m2} m²  =  {min_px} px")
        self._print(f"  Área mínima de buracos: {self.config.hole_area_m2} m²  =  {min_hole_px} px\n")
        self._print("  Distribuição das classes preditas:")
        total_px = int((data > 0).sum())
        for cid, cname in self.config.class_names.items():
            n = int((data == cid).sum())
            pct = 100 * n / max(total_px, 1)
            self._print(f"    {cname:10s}: {n:>12,} px  ({pct:5.1f}%)")
        self._print("")

        struct = np.ones((3, 3), dtype=bool)

        schema = {
            "geometry": "Polygon",
            "properties": {"class_id": "int", "class_name": "str", "area_m2": "float"},
        }

        total_poligonos = 0
        total_area = 0
        class_names = self.config.class_names
        n_classes = len(class_names)
        class_done = 0

        for cid, cname in class_names.items():
            self._check_stop()
            mask = (data == cid).astype(np.uint8)
            if mask.sum() == 0:
                self._print(f"  {cname}: sem pixels, pulando.")
                class_done += 1
                continue

            mb = mask.astype(bool)
            mb = binary_opening(mb, structure=struct, iterations=self.config.smooth_iter)
            mb = binary_closing(mb, structure=struct, iterations=self.config.smooth_iter)
            mask = mb.astype(np.uint8)

            if mask.sum() == 0:
                self._print(f"  {cname}: sem pixels após limpeza, pulando.")
                class_done += 1
                continue

            raw_polys = []
            for geom_dict, val in rio_shapes(mask, mask=mask, transform=transform):
                if int(val) != 1:
                    continue
                p = shape(geom_dict)
                raw_polys.append(p)

            if not raw_polys:
                self._print(f"  {cname}: sem polígonos, pulando.")
                class_done += 1
                continue

            polys = []
            holes_removed = 0
            small_removed = 0

            for p in raw_polys:
                if not p.is_valid:
                    p = p.buffer(0)

                if p.area < self.config.min_area_m2:
                    small_removed += 1
                    continue

                if p.interiors:
                    p_cleaned = self._remove_small_holes(p, self.config.hole_area_m2)
                    holes_removed += len(p.interiors) - len(p_cleaned.interiors) if hasattr(p_cleaned, 'interiors') else 0
                    polys.append(p_cleaned)
                else:
                    polys.append(p)

            if not polys:
                self._print(f"  {cname}: sem polígonos acima de {self.config.min_area_m2} m², pulando.")
                class_done += 1
                continue

            tol = res_m * 0.5
            feats = []
            for p in polys:
                ps = p.simplify(tol, preserve_topology=True)
                if ps.is_empty:
                    continue
                if not ps.is_valid:
                    ps = ps.buffer(0)
                feats.append({
                    "geometry": mapping(ps),
                    "properties": {
                        "class_id": cid,
                        "class_name": cname,
                        "area_m2": round(p.area, 3),
                    },
                })

            if not feats:
                class_done += 1
                continue

            geojson_path = os.path.join(vec_dir, f"classe_{cname}.geojson")
            with open(geojson_path, "w", encoding="utf-8") as fp:
                json.dump({
                    "type": "FeatureCollection",
                    "crs": {"type": "name",
                            "properties": {"name": crs.to_string()}},
                    "features": [{"type": "Feature", **f} for f in feats],
                }, fp, ensure_ascii=False)

            shp_path = os.path.join(vec_dir, f"classe_{cname}.shp")
            with fiona.open(shp_path, "w",
                            driver="ESRI Shapefile",
                            crs=crs.to_wkt(),
                            schema=schema) as dst:
                dst.writerecords(feats)

            class_ha = sum(f["properties"]["area_m2"] for f in feats) / 10_000
            total_poligonos += len(feats)
            total_area += class_ha

            self._print(f"  {cname}: {len(feats):>6,} polígonos  |  {class_ha:.2f} ha")
            if small_removed > 0:
                self._print(f"    → {small_removed} polígonos pequenos (< {self.config.min_area_m2} m²) removidos")
            if holes_removed > 0:
                self._print(f"    → {holes_removed} buracos pequenos (< {self.config.hole_area_m2} m²) preenchidos")
            self._print(f"    → {shp_path}")

            class_done += 1
            pct = 100 * class_done / n_classes
            self._progress(pct, f"Vetorização: {cname} concluída")

        self._print(f"\n  RESUMO DA VETORIZAÇÃO:")
        self._print(f"  • Total de polígonos: {total_poligonos:,}")
        self._print(f"  • Área total mapeada: {total_area:.2f} ha")
        self._print(f"\n  Todos os vetores em: {vec_dir}")

    # ── Pipeline completo ──────────────────────────────────────────────────

    def run(self):
        """Executa o pipeline completo: treino → predição → vetorização."""
        self._print("=" * 60)
        self._print("  CLASSIFICADOR DE USO DO SOLO — Random Forest")
        self._print("=" * 60)

        image_path = self.config.input_tiff
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"TIFF de entrada não encontrado: {image_path}")
        self._print(f"  Imagem : {image_path}")
        self._print(f"  Saída  : {self.config.output_tiff}")
        self._print(f"  Pasta  : {self.config.out_dir}")

        force_retrain = self.config.force_retrain or not os.path.exists(self.config.model_path)
        self._print(f"  Modelo : {'[novo treino]' if force_retrain else '[reutilizando modelo salvo]'}")
        self._print("=" * 60)

        if force_retrain:
            clf = self.train(image_path)
        else:
            self._print(f"\n  Carregando modelo: {self.config.model_path}")
            self._print("  (marque 'Forçar re-treino' para treinar novamente)")
            clf = joblib.load(self.config.model_path)

        pred_path = self.predict(clf, image_path)
        self.vectorize(pred_path)

        self._print("\n" + "=" * 3)
        self._print("  PIPELINE CONCLUÍDO")
        self._print("=" * 60)
        self._print(f"  TIFF classificado  →  {self.config.output_tiff}")
        self._print(f"  Vetores por classe →  {os.path.join(self.config.out_dir, 'vetores')}/classe_<nome>.shp")
        self._print(f"                        {os.path.join(self.config.out_dir, 'vetores')}/classe_<nome>.geojson")
        self._print("")
        self._print("  Configurações aplicadas:")
        self._print(f"  • Polígonos menores que {self.config.min_area_m2} m² → REMOVIDOS")
        self._print(f"  • Buracos internos menores que {self.config.hole_area_m2} m² → PREENCHIDOS")
        self._print("=" * 60)