import rasterio
import numpy as np
from skimage import measure
from shapely.geometry import Polygon
from shapely.geometry import mapping
from shapely.ops import unary_union
import geopandas as gpd

# abrir raster classificado
with rasterio.open("raster2.tif") as src:
    img = src.read(1)
    transform = src.transform
    crs = src.crs

poligonos = []

classes = np.unique(img)

for classe in classes:
    if classe == 0:
        continue

    mask = img == classe

    contours = measure.find_contours(mask.astype(float), 0.5)

    for contour in contours:
        coords = []

        for y, x in contour:
            px, py = rasterio.transform.xy(transform, y, x)
            coords.append((px, py))

        if len(coords) > 3:
            poly = Polygon(coords)

            if poly.is_valid:
                poly = poly.simplify(1.0)
                poligonos.append({
                    "geometry": poly,
                    "classe": int(classe)
                })

gdf = gpd.GeoDataFrame(poligonos, crs=crs)
gdf.to_file("saida.gpkg", driver="GPKG")