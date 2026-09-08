"""Raster-to-vector conversion of probability maps with per-polygon statistics."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

_SHAPEFILE_SIDE_CARS = ('.shp', '.shx', '.dbf', '.prj', '.cpg')


def _remove_vector(path: Path) -> None:
    if path.suffix.lower() == '.shp':
        for ext in _SHAPEFILE_SIDE_CARS:
            path.with_suffix(ext).unlink(missing_ok=True)
    else:
        path.unlink(missing_ok=True)


def _binarize(prob_path: Path, tmp_bin: Path, thresh: float, scale: float) -> None:
    import rasterio
    from tqdm import tqdm
    with rasterio.open(prob_path) as src:
        meta = src.meta.copy()
        meta.update(dtype='uint8', count=1, nodata=None, compress='deflate')
        with rasterio.open(tmp_bin, 'w', **meta) as dst:
            for _, win in tqdm(list(src.block_windows(1)), desc='   binarize', leave=False):
                p = src.read(1, window=win)
                dst.write((p >= thresh * scale).astype(np.uint8), 1, window=win)


def _polygonize_gdal(tmp_bin: Path, tmp_vec: Path, connectivity: int) -> bool:
    try:
        from osgeo import gdal, ogr, osr
    except ImportError:
        return False
    gdal.UseExceptions()
    src_ds = gdal.Open(str(tmp_bin), gdal.GA_ReadOnly)
    band = src_ds.GetRasterBand(1)
    drv = ogr.GetDriverByName('GPKG')
    if tmp_vec.exists():
        drv.DeleteDataSource(str(tmp_vec))
    dst_ds = drv.CreateDataSource(str(tmp_vec))
    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_ds.GetProjection())
    layer = dst_ds.CreateLayer('polygons', srs=srs, geom_type=ogr.wkbMultiPolygon)
    layer.CreateField(ogr.FieldDefn('value', ogr.OFTInteger))
    opts = ['8CONNECTED=8'] if connectivity == 8 else []
    # the band itself is used as mask -> only non-zero pixels are polygonised
    gdal.Polygonize(band, band, layer, 0, opts, callback=None)
    layer = None; dst_ds = None; band = None; src_ds = None  # noqa: E702 (flush)
    return True


def _polygonize_rasterio(tmp_bin: Path, connectivity: int, max_pixels: int):
    """In-memory fallback (no GDAL Python bindings). Requires the whole mask."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape
    with rasterio.open(tmp_bin) as src:
        if src.width * src.height > max_pixels:
            raise RuntimeError(
                'GDAL Python bindings (osgeo) are not installed and the mask '
                f'({src.width} x {src.height} px) exceeds the in-memory limit of '
                f'{max_pixels} px. Install GDAL (conda-forge) or raise --max-inmem-pixels.')
        mask = src.read(1)
        geoms = [shape(g) for g, v in shapes(mask, mask=mask == 1, connectivity=connectivity,
                                             transform=src.transform)]
        return gpd.GeoDataFrame(geometry=geoms, crs=src.crs)


def _mean_prob(prob_ds, geom, scale: float) -> float:
    from rasterio.features import geometry_mask, geometry_window
    from rasterio.windows import transform as win_transform
    try:
        win = geometry_window(prob_ds, [geom], pad_x=2, pad_y=2, north_up=True, pixel_precision=3)
        block = prob_ds.read(1, window=win, boundless=True, fill_value=0)
        m = geometry_mask([geom], out_shape=(win.height, win.width),
                          transform=win_transform(win, prob_ds.transform), invert=True)
    except Exception:  # degenerate geometry -> full read (rare)
        block = prob_ds.read(1)
        m = geometry_mask([geom], out_shape=(prob_ds.height, prob_ds.width),
                          transform=prob_ds.transform, invert=True)
    vals = block[m]
    return float(vals.mean() / scale) if vals.size else 0.0


def polygonize_probability(prob_path: str,
                           out_path: str,
                           thresh: float = 0.35,
                           min_area: float = 0.0,
                           connectivity: int = 4,
                           compute_mean_prob: bool = True,
                           prob_scale: Optional[float] = None,
                           max_inmem_pixels: int = 1_500_000_000) -> int:
    """Threshold a probability raster and export crown polygons.

    Args:
        prob_path: Single-band probability raster (float in [0,1] or uint8 in
            [0,255]).
        out_path: ``.gpkg`` or ``.shp`` destination.
        thresh: Probability threshold in [0,1].
        min_area: Minimum polygon area in CRS units (m^2 for projected CRS).
        connectivity: 4 or 8 for pixel connectivity.
        compute_mean_prob: Attach the mean probability of each polygon.
        prob_scale: 1.0 for float rasters, 255.0 for uint8 (auto if None).
    Returns:
        Number of polygons written.
    """
    import geopandas as gpd
    import rasterio
    from tqdm import tqdm

    prob_path, out_path = Path(prob_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(prob_path) as src:
        if prob_scale is None:
            prob_scale = 255.0 if np.dtype(src.dtypes[0]) == np.uint8 else 1.0
        crs = src.crs

    tmp_bin = out_path.with_name(out_path.stem + '_bin.tmp.tif')
    tmp_vec = out_path.with_name(out_path.stem + '_raw.tmp.gpkg')
    print(f'-> Vectorising @ threshold={thresh:.3f} (connectivity={connectivity}) ...')
    _binarize(prob_path, tmp_bin, thresh, prob_scale)

    if _polygonize_gdal(tmp_bin, tmp_vec, connectivity):
        gdf = gpd.read_file(tmp_vec)
        if 'value' in gdf.columns:
            gdf = gdf[gdf['value'] == 1].drop(columns=['value'])
        if gdf.crs is None and crs is not None:
            gdf = gdf.set_crs(crs)
    else:
        print('   (osgeo not available: using in-memory rasterio.features.shapes)')
        gdf = _polygonize_rasterio(tmp_bin, connectivity, max_inmem_pixels)

    n_raw = len(gdf)
    if n_raw:
        gdf = gdf.reset_index(drop=True)
        gdf['area'] = gdf.geometry.area.astype('float64').round(3)
        if min_area > 0:
            gdf = gdf[gdf['area'] >= min_area].reset_index(drop=True)
        if compute_mean_prob and len(gdf):
            with rasterio.open(prob_path) as pds:
                means = [_mean_prob(pds, g, prob_scale)
                         for g in tqdm(gdf.geometry, desc='   mean_prob', leave=False)]
            gdf['mean_prob'] = np.round(np.asarray(means, dtype=np.float32), 4)

    n_out = len(gdf)
    if n_out:
        driver = 'ESRI Shapefile' if out_path.suffix.lower() == '.shp' else 'GPKG'
        _remove_vector(out_path)
        gdf.to_file(out_path, driver=driver)
        print(f'   -> {out_path}: {n_out} polygons (raw {n_raw}, min_area={min_area})')
    else:
        print('   -> no polygons at this threshold; no vector file written')

    tmp_bin.unlink(missing_ok=True)
    tmp_vec.unlink(missing_ok=True)
    return n_out
