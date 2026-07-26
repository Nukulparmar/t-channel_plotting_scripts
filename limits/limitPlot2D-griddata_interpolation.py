import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import mplhep as hep
import pandas as pd
import os
import re
import scipy.interpolate

# import numpy as np
# import scipy.interpolate
import scipy.ndimage
plt.rcParams["mathtext.fontset"] = "cm"
# plt.rcParams['text.usetex'] = True
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })
def plot_limit_contour_smooth(
    ax,
    interp,          # Your StableGridInterpolator from previous step
    x_range,
    y_range,
    n_entries=200,   # Increased resolution for smoother mesh lines
    sigma=2,       # Controls smoothness (1.0 - 2.0 is ideal for CMS plots)
    color="red",
    valid_mask_fn=None,
    linewidth=3,
    level=1,
    linestyle="-",
    **kwargs,
):
    # 1. Generate a high-resolution mesh grid
    x_grid = np.linspace(x_range[0], x_range[1], n_entries)
    y_grid = np.linspace(y_range[0], y_range[1], n_entries)
    xi, yi = np.meshgrid(x_grid, y_grid)
    
    # 2. Get the stable, linear griddata interpolation matrix
    zi = interp(xi, yi)
    
    # 3. Create a mask of exactly where valid data or physics constraints exist
    # This ensures the smoothing filter doesn't bleed into empty spaces
    nan_mask = np.isnan(zi)
    if valid_mask_fn is not None:
        nan_mask |= ~valid_mask_fn(xi, yi)
    
    # 4. Handle NaNs before smoothing (Gaussian filters choke on NaNs)
    # Temporarily fill NaNs using the nearest valid data point
    zi_filled = zi.copy()
    if np.any(nan_mask):
        # Find coordinates of valid entries
        valid_coords = np.argwhere(~nan_mask)
        valid_values = zi[~nan_mask]
        # Find coordinates of all grid pixels
        all_coords = np.argwhere(np.ones_like(zi))
        # Map nearest neighbors to fill the empty space smoothly
        nearest_interp = scipy.interpolate.NearestNDInterpolator(valid_coords, valid_values)
        zi_filled = nearest_interp(all_coords).reshape(zi.shape)

    # 5. Apply a 2D Gaussian Filter to smooth the contours
    zi_smoothed = scipy.ndimage.gaussian_filter(zi_filled, sigma=sigma)
    
    # 6. Re-apply the strict boundary mask so nothing leaks past the kinematic cutoff
    zi_smoothed[nan_mask] = np.nan
    
    # 7. Plot the beautiful, smooth contour lines
    ax.contour(
        xi, yi, zi_smoothed, 
        levels=np.array([level]), 
        colors=color, 
        linewidths=linewidth, 
        linestyles=linestyle, 
        **kwargs
    )
    
    return ax.plot([], [], color=color, linewidth=linewidth, linestyle=linestyle, **kwargs)

def make_stable_interpolator(
    table,
    poi_x,
    poi_y,
    quantile=None,
    observed=None,
    xsec=False,
    **kwargs  # Absorbs unused triangulation arguments cleanly
):
    assert not (quantile is None and observed is None)
    assert not (quantile is not None and observed is not None)

    # 1. Extract and filter data exactly like your original script
    x = np.array(table[poi_x], dtype=float)
    y = np.array(table[poi_y], dtype=float)
    z = np.array(table["limit"], dtype=float)
    if xsec:
        z = z * np.array(table["xsec"], dtype=float)
    if observed is True:
        quantile = -1

    mask = table["quantileExpected"] == quantile
    x, y, z = x[mask], y[mask], z[mask]
    
    points = np.column_stack((x, y))

    # 2. Build a localized Callable Wrapper class
    class StableGridInterpolator:
        def __init__(self, points, z):
            self.points = points
            self.z = z
            
        def __call__(self, xi, yi):
            # Formats mesh arrays to ensure compatibility with scipy
            xi = np.array(xi, dtype=float)
            yi = np.array(yi, dtype=float)
            
            # Localized linear interpolation (Never overshoots or creates random lines)
            zi_linear = scipy.interpolate.griddata(
                self.points, self.z, (xi, yi), method='linear'
            )
            
            # Optional: Enforce your physical mMed-mDark constraints on the mesh grid itself
            if poi_x == "mMed" and poi_y == "mDark":
                valid = mmed_mdark_constraint(xi, yi)
                zi_linear[~valid] = np.nan
            elif poi_x == "mDark" and poi_y == "mMed":
                valid = mmed_mdark_constraint(yi, xi)
                zi_linear[~valid] = np.nan
                
            return zi_linear

    return StableGridInterpolator(points, z)


matplotlib.style.use(hep.style.CMS)

def latex_readable_varibles(name):
    """
    Converts variable names to human-readable format.

    Args:
        name (str): Variable name (e.g., 'mZprime', 'mDark', 'rinv', 'alpha').

    Returns:
        str: Human-readable variable name.
    """
    mapping = {
        "mZprime": r"$M_{Z'}$ ",
        "mDark": r"$m_{dark}$ ",
        "rinv": r"$r_{inv}$",
        "alpha": r"$\alpha_{D}$",
        "yukawa": r"$\lambda$",
        "mMed": r"$m_{\Phi}$ [GeV]",
    }
    return mapping.get(name, name)

POI_CONFIG = {
    "mMed": {
        "regex": r"mMed[_\-]?([0-9]+(?:p[0-9]+)?)",
        "label": r"$m_{\Phi}$ [GeV]",
        "range": (500, 4000),
    },
    "rinv": {
        "regex": r"rinv[_\-]?([0-9]+(?:p[0-9]+)?)",
        "label": r"$r_{inv}$",
        "range": (0, 1),
    },
    "mDark": {
        "regex": r"mDark[_\-]?([0-9]+(?:p[0-9]+)?)",
        "label": r"$m_{dark}$ [GeV]",
        "range": (1, 100),
    },
    "yukawa": {
        "regex": r"yukawa[_\-]?([0-9]+(?:p[0-9]+)?)",
        "label": r"$\lambda$",
        "range": (0.5, 3.5),
    },
}

LIMIT_KEY_MAP = {
    "y_down_points2": "expected_m2sigma",
    "y_down_points1": "expected_m1sigma",
    "y_vals": "expected",
    "y_up_points1": "expected_p1sigma",
    "y_up_points2": "expected_p2sigma",
    "y_observed": "obs_lim",
}


def parse_limits_txt(txt_path):
    with open(txt_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    data = {}
    for ln in lines:
        parts = ln.split()
        key = parts[0]
        vals = []
        for v in parts[1:]:
            try:
                vals.append(float(v))
            except ValueError:
                vals.append(np.nan)
        data[key] = vals
    return data


def infer_param_from_filename(fname, poi_name):
    """Extract a parameter value token (e.g. '0p1') from a filename."""
    base = os.path.basename(fname)
    cfg = POI_CONFIG.get(poi_name)
    if cfg:
        m = re.search(cfg["regex"], base, flags=re.IGNORECASE)
        if m:
            return m.group(1).replace(".", "p")
    generic = re.search(rf"{poi_name}[_\-]?([0-9]+(?:p[0-9]+)?)", base, flags=re.IGNORECASE)
    if generic:
        return generic.group(1).replace(".", "p")
    return "0p0"


def token_to_float(token):
    return float(str(token).replace("p", "."))


def add_cross_sections(df, xsec_csv):
    xsec_df = pd.read_csv(xsec_csv)
    if "cross_section" not in xsec_df.columns:
        raise ValueError(f"Cross-section CSV {xsec_csv} must contain a 'cross_section' column")

    match_cols = [c for c in ["mMed", "mDark", "rinv", "yukawa"] if c in df.columns and c in xsec_df.columns]
    if not match_cols:
        raise ValueError(f"Cross-section CSV {xsec_csv} has no parameter columns matching the limits CSV")

    df = df.copy()
    xsec_df = xsec_df.copy()
    key_cols = []
    for col in match_cols:
        key_col = f"__xsec_key_{col}"
        df[key_col] = df[col].apply(lambda v: token_to_float(v) if isinstance(v, str) else float(v)).round(10)
        xsec_df[key_col] = xsec_df[col].apply(lambda v: token_to_float(v) if isinstance(v, str) else float(v)).round(10)
        key_cols.append(key_col)

    if xsec_df.duplicated(key_cols).any():
        dupes = xsec_df[xsec_df.duplicated(key_cols, keep=False)][match_cols]
        raise ValueError(
            f"Cross-section CSV {xsec_csv} contains duplicate entries for matched parameters:\n"
            + dupes.drop_duplicates().to_string(index=False)
        )

    merged = df.merge(
        xsec_df[key_cols + ["cross_section"]],
        on=key_cols,
        how="left",
        validate="many_to_one",
    )
    missing = merged["cross_section"].isna()
    if missing.any():
        missing_points = merged.loc[missing, match_cols].drop_duplicates()
        raise ValueError(
            f"Cross-section CSV {xsec_csv} is missing {len(missing_points)} matched scan point(s):\n"
            + missing_points.to_string(index=False)
        )

    merged["xsec"] = merged["cross_section"].astype(float)
    return merged.drop(columns=key_cols + ["cross_section"])


def construct_limits_table(csv_file, poi_x, poi_y, fixed_params=None, xsec_csv=None):
    df = pd.read_csv(csv_file)

    # convert scan axes
    for col in [poi_x, poi_y]:
        df[col] = df[col].apply(lambda v: token_to_float(v) if isinstance(v, str) else float(v))

    if fixed_params is None:
        fixed_params = {}

    # convert and filter all non-scanned fixed parameters
    for pname, pval in fixed_params.items():
        if pname == "xsec":
            continue
        if pname not in df.columns:
            continue
        df[pname] = df[pname].apply(lambda v: token_to_float(v) if isinstance(v, str) else float(v))
        df = df[np.isclose(df[pname], float(pval))]

    if xsec_csv is not None:
        df = add_cross_sections(df, xsec_csv)
    else:
        df["xsec"] = fixed_params.get("xsec", 1.0)

    rows = []
    for _, row in df.iterrows():
        x_val = row[poi_x]
        y_val = row[poi_y]

        quantiles = [
            (0.025, row["expected_m2sigma"]),
            (0.16, row["expected_m1sigma"]),
            (0.5, row["expected"]),
            (0.84, row["expected_p1sigma"]),
            (0.975, row["expected_p2sigma"]),
            (-1, row["obs_lim"]),
        ]

        for quantile, limit in quantiles:
            entry = {
                poi_x: x_val,
                poi_y: y_val,
                "quantileExpected": quantile,
                "limit": limit,
                "xsec": row["xsec"],
            }
            rows.append(entry)

    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


## Custon triangulation for rectangular grids
def make_scan_triangulation(x, y):
    """
    Deterministic triangulation for adaptive mMed-mDark scans.
    Connects neighbouring mMed columns without Delaunay.
    """

    points = list(zip(map(float, x), map(float, y)))

    columns = {}
    for idx, (px, py) in enumerate(points):
        columns.setdefault(px, []).append((py, idx))

    x_values = sorted(columns.keys())

    for px in x_values:
        columns[px] = sorted(columns[px])


    triangles = []

    for xl, xr in zip(x_values[:-1], x_values[1:]):

        left = columns[xl]
        right = columns[xr]

        # make all possible vertical strips
        for yl, il in left:
            for yr, ir in right:

                # connect points with similar y values
                if abs(yl - yr) == 0:

                    # find next points above
                    left_up = [
                        p for p in left
                        if p[0] > yl
                    ]

                    right_up = [
                        p for p in right
                        if p[0] > yr
                    ]

                    if left_up and right_up:

                        yl2, il2 = left_up[0]
                        yr2, ir2 = right_up[0]

                        triangles.append(
                            [il, ir, ir2]
                        )
                        triangles.append(
                            [il, ir2, il2]
                        )


    triangles = np.asarray(triangles, dtype=int)

    # remove duplicates
    triangles = np.unique(
        np.sort(triangles, axis=1),
        axis=0
    )

    # remove degenerate
    triangles = triangles[
        (triangles[:,0] != triangles[:,1]) &
        (triangles[:,1] != triangles[:,2])
    ]

    if len(triangles) == 0:
        raise RuntimeError("No valid triangles created")

    return matplotlib.tri.Triangulation(
        np.asarray(x),
        np.asarray(y),
        triangles
    )

def make_rectangular_triangulation(x, y, diagonal="bl-tr"):
    """Build a deterministic triangulation by splitting complete rectangular cells."""
    x_unique = np.array(sorted({float(v) for v in x}))
    y_unique = np.array(sorted({float(v) for v in y}))
    point_index = {(float(px), float(py)): i for i, (px, py) in enumerate(zip(x, y))}

    triangles = []
    for x0, x1 in zip(x_unique[:-1], x_unique[1:]):
        for y0, y1 in zip(y_unique[:-1], y_unique[1:]):
            corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
            if not all(corner in point_index for corner in corners):
                continue

            bl = point_index[(x0, y0)]
            br = point_index[(x1, y0)]
            tl = point_index[(x0, y1)]
            tr = point_index[(x1, y1)]

            if diagonal == "bl-tr":
                triangles.extend([[bl, br, tr], [bl, tr, tl]])
            elif diagonal == "tl-br":
                triangles.extend([[bl, br, tl], [br, tr, tl]])
            else:
                raise ValueError(f"Unknown rectangular diagonal: {diagonal}")

    if not triangles:
        raise ValueError("Rectangular triangulation found no complete cells in the selected scan slice")

    return matplotlib.tri.Triangulation(x, y, np.array(triangles, dtype=int))


def make_interpolator(
    table,
    poi_x,
    poi_y,
    quantile=None,
    observed=None,
    xsec=False,
    triangulation="delaunay",
    rectangular_diagonal="bl-tr",
):
    assert not (quantile is None and observed is None)
    assert not (quantile is not None and observed is not None)

    x = table[poi_x]
    y = table[poi_y]
    z = table["limit"]
    if xsec:
        z = z * table["xsec"]
    if observed is True:
        quantile = -1

    mask = table["quantileExpected"] == quantile
    x, y, z = x[mask], y[mask], z[mask]
    if triangulation == "delaunay":
        tri = matplotlib.tri.Triangulation(x, y)
    elif triangulation == "rectangular":
        tri = make_rectangular_triangulation(x, y, diagonal=rectangular_diagonal)
    elif triangulation == "scan":
        tri = make_scan_triangulation(x, y)
    else:
        raise ValueError(f"Unknown triangulation mode: {triangulation}")
    # mask unphysical regions for mMed-mDark scans
    # Mask triangles outside the physical mMed-mDark region
    if poi_x == "mMed" and poi_y == "mDark":
        triangle_x = x[tri.triangles]
        triangle_y = y[tri.triangles]
        valid = mmed_mdark_constraint(
            triangle_x,
            triangle_y
        )
        # Keep only triangles where all three vertices are physical
        tri.set_mask(~np.all(valid, axis=1))
    elif poi_x == "mDark" and poi_y == "mMed":
        triangle_x = x[tri.triangles]
        triangle_y = y[tri.triangles]
        valid = mmed_mdark_constraint(
            triangle_y,
            triangle_x
        )

        tri.set_mask(~np.all(valid, axis=1))
    return matplotlib.tri.LinearTriInterpolator(tri, z)


def _nearest_axis_neighbors(sorted_vals, val):
    idx = np.where(np.isclose(sorted_vals, val))[0]
    if len(idx) == 0:
        return None, None
    i = idx[0]
    left = sorted_vals[i - 1] if i - 1 >= 0 else None
    right = sorted_vals[i + 1] if i + 1 < len(sorted_vals) else None
    return left, right


def _value_sign(x, level):
    if np.isclose(x, level):
        return 0
    return 1 if x > level else -1


def find_suspicious_observed_points(table, poi_x, poi_y, contour_level=1.0, neighbor_ratio=2.5):
    """Return observed points that are locally inconsistent with their 4-neighborhood."""
    obs_mask = table["quantileExpected"] == -1
    x_vals = table[poi_x][obs_mask]
    y_vals = table[poi_y][obs_mask]
    z_vals = table["limit"][obs_mask]

    grid = {}
    for x, y, z in zip(x_vals, y_vals, z_vals):
        grid[(float(x), float(y))] = float(z)

    ux = np.array(sorted({float(x) for x in x_vals}))
    uy = np.array(sorted({float(y) for y in y_vals}))

    flagged = []
    for x, y, z in zip(x_vals, y_vals, z_vals):
        x = float(x)
        y = float(y)
        z = float(z)

        x_lo, x_hi = _nearest_axis_neighbors(ux, x)
        y_lo, y_hi = _nearest_axis_neighbors(uy, y)
        if x_lo is None or x_hi is None or y_lo is None or y_hi is None:
            continue

        n_left = grid.get((float(x_lo), y))
        n_right = grid.get((float(x_hi), y))
        n_down = grid.get((x, float(y_lo)))
        n_up = grid.get((x, float(y_hi)))
        neighbors = [n_left, n_right, n_down, n_up]
        if any(v is None for v in neighbors):
            continue

        neigh_med = float(np.median(neighbors))
        ratio = max(z / neigh_med, neigh_med / z) if z > 0 and neigh_med > 0 else np.nan

        s0 = _value_sign(z, contour_level)
        sx_lo = _value_sign(n_left, contour_level)
        sx_hi = _value_sign(n_right, contour_level)
        sy_lo = _value_sign(n_down, contour_level)
        sy_hi = _value_sign(n_up, contour_level)

        horizontal_flip = s0 != 0 and sx_lo != 0 and sx_hi != 0 and s0 != sx_lo and s0 != sx_hi
        vertical_flip = s0 != 0 and sy_lo != 0 and sy_hi != 0 and s0 != sy_lo and s0 != sy_hi

        reasons = []
        if np.isfinite(ratio) and ratio >= float(neighbor_ratio):
            reasons.append(f"neighbor-ratio={ratio:.2f}")
        if horizontal_flip:
            reasons.append("horizontal-sign-flip")
        if vertical_flip:
            reasons.append("vertical-sign-flip")

        if reasons:
            direction = "up" if z > neigh_med else "down"
            flagged.append(
                {
                    poi_x: x,
                    poi_y: y,
                    "obs_limit": z,
                    "neighbor_median": neigh_med,
                    "direction": direction,
                    "delta": z - neigh_med,
                    "reasons": ",".join(reasons),
                }
            )

    flagged.sort(key=lambda r: (r[poi_y], r[poi_x]))
    return flagged


def mmed_mdark_constraint(mmed, mdark):
    """Returns True where (mMed, mDark) satisfies the physical constraint."""
    max_mdark = np.maximum(20, (mmed / 96.0) ** 1.25) # should apply ceil function here ?
    max_mdark = np.maximum(20, np.ceil((mmed / 96.0) ** 1.25 /10 ) * 10    )
    return mdark <= max_mdark


def _make_mesh(x_range, y_range, n_entries):
    return np.meshgrid(
        np.linspace(x_range[0], x_range[1], n_entries),
        np.linspace(y_range[0], y_range[1], n_entries),
    )


def get_log_color_range(z, colorbar_range=None):
    if colorbar_range is not None:
        vmin, vmax = colorbar_range
        if vmin <= 0 or vmax <= 0 or vmin >= vmax:
            raise ValueError("--colorbar-range requires two positive values with MIN < MAX")
        return vmin, vmax

    z_masked = np.ma.masked_invalid(z)
    positive_values = z_masked.compressed()
    positive_values = positive_values[positive_values > 0]
    if len(positive_values) == 0:
        raise ValueError("Cannot make a log color scale: no positive finite shaded values")

    vmin = float(np.min(positive_values))
    vmax = float(np.max(positive_values))
    if np.isclose(vmin, vmax):
        scale = np.sqrt(10.0)
        vmin /= scale
        vmax *= scale
    return vmin, vmax


def plot_2d_color(ax, interp, x_range, y_range, n_entries=200, valid_mask_fn=None, colorbar_range=None):
    x, y = _make_mesh(x_range, y_range, n_entries)
    z = interp(x, y)
    if valid_mask_fn is not None:
        z = np.where(valid_mask_fn(x, y), z, np.nan)
    vmin, vmax = get_log_color_range(z, colorbar_range=colorbar_range)
    im = ax.pcolormesh(
        x, y, z,
        norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
        linewidth=0.0,
        edgecolors="None",
    )
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(r"95% CL upper limit on $\sigma_{\Phi}/\sigma_{\Phi_{theory}}$  [pb]", va="top")
    return ax, z


def plot_limit_contour(
    ax,
    interp,
    x_range,
    y_range,
    n_entries=200,
    color="red",
    valid_mask_fn=None,
    linewidth=3,
    level=1,
    linestyle="-",
    **kwargs,
):
    x, y = _make_mesh(x_range, y_range, n_entries)
    z = interp(x, y)
    if valid_mask_fn is not None:
        z = np.where(valid_mask_fn(x, y), z, np.nan)
    ax.contour(x, y, z, levels=np.array([level]), colors=color, linewidths=linewidth, linestyles=linestyle, **kwargs)
    return ax.plot([], [], color=color, linewidth=linewidth, linestyle=linestyle, **kwargs)


def plot_limit_band(ax, interp1, interp2, x_range, y_range, n_entries=200, color="red", valid_mask_fn=None, **kwargs):
    x, y = _make_mesh(x_range, y_range, n_entries)
    z1 = interp1(x, y)
    z2 = interp2(x, y)
    if valid_mask_fn is not None:
        invalid = ~valid_mask_fn(x, y)
        z1 = np.where(invalid, np.nan, z1)
        z2 = np.where(invalid, np.nan, z2)
    z = (z1 < 1) ^ (z2 < 1)
    ax.contourf(
        x, y, z,
        levels=np.array([0.5, 2]),
        cmap=matplotlib.colors.ListedColormap([color]),
        alpha=0.2,
    )
    return ax.fill(np.nan, np.nan, color=color, alpha=0.2)


def get_poi_label(poi_name):
    return latex_readable_varibles(poi_name)


def get_poi_range(poi_name):
    cfg = POI_CONFIG.get(poi_name)
    if cfg:
        r = list(cfg["range"])
        if poi_name == "mMed":
            r[0] = r[0] + 1  # avoid axis tick clashing
        return tuple(r)
    return (0, 1)

def get_contour_max(ax):
    """
    Extract the maximum x-value from the last contour-like artist on the axes.
    Supports ContourSet and PathCollection.
    Returns (max_x, (x, y)).
    """
    cs = ax.collections[-1]  # last drawn contour/collection

    vertices = []

    # Case 1: ContourSet (has .collections)
    if hasattr(cs, "collections"):
        for coll in cs.collections:
            for path in coll.get_paths():
                vertices.extend(path.vertices)

    # Case 2: PathCollection (scatter-like or contour fallback)
    elif hasattr(cs, "get_paths"):
        for path in cs.get_paths():
            vertices.extend(path.vertices)

    else:
        raise TypeError(f"Unsupported contour type: {type(cs)}")

    vertices = np.asarray(vertices)
    xs = vertices[:, 0]
    ys = vertices[:, 1]

    idx = xs.argmax()
    idy = ys.argmax()
    max_x = xs[idx]
    max_y = ys[idy]

    return max_x, (xs[idx], ys[idx]), max_y, (xs[idy], ys[idy])


def make_2d_limit_plot(
    folder_path,
    poi_x="mMed",
    poi_y="rinv",
    fixed_params=None,
    observed=True,
    label="Preliminary",
    lumi_text=r"138 $fb^{-1}$ (13 TeV)",
    output_csv=None,
    output_pdf=None,
    flag_outliers=False,
    outlier_ratio=2.5,
    contour_level=1.0,
    annotate_outliers=False,
    overlay_scan_grid=False,
    xsec_csv=None,
    colorbar_range=None,
    triangulation="delaunay",
    rectangular_diagonal="bl-tr",
    output_suffix = "",
    tagger = "wnae",
    smooth_contour = False,
):
    if fixed_params is None:
        fixed_params = {}
    if output_csv is None:
        output_csv = f"limits_2D_{poi_x}_{poi_y}_{output_suffix}.csv"
    if output_pdf is None:
        output_pdf = f"limits2d_{label}_{poi_x}_{poi_y}_{output_suffix}.pdf"

    # build_csv_from_txt_path(folder_path, poi_x, poi_y, output_csv)
    
    table = construct_limits_table(output_csv, poi_x, poi_y, fixed_params=fixed_params, xsec_csv=xsec_csv)

    x_range = get_poi_range(poi_x)
    y_range = get_poi_range(poi_y)

    valid_mask_fn = None
    if poi_x == "mMed" and poi_y == "mDark":
        valid_mask_fn = mmed_mdark_constraint
    elif poi_x == "mDark" and poi_y == "mMed":
        valid_mask_fn = lambda x, y: mmed_mdark_constraint(y, x)

    fig = plt.figure(constrained_layout=True, figsize=(11, 11))
    ax = fig.add_subplot(111)

    legend_entries = {}

    exp_central_fb = make_stable_interpolator(
        table,
        poi_x,
        poi_y,
        quantile=0.5,
        xsec=True,
        triangulation=triangulation,
        rectangular_diagonal=rectangular_diagonal,
    )
    ax, z = plot_2d_color(
        ax,
        exp_central_fb,
        x_range,
        y_range,
        valid_mask_fn=valid_mask_fn,
        n_entries=200,
        colorbar_range=colorbar_range,
    )



    if overlay_scan_grid:
        grid_mask = table["quantileExpected"] == 0.5
        p_grid = ax.scatter(
            table[poi_x][grid_mask],
            table[poi_y][grid_mask],
            marker="o",
            s=16,
            facecolors="none",
            edgecolors="white",
            linewidths=0.8,
            alpha=0.8,
            zorder=7,
        )
        legend_entries["Scan grid points"] = p_grid

    if observed and observed != "False":
        obs = make_stable_interpolator(
            table,
            poi_x,
            poi_y,
            quantile=-1 if observed else 0.16,
            triangulation=triangulation,
            rectangular_diagonal=rectangular_diagonal,
        )
        if smooth_contour:
            po = plot_limit_contour_smooth(
            ax,
            obs,
            x_range,
            y_range,
            color="black",
            valid_mask_fn=valid_mask_fn,
            level=contour_level,
        )
        else:
            po = plot_limit_contour(
                ax,
                obs,
                x_range,
                y_range,
                color="black",
                valid_mask_fn=valid_mask_fn,
                level=contour_level,
            )
        obs_label = "Obs. limit (dummy)" if observed == "Dummy" else "Observed limit"
        legend_entries[obs_label] = po[0]
        max_x, (x_point_x, y_point_y), max_y, (x_point_y, y_point_y) = get_contour_max(ax)
        print(f"Observed max contour point: max x = {max_x:.2f}, (x,y) = ({x_point_x:.2f}, {y_point_y:.2f}), max y = {max_y:.2f}, (x,y) = ({x_point_y:.2f}, {y_point_y:.2f})")
       

    exp_central = make_stable_interpolator(
        table, poi_x, poi_y, quantile=0.5, triangulation=triangulation, rectangular_diagonal=rectangular_diagonal
    )
    exp_up = make_stable_interpolator(
        table, poi_x, poi_y, quantile=0.16, triangulation=triangulation, rectangular_diagonal=rectangular_diagonal
    )
    exp_lo = make_stable_interpolator(
        table, poi_x, poi_y, quantile=0.84, triangulation=triangulation, rectangular_diagonal=rectangular_diagonal
    )
    if smooth_contour:
        p1 = plot_limit_contour_smooth(ax, exp_central, x_range, y_range, color="#B30000", valid_mask_fn=valid_mask_fn, linestyle="--")
    else:
        p1 = plot_limit_contour(ax, exp_central, x_range, y_range, color="#B30000", valid_mask_fn=valid_mask_fn, linestyle="--")
    p2 = plot_limit_band(ax, exp_up, exp_lo, x_range, y_range, color="#F7AAAA", valid_mask_fn=valid_mask_fn)
 #B30000. #F7AAAA
    legend_entries[r"Expected limit $\pm 1\sigma_{exp}$"] = (p1[0], p2[0])

    
    
    if flag_outliers:
        flagged_points = find_suspicious_observed_points(
            table,
            poi_x,
            poi_y,
            contour_level=contour_level,
            neighbor_ratio=outlier_ratio,
        )
        if flagged_points:
            up_points = [p for p in flagged_points if p["direction"] == "up"]
            down_points = [p for p in flagged_points if p["direction"] == "down"]

            if up_points:
                p_up = ax.scatter(
                    [p[poi_x] for p in up_points],
                    [p[poi_y] for p in up_points],
                    marker="^",
                    s=95,
                    color="tab:red",
                    linewidths=1.0,
                    edgecolors="black",
                    zorder=8,
                )
                legend_entries["Flagged upward fluctuations"] = p_up

            if down_points:
                p_down = ax.scatter(
                    [p[poi_x] for p in down_points],
                    [p[poi_y] for p in down_points],
                    marker="v",
                    s=95,
                    color="tab:blue",
                    linewidths=1.0,
                    edgecolors="black",
                    zorder=8,
                )
                legend_entries["Flagged downward fluctuations"] = p_down

            if annotate_outliers:
                for p in flagged_points:
                    is_up = p["direction"] == "up"
                    label_color = "tab:red" if is_up else "tab:blue"
                    y_offset = 6 if is_up else -10
                    ax.annotate(
                        f"{p['delta']:+.2g}",
                        (p[poi_x], p[poi_y]),
                        textcoords="offset points",
                        xytext=(6, y_offset),
                        ha="left",
                        va="center",
                        fontsize=9,
                        color=label_color,
                        zorder=9,
                    )
                legend_entries[r"Label: $\Delta = Obs. - median(neighbors)$"] = ax.plot([], [], alpha=0)[0]

            print("Suspicious observed points:")
            for p in flagged_points:
                print(
                    f"  {poi_x}={p[poi_x]:g}, {poi_y}={p[poi_y]:g}, "
                    f"obs={p['obs_limit']:.4g}, neigh_med={p['neighbor_median']:.4g}, "
                    f"delta={p['delta']:+.4g}, direction={p['direction']}, reasons={p['reasons']}"
                )
            print(f"Flagged totals: up={len(up_points)}, down={len(down_points)}")
        else:
            print("No suspicious observed points were flagged.")

    
    hep.cms.text(text=label, ax=ax, loc=0, fontsize=34)
    hep.cms.lumitext(text=lumi_text)

    if (poi_x == "mMed" and poi_y == "mDark") or (poi_x == "mDark" and poi_y == "mMed"):
        if poi_x == "mMed" and poi_y == "mDark":
            mmed_vals = np.linspace(500, 4000, 200)
            mdark_vals = np.minimum(np.maximum(20, np.ceil((mmed_vals / 96.0) ** 1.25 / 10) * 10), 100)
            mask = mmed_mdark_constraint(mmed_vals, mdark_vals)
            p_theory = ax.plot(mmed_vals[mask], mdark_vals[mask], 'k--', linewidth=2, zorder=5)
            # legend_entries["Jettiness condition"] = p_theory[0]
        else:
            mdark_vals = np.linspace(max(x_range[0], 20), x_range[1], 200)
            mmed_vals = 96.0 * mdark_vals ** 0.8
            mask = (mmed_vals <= y_range[1])
            p_theory = ax.plot(mdark_vals[mask], mmed_vals[mask], 'k--', linewidth=2, zorder=5)
            # legend_entries["Jettiness condition"] = p_theory[0]

    if tagger == "wnae":
        tagger_name = "Unsupervised"
    else:
        tagger_name = "Supervised"
    fixed_str = ", ".join(
        f"{latex_readable_varibles(k)} = {int(v) if isinstance(v, float) and v.is_integer() else v}" + (" GeV" if k == "mDark" else "") 
        for k, v in fixed_params.items() if k != "xsec"
    ) if fixed_params else ""
    
    legend_title = None
    if fixed_str:
        legend_title = tagger_name + "\n" + fixed_str
    
    if (poi_x == "mMed" and poi_y == "mDark") or (poi_x == "mDark" and poi_y == "mMed"):
        default_legend_loc = "upper left"
        frameon = False
        legend_entries[r"$m_{\Phi} \gtrsim 30 \Lambda_{dark}, \Lambda_{dark} = 3.2 m_{dark}^{0.8}$"] = p_theory[0]
    else:
        default_legend_loc = "upper right"
        frameon = True
    
    legend_loc = getattr(args, 'legend_position', default_legend_loc)
    # if fixed_str:
    #     fixed_str = fixed_str 
    #     legend_entries[fixed_str] = matplotlib.lines.Line2D([], [], linestyle="None")
    #     legend_entries[tagger_name] = matplotlib.lines.Line2D([], [], linestyle="None")
    
    leg = ax.legend(
        list(legend_entries.values()),
        list(legend_entries.keys()),
        title=legend_title,
        loc=legend_loc,
        frameon=frameon,
    )
    if legend_title:
        leg.get_title().set_multialignment("left")
    leg._legend_box.align = "left"

    ax.set_xlabel(get_poi_label(poi_x) + (" [GeV]" if poi_x == "mDark" else ""), fontsize=40)
    ax.set_ylabel(get_poi_label(poi_y) + (" [GeV]" if poi_y == "mDark" else ""), fontsize=44)
    ax.tick_params(axis="both", labelsize=30)
    fig.savefig(output_pdf)
    print(f"Saved: {output_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D limit plot for two arbitrary POIs")
    # parser.add_argument("--folder", required=True, help="Path to folder with limit txt files")
    parser.add_argument("--poi-x", default="mMed", help="X-axis POI name (default: mMed)")
    parser.add_argument("--poi-y", default="rinv", help="Y-axis POI name (default: rinv)")
    parser.add_argument("--fixed", nargs="*", default=[], help="Fixed params as key=value pairs, e.g. mDark=20 xsec=1.0")
    parser.add_argument("--observed", default=True, type=lambda x: x if x in ("Dummy", "False") else x.lower() == "true")
    parser.add_argument("--label", default="")
    parser.add_argument("--lumi", default=r"138 $fb^{-1}$ (13 TeV)")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-pdf", default=None)
    parser.add_argument("--xsec-csv", default=None, help="CSV with per-signal theoretical cross sections for shading")
    parser.add_argument(
        "--colorbar-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=None,
        help="Manual log colorbar range; default is inferred from positive finite shaded values",
    )
    parser.add_argument("--flag-outliers", action="store_true", help="Flag suspicious observed points in the scan grid")
    parser.add_argument("--annotate-outliers", action="store_true", help="Annotate flagged points with signed local delta values")
    parser.add_argument("--outlier-ratio", type=float, default=2.5, help="Neighbor-median ratio threshold for outlier flagging")
    parser.add_argument("--contour-level", type=float, default=1.0, help="Contour threshold used for observed-limit diagnostics")
    parser.add_argument("--overlay-scan-grid", action="store_true", help="Overlay scan grid points on top of the plot")
    parser.add_argument(
        "--triangulation",
        choices=("delaunay", "rectangular", "scan"),
        default="delaunay",
        help="Triangulation used for interpolation: delaunay or deterministic rectangular-cell splits",
    )
    parser.add_argument(
        "--rectangular-diagonal",
        choices=("bl-tr", "tl-br"),
        default="bl-tr",
        help="Diagonal used when --triangulation rectangular splits each complete scan cell",
    )
    parser.add_argument("--legend-position", default="upper left", help="Legend position (default: upper left)")
    parser.add_argument("--tagger", default="wnae", help="Tagger type (default: wnae)")
    parser.add_argument("--output-suffix", default="", help="Suffix for output files (default: empty)")
    parser.add_argument("--smooth", action="store_true", help="Smooth the interpolated limit contours (default: False)")
    
    args = parser.parse_args()

    fixed_params = {}
    for item in args.fixed:
        k, v = item.split("=", 1)
        try:
            fixed_params[k] = float(v)
        except ValueError:
            fixed_params[k] = v

    make_2d_limit_plot(
        # folder_path=args.folder,
        folder_path=None,  # not used since we are directly providing the CSV
        poi_x=args.poi_x,
        poi_y=args.poi_y,
        fixed_params=fixed_params,
        observed=args.observed,
        label=args.label,
        lumi_text=args.lumi,
        output_csv=args.output_csv,
        output_pdf=args.output_pdf,
        flag_outliers=args.flag_outliers,
        annotate_outliers=args.annotate_outliers,
        outlier_ratio=args.outlier_ratio,
        contour_level=args.contour_level,
        overlay_scan_grid=args.overlay_scan_grid,
        xsec_csv=args.xsec_csv,
        colorbar_range=args.colorbar_range,
        triangulation=args.triangulation,
        rectangular_diagonal=args.rectangular_diagonal,
        output_suffix=args.output_suffix,
        tagger=args.tagger,
        smooth_contour=args.smooth,
    )