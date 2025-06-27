# Import Libraries
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import gaussian_kde
from noise import pnoise3
from mayavi import mlab
from rasterio.plot import reshape_as_image
from tvtk.api import tvtk
from tvtk.util.ctf import PiecewiseFunction
from tvtk.util.ctf import ColorTransferFunction
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------------------------------------------- #
#                                                  1. Data Loading                                                 #
# ---------------------------------------------------------------------------------------------------------------- #

def load_event_data(file_path):
    df = pd.read_csv(file_path).dropna(subset=['X', 'Y', 'Z'])
    return df, df['spatialdist'].to_numpy()

def load_basemap(tif_path):
    with rasterio.open(tif_path) as src:
        image = reshape_as_image(src.read())
        
        image_rgb = (image.astype(np.float32) - image.min()) / (image.max() - image.min())
        image_rgb = (image_rgb * 255).astype(np.uint8)
        #image_rgb = np.transpose(image_rgb, (1, 0, 2))  # Rotate 90° counter-clockwise
        #image = np.flipud(image)

        rows, cols = image_rgb.shape[:2]
        if image_rgb.shape[2] == 3:
            alpha = np.full((rows, cols, 1), 255, dtype=np.uint8)
            image_rgba = np.concatenate((image_rgb, alpha), axis=2)
        else:
            image_rgba = image_rgb

        colors = tvtk.UnsignedCharArray()
        colors.from_array(image_rgba.transpose((1, 0, 2)).reshape(-1, 4))
        transform = src.transform
        #bounds = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        return image_rgba, colors, transform, (rows, cols)


# ---------------------------------------------------------------------------------------------------------------- #
#                                            2. Relative Coordinate System                                         #
# ---------------------------------------------------------------------------------------------------------------- #
def create_relative_system(transform, cols, rows):
    x_coords = np.arange(cols) + 0.5
    y_coords = np.arange(rows) + 0.5

    # Real-world pixel centers (in feet)
    X_full = transform.c + x_coords * transform.a
    Y_full = transform.f + y_coords * transform.e

    X_grid, Y_grid = np.meshgrid(X_full, Y_full)

    # Set origin to bottom-left of the basemap
    origin_x, origin_y = X_grid.min(), Y_grid.min()

    # Relative coordinates in feet
    X = X_grid - origin_x
    Y = Y_grid - origin_y
    Z = np.zeros_like(X)
    return X, Y, Z, origin_x, origin_y

def compute_z_scale(points):
    """
    Computes a Z scaling factor to equalize Z range with the average XY range.
    
    Parameters:
        points: (N, 3) ndarray of relative [X, Y, Z] coordinates.
        
    Returns:
        z_scale: float — value to multiply Z by for balanced spatial-temporal scaling.
    """
    x_range = np.ptp(points[:, 0])
    y_range = np.ptp(points[:, 1])
    z_range = np.ptp(points[:, 2])

    avg_xy_range = (x_range + y_range) / 2.0
    z_scale = avg_xy_range / z_range if z_range > 0 else 1.0
    return z_scale

def convert_events(df, x_origin, y_origin):
    """
    Convert absolute event coordinates to a relative system and apply Z scaling
    based on average X/Y range to Z range ratio.
    """
    # Shift coordinates to relative space
    df['X_rel'] = df['X'] - x_origin
    df['Y_rel'] = df['Y'] - y_origin
    df['Z_rel'] = df['Z']

    # Convert to array
    relative_points = df[['X_rel', 'Y_rel', 'Z_rel']].values

    # Compute ranges
    x_range = np.ptp(relative_points[:, 0])
    y_range = np.ptp(relative_points[:, 1])
    z_range = np.ptp(relative_points[:, 2])

    # Average XY range
    xy_range_avg = (x_range + y_range) / 2

    # Compute scale factor
    if z_range > 0:
        z_scale = xy_range_avg / z_range
        relative_points[:, 2] *= z_scale
        #print(f"[✔] Z scaling applied: scale = {z_scale:.2f}")
        #print(f"[✓] Z Scaling Factor: {z_scale:.3f} (XY range = {xy_range_avg:.1f}, Z range = {z_range})")
    else:
        print("[!] Z scaling skipped: Z range is 0")

    return relative_points


# unused
'''
def compute_ann(points_2d):
    nbrs = NearestNeighbors(n_neighbors=2).fit(points_2d)
    distances, _ = nbrs.kneighbors(points_2d)
    # distances[:, 0] is 0 (self), take distances[:, 1]
    return distances[:, 1].mean()
'''

# ---------------------------------------------------------------------------------------------------------------- #
#                                                    3. Data Processing                                            #
# ---------------------------------------------------------------------------------------------------------------- #
def sphere_fade(distance, radius=5.0):
    """
    Returns a sphere-like contribution for each voxel distance.
    Larger alpha => sharper drop. 
    radius       => approximate radius for "fade-out."
    """
    alpha = 4.0 / (radius**2)  # tune to control how quickly it fades to zero
    return np.exp(-alpha * distance**2)

def build_grid_from_basemap(X, Y, relative_points, grid_size=120, pad_factor=0.3):
    """
    Build a 3D voxel grid for KDE/metaball analysis using the basemap extent for X/Y
    and the event time extent for Z. This ensures spatial alignment with the basemap.

    Parameters:
        X, Y: 2D np.arrays from basemap meshgrid (already relative)
        relative_points: (N, 3) array of event points in relative coordinates
        grid_size: int – number of voxels along each axis
        pad_factor: float – extra padding for Z axis (optional)

    Returns:
        gridX, gridY, gridZ: voxel grid coordinates
        grid_coords: flattened coordinates (3, N)
        shape: shape of the 3D volume
    """
    # Use basemap X/Y extent
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()

    # Use event Z extent
    z_min = relative_points[:, 2].min()
    z_max = relative_points[:, 2].max()
    z_range = z_max - z_min

    # Optional padding for Z
    z_min -= pad_factor * z_range
    z_max += pad_factor * z_range

    # Create voxel grid
    gridX, gridY, gridZ = np.mgrid[
        x_min:x_max:grid_size*1j,
        y_min:y_max:grid_size*1j,
        z_min:z_max:grid_size*1j
    ]

    # Flattened coordinates for KDE
    grid_coords = np.vstack([gridX.ravel(), gridY.ravel(), gridZ.ravel()])
    shape = gridX.shape

    return gridX, gridY, gridZ, grid_coords, shape



def build_volume(
    relative_points, gridX, gridY, gridZ, grid_coords, shape,
    approach='KDE', bw=0.3, radii=None,
    plume_ids=None, subdivide=False,
    p_cfg=None
):
    """
    Builds a 3D volume using either KDE or Metaballs.
    If subdivide=True and plume_ids is provided, returns a dominant ID grid.
    """

    # -------------------------------
    # Step 1: Spatial Isotropy Scaling
    # -------------------------------
    spread_x = np.ptp(relative_points[:, 0])
    spread_y = np.ptp(relative_points[:, 1])
    spread_z = np.ptp(relative_points[:, 2])
    mean_spread = np.mean([spread_x, spread_y, spread_z])

    scale_x = mean_spread / spread_x if spread_x > 0 else 1.0
    scale_y = mean_spread / spread_y if spread_y > 0 else 1.0
    scale_z = mean_spread / spread_z if spread_z > 0 else 1.0

    points_scaled = relative_points.copy()
    points_scaled[:, 0] *= scale_x
    points_scaled[:, 1] *= scale_y
    points_scaled[:, 2] *= scale_z

    grid_coords_scaled = grid_coords.copy()
    grid_coords_scaled[0, :] *= scale_x
    grid_coords_scaled[1, :] *= scale_y
    grid_coords_scaled[2, :] *= scale_z

    # -------------------------------
    # Step 2A: KDE Volume
    # -------------------------------
    if approach == 'KDE':
        print(f">> Computing KDE")
        kde = gaussian_kde(points_scaled.T, bw_method=bw)
        density = kde(grid_coords_scaled).reshape(shape)

        if p_cfg and p_cfg.get('enabled', False):
            density = add_perlin_noise(density, scale=p_cfg['scale'], amplitude=p_cfg['amplitude'])

        density_max = density.max()
        norm_density = density / density_max if density_max > 0 else density

        if subdivide and plume_ids is not None:
            print(">> Building dominant ID grid")
            id_grid = np.full(shape, -1, dtype=int)

            for pid in np.unique(plume_ids):
                points_pid = relative_points[plume_ids == pid].copy()
                points_pid[:, 0] *= scale_x
                points_pid[:, 1] *= scale_y
                points_pid[:, 2] *= scale_z

                kde_pid = gaussian_kde(points_pid.T, bw_method=bw)
                local_density = kde_pid(grid_coords_scaled).reshape(shape)

                update_mask = local_density > density
                id_grid[update_mask] = pid
                density[update_mask] = local_density[update_mask]

            norm_density = density / density.max()
            return gridX, gridY, gridZ, norm_density, id_grid

        return gridX, gridY, gridZ, norm_density, 0

    # -------------------------------
    # Step 2B: Metaballs Volume
    # -------------------------------
    elif approach == 'METABALLS':
        print(f">> Initialize Metaball Volume")

        if radii is None:
            raise ValueError("Radii must be provided for metaballs volume.")

        avg_scale = np.mean([scale_x, scale_y, scale_z])
        scaled_radii = radii * avg_scale

        # Case 1: Unified Metaballs Volume
        if not subdivide or plume_ids is None:
            metaballs_volume = np.zeros(shape, dtype=np.float32)

            for i, (px, py, pz) in enumerate(points_scaled):
                r = scaled_radii[i]
                dist = np.sqrt(
                    (grid_coords_scaled[0].reshape(shape) - px) ** 2 +
                    (grid_coords_scaled[1].reshape(shape) - py) ** 2 +
                    (grid_coords_scaled[2].reshape(shape) - pz) ** 2
                )
                metaballs_volume += sphere_fade(dist, radius=r)

            if p_cfg and p_cfg.get('enabled', False):
                metaballs_volume = add_perlin_noise(metaballs_volume, scale=p_cfg['scale'], amplitude=p_cfg['amplitude'])

            norm_density = metaballs_volume / np.max(metaballs_volume)
            return gridX, gridY, gridZ, norm_density, 0

        # Case 2: Subdivided Metaballs Volume by PlumeID
        else:
            print(">> Building dominant ID grid (Metaballs)")
            density = np.zeros(shape, dtype=np.float32)
            id_grid = np.full(shape, -1, dtype=int)

            unique_ids = np.unique(plume_ids)

            for pid in unique_ids:
                mask = plume_ids == pid
                points_pid = points_scaled[mask]
                radii_pid = scaled_radii[mask]

                temp_volume = np.zeros(shape, dtype=np.float32)

                for i in range(len(points_pid)):
                    px, py, pz = points_pid[i]
                    r = radii_pid[i]
                    dist = np.sqrt(
                        (grid_coords_scaled[0].reshape(shape) - px) ** 2 +
                        (grid_coords_scaled[1].reshape(shape) - py) ** 2 +
                        (grid_coords_scaled[2].reshape(shape) - pz) ** 2
                    )
                    temp_volume += sphere_fade(dist, radius=r)

                # Dominance mask
                update_mask = temp_volume > density
                density[update_mask] = temp_volume[update_mask]
                id_grid[update_mask] = pid

            if p_cfg and p_cfg.get('enabled', False):
                density = add_perlin_noise(density, scale=p_cfg['scale'], amplitude=p_cfg['amplitude'])

            norm_density = density / np.max(density)
            return gridX, gridY, gridZ, norm_density, id_grid
    else:
        raise ValueError("Invalid approach.")



def add_perlin_noise(volume, scale=0.01, amplitude=0.2):
    noisy_volume = np.empty_like(volume)
    shape = volume.shape
    #print("Test noise:", [pnoise3(i * 0.01, i * 0.01, i * 0.01) for i in range(5)])

    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                n = pnoise3(x * scale, y * scale, z * scale)
                n = 0.5 * (1 + n)  # normalize to [0, 1]
                noisy_volume[x, y, z] = volume[x, y, z] * (1 + amplitude * (n - 0.5))

    return noisy_volume

# ---------------------------------------------------------------------------------------------------------------- #
#                                                    4. Plotting                                                   #
# ---------------------------------------------------------------------------------------------------------------- #
def auto_ticks(start, stop, approx_ticks=5):
    """Generate 'nice' rounded ticks between start and stop (supports reverse order)."""
    # Compute direction-aware range
    span = abs(stop - start)
    if span == 0:
        return np.array([start])  # Single tick if no range

    raw_step = span / (approx_ticks - 1)
    magnitude = 10 ** np.floor(np.log10(raw_step))
    nice_steps = [1, 2, 5, 10]
    step = min(nice_steps, key=lambda x: abs(x * magnitude - raw_step)) * magnitude

    # Generate ticks in correct direction
    if start < stop:
        tick_start = np.ceil(start / step) * step
        tick_end = np.floor(stop / step) * step
        return np.arange(tick_start, tick_end + step, step)
    else:
        tick_start = np.floor(start / step) * step
        tick_end = np.ceil(stop / step) * step
        return np.arange(tick_start, tick_end - step, -step)

def apply_opacity_flicker(otf, magnitude=0.1):
    n = otf._get_size()
    for i in range(n):
        vals = [0.0] * 4  # [x, y, midpoint, sharpness]
        otf.get_node_value(i, vals)

        # Flicker Y (opacity)
        vals[1] = max(0.0, min(1.0, vals[1] + np.random.uniform(-magnitude, magnitude)))

        # Update the same node
        otf.set_node_value(i, vals)


def base_volume_style(vol, lighting,  unit_distance, style_cfg, opac_cfg):
    """
    Apply range-compressed OTF and CTF to the volume property based on data percentiles.
    """
    vp = vol._volume_property

    # --- Color Transfer Function (CTF): glow through
    ctf = ColorTransferFunction()
    for x, r, g, b in style_cfg.get('ctf_points', []):  # fallback if not defined
        ctf.add_rgb_point(x, r, g, b)
    vp.set_color(ctf)

    # --- Opacity Transfer Function (OTF): wispiness
    otf = PiecewiseFunction()
    for x, y in style_cfg['otf_points']:
        otf.add_point(x, y)

    if opac_cfg['enabled']:
        apply_opacity_flicker(otf, opac_cfg['magnitude'])

    vp.set_scalar_opacity(otf)
    vp.scalar_opacity_unit_distance = unit_distance  # decrease to make plume more transparent (try 0.02–0.1)

    # --- Lighting: ethereal glow
    vp.shade = lighting['shade']
    vp.ambient = lighting['ambient']
    vp.diffuse = lighting['diffuse']
    vp.specular = lighting['specular']

def sub_volume_Style(vol,rgb_color, unit_distance, lighting, style_cfg, opac_cfg):
    vp = vol._volume_property

    # --- Opacity Transfer Function ---
    otf = PiecewiseFunction()
    for x, y in style_cfg['otf_points']:
        otf.add_point(x, y)

    if opac_cfg['enabled']:
        apply_opacity_flicker(otf, opac_cfg['magnitude'])

    vp.set_scalar_opacity(otf)
    vp.scalar_opacity_unit_distance = unit_distance

    # --- Color Transfer Function ---
    r, g, b = rgb_color[:3]
    ctf = ColorTransferFunction()
    ctf.add_rgb_point(0.0,   0.1, 0.1, 0.1)
    ctf.add_rgb_point(0.1,  r, g, b)
    ctf.add_rgb_point(1.0,  r, g, b)
    vp.set_color(ctf)

    # --- Lighting ---
    vp.shade = lighting['shade']
    vp.ambient = lighting['ambient']
    vp.diffuse = lighting['diffuse']
    vp.specular = lighting['specular']

def render_basemap(rows, cols, colors, X, Y, fig=None):
    # Create placeholder image object (greyscale dummy to start)
    m_image = mlab.imshow(np.ones((rows, cols)), colormap='gray', figure=fig)
    m_image.actor.input.point_data.scalars = colors
    m_image.actor.input.point_data.scalars.name = 'RGBA'

    # Compute spatial extent from mesh
    x_rel_min, x_rel_max = X.min(), X.max()
    y_rel_min, y_rel_max = Y.min(), Y.max()

    # Compute scale
    scale_x = (x_rel_max - x_rel_min) / cols
    scale_y = (y_rel_max - y_rel_min) / rows

    # Position at center of mesh
    m_image.actor.position = (
        x_rel_min + (scale_x * cols) / 2,
        y_rel_min + (scale_y * rows) / 2,
        -10
    )

    # Correct rotation
    m_image.actor.orientation = (0, 0, -90)

    # Set scale to match mesh
    m_image.actor.scale = (scale_x, scale_y, 1)

    # Make lighting flat and full color
    m_image.actor.property.ambient = 1.0
    m_image.actor.property.diffuse = 0.0

    return m_image

def set_smart_camera(xmin, xmax, ymin, ymax, zmin=0, zmax=1000,
                     azimuth=225, elevation=30, margin=0.2):
    """
    Auto-set Mayavi camera to frame your scene smartly.
    """
    import numpy as np
    from mayavi import mlab

    # Center of scene
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    cz = (zmin + zmax) / 2

    # Size of scene
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    diag = np.sqrt(dx**2 + dy**2 + dz**2)

    # Add margin to distance
    dist = diag * (1 + margin)

    # Set view
    mlab.view(azimuth=azimuth, elevation=elevation, distance=dist, focalpoint=(cx, cy, cz))

def draw_camera_relative_axes(x_range, y_range, z_range, tick_len=50, label_scale=100):
    """
    Draw custom 3D axes with ticks and labels in a camera-relative, scientific style.
    """
    x0, x1 = x_range
    y0, y1 = y_range
    z0, z1 = z_range

    # Origin location (customize to suit)
    origin = (x0, y0, z0)

    # Tick positions
    x_ticks = auto_ticks(x0, x1)
    y_ticks = auto_ticks(y0, y1)
    z_ticks = auto_ticks(z0, z1)

    # Move Z axis to the end of the X axis
    z_origin_x = x1      # End of X axis
    z_origin_y = y0      # Same Y as base
    z_origin_z = z0      # Ground level

    # --- X Axis ---
    # Ticks an Tick Labels
    for xt in x_ticks:
        mlab.plot3d([xt, xt], [y0, y0 - tick_len], [z0, z0], color=(0, 0, 0), tube_radius=None)
        mlab.text3d(xt, y0 - 3 * tick_len, z0, f'{xt:.0f}', scale=label_scale, color=(0, 0, 0))

    # Main Bar and Label;
    mlab.plot3d([x0, x1], [y0, y0], [z0, z0], color=(0, 0, 0), tube_radius=None, line_width=2)
    mlab.text3d(x1 + tick_len, y0, z0, 'X', scale=label_scale*1.2, color=(0, 0, 0))

    # --- Y Axis ---
    # Ticks an Tick Labels
    for yt in y_ticks:
        label = f'{yt:.0f}'
        if label == '0':
            shift = -3.5 * tick_len  # adjust this value as needed
        else:
            shift = 0.0  # no shift for other labels

        mlab.plot3d([x0, x0 - tick_len], [yt, yt], [z0, z0], color=(0, 0, 0), tube_radius=None)
        mlab.text3d(x0 - 7.5 * tick_len - shift, yt, z0, f'{yt:.0f}', scale=label_scale, color=(0, 0, 0))
    
    # Main Bar and Label;
    mlab.plot3d([x0, x0], [y0, y1], [z0, z0], color=(0, 0, 0), tube_radius=None, line_width=2)
    mlab.text3d(
    x0 - 3 * tick_len,  # push along X
    y1 + 2 * tick_len,  # push away from axis
    z0,
    'Y',
    scale=label_scale * 1.2,
    color=(0, 0, 0)
    )

    # --- Z Axis ---
    # Z-axis tick marks and labels
    for zt in z_ticks:
        mlab.plot3d(
            [z_origin_x, z_origin_x - tick_len],  # small horizontal tick
            [z_origin_y, z_origin_y],
            [zt, zt],
            color=(0, 0, 0),
            tube_radius=None
        )
        mlab.text3d(z_origin_x - 5 * tick_len, z_origin_y + 5 * tick_len, zt, f'{zt:.0f}', scale=label_scale, color=(0, 0, 0))
    
    # Main Bar and Label;
    mlab.plot3d([z_origin_x, z_origin_x], [z_origin_y, z_origin_y], [z0, z1], color=(0, 0, 0), tube_radius=None, line_width=2)
    mlab.text3d(
    z_origin_x + tick_len,  # push outward in X
    z_origin_y + tick_len,  # push in Y too if needed
    z1 + tick_len,          # label above top tick
    'Z',
    scale=label_scale * 1.2,
    color=(0, 0, 0)
    )

def save_plume_legend(plume_colors, output_path="legend.png", title="Plume Legend"):
    """
    Generate and save a legend image for PlumeID colors.

    Parameters:
        plume_colors (dict): Dictionary mapping PlumeID to RGB tuple or hex.
        output_path (str): File path to save the legend image (e.g., 'legend.png').
        title (str): Optional title to display above the legend.
    """
    # Convert RGB floats to hex if needed
    labels = []
    patches = []
    for plume_id, color in plume_colors.items():
        if isinstance(color, tuple):
            color_hex = '#%02x%02x%02x' % tuple(int(255 * c) for c in color[:3])
        else:
            color_hex = color
        labels.append(f'Plume {plume_id}')
        patches.append(Patch(facecolor=color_hex, edgecolor='black'))

    # Create a figure just for the legend
    fig, ax = plt.subplots(figsize=(3, len(labels) * 0.4))
    ax.axis('off')
    legend = ax.legend(handles=patches, labels=labels, loc='center left', frameon=True, title=title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)

# ---------------------------------------------------------------------------------------------------------------- #
#                                                    X. Debugging                                                  #
# ---------------------------------------------------------------------------------------------------------------- #

# Double check relative_points
#print(relative_points.shape)
#print(relative_points[:5])  # inspect first few rows

'''
print(f">> Voxel Size (dx, dy, dz): {dx:.2f}, {dy:.2f}, {dz:.2f}")
print(">> Grid Shape:", gridX.shape)
print(">> Grid Ranges:")
print(f"   X: {gridX.min():.2f} to {gridX.max():.2f} → Δx = {dx:.2f}")
print(f"   Y: {gridY.min():.2f} to {gridY.max():.2f} → Δy = {dy:.2f}")
print(f"   Z: {gridZ.min():.2f} to {gridZ.max():.2f} → Δz = {dz:.2f}")
'''


#print("Density min/max:", density_values.min(), density_values.max())
# Maybe norm_density = np.clip(density_values, 0, np.percentile(density_values, 99))

#print("Volume shape:", density_values.shape)

# Display grayscale placeholder, then overwrite scalars with RGBA
# ALignment Mesh
#mlab.mesh(X, Y, Z, color=(1, 0, 0), opacity=0.05)  # faint red mesh on top of basemap

#print("Image shape (rows, cols):", image_rgba.shape[:2])
#print("Mesh shape X:", X.shape, "Y:", Y.shape)

#print("Min:", density_values.min())
#print("Max:", density_values.max())
#for p in [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]:
#    print(f"P{p}: {np.percentile(density_values, p)}")

#xy_points = df[['X', 'Y']].to_numpy()
#ann = PF.compute_ann(xy_points)
#print(f"Average Nearest Neighbor (ANN): {ann:.2f} ft")

#print("ID Grid Shape:", id_grid.shape)
#print("Grid Shape:", gridX.shape)
#print("PlumeID Range in Grid:", np.unique(id_grid))


'''
def convert_events(df, x_origin, y_origin, ann_scale=1.0):
    df['X_rel'] = df['X'] - x_origin
    df['Y_rel'] = df['Y'] - y_origin
    df['Z_rel'] = df['Z'] * ann_scale  # Days → feet using ANN

    relative_points = df[['X_rel', 'Y_rel', 'Z_rel']].to_numpy()
    return relative_points
'''


# Original Build Volume:
'''
def build_volume(relative_points, approach='KDE', bw=0.3, radii=None,
                 grid_size=120, pad=True, pad_factor=0.3, 
                 plume_ids=None, subdivide=False, p_cfg=None):
    """
    Builds a 3D volume using either KDE or Metaballs.
    If subdivide=True and plume_ids is provided, returns a dominant ID grid.
    """
    # Padding
    if pad:
        x_range, y_range, z_range = relative_points.ptp(axis=0)
        x_min = relative_points[:,0].min() - pad_factor * x_range
        x_max = relative_points[:,0].max() + pad_factor * x_range
        y_min = relative_points[:,1].min() - pad_factor * y_range
        y_max = relative_points[:,1].max() + pad_factor * y_range
        z_min = relative_points[:,2].min() - pad_factor * z_range
        z_max = relative_points[:,2].max() + pad_factor * z_range
    else:
        x_min, x_max = relative_points[:,0].min(), relative_points[:,0].max()
        y_min, y_max = relative_points[:,1].min(), relative_points[:,1].max()
        z_min, z_max = relative_points[:,2].min(), relative_points[:,2].max()

    # Grid
    gridX, gridY, gridZ = np.mgrid[
        x_min:x_max:grid_size*1j,
        y_min:y_max:grid_size*1j,
        z_min:z_max:grid_size*1j
    ]
    grid_coords = np.vstack([gridX.ravel(), gridY.ravel(), gridZ.ravel()])
    shape = gridX.shape

    # KDE Volume
    if approach == 'KDE':
        print(f">> Computing KDE")
        # Standardize the input points
        scaler = StandardScaler()
        points_scaled = scaler.fit_transform(relative_points)

        # Standardize the grid coordinates
        grid_coords = np.vstack([gridX.ravel(), gridY.ravel(), gridZ.ravel()])
        grid_coords_scaled = scaler.transform(grid_coords.T).T

        # Compute KDE in scaled space
        kde = gaussian_kde(points_scaled.T, bw_method=bw)
        density = kde(grid_coords_scaled).reshape(gridX.shape)

        # Optional: Add Perlin noise *before* normalization
        if p_cfg['enabled']:
            density = add_perlin_noise(density, scale=p_cfg['scale'], amplitude=p_cfg['amplitude'])

        # Normalize
        density_max = density.max()
        norm_density = density / density_max if density_max > 0 else density

        if subdivide and plume_ids is not None:
            print(">> Building dominant ID grid")

            id_grid = np.full(shape, -1, dtype=int)
            for pid in np.unique(plume_ids):
                points_pid = relative_points[plume_ids == pid]
                kde_pid = gaussian_kde(points_pid.T, bw_method=bw)
                local_density = kde_pid(grid_coords).reshape(shape)

                update_mask = local_density > density
                id_grid[update_mask] = pid
                density[update_mask] = local_density[update_mask]  # override

            norm_density = density / density.max()
            return gridX, gridY, gridZ, norm_density, id_grid

        return gridX, gridY, gridZ, norm_density, 0

    # Metaballs approach

    else:
        raise ValueError("Invalid approach.")

'''