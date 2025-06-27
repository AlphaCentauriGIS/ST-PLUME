# ---------------------------------------------------------------------------------------------------------------- #
#   Plume 3D Visualization
#   Goal: Generate 3D visualization of plumes. The visualization should represent plumes/places as produces 
#           a fuzzy cloud around the events, mirroring the way smoke plumes billow and disperse 
#           under external pressures. The events themselves function as factor points, defining the core of each
#            plume while allowing for smooth transitions at the edges rather than abrupt cutoffs.
#   Approach: Two appraoches to generating the volume around the event cloud: 3D KDE from a grid encapsulating the points, a union of spheres - metaballs where the sphere radii are the zidx
#   Implimentation: Create and Plot Event Cloud, 3D Volume with transient boundaries, as well as Full color basemap
#   Pipelines: (1) Data Load, (2) Relative Coordinate Conversion, (3) Data Processing:, (4) Plotting
# ---------------------------------------------------------------------------------------------------------------- #

# Import Libraries
import numpy as np
from mayavi import mlab
import os
import PlumeFunctions as PF
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
import yaml

# ---------------------------------------------------------------------------------------------------------------- #
#                                                  0. Configs & Parameters                                         #
# ---------------------------------------------------------------------------------------------------------------- #
print(f'Section 0 - Setting up Configs')
# Load Configuration from YAML
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Filepaths
event_path = config['event_path']
tif_path = config['tif_path']
save = config['save_plot']
outputfile = config['outputfile']
legend_out = config['legend_output']

# General Settings
subdivide = config['subdivide']
option = config['approach']
grid_size = config['grid_size']
pad = config['pad']
pad_factor = config['pad_factor']
bw = config['bandwidth']

# Camera Configs
azimuth = config['camera']['azimuth']
elevation = config['camera']['elevation']
camdist = config['camera']['distance']

# Event Cloud Style
scale_factor = config['event_cloud']['scale_factor']
single_color = config['event_cloud']['single_color']

# Volume stylesets
lighting = config["volume_styles"]["lighting"]
unified_style = config["volume_styles"]["unified"]
subdivided_style = config["volume_styles"]["subdivided"]
p_cfg = config['volume_styles']['effects']['perlin_noise']
opac_cfg = config['volume_styles']['effects']['opacity_flicker']

# Environment Settings
os.environ['ETS_TOOLKIT'] = 'qt5'  # Ensure Qt5 is used
os.environ['QT_API'] = 'pyqt5'  

# ---------------------------------------------------------------------------------------------------------------- #
#                                                  1. Data Loading                                                 #
# ---------------------------------------------------------------------------------------------------------------- #
print(f'Section 1 - Data Load')
# Read Events
df, radii = PF.load_event_data(event_path)

# Get colors for temporal fragmentation
if subdivide:
    # PlumeID Colors
    # Extract unique plume IDs and assign each a color
    plume_ids = df['PlumeID'].unique()
    plume_ids.sort()
    n_plumes = len(plume_ids)

    # Use a matplotlib colormap to generate N distinct colors
    colormap = plt.colormaps['tab20']
    colors = [colormap(i / n_plumes) for i in range(n_plumes)]
    plume_colors = {pid: to_rgba(colors[i]) for i, pid in enumerate(plume_ids)}

    # Assign colors to each event
    df['Color'] = df['PlumeID'].map(plume_colors)

# Read Basemap
image_rgba, colors, transform, rowcols = PF.load_basemap(tif_path)

# ---------------------------------------------------------------------------------------------------------------- #
#                                            2. Relative Coordinate System                                         #
# ---------------------------------------------------------------------------------------------------------------- #
# In order for plotting to generate correct looking plots, we need to scale the axes together. Other               #
# considerations include the x,y spatial coordinate system (EPSG:2376) vs the Z scale which is days. We need these #
# axes to match in order to produce correct spatio-temporal distances. This section will adjust the coordinate     #
# values of the events as well as the basemap to a relative system and ensure alignment as well                    #
# ---------------------------------------------------------------------------------------------------------------- #
print(f'Section 2 - Relative Coordinates')

# Choose a reference point (bottom-left corner as (0,0)). This 0,0 is of the basemap! since the basemap has a
# larger spatial extent than the event data
rows, cols = rowcols
X, Y, Z, x_ref, y_ref = PF.create_relative_system(transform, cols, rows)

# Shift event points to relative system
relative_points = PF.convert_events(df, x_ref, y_ref)

# ---------------------------------------------------------------------------------------------------------------- #
#                                                    3. Data Processing                                            #
# ---------------------------------------------------------------------------------------------------------------- #
# This section focuses on two workflows: (a) generating a 3D volume characterizing the region around events        #
# and (b) implementing a workaround procedure to displaying the basemap in full color as it is not directly        #
# supported out of the box for mayavi. Two options are chosen to represent the cloud structure region around       #
# the event-swarm: a 3D Kernel Density Estimation, and a Metaballs Function on the Union of Spheres.               #
# ---------------------------------------------------------------------------------------------------------------- #
# Build Grid
gridX, gridY, gridZ, grid_coords, shape = PF.build_grid_from_basemap(X, Y, relative_points, grid_size, pad_factor)

print(f'Section 3 - Data Processing')
gridX, gridY, gridZ, density_values, id_grid = PF.build_volume(
    relative_points,
    gridX, gridY, gridZ, grid_coords, shape,
    approach=option,
    bw=bw,
    radii=radii,
    plume_ids=df['PlumeID'].to_numpy(),
    subdivide=subdivide,
    p_cfg=p_cfg
)

# ---------------------------------------------------------------------------------------------------------------- #
#                                                    4. Mayavi Plotting                                            #
# ---------------------------------------------------------------------------------------------------------------- #
# This section initializes a 3D plot, plots the volume, events, basemap, and finally applies scientific            #
# aesthetics for output                                                                                            #
# ---------------------------------------------------------------------------------------------------------------- #
print(f'Section 4 - Plotting')
fig = mlab.figure("Plume Volume", size=(1000,800), bgcolor=(1,1,1), fgcolor=(0,0,1))

# -------------------------------------------------------------------
print(">> Render Basemap")
# Plot the image as a 3D surface
bmap = PF.render_basemap(rows, cols, colors, X, Y, fig=fig)

# -------------------------------------------------------------------
# Render the Volume
print(f">> Render Volume")

# --- Compute voxel dimensions ---
dx = np.ptp(gridX) / gridX.shape[0]
dy = np.ptp(gridY) / gridY.shape[1]
dz = np.ptp(gridZ) / gridZ.shape[2]
avg_voxel_size = np.mean([dx, dy, dz])
unit_distance = avg_voxel_size * 2  # This scales opacity unit distance

if subdivide:
    print(">> Render Subdivided Volume")
    for i, pid in enumerate(plume_ids):
        if pid == -1: continue

        # Mask and extract per-plume volume
        mask = id_grid == pid
        density_masked = np.where(mask, density_values, 0.0)
        density_clipped = density_masked / np.max(density_masked) if np.max(density_masked) > 0 else density_masked

        src = mlab.pipeline.scalar_field(gridX, gridY, gridZ, density_clipped)
        src.image_data.point_data.scalars.name = f'density_{pid}'

        vol = mlab.pipeline.volume(src)
        PF.sub_volume_Style(vol, plume_colors[pid], unit_distance, lighting=lighting, style_cfg=subdivided_style, opac_cfg=opac_cfg)

else:
    print(">> Render Unified Volume")
    # Create mask where Z values are below the threshold
    mask = gridZ >= 0

    # Apply mask to density values
    density_clipped = np.where(mask, density_values, 0.0)
    src = mlab.pipeline.scalar_field(gridX, gridY, gridZ, density_clipped)
    src.image_data.point_data.scalars.name = 'density'

    vol = mlab.pipeline.volume(src)
    PF.base_volume_style(vol, unit_distance=unit_distance, lighting=lighting, style_cfg=unified_style, opac_cfg=opac_cfg)


# -------------------------------------------------------------------
# Render the Event Cloud

if subdivide:
    print(">> Render Event Cloud (by PlumeID)")

    for pid in plume_ids:
        sub = df[df['PlumeID'] == pid]
        color = plume_colors[pid][:3]
        mask = df['PlumeID'] == pid
        pts = relative_points[mask]

        mlab.points3d(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            scale_mode='none',
            scale_factor=scale_factor,
            color=color
        )
else:
    print(">> Render Event Cloud (single color)")
    mlab.points3d(
        relative_points[:, 0],
        relative_points[:, 1],
        relative_points[:, 2],
        scale_mode='none',
        scale_factor=scale_factor,
        color=single_color
    )


# -------------------------------------------------------------------
# AESTHETICS
print(f'>> Plot Aesthetics')
# box around the data
#mlab.outline()     

# Compute axis bounds from current relative data
x0, x1 = X.min(), X.max()
y0, y1 = Y.min(), Y.max()
z0 = relative_points[:, 2].min()
z1 = relative_points[:, 2].max()

# Automatically determine a reasonable label scale
extent_diag = np.linalg.norm([x1 - x0, y1 - y0, z1 - z0])
label_scale = extent_diag * 0.015       # Or try 0.04 '352163''564               opop/ 0.05
tick_len = extent_diag * 0.01

# Draw Axes            
PF.draw_camera_relative_axes((x0, x1), (y0, y1), (z0, z1), tick_len=tick_len, label_scale=label_scale)

# Camera Adjustments:
mlab.view(azimuth=azimuth, elevation=elevation, distance=camdist)  # rotate camera around to front-left

# Save Fig
if save:
    mlab.savefig(outputfile, size=(1800, 1200))
    PF.save_plume_legend(plume_colors, output_path=legend_out)

# Visualize
print(f'Visualize')
mlab.show()
