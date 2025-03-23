# 1. input peak coord -> map area to search
# 2. download from database if needed
# 3. search the x,y grid of peak (within a radius of 1km) -> print out max value found

import rasterio
from rasterio.transform import rowcol
import numpy as np
from math import floor, cos, pi
from itertools import product
from os.path import exists, join
import subprocess
import matplotlib.pyplot as plt


def get_center_grid(tif_file, lon, lat):
    # Open the DEM file
    with rasterio.open(tif_file) as dataset:
        # Convert lat/lon to row/col
        row, col = rowcol(dataset.transform, lon, lat)
        # Read elevation value
        ndarray = dataset.read(1)
        w, h = ndarray.shape
        need_east_tif = 0
        if col >= w:
            need_east_tif = 1
        elif col < 0:
            need_east_tif = -1
        need_north_tif = 0
        if row > h:
            need_north_tif = -1
        elif row < 0:
            need_north_tif = 1
        if need_east_tif == 0 and need_north_tif == 0:
            # elevation = dataset.read(1)[row, col]
            return True, row, col
        else:
            return False, need_east_tif, need_north_tif

def get_grid_data(lat: int, lon: int):
    code = lat_lon_to_code(lat, lon)
    tif_file = join('download', f'ASTGTMV003_{code}_dem.tif')
    with rasterio.open(tif_file) as dataset:
        return dataset.read(1)

def get_max_elevation(tif_file):
    with rasterio.open(tif_file) as dataset:
        elevation_data = dataset.read(1)  # Read the first (and only) band
        print(elevation_data.shape)
        max_elevation = np.nanmax(elevation_data)  # Get max, ignoring NaNs if present
    return max_elevation

def get_array_block(arr:np.ndarray, row_range, col_range):
    row_lo, row_hi = row_range
    col_lo, col_hi = col_range
    if row_hi == -1:
        row_hi = None
    else:
        row_hi += 1
    if col_hi == -1:
        col_hi = None
    else:
        col_hi += 1
    return arr[row_lo: row_hi, col_lo: col_hi]

def lat_lon_to_code(lat, lon):
    lat_i = floor(lat)
    lon_i = floor(lon)
    ns = 'N' if lat_i >= 0 else 'S'
    ew = 'E' if lon_i >= 0 else 'W'
    lat_s = str(abs(lat_i))
    while len(lat_s) < 2:
        lat_s = '0' + lat_s
    lon_s = str(abs(lon_i))
    while len(lon_s) < 3:
        lon_s = '0' + lon_s
    code = ns + lat_s + ew + lon_s
    return code

SEARCH_RADIUS_DEGREE = 0.05 # degree, ~ 5km
SEARCH_RADIUS_GRIDS = int(SEARCH_RADIUS_DEGREE * 3600) - 1 # 1 grid = 1/3600 degree


def plot_terrain(elevation_data):
    # TODO: plot grids above threshold
    # Get the shape of the elevation data
    rows, cols = elevation_data.shape

    # Generate coordinate grids
    x = np.linspace(0, cols, cols)  # Simulated longitude indices
    y = np.linspace(0, rows, rows)  # Simulated latitude indices
    X, Y = np.meshgrid(x, y)

    # Create 3D figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface with a colormap
    ax.plot_surface(X, Y, elevation_data, cmap='terrain', edgecolor='none')

    # Labels and title
    ax.set_xlabel('Longitude Index')
    ax.set_ylabel('Latitude Index')
    ax.set_zlabel('Elevation (m)')
    ax.set_title('3D Terrain Visualization')

    plt.show()


def solution(lat, lon):
    lat_is = [lat - SEARCH_RADIUS_DEGREE, lat + SEARCH_RADIUS_DEGREE]
    lon_is = [lon - SEARCH_RADIUS_DEGREE, lon + SEARCH_RADIUS_DEGREE]
    request_grids = set() # e.g. 'N01E001'
    for lat_i, lon_i in product(lat_is, lon_is):
        request_grids.add(lat_lon_to_code(lat_i, lon_i))

    for code in request_grids:
        expected = f'ASTGTMV003_{code}_dem.tif'
        if exists(join('download', expected)):
            print("Reuse", expected, ":already in database")
        else:
            print("Downloading", expected)

            with open('download\\download_one_template.sh', 'r') as f:
                download_script = f.read()
            download_script = download_script.replace('<>', expected)
            with open('download\\download_one.sh', 'w') as f:
                f.write(download_script)
            subprocess.run(["C:\\Program Files\\Git\\bin\\bash.exe", 'download_one.sh'], cwd="download")
            # TODO: move this downward

    center_code = lat_lon_to_code(lat, lon)
    center_tif = join('download', f'ASTGTMV003_{center_code}_dem.tif')
    success, row, col = get_center_grid(center_tif, lon, lat)

    # col < 0: need to go west
    col_range_r = col - SEARCH_RADIUS_GRIDS, col + SEARCH_RADIUS_GRIDS # inclusive
    col_range_w = 0, 2 * SEARCH_RADIUS_GRIDS

    west_col_range_r = None
    west_col_range_w = None
    east_col_range_r = None
    east_col_range_w = None
    if col_range_r[0] < 0:
        west_col_range_r = col_range_r[0], -1
        west_col_range_w = 0, -col_range_r[0] - 1
        col_range_r = 0, col_range_r[1]
        col_range_w = west_col_range_w[1] + 1, 2 * SEARCH_RADIUS_GRIDS
    elif col_range_r[1] > 3600:
        east_col_range_r = 0, col_range_r[1] - 3601
        east_col_range_w = 2 * SEARCH_RADIUS_GRIDS - (col_range_r[1] - 3601), 2 * SEARCH_RADIUS_GRIDS
        col_range_r = col_range_r[0], 3600
        col_range_w = 0, east_col_range_w[0] - 1

    # row < 0: need to go north
    row_range_r = row - SEARCH_RADIUS_GRIDS, row + SEARCH_RADIUS_GRIDS # inclusive
    row_range_w = 0, 2 * SEARCH_RADIUS_GRIDS
    north_row_range_r = None
    north_row_range_w = None
    south_row_range_r = None
    south_row_range_w = None

    if row_range_r[0] < 0:
        north_row_range_r = row_range_r[0], -1
        north_row_range_w = 0, -row_range_r[0] - 1
        row_range_r = 0, row_range_r[1]
        row_range_w = north_row_range_w[1] + 1, 2 * SEARCH_RADIUS_GRIDS
    elif row_range_r[1] > 3600:
        south_row_range_r = 0, row_range_r[1] - 3601
        south_row_range_w = 2 * SEARCH_RADIUS_GRIDS - (row_range_r[1] - 3601), 2 * SEARCH_RADIUS_GRIDS
        row_range_r = row_range_r[0], 3600
        row_range_w = 0, south_row_range_w[0] - 1

    # =============# =============
    # Index verification
    # =============# =============

    col_count_r = col_range_r[1] - col_range_r[0] + 1
    col_count_w = col_range_w[1] - col_range_w[0] + 1
    assert col_count_r > 0
    assert col_count_w > 0

    # Verify col range
    if west_col_range_r is not None:
        west_count_r = west_col_range_r[1] - west_col_range_r[0] + 1
        west_count_w = west_col_range_w[1] - west_col_range_w[0] + 1

        assert west_count_r > 0
        assert west_count_w > 0
        assert west_count_r + col_count_r == 2 * SEARCH_RADIUS_GRIDS + 1
        assert west_count_w + col_count_w == 2 * SEARCH_RADIUS_GRIDS + 1

    elif east_col_range_r is not None:
        east_count_r = east_col_range_r[1] - east_col_range_r[0] + 1
        east_count_w = east_col_range_w[1] - east_col_range_w[0] + 1

        assert east_count_r > 0
        assert east_count_w > 0
        assert east_count_r + col_count_r == 2 * SEARCH_RADIUS_GRIDS + 1, str(east_count_r + col_count_r)
        assert east_count_w + col_count_w == 2 * SEARCH_RADIUS_GRIDS + 1, str(east_count_w) + " + " + str(col_count_w)

    else:
        assert col_count_r == 2 * SEARCH_RADIUS_GRIDS + 1, str(col_count_r)
        assert col_count_w == 2 * SEARCH_RADIUS_GRIDS + 1, str(col_count_r)

    # Verify row range
    row_count_r = row_range_r[1] - row_range_r[0] + 1
    row_count_w = row_range_w[1] - row_range_w[0] + 1
    assert row_count_r > 0
    assert row_count_w > 0
    if north_row_range_r is not None:
        north_count_r = north_row_range_r[1] - north_row_range_r[0] + 1
        north_count_w = north_row_range_w[1] - north_row_range_w[0] + 1

        assert north_count_r > 0
        assert north_count_w > 0
        assert row_count_r + north_count_r == 2 * SEARCH_RADIUS_GRIDS + 1
        assert row_count_w + north_count_w == 2 * SEARCH_RADIUS_GRIDS + 1, str(row_count_w) + " + " + str(north_count_w)

    elif south_row_range_r is not None:
        south_count_r = south_row_range_r[1] - south_row_range_r[0] + 1
        south_count_w = south_row_range_w[1] - south_row_range_w[0] + 1

        assert south_count_r > 0
        assert south_count_w > 0
        assert row_count_r + south_count_r == 2 * SEARCH_RADIUS_GRIDS + 1
        assert row_count_w + south_count_w == 2 * SEARCH_RADIUS_GRIDS + 1
    else:
        assert row_count_r == 2 * SEARCH_RADIUS_GRIDS + 1
        assert row_count_w == 2 * SEARCH_RADIUS_GRIDS + 1

    # Get data from center
    collected = np.empty([2 * SEARCH_RADIUS_GRIDS + 1, 2 * SEARCH_RADIUS_GRIDS + 1], dtype=np.int16)
    collected[:,:] = np.iinfo(np.int16).min

    center_data = get_grid_data(lat, lon)
    get_array_block(collected, row_range_w, col_range_w)[:,:] = \
        get_array_block(center_data, row_range_r, col_range_r)

    # Get data from N
    if north_row_range_r is not None:
        print("Need N")
        n_data = get_grid_data(lat + 1, lon)
        get_array_block(collected, north_row_range_w, col_range_w)[:, :] = \
            get_array_block(n_data, north_row_range_r, col_range_r)

    # Get data from NW
    if west_col_range_r is not None and north_row_range_r is not None:
        print("Need NW")
        nw_data = get_grid_data(lat + 1, lon - 1)
        get_array_block(collected, north_row_range_w, west_col_range_w)[:, :] = \
            get_array_block(nw_data, north_row_range_r, west_col_range_r)

    # Get data from W
    if west_col_range_r is not None:
        print("Need W")
        w_data = get_grid_data(lat, lon - 1)
        get_array_block(collected, row_range_w, west_col_range_w)[:,:] = \
            get_array_block(w_data, row_range_r, west_col_range_r)

    # Get data from SW
    if west_col_range_r is not None and south_row_range_r is not None:
        print("Need SW")
        sw_data = get_grid_data(lat - 1, lon - 1)
        get_array_block(collected, south_row_range_w, west_col_range_w)[:,:] = \
            get_array_block(sw_data, south_row_range_r, west_col_range_r)

    # Get data from S
    if south_row_range_r is not None:
        print("Need S")
        s_data = get_grid_data(lat - 1, lon)
        get_array_block(collected, south_row_range_w, col_range_w)[:, :] = \
            get_array_block(s_data, south_row_range_r, col_range_r)

    # Get data from SE
    if south_row_range_r is not None and east_col_range_r is not None:
        print("Need SE")
        se_data = get_grid_data(lat - 1, lon + 1)
        get_array_block(collected, south_row_range_w, east_col_range_w)[:,:] = \
            get_array_block(se_data, south_row_range_r, east_col_range_r)

    # Get data from E
    if east_col_range_r is not None:
        print("Need E")
        e_data = get_grid_data(lat, lon + 1)
        get_array_block(collected, row_range_w, east_col_range_w)[:,:] = \
            get_array_block(e_data, row_range_r, east_col_range_r)

    # Get data from NE
    if east_col_range_r is not None and north_row_range_r is not None:
        print("Need NE")
        ne_data = get_grid_data(lat + 1, lon + 1)
        get_array_block(collected, north_row_range_w, east_col_range_w)[:,:] = \
            get_array_block(ne_data, north_row_range_r, east_col_range_r)

    assert not np.iinfo(np.int16).min in collected
    peak_elevation = np.max(collected)
    print("Max elevation:", peak_elevation)

    xs, ys = np.where(collected == peak_elevation)
    peak_x = xs[0]
    peak_y = ys[0]
    threshold = peak_elevation - 500
    cut_edges = set() # (small_x,y, larger_x,y): have ends >= & < threshold, respectively
    calc_area_grids = set() # (bottom-left corner x,y): has at least one corner >= threshold

    discovered = {(peak_x, peak_y)}  # those >= threshold
    discovery_queue = [(peak_x, peak_y)]  # those >= threshold
    while discovery_queue:
        node_x, node_y = discovery_queue.pop()
        calc_area_grids.add((node_x, node_y))
        calc_area_grids.add((node_x - 1, node_y))
        calc_area_grids.add((node_x, node_y - 1))
        calc_area_grids.add((node_x - 1, node_y - 1))

        for offset_x, offset_y in [(1,0), (-1,0), (0,1), (0,-1)]:
            neighbor_x = node_x + offset_x
            neighbor_y = node_y + offset_y
            # TODO: handle out-of-border
            if collected[neighbor_x, neighbor_y] >= threshold:
                if (neighbor_x, neighbor_y) not in discovered:
                    discovery_queue.append((neighbor_x, neighbor_y))
                    discovered.add((neighbor_x, neighbor_y))
            else:
                n1 = (node_x, node_y)
                n2 = (neighbor_x, neighbor_y)
                cut_edges.add((min(n1, n2), max(n1, n2)))

    total_area = 0
    for x, y in calc_area_grids:
        local_edges = {((x, y), (x, y + 1)),
                     ((x, y), (x + 1, y)),
                     ((x, y + 1), (x + 1, y + 1)),
                     ((x + 1, y), (x + 1, y + 1))}
        local_cut_edges = local_edges & cut_edges
        if len(local_cut_edges) == 0:
            assert collected[x, y] >= threshold
            assert collected[x+1, y] >= threshold
            assert collected[x, y+1] >= threshold
            assert collected[x+1, y+1] >= threshold
            total_area += 1
        elif len(local_cut_edges) == 2:
            (n1, n2), (n3, n4) = list(local_cut_edges)
            n_set = {n1, n2, n3, n4}
            if len(n_set) == 4:
                # trapezoid
                portion_12 = (max(collected[*n1], collected[*n2]) - threshold) / abs(collected[*n1] - collected[*n2])
                portion_34 = (max(collected[*n3], collected[*n4]) - threshold) / abs(collected[*n3] - collected[*n4])
                local_area = (portion_12 + portion_34) / 2
                assert 0 <= local_area <= 1
                total_area += local_area
            elif len(n_set) == 3:
                nc = {n1, n2} & {n3, n4} # find right angle vertex
                na = {n1, n2} - nc
                nb = {n3, n4} - nc
                nc, na, nb = [list(x)[0] for x in (nc, na, nb)]
                portion_ca = (max(collected[*nc], collected[*na]) - threshold) / abs(collected[*nc] - collected[*na])
                portion_cb = (max(collected[*nc], collected[*nb]) - threshold) / abs(collected[*nc] - collected[*nb])
                if collected[nc] >= threshold:
                    assert collected[na] < threshold
                    assert collected[nb] < threshold
                    # triangle
                    local_area = portion_ca * portion_cb / 2
                    assert 0 <= local_area <= 1/2
                else:
                    assert collected[na] >= threshold
                    assert collected[nb] >= threshold
                    # rect - triangle
                    local_area = 1 - (1 - portion_ca) * (1 - portion_cb) / 2
                    assert 1/2 <= local_area <= 1
                total_area += local_area
            else:
                assert False
        elif len(local_cut_edges) == 4:
            print("4 cut edge: Rare case, noise in data?")
            total_area += 1 / 2
        else:
            assert False, str(len(local_cut_edges)) + " grid-of-interest can only border even cut edges!"

    meters_per_second = 30.92 * cos(lat * pi / 180)
    total_area *= pow(meters_per_second, 2) / 1e6
    print("Area above", threshold, '=', total_area, 'km2')
    plot_terrain(collected)

if __name__ == '__main__':
    solution(27.99, 86.92) # Everest, 1.24
    # solution(35.88, 76.52) # K2, 0.60
    # solution(27.70, 88.15) # Kan, 0.78
    # solution(29.63, 95.055) # Namcha Barwa, 0.55
    # solution(29.59, 101.88) # Konka, 0.92