import rasterio
from rasterio.transform import rowcol
import numpy as np
from math import floor
from os.path import exists, join
import subprocess

def get_dataset_at(lat: float, lon: float):
    # Retrieve or download dem.tif
    # Return: file path
    code = lat_lon_to_code(lat, lon)
    expected = f'ASTGTMV003_{code}_dem.tif'
    if exists(join('download', expected)):
        # print("Reuse", expected, ":already in database")
        ...
    else:
        print("Downloading", expected)

        with open('download\\download_one_template.sh', 'r') as f:
            download_script = f.read()
        download_script = download_script.replace('<>', expected)
        with open('download\\download_one.sh', 'w') as f:
            f.write(download_script)
        subprocess.run(["C:\\Program Files\\Git\\bin\\bash.exe", 'download_one.sh'], cwd="download")
    return join('download', expected)

def get_h_array_at(lat: float, lon: float):
    # Returns: numpy.ndarray shape=(3601, 3601)
    downloaded = get_dataset_at(lat, lon)
    if not exists(downloaded):
        print("Download failed. Assuming sea level (all zero).")
        return np.zeros((3601, 3601), dtype=np.int16)
    with rasterio.open(downloaded) as dataset:
        return dataset.read(1)

def get_row_col_at(lat: float, lon: float):
    # Convert lat/lon to row/col
    with rasterio.open(get_dataset_at(lat, lon)) as dataset:
        row, col = rowcol(dataset.transform, lon, lat)
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
            return row, col
        assert False, f"Need a different tif: {need_east_tif} to east, {need_north_tif} to north!"

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

def collect_heightmap(lat: float, lon: float, radius: int):
    # Return: np.ndarray[2 * radius + 1, 2 * radius + 1]

    # col < 0: need to go west
    row, col = get_row_col_at(lat, lon)
    col_range_r = col - radius, col + radius  # inclusive
    col_range_w = 0, 2 * radius

    west_col_range_r = None
    west_col_range_w = None
    east_col_range_r = None
    east_col_range_w = None
    if col_range_r[0] < 0:
        west_col_range_r = col_range_r[0], -1
        west_col_range_w = 0, -col_range_r[0] - 1
        col_range_r = 0, col_range_r[1]
        col_range_w = west_col_range_w[1] + 1, 2 * radius
    elif col_range_r[1] > 3600:
        east_col_range_r = 0, col_range_r[1] - 3601
        east_col_range_w = 2 * radius - (col_range_r[1] - 3601), 2 * radius
        col_range_r = col_range_r[0], 3600
        col_range_w = 0, east_col_range_w[0] - 1

    # row < 0: need to go north
    row_range_r = row - radius, row + radius  # inclusive
    row_range_w = 0, 2 * radius
    north_row_range_r = None
    north_row_range_w = None
    south_row_range_r = None
    south_row_range_w = None

    if row_range_r[0] < 0:
        north_row_range_r = row_range_r[0], -1
        north_row_range_w = 0, -row_range_r[0] - 1
        row_range_r = 0, row_range_r[1]
        row_range_w = north_row_range_w[1] + 1, 2 * radius
    elif row_range_r[1] > 3600:
        south_row_range_r = 0, row_range_r[1] - 3601
        south_row_range_w = 2 * radius - (row_range_r[1] - 3601), 2 * radius
        row_range_r = row_range_r[0], 3600
        row_range_w = 0, south_row_range_w[0] - 1

    # ==========================
    # Index verification
    # ==========================

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
        assert west_count_r + col_count_r == 2 * radius + 1
        assert west_count_w + col_count_w == 2 * radius + 1

    elif east_col_range_r is not None:
        east_count_r = east_col_range_r[1] - east_col_range_r[0] + 1
        east_count_w = east_col_range_w[1] - east_col_range_w[0] + 1

        assert east_count_r > 0
        assert east_count_w > 0
        assert east_count_r + col_count_r == 2 * radius + 1, str(east_count_r + col_count_r)
        assert east_count_w + col_count_w == 2 * radius + 1, str(east_count_w) + " + " + str(col_count_w)

    else:
        assert col_count_r == 2 * radius + 1, str(col_count_r)
        assert col_count_w == 2 * radius + 1, str(col_count_r)

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
        assert row_count_r + north_count_r == 2 * radius + 1
        assert row_count_w + north_count_w == 2 * radius + 1, str(row_count_w) + " + " + str(north_count_w)

    elif south_row_range_r is not None:
        south_count_r = south_row_range_r[1] - south_row_range_r[0] + 1
        south_count_w = south_row_range_w[1] - south_row_range_w[0] + 1

        assert south_count_r > 0
        assert south_count_w > 0
        assert row_count_r + south_count_r == 2 * radius + 1
        assert row_count_w + south_count_w == 2 * radius + 1
    else:
        assert row_count_r == 2 * radius + 1
        assert row_count_w == 2 * radius + 1

    # Get data from center
    collected = np.empty([2 * radius + 1, 2 * radius + 1], dtype=np.int16)
    collected[:, :] = np.iinfo(np.int16).min

    center_data = get_h_array_at(lat, lon)
    get_array_block(collected, row_range_w, col_range_w)[:, :] = \
        get_array_block(center_data, row_range_r, col_range_r)

    # Get data from N
    if north_row_range_r is not None:
        n_data = get_h_array_at(lat + 1, lon)
        get_array_block(collected, north_row_range_w, col_range_w)[:, :] = \
            get_array_block(n_data, north_row_range_r, col_range_r)

    # Get data from NW
    if west_col_range_r is not None and north_row_range_r is not None:
        nw_data = get_h_array_at(lat + 1, lon - 1)
        get_array_block(collected, north_row_range_w, west_col_range_w)[:, :] = \
            get_array_block(nw_data, north_row_range_r, west_col_range_r)

    # Get data from W
    if west_col_range_r is not None:
        w_data = get_h_array_at(lat, lon - 1)
        get_array_block(collected, row_range_w, west_col_range_w)[:, :] = \
            get_array_block(w_data, row_range_r, west_col_range_r)

    # Get data from SW
    if west_col_range_r is not None and south_row_range_r is not None:
        sw_data = get_h_array_at(lat - 1, lon - 1)
        get_array_block(collected, south_row_range_w, west_col_range_w)[:, :] = \
            get_array_block(sw_data, south_row_range_r, west_col_range_r)

    # Get data from S
    if south_row_range_r is not None:
        s_data = get_h_array_at(lat - 1, lon)
        get_array_block(collected, south_row_range_w, col_range_w)[:, :] = \
            get_array_block(s_data, south_row_range_r, col_range_r)

    # Get data from SE
    if south_row_range_r is not None and east_col_range_r is not None:
        se_data = get_h_array_at(lat - 1, lon + 1)
        get_array_block(collected, south_row_range_w, east_col_range_w)[:, :] = \
            get_array_block(se_data, south_row_range_r, east_col_range_r)

    # Get data from E
    if east_col_range_r is not None:
        e_data = get_h_array_at(lat, lon + 1)
        get_array_block(collected, row_range_w, east_col_range_w)[:, :] = \
            get_array_block(e_data, row_range_r, east_col_range_r)

    # Get data from NE
    if east_col_range_r is not None and north_row_range_r is not None:
        ne_data = get_h_array_at(lat + 1, lon + 1)
        get_array_block(collected, north_row_range_w, east_col_range_w)[:, :] = \
            get_array_block(ne_data, north_row_range_r, east_col_range_r)

    assert not np.iinfo(np.int16).min in collected
    assert collected.shape[0] == collected.shape[1]
    assert collected.shape[0] % 2 == 1
    return collected