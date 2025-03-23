# 1. input peak coord -> map area to search
# 2. download from database if needed
# 3. search the x,y grid of peak (within a radius of 1km) -> print out max value found
import itertools
import numpy as np
from math import cos, pi, sqrt, atan
from plotting import plot_all
from dataset_utils import collect_heightmap

# from discovered peak, search area within radius
SEARCH_RADIUS_DEGREE_LIMIT = 0.30 # degree, ~ 30km
SEARCH_RADIUS_GRIDS_LIMIT = int(SEARCH_RADIUS_DEGREE_LIMIT * 3600) - 1 # 1 grid = 1/3600 degree


def calc_ngrids_above(collected: np.ndarray, peak_elevation, threshold):
    """
    :return: None if collected area doesn't fit satisfying area;
             0 if supplied threshold is below key col
             Number of grids around peak higher than threshold. Each grid is one sqruare arc sec.
    """
    data_radius = collected.shape[0] // 2
    peak_x = data_radius
    peak_y = data_radius

    cut_edges = set()  # (small_x,y, larger_x,y): have ends >= & < threshold, respectively
    calc_area_grids = set()  # (bottom-left corner x,y): has at least one corner >= threshold. x/y range: [0,2N)

    discovered = {(peak_x, peak_y)}  # those >= threshold
    discovery_queue = [(peak_x, peak_y)]  # those >= threshold
    while discovery_queue:
        node_x, node_y = discovery_queue.pop()
        if collected[node_x, node_y] > peak_elevation:
            print(f"Supplied threshold {threshold} is below peak col.")
            return None

        for gx, gy in itertools.product([node_x, node_x - 1], [node_y, node_y - 1]):
            if 0 <= gx < 2 * data_radius and \
                    0 <= gy < 2 * data_radius:
                calc_area_grids.add((gx, gy))

        for offset_x, offset_y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor_x = node_x + offset_x
            neighbor_y = node_y + offset_y
            if neighbor_x < 0 or neighbor_x >= 2 * data_radius + 1 or \
                    neighbor_y < 0 or neighbor_y >= 2 * data_radius + 1:
                # print("Search area out of collected range!")
                return 0

            if collected[neighbor_x, neighbor_y] >= threshold:
                if (neighbor_x, neighbor_y) not in discovered:
                    discovery_queue.append((neighbor_x, neighbor_y))
                    discovered.add((neighbor_x, neighbor_y))
            else:
                n1 = (node_x, node_y)
                n2 = (neighbor_x, neighbor_y)
                cut_edges.add((min(n1, n2), max(n1, n2)))

    total_area = 0
    face_color = np.zeros([2 * data_radius, 2 * data_radius, 3])
    face_color[:, :, :] = [0, 1, 0]
    for x, y in calc_area_grids:
        local_edges = {((x, y), (x, y + 1)),
                       ((x, y), (x + 1, y)),
                       ((x, y + 1), (x + 1, y + 1)),
                       ((x + 1, y), (x + 1, y + 1))}
        local_cut_edges = local_edges & cut_edges
        if len(local_cut_edges) == 0:
            assert collected[x, y] >= threshold
            assert collected[x + 1, y] >= threshold
            assert collected[x, y + 1] >= threshold
            assert collected[x + 1, y + 1] >= threshold
            total_area += 1
            face_color[x, y] = [1, 0, 0]
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
                nc = {n1, n2} & {n3, n4}  # find right angle vertex
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
                    assert 0 <= local_area <= 1 / 2
                else:
                    assert collected[na] >= threshold
                    assert collected[nb] >= threshold
                    # rect - triangle
                    local_area = 1 - (1 - portion_ca) * (1 - portion_cb) / 2
                    assert 1 / 2 <= local_area <= 1
                total_area += local_area
            else:
                assert False
            face_color[x, y] = [0, 0, 1]
        elif len(local_cut_edges) == 4:
            print("  Info: 4 cut edge, rare case. Noise in data.")
            total_area += 1 / 2
            face_color[x, y] = [1, 0, 1]
        else:
            assert False, f"grid-of-interest at {(x, y)} can only border even cut edges, but got {len(local_cut_edges)}"

    plot_data = (collected, face_color, f"ArcSec^2 above {threshold}: {total_area:.1f}")
    return total_area, plot_data

def analysis_peak(rough_lat, rough_lon, true_elevation=None,
                  alpha_threshold_deg=20.0, discover_radius_degree=0.05):
    # from supplied lat, lon, discover peak:= highest point within discover_radius
    discover_radius_grids = int(discover_radius_degree * 3600) - 1
    discover_area = collect_heightmap(rough_lat, rough_lon, discover_radius_grids)
    peak_elevation = np.max(discover_area)
    if true_elevation is not None and \
            (peak_elevation > true_elevation or peak_elevation < true_elevation * 0.95):
        print(f"Error: Collected peak elevation ({peak_elevation}) is far from truth value ({true_elevation})")
        return

    xs, ys = np.where(discover_area == peak_elevation)
    peak_x = xs[0]
    peak_y = ys[0]
    if not discover_radius_grids * 0.1 < peak_x < discover_radius_grids * 1.9 or \
       not discover_radius_grids * 0.1 < peak_y < discover_radius_grids * 1.9:
        print("Warning: peak found at the border of discover area. Please double check lat/lon.")
        return

    offset_row = discover_radius_grids - peak_x
    offset_col = discover_radius_grids - peak_y
    lat = rough_lat + offset_row / 3600
    lon = rough_lon - offset_col / 3600
    print("Use peak elevation:", peak_elevation, "at", lat, lon)

    search_radius = 150 # start from 0.04 arc second
    height_below = 300
    search_step = 200
    collected = collect_heightmap(lat, lon, search_radius)
    plots = []

    while height_below < peak_elevation and search_radius < SEARCH_RADIUS_GRIDS_LIMIT:
        threshold = peak_elevation - height_below
        result = calc_ngrids_above(collected, peak_elevation, threshold)
        if result is None:
            break
        if result == 0:
            search_radius += 100
            collected = collect_heightmap(lat, lon, search_radius)
            print("Increase search radius to", search_radius)
            continue

        total_area, plot_data = result
        plots.append(plot_data)
        meters_per_grid = 30.92 * cos(lat * pi / 180)
        total_area *= pow(meters_per_grid, 2)
        alpha_rad = atan(height_below / sqrt(total_area / pi))
        alpha_deg = alpha_rad / pi * 180
        print(threshold, ": Alpha=", alpha_deg, "degree")
        if alpha_deg < alpha_threshold_deg:
            break
        height_below += search_step

    print("Generating plots...")
    plot_all(plots)

if __name__ == '__main__':
    # North-East
    analysis_peak(27.98, 86.92) # Everest, 1.24
    # analysis_peak(35.88, 76.52) # K2, 0.60
    # analysis_peak(27.70, 88.15) # Kan, 0.78
    # analysis_peak(27.96, 86.93,
    #                 true_elevation=8516, discover_radius_degree=0.02) # Lhotse, 1.22

    # North-West
    # analysis_peak(63.07, -151.01) # McKinley, 1,55

    # South-East
    # analysis_peak(-9.12, -77.61) # Huascaran, 2.62