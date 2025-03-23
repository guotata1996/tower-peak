# 1. input peak coord -> map area to search
# 2. download from database if needed
# 3. search the x,y grid of peak (within a radius of 1km) -> print out max value found
import itertools
import numpy as np
from math import cos, pi
from plotting import plot_elevation
from dataset_utils import collect_heightmap

# from discovered peak, search area within radius
SEARCH_RADIUS_DEGREE = 0.04 # degree, ~ 10km
SEARCH_RADIUS_GRIDS = int(SEARCH_RADIUS_DEGREE * 3600) - 1 # 1 grid = 1/3600 degree

def area_above_height(rough_lat, rough_lon, max_elevation_drop=500,
                      true_elevation=None, discover_radius_degree=0.05):
    # from supplied lat, lon, discover peak:= highest point within discover_radius
    discover_radius_grids = int(discover_radius_degree * 3600) - 1
    discover_area = collect_heightmap(rough_lat, rough_lon, discover_radius_grids)
    peak_elevation = np.max(discover_area)
    if true_elevation is not None and \
            (peak_elevation > true_elevation or peak_elevation < true_elevation * 0.95):
        print(f"Error: Collected peak elevation ({peak_elevation}) is far from truth value ({true_elevation})")
        return None

    xs, ys = np.where(discover_area == peak_elevation)
    peak_x = xs[0]
    peak_y = ys[0]
    offset_row = discover_radius_grids - peak_x
    offset_col = discover_radius_grids - peak_y
    lat = rough_lat + offset_row / 3600
    lon = rough_lon - offset_col / 3600
    print("Use peak elevation:", peak_elevation, "at", lat, lon)

    collected = collect_heightmap(lat, lon, SEARCH_RADIUS_GRIDS)
    peak_x = SEARCH_RADIUS_GRIDS
    peak_y = SEARCH_RADIUS_GRIDS
    threshold = peak_elevation - max_elevation_drop
    cut_edges = set() # (small_x,y, larger_x,y): have ends >= & < threshold, respectively
    calc_area_grids = set() # (bottom-left corner x,y): has at least one corner >= threshold. x/y range: [0,2N)

    discovered = {(peak_x, peak_y)}  # those >= threshold
    discovery_queue = [(peak_x, peak_y)]  # those >= threshold
    while discovery_queue:
        node_x, node_y = discovery_queue.pop()
        if collected[node_x, node_y] > peak_elevation:
            print(f"Supplied {max_elevation_drop} is beyond peak prominence. Abort!")
            return None

        for gx, gy in itertools.product([node_x, node_x-1], [node_y, node_y-1]):
            if 0 <= gx < 2 * SEARCH_RADIUS_GRIDS and \
                0 <= gy < 2 * SEARCH_RADIUS_GRIDS:
                calc_area_grids.add((gx, gy))

        for offset_x, offset_y in [(1,0), (-1,0), (0,1), (0,-1)]:
            neighbor_x = node_x + offset_x
            neighbor_y = node_y + offset_y
            if neighbor_x < 0 or neighbor_x >= 2 * SEARCH_RADIUS_GRIDS + 1 or \
                    neighbor_y < 0 or neighbor_y >= 2 * SEARCH_RADIUS_GRIDS + 1:
                print("Warning: Search area out of collected range!")
                continue

            if collected[neighbor_x, neighbor_y] >= threshold:
                if (neighbor_x, neighbor_y) not in discovered:
                    discovery_queue.append((neighbor_x, neighbor_y))
                    discovered.add((neighbor_x, neighbor_y))
            else:
                n1 = (node_x, node_y)
                n2 = (neighbor_x, neighbor_y)
                cut_edges.add((min(n1, n2), max(n1, n2)))

    total_area = 0
    face_color = np.zeros([2 * SEARCH_RADIUS_GRIDS, 2 * SEARCH_RADIUS_GRIDS, 3])
    face_color[:,:,:] = [0, 1, 0]
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
            face_color[x, y] = [0, 0, 1]
        elif len(local_cut_edges) == 4:
            print("4 cut edge: Rare case, noise in data?")
            total_area += 1 / 2
            face_color[x, y] = [1, 0, 1]
        else:
            assert False, f"grid-of-interest at {(x,y)} can only border even cut edges, but got {len(local_cut_edges)}"

    meters_per_grid = 30.92 * cos(lat * pi / 180)
    total_area *= pow(meters_per_grid, 2) / 1e6
    print("Area above", threshold, '=', total_area, 'km2')
    plot_elevation(collected, face_color)
    return total_area

if __name__ == '__main__':
    # North-East
    # area_above_height(27.98, 86.92, 500) # Everest, 1.24
    # area_above_height(35.88, 76.52, 500) # K2, 0.60
    # area_above_height(27.70, 88.15, 500) # Kan, 0.78
    area_above_height(27.96, 86.93, 500,
                      true_elevation=8516, discover_radius_degree=0.02) # Lhotse, 1.22

    # North-West
    # area_above_height(63.07, -151.01, 500) # McKinley, 1,55

    # South-East
    # area_above_height(-9.12, -77.61, 500) # Huascaran, 2.62