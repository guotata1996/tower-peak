import xml.etree.ElementTree as ET
import re

def kml_to_csv(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Define the KML namespace
    namespace = {'kml': 'http://earth.google.com/kml/2.1'}

    # Find all Placemark elements
    placemarks = root.findall(".//kml:Placemark", namespace)

    results = []
    for placemark in placemarks:
        name_raw = placemark.find("kml:name", namespace).text if placemark.find("kml:name",
                                                                            namespace) is not None else "Unknown"
        coordinates = placemark.find(".//kml:coordinates", namespace).text.strip() if placemark.find(
            ".//kml:coordinates", namespace) is not None else "Unknown"
        if coordinates == "Unknown":
            print("Skip name", name_raw, "due to invalid coordinates")
        lon, lat, _ = coordinates.split(',')
        coordinates = (float(lat), float(lon))
        # Extract name and elevation
        match = re.search(r"\d+\.\s*(.*?)\s*\((\d+\.?\d*)\s*m\)", name_raw)
        if match:
            name = match.group(1)
            meters = float(match.group(2))
        else:
            print("Skip name", name_raw, "due to invalid elevation")
            continue

        results.append((name, meters, coordinates))

    with open('Highest.csv', 'w') as out:
        for name, h, (lat, long) in results:
            out.write(f"{name},{float(lat)},{float(long)},{float(h)}\n")
    return results

def convert_wiki_csv(file_path):
    with open(file_path, 'r') as f:
        with open('MostProminent.csv', 'w') as out:
            for line in f.readlines()[1:]:
                name, coord, h = line.split(',')
                coord = coord[:coord.index('(')]
                coord = coord.split('/')[-1]
                lat, long = coord.split(';')

                out.write(f"{name},{float(lat)},{float(long)},{float(h)}\n")

def parse_csv(file_path):
    results = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            name, lat, lon, elevation = line.split(',')
            results.append((name, float(elevation), (float(lat), float(lon))))
    return results

def parse_coord_input(file_path):
    if file_path.endswith('kml'):
        return kml_to_csv(file_path)
    elif file_path.endswith('csv'):
        return parse_csv(file_path)
    assert False


if __name__ == "__main__":
    file_path = "data\\HighestPeaks.kml"
    places = parse_coord_input(file_path)

    for name, elevation, coords in places:
        assert len(name) > 0
        assert elevation > 2000.0, name
        assert len(coords) == 2
        print(f"Name: {name}\nMeters: {elevation} m\nCoordinates: {coords}\n")