import xml.etree.ElementTree as ET
import re

def parse_kml(file_path):
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

    return results


if __name__ == "__main__":
    file_path = "HighestPeaks.kml"  # Change this to your actual file path
    places = parse_kml(file_path)

    for name, elevation, coords in places:
        assert len(name) > 0
        assert elevation > 2000.0, name
        assert len(coords) == 2
        # print(f"Name: {name}\nMeters: {elevation} m\nCoordinates: {coords}\n")