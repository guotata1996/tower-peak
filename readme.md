# Peak Analysis Tool

This repository contains tools that computes EquivalentTowerHeight (ETH) for mountain peaks. 

## Features

- Automatic data collection and processing from ASTER Global Digital Elevation Model V003 database
- Grid-based peak contour apporximation
- Result visualization

## Usage

By default, `main.py` computes ETH for all peaks in `data\Combined.csv`.

You can also perform a single analysis on the peak of your choice.

```python
from main import analysis_peak

# Analyze Mt Everest at given coordinates
analysis_peak(
    rough_lat=27.99,     # Latitude
    rough_lon=86.92,     # Longitude
    true_elevation=8849, # Optional known elevation in meters, for data validation
)
```

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.