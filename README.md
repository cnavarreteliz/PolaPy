# PolaPy

[![PyPI version](https://badge.fury.io/py/polapy.svg)](https://badge.fury.io/py/polapy)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

**PolaPy** (Polarization for Python) is a collection of algorithmic implementations of polarization and electoral competition metrics in Python.

## Features

- 📊 **Polarization Metrics**: Esteban-Ray, Reynal-Querol, Wang-Tsui, and Electoral Divisiveness
- 🗳️ **Electoral Competition**: Blais-Lago, Grofman-Selb, and Laakso-Taagepera indices
- 📈 **Electoral Systems**: D'Hondt method and proportional allocation
- 🐼 **Pandas Integration**: Works seamlessly with DataFrames

## Installation

```bash
pip install polapy
```

## Quick Start

### Polarization Metrics

```python
import pandas as pd
from polapy.polarization import esteban_ray, reynal_querol, electoral_divisiveness

# Esteban-Ray polarization
df = pd.DataFrame({'pi': [0.5, 0.5], 'y': [1, 2]})
value = esteban_ray(df, alpha=0.5)

# Reynal-Querol polarization
df = pd.DataFrame({'rate': [0.4, 0.3, 0.2, 0.1]})
value = reynal_querol(df)

# Electoral Divisiveness
df = pd.DataFrame({
    'unit': ['A', 'A', 'B', 'B'],
    'candidate': ['X', 'Y', 'X', 'Y'],
    'votes': [100, 50, 60, 90]
})
value, details = electoral_divisiveness(df)
```

### Electoral Competition

```python
from polapy.competitiveness import blais_lago, grofman_selb, laackso_taagepera

# Effective Number of Parties
df = pd.DataFrame({'share': [0.4, 0.35, 0.25]})
enp = laackso_taagepera(df)

# Blais-Lago competition index
df = pd.DataFrame({'party': ['A', 'B', 'C'], 'votes': [5000, 3000, 2000]})
value, details = blais_lago(df, n_seats=5)
```

## Available Metrics

| Module | Metric | Reference |
|--------|--------|-----------|
| `polarization` | `electoral_divisiveness` | Navarrete et al. (2023) |
| `polarization` | `esteban_ray` | Esteban & Ray (1994) |
| `polarization` | `reynal_querol` | Reynal-Querol (2002) |
| `polarization` | `wang_tsui` | Wang & Tsui (2000) |
| `competitiveness` | `blais_lago` | Blais & Lago (2009) |
| `competitiveness` | `grofman_selb` | Grofman & Selb (2009) |
| `competitiveness` | `laackso_taagepera` | Laakso & Taagepera (1979) |
| `aggregate` | `dhondt` | D'Hondt/Jefferson method |
| `aggregate` | `proportional` | Largest Remainder Method |

## Requirements

- Python ≥ 3.7
- numpy
- pandas
- scipy

## About

PolaPy is developed and maintained by researchers at the University of Concepción (Chile). For questions, issues, or contributions, visit our [GitHub repository](https://github.com/cnavarreteliz/polapy).

## License

MIT License - see [LICENSE.md](LICENSE.md) for details.
