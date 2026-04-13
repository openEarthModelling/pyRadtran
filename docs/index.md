# pyRadtran

A complete Python wrapper for [libRadtran](https://www.libradtran.org) radiative transfer simulations.

## Quick Start

```python
from pyradtran import Scene, Runner

scene = (
    Scene()
    .set_atmosphere(profile="us", altitude=2.663)
    .set_source_solar(sza=30.0)
    .set_wavelength(250.0, 1200.0)
    .set_solver(method="disort", streams=16)
    .set_output(quantities=["lambda", "edir"], quantity="transmittance")
)

result = Runner.execute(scene, data_path="/usr/local/share/libRadtran/data")
result.edir.plot()
```
