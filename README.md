[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/quantumjot/cellx/actions/workflows/cellx.yml/badge.svg)](quantumjot/cellx/actions)

# CellX

Code for CellX and related projects.

---

### Local installation

```sh
git clone https://github.com/quantumjot/cellx.git
cd cellx
pip install -e .
```

### Pull requests/Contributions
See the contributing [guide](CONTRIBUTING.md).


### Running in a Docker container
Build the image:
```sh
docker build . -t cellx/cellx:latest
```

Run a local script using the container:
```sh
docker run -it --runtime=nvidia  --rm -v $PWD:/tmp -w /tmp cellx/cellx:latest python ./script.py
```
---

### Contributors
* Alan R. Lowe (quantumjot, arl)
* T. L. Laure Ho (laureho, tllh)
* Christopher J. Soelistyo (chris-soelistyo, cjs)
* Kristina Ulicna (KristinaUlicna, ku)
