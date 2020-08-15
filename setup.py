#!/usr/bin/env python
from setuptools import setup

setup(name='cellx',
      version='0.1',
      description='CellX libraries',
      author='Alan R. Lowe, Christopher Soelistyo, Laure Ho',
      author_email='a.lowe@ucl.ac.uk',
      url='https://github.com/quantumjot/cellx',
      packages=['cellx'],
      install_requires=['matplotlib',
                        'numpy',
                        'scikit-image',
                        'scikit-learn',
                        'scipy',
                        'tensorflow',
                        'tqdm'],
      python_requires='>=3.6'
     )
