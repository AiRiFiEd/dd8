# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:30:15 2019

@author: yuanq
"""

import setuptools
from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('modStats3.pyx'))