# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 22:59:42 2019

@author: yuanq
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
      ext_modules=cythonize('modBlackScholesModel3.pyx', annotate=True)
      )