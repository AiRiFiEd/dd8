# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 15:16:54 2019

@author: yuanq
"""
import enum

@enum.unique
class ENUM_OPTION_TYPE(enum.Enum):
    PUT = 1
    CALL = 2

class ENUM_OPTION_STYLE(enum.Enum):
    EUROPEAN = 1
    AMERICAN = 2