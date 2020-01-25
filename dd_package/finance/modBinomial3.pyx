# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 00:17:40 2019

@author: yuanq
"""

import numpy as np
import math

from modFinanceEnums3 import ENUM_OPTION_STYLE, ENUM_OPTION_TYPE

class JarrowRudd(object):
    def __init__(self, int_binomial_steps):
        self.binomial_steps = int_binomial_steps
        self.option_type = None
        self.option_style = None
        self.initial_price = None
        self.strike_price = None
        self.time_to_expiry = None
        self.volatility = None
        self.risk_free_rate = None
        
        
    def price(enum_option_type,
                 enum_option_style,
                 dbl_initial_price, 
                 dbl_strike_price, 
                 dbl_time_to_expiry, 
                 dbl_volatility,
                 dbl_risk_free_rate, 
                 ):
        