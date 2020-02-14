# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 23:37:15 2019

@author: yuanq
"""

from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.tsa.seasonal
import astropy
import pmdarima
import enum

# custom packages
from .. import modUtils3 as utils

logger = utils.get_basic_logger(__name__,utils.logging.DEBUG,utils.logging.INFO)

@enum.unique
class ENUM_LSSA_METHOD(enum.Enum):
    LOMB_SCARGLE = 1
    VANICEK = 2

class TimeSeries(object):
    def __init__(self, df_y, df_X=None, bln_is_ascending=True):
        self.__df_y = df_y
        self.__df_X = df_X
        self.__bln_is_ascending = bln_is_ascending    
    
    def __repr__(self):
        pass
    
    def __len__(self):
        return self.__df_y.shape[0]
    
    def is_stationary(self):
        pass
    
    def seasonal_decompose(self,
                           str_model='additive', 
                           arr_filt=None, 
                           int_freq=None, 
                           bln_two_sided=True, 
                           int_extrapolate_trend=0):
        return statsmodels.tsa.seasonal.seasonal_decompose(self.__df_y,
                                                           model=str_model, 
                                                           filt=arr_filt, 
                                                           freq=int_freq, 
                                                           two_sided=bln_two_sided, 
                                                           extrapolate_trend=int_extrapolate_trend)
    
    
    def auto_arima(self, lst_start_params=None):
        return pmdarima.auto_arima(self.__df_y, seasonal=True, m=365, transparams=False, start_params=lst_start_params).summary()
    
    def gen_evenly_spaced_series(self, 
                                 enum_method = ENUM_LSSA_METHOD.LOMB_SCARGLE,
                                 **kwargs):
        if enum_method == ENUM_LSSA_METHOD.LOMB_SCARGLE:
            if '3.1' in astropy.__version__:
                pass
            elif '3.2' in astropy.__version__:
                if kwargs is not None:
                    model = astropy.timeseries.LombScargle(self.__npa_x, 
                                                           self.__npa_y, 
                                                           **kwargs)
                else:
                    model = astropy.timeseries.LombScargle(self.__npa_x,
                                                           self.__npa_y)
        