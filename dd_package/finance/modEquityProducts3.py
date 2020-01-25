# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:48:08 2019

@author: LIM YUAN QING
"""

import re
from ..modUtils3 import  get_basic_logger
from ..modGlobal3 import LOG_PRINT_LEVEL, LOG_WRITE_LEVEL, TODAY

logger = get_basic_logger(__name__, LOG_PRINT_LEVEL)

class Derivative(object):
    def __init__(self, int_trade_id = None,
                 dte_trade_date = None,
                 dte_issue_date = None,
                 dte_final_valuation_date = None,
                 dte_maturity_date = None,
                 str_currency = None,                 
                 str_underlying = None,
                 dbl_notional = None):
        self.trade_id = int_trade_id
        if dte_trade_date:
            self.trade_date = dte_trade_date
        else:
            self.trade_date = TODAY
            
        self.issue_date = dte_issue_date
        
        if dte_final_valuation_date:
            self.final_valuation_date = dte_final_valuation_date
        else:
            logger.error('final valuation date missing...')
            
        self.maturity_date = dte_maturity_date
        if str_currency:
            self.currency = str_currency
        else:
            logger.error('currency missing...')
        delimiter = re.findall('[^\w\s]', str_underlying)
        
        if not delimiter:
            self.underlying = str_underlying
        elif len(set(delimiter)) == 1:
            self.underlying = [und for und in str_underlying.split(delimiter)
                                    if und != ' ']
        else:
            logger.error('multiple delimiters found in `str_underlying`.') 
            
        if dbl_notional:
            self.notional = dbl_notional
        
    def price(self):
        pass
    
    def risks(self):
        pass
    
    def gen_days(self):
        pass
    
    
    
    
    
class FixedCoupon(Derivative):
    def __init__(self, int_trade_id,
                 str_trade_date,
                 bln_is_note_form,
                 int_issue_delay,
                 str_underlying,
                 str_currency,
                 int_tenor,
                 dbl_coupon,
                 dbl_strike,
                 dbl_knock_out,                 
                 str_knock_out_freq,
                 bln_knock_out_mem,
                 dbl_knock_in,                 
                 str_knock_in_freq,
                 dbl_pv):        
        
        super().__init__(int_trade_id)
        self.trade_date = str_trade_date
        self.is_note_form = bln_is_note_form
        self.issue_delay = int_issue_delay       
        
        
        
    def get_trade_id(self):
        return self.trade_id
        
        
if __name__ == '__main__':
    FixedCoupon(123)