# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 23:14:17 2019

@author: yuanq
"""

import xlwings as xw


@xw.func
@xw.arg('xl_app', vba='Application')
def get_range_names(xl_app):
    return [str(name).replace('=','') for name in xl_app.Caller.Parent.Parent.names]