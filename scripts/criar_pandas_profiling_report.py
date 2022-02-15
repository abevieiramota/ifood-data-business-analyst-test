#!/usr/bin/env python
# coding: utf-8

from toolbelt import *
from pandas_profiling import ProfileReport
import os

# dados base

df = read_data()

profile = ProfileReport(df, title="iFood Data Analyst Case", minimal=True)

profile.to_file(os.path.join(REPORTS_DIR, "1 - dados_base.html"))

# features iteração 1

df2 = apply_feature_engineering_1(df)

profile = ProfileReport(df2, title="iFood Data Analyst Case", minimal=True)

profile.to_file(os.path.join(REPORTS_DIR, "2 - feature_engineering_1.html"))