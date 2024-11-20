# This script is for trying the code written in the model.ipynb file inside neovim for neovim practice.

import polars as pl # Written in rust to make use of the borrowing for memeory efficiency.
import pandas as pd # Pandas for 
import matplotlib.pyplot as plt
import math
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from jormund import *

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

#%matplotlib inline

setup_plots()

spy = pl.read_csv('spy_analysis/spy.csv')
spy = spy.with_columns([
    pl.col("Date").str.strptime(pl.Date, "%Y-%m-%d").alias("date")
])
spy = spy.drop('Date')
train = spy.filter(pl.col("Year") < 2024)
test = spy.filter(pl.col("Year") == 2024)

#display(train)
#display(train.describe())


