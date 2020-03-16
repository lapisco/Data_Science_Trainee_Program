import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path

from feature_extraction_signal.src import feature_extraction

DATAFOLDER = path.join('..', '..', 'data')
DATAFILES = [
    'v000_FAULT_SC_HI_LVL2_FR6000_FG5942_L000_0,6IN_SENSORC.csv',
    'v000_FAULT_SC_LI_LVL3_FR4500_FG4365_L000_0,8IN_SENSORC.csv',
    'v000_FAULT_SC_LI_LVL3_FR6000_FG5927_L000_0,4IN_SENSORC.csv',
    'v000_NORMAL_FR4500_FG4385_L000_1,0IN_SENSORC.csv',
    'v000_NORMAL_FR6000_FG5955_L000_0,5IN_SENSORC.csv'
]

SEED = 1987298712

data_normal = pd.read_csv(path.join(DATAFOLDER, DATAFILES[4]))

fe_fourier = feature_extraction.Fourier(fundamental=60.0, fs=5000.0, harmonics=(0.5, 1, 1.5, 3, 5, 7))

out_fourier = fe_fourier.transform(data_normal['Current_R'])