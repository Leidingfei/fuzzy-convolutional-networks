
import pandas as pd
import numpy as np


DATA_DIR = "data/"




def load_data():
	
	sgx = pd.read_csv("{}sti.csv".format(DATA_DIR))
	print(sgx.head())
	gspc = pd.read_csv("{}GSPC.csv".format(DATA_DIR))
	print(gspc.head())


