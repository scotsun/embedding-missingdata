"""Python script."""

# import os

# os.system("cd ~/Desktop/Duke\ Univeristy\ MB\ Program/research/embedding/python-r")
# os.system("Rscript ./rscript.r 5 100")
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# Loading R session and R package
r = robjects.r
utils = importr("utils")

dataframe = utils.read_csv("../embedding_missingdata/car.data", header=False)
r = robjects.r

# Loading the function we have defined in R.
r["source"]("./rfun.r")
rfun = robjects.globalenv["myshape"]
rfun(dataframe)

# convert R df to pd.DataFrame
with localconverter(robjects.default_converter + pandas2ri.converter):
    df = robjects.conversion.rpy2py(dataframe)
