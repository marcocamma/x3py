""" This is a light version of the x3py package. Can simply acess the
   data """
from x3py.toolsConf import config
import x3py.lclsH5

class Dataset(x3py.lclsH5.H5):
  def __init__(self,
    inputFilesOrExpRunTuple='',
    detectors = [],
    ):
    x3py.lclsH5.H5.__init__(self,inputFilesOrExpRunTuple)
