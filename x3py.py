""" This is a light version of the x3py package. Can simply acess the
   data """
from x3py.toolsConf import config
from x3py import toolsMatchTimeStamps,toolsVarious
import x3py.lclsH5
import numpy as np

class Dataset(x3py.lclsH5.H5):
  config = config
  def __init__(self,
    inputFilesOrExpRunTuple='',
    matchTimeStamps=True,
    detectors = [],
    ):
    x3py.lclsH5.H5.__init__(self,inputFilesOrExpRunTuple)
    self.detectors = config.detectors
    if matchTimeStamps: self._matchTimeStamps()

  def _matchTimeStamps(self,detectorList=None):
    c=toolsVarious.CodeBlock("Time stamp matching started ...")
    if detectorList is None:
      dets = self.detectors.values()
    else:
      dets = detectorList
    # TODO This should be improved since many detectors share the same time timestamps
    # (for example all 'fields' of IPMs, all event codes
    idx = toolsMatchTimeStamps.matchTimeStamps(*dets,returnCommonTimeStamp=False)
    for (d,f) in zip(dets,idx):
      d.defineFilter(f)
    c.done()
    
