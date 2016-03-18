""" This is a light version of the x3py package. Can simply acess the
   data """
from x3py.toolsConf import config
from x3py import toolsMatchTimeStamps,toolsVarious
import x3py.lclsH5
import x3py.abstractDet
import numpy as np

class Dataset(x3py.lclsH5.H5):
  config = config
  def __init__(self,
    inputFilesOrExpRunTuple='',
    matchTimeStamps=True,
    detectors = "all",
    exclude_dets = None,
    ):
    x3py.lclsH5.H5.__init__(self,inputFilesOrExpRunTuple,detectors=detectors,\
    exclude=exclude_dets)
    self.detectors = config.detectors
    if matchTimeStamps: self._matchTimeStamps()

  def _matchTimeStampsTOFINISH(self,detectorList=None):
    # TODO: to be done: find kids and apply filter
    c=toolsVarious.CodeBlock("Time stamp matching started ...")
    # find parents; set is used for unique list; TODO: what about orphans ?
    parents = list(set([di.parent for di in self.detectors.values() if di.parent is not None]))
    # make sure parents don't have grandparents ...
    for i,d in enumerate(parents):
      if hasattr(d,"parent") and (d.parent is not None): parents[i] = d.parent
    # det time stamp for each parent ... it is a bit of an hack ...
    times = [ np.hstack([d.getTimeStamp(i) for i in range(d.nCalib)]) for d in parents ]
    #print(len(times))
    idx = toolsMatchTimeStamps.matchTimeStamps(*times,returnCommonTimeStamp=False)
    for (p,filter) in zip(parents,idx):
      kids = [d for d in p.__dict__.values() if isinstance(d,x3py.abstractDet.Detector) ]
      print(p,kids)
      for k in kids: k.defineFilter(filter)
    c.done()
    
  def _matchTimeStamps(self,detectorList=None):
    c=toolsVarious.CodeBlock("Time stamp matching started ...")
    if detectorList is None:
      dets = self.detectors.values()
    else:
      dets = detectorList
    #print(len(dets))
    # TODO This should be improved since many detectors share the same time timestamps
    # (for example all 'fields' of IPMs, all event codes
    idx = toolsMatchTimeStamps.matchTimeStamps(*dets,returnCommonTimeStamp=False)
    for (d,f) in zip(dets,idx):
      d.defineFilter(f)
    c.done()
    
