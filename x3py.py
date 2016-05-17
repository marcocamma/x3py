""" This is a light version of the x3py package. Can simply acess the
   data """
import os
import numpy as np

from x3py.toolsConf import config
from x3py import toolsMatchTimeStamps,toolsVarious,toolsOS,toolsDetectors
import h5py
import x3py.lclsH5
import x3py.abstractDet

class Dataset(x3py.lclsH5.H5):
  config = config
  def __init__(self,
    inputFilesOrExpRunTuple='',
    matchTimeStamps=True,
    detectors = "all",
    exclude_dets = None,
    load_cache   = True
    ):
    x3py.lclsH5.H5.__init__(self,inputFilesOrExpRunTuple,detectors=detectors,\
    exclude=exclude_dets)
    setattr(config,"h5handles",self.h5); # useful for getting this info in other classes
    if load_cache: self.loadCache()
    if matchTimeStamps: self._matchTimeStamps()

  @property
  def detectors(self):
    return toolsDetectors.findDetectorsInsideObj(self)

  # methods for making the Dataset iterables
  def __iter__(self):
    self._itercount = -1
    self._iteritems = list(self.detectors.keys())
    return self
  def __next__(self):
    self._itercount+=1
    if self._itercount < len(self._iteritems):
      return self._iteritems[self._itercount]
    else:
      raise StopIteration

  def loadCache(self,path=None):
    if path is None: path= os.path.join(config.cachepath,os.path.basename(self.h5[0].filename))
    # find files
    files = toolsOS.getCMD("find %s -name abstractDetector.npy"%path)
    files.sort( key=len ); # to have child (like .profile.timeTool) after .profile
    relfiles = [os.path.relpath(f,path) for f in files]
    for f,relf in zip(files,relfiles):
      dets = relf.split(os.path.sep)
      # make sure intermediate layers are available
      start = self
      for d in dets[:-2]:
        if not hasattr(start,d): setattr(start,d,toolsVarious.DropObject(name=d))
        start = getattr(start,d)
      parent = start if start != self else None
      toolsDetectors.wrapArray(dets[-2],f,parent=parent)
      #self.detectors[dets[-1]]=det
      #setattr(start,dets[-2],det); done by wraparray

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

  def get(self,x): return eval('self.%s' % x)

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
      d.defineTimeStampFilter(f)
    c.done()

  def addFilter(self,name,boolIdx,detectors=None):
    """ Add the filter defined by boolIdx to all detectors """
    if detectors is None: detectors=self.detectors
    for d in detectors.values(): d.addFilter(name,boolIdx)
