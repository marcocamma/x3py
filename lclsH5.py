import h5py
import functools
import re
import numpy as np

import x3py.lclsDet as lclsDet
from   x3py.toolsLog import log
from   x3py.toolsConf import config
from   x3py.toolsVarious import iterfy,DropObject,CodeBlock,filterList
from   x3py import lclsSpecialDet

memory = config.joblibcache

@memory.cache
def visitH5(h5file):
  subFolders = []
  h5data = h5py.File(h5file,"r")
  subFolders = []
  h5data.visit(subFolders.append)
  return subFolders

#@functools.lru_cache(maxsize=None)
def getSubFields(h5data,basepath="Configure:0000/Run:0000",
  regex="/time$",returnAbsolute=False,returnHandle=False,strip=True):
  if isinstance(h5data,str):
    h5filename = h5data
    h5data = h5py.File(h5filename,"r")
  else:
    h5filename = h5data.filename
  subFolders = visitH5(h5filename)
  if basepath is not None: subFolders=filterList(subFolders,basepath,strip=False)
  if regex    is not None: subFolders=filterList(subFolders,regex,strip=strip)
  if returnAbsolute:
    subFolders = ["%s/%s" % (h5data.name,f) for f in subFolders]
  if returnHandle:
    subFolders = [ (h5data,f) for f in subFolders ]
  return subFolders

def makeLists(h5handles,regex="/time$",strip=True):
  pathlist = getSubFields(h5handles[0],regex=regex,strip=strip)
  pathlist.sort()
  h5list   = (h5handles[0],)*len(pathlist)
  for h5 in h5handles[1:]:
    temp = getSubFields(h5,regex=regex,strip=strip)
    temp.sort()
    pathlist += temp
    h5list   += (h5,)*len(temp)
  return h5list,pathlist
  
class Dataset(object):
  def __init__(self,handle,pathTemplate):
    self.handle = handle
    self.pathTemplate = pathTemplate
    
  def get(self,calibList,shotSlice):
    path = [self.pathTemplate % c for c in calibList]
    return [self.handle[p] for p in path] 
    
class H5(object):
  def __init__(self,fnames,detectors="all",exclude=None,mode="r",driver=None):
    """ exclude : list of detectors to exclude """
    fnames  = iterfy(fnames)
    
    self.h5 = [h5py.File(fname,mode=mode,driver=driver) for fname in fnames]
    c = CodeBlock("Building paths list of HDF5 file(s)")
    self._h5list,self._pathlist = makeLists(self.h5)
    c.done()
    self._detectorsToLookFor = detectors
    self._detectorsToExclude = exclude
    self.findDetectors()
    try:
      self.findScan()
    # at least for xppa1714-r0107.h5, this is the error given when trying to open:
    #  e = h5[0]["Configure:0000/Run:0000/CalibCycle:0000/ControlData::ConfigV3/Control/pvControls"]
    # e[...]
    # *** OSError: Can't read data (Src and dest data spaces have different sizes)
    except OSError:
      log.warn("No scan variable could be found")
    self.epics = lclsSpecialDet.Epics(self.h5)
    

  def findScan(self,conf=config):
    regex = conf.scan
    r = regex.replace("*","\S+")
    h5,path=makeLists(self.h5,r,strip=False)
    v = np.asarray( [h[p]["value"] for (h,p) in zip(h5,path)] )
    n = np.asarray( [h[p]["name"]  for (h,p) in zip(h5,path)] )
    self.__dict__["scan"] = DropObject()
    for i in range(n.shape[1]):
      name = n[0,i].decode()
      setattr(self.scan,name,v[:,i])
      setattr(self.scan,"scanmotor%d"%i,name)
      setattr(self.scan,"scanmotor%d_values"%i,v[:,i])

  def findDetectors(self,conf=config):
    dets = self._detectorsToLookFor
    detsconf = conf.detectorsToMatch
    if dets == "all":
      detList = conf.detectorsToMatch
    else:
      dets = iterfy(dets)
      detList = dict( [(regex,det) for (regex,det) in conf.detectorsToMatch.items() if det in dets] )
    if self._detectorsToExclude is not None:
      exl = iterfy(self._detectorsToExclude)
      detList = dict( [(regex,det) for (regex,det) in detList.items() if det not in exl ] )
    h5list   = self._h5list
    pathlist = self._pathlist
    # TODO: this part is not very efficient since I loop for every detector 
    # trough the entire list ...
    for regex,mne in detList.items():
      calibs = []
      r    = re.compile( regex.replace("*","\S+") )
      for h5,path in zip(h5list,pathlist):
        if r.search(path) is not None:
          calibs.append( (h5,path) )
      #log.info("Found %s (regex: %s)",mne,r)
      if len(calibs) > 0:
        if mne in self.__dict__:
          log.warning("Overwriting detector %s, pattern %s",mne,regex)
          #log.warning("Warning: in findDetectors, more than one templates matches %s",patterns)
        self.__dict__[mne] = lclsDet.defineDetector(mne,calibs)
