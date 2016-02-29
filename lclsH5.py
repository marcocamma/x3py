import h5py
import functools
import re
import numpy as np

import x3py.lclsDet as lclsDet
from   x3py.toolsLog import log
from   x3py.toolsConf import config
from   x3py.toolsVarious import iterfy,DropObject

memory = config.joblibcache
g_epicsSignature = "Epics::EpicsPv"

@functools.lru_cache(maxsize=None)
@memory.cache
def getSubFields(h5data,regex="/time$",returnAbsolute=False,returnHandle=False):
  if isinstance(h5data,str): h5data = h5py.File(h5data,"r")
  subFolders = []
  h5data.visit(subFolders.append)
  if returnAbsolute:
    subFolders = ["%s/%s" % (h5data.name,f) for f in subFolders]
  if returnHandle:
    subFolders = [ (h5data,f) for f in subFolders ]
  return subFolders

def filterList(list,regex,strip=False):
  r = re.compile(regex)
  if strip:
    return [r.sub("",l) for l in list if r.search(l) is not None ]
  else:
    return [l for l in list if r.search(l) is not None ]

def makeLists(h5handles,regex="/time$",strip=True):
  pathlist = filterList(getSubFields(h5handles[0]),regex,strip=strip)
  pathlist.sort()
  h5list   = (h5handles[0],)*len(pathlist)
  for h5 in h5handles[1:]:
    temp = filterList(getSubFields(h5),regex,strip=strip)
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
  def __init__(self,fnames,mode="r",driver=None):
    fnames  = iterfy(fnames)
    self.h5 = [h5py.File(fname,mode=mode,driver=driver) for fname in fnames]
    self._h5list,self._pathlist = makeLists(self.h5)
    self.findDetectors()
    self.findScan()
    

  def findScan(self,conf=config):
    regex = conf.scan
    r = regex.replace("*","\S+")
    h5,path=makeLists(self.h5,r,strip=False)
    v = np.asarray( [h[p]["value"] for (h,p) in zip(h5,path)] )
    n = np.asarray( [h[p]["name"]  for (h,p) in zip(h5,path)] )
    self.__dict__["scan"] = DropObject()
    for i in range(n.shape[1]):
      name = n[0,i].decode()
      self.__dict__["scan"]._add(name,v[:,i])

  def findDetectors(self,conf=config):
    detList = conf.detectors
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
          log.warning("Overwriting detector",mne)
          #log.warning("Warning: in findDetectors, more than one templates matches %s",patterns)
        self.__dict__[mne] = lclsDet.defineDetector(mne,calibs)
