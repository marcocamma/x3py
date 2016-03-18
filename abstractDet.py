""" this module defines abstract detector class. To define a detector we need 
    2 functions:
     1. getData(calib,shotSlice=optional) -> must return data
     2. getTimeStamp(calib,shotSlice=optional) -> as above for timestamp
    We need also:
     3. the number of calib (can be inferred from the parent detector if given)
     4. the dtype of the single shot (useful to understand shape and memoryusage).
        If dtype is not given, one shot is read and the shape/size read from it
    The abstract detector class will take care of raveling the indices to provide
    access independent on the calibcycle
"""
import functools
import numpy as np
import time
from x3py.toolsLog import log
from x3py import toolsVarious
from x3py.toolsConf import config

class DetectorFilter(object):
  def __init__(self,det,boolIdx):
    self._det = det
    self._boolidx = boolIdx
    self._idx     = np.argwhere(boolIdx).ravel()

  def _getData(self,x):
    return self._det.getShots(self._idx[x],what="data")

  def _getTime(self,x):
    return self._det.getShots(self._idx[x],what="time")


  def __getitem__(self,x):
    return self._getData(x)

class Detector(object):
  """ Abstract detector class, should be smart enough to handle 0D,1D and 2D data
      (caching based on size); """
  _datacache = None
  _timecache = None
  def __init__(self,mne,getData,getTimeStamp,isOkFilter=None,nCalib=None,
    parent=None,dtype=None):
    self.name = mne
    self._getData = getData
    self._getTimeStamp = getTimeStamp
    if parent is None:
      self.nCalib = nCalib
    else:
      self.nCalib = parent.nCalib
    if self.nCalib is None:
      log.warning("In det %s,nCalib is None",mne)
    # get the number of shots per calibcycle
    self._unFilteredLens = \
      np.asarray( [getTimeStamp(i).shape[0] for i in range(self.nCalib) ] )
    self._unFilteredCumLens = np.cumsum(self._unFilteredLens)
    self._unFilteredNShots = np.sum(self._unFilteredLens)
    # define filter
    self.defineFilter(isOkFilter)
    self.parent   = parent
    # register detector in database
#    if parent is not None:
#      fullname = parent.name + "." + mne
#    else:
#      fullname = mne
    config.detectors[mne] = self
    self.dtype    = dtype

    # try to determine internal hdf5 chunking
    if parent is None:
      self._hdf5Chunks = 1
    else:
      det = self
      while ( "parent" in det.__dict__):
        det = det.__dict__["parent"]
      _calib = det.getDataPointer(0)
      if hasattr(_calib,"chunks") and (_calib.chunks is not None):
        self._hdf5Chunks = _calib.chunks[0]
      else:
        self._hdf5Chunks = 1

    # filters container
    self.filter = toolsVarious.DropObject()
      
    # what follows is for determining shapes and sizes...
    if dtype is not None:
      self.itemshape = dtype.shape
      self.itemsize  = dtype.itemsize
    else:
      oneShot = getData(0,0)
      self.itemshape = oneShot.shape
      self.itemsize  = oneShot.nbytes
    self._calibsizes = self.itemsize*self._unFilteredLens
    self._datasetGB   = np.sum(self._calibsizes)/1024**3

  @property
  def data(self):
    return getitemWrapper(self,"data")

  @property
  def time(self):
    return getitemWrapper(self,"time")

  def getCalibs(self,calibSlice=None,shotSlice=None,what="data",ravel=True):
    nC = self.nCalib
    getD = self._getData
    getT = self._getTimeStamp
    slices = ( calibSlice,shotSlice )
    itemshape = self.itemshape
    if   what == "data":
      return _interpretSlices(getD,slices,nC,ravel,itemshape)
    elif what == "time":
      return _interpretSlices(getT,slices,nC,ravel,itemshape)
    else:
      data = _interpretSlices(getD,slices,nC,ravel,itemshape)
      time = _interpretSlices(getT,slices,nC,ravel,itemshape)
      return data,time

  def getShots(self,shotSlice,what="data"):
    """ this function returns the shots independently of the calibcycle. In 
        other words if the first calibcycle has 100 shots, the shot 101 will
        be the first one of the second calibcycle;
        The idea behind such kind of access is data storage independent of
        calibcycles """
    shotSlice = self._applyFilterToShotSlice(shotSlice)
    if what == "time":
      if self._timecache is None:
        self._timecache = self.getCalibs(what="time",ravel=True)
      ret = self._timecache[shotSlice]
    else:
      if self._datasetGB < config.cachesizeGB:
        if self._datacache is None:
          self._datacache = self.getCalibs(what="data",ravel=True)
        ret = self._datacache[shotSlice]
      elif self.nCalib == 1:
        ret = self._getData(0,shotSlice)
      else:
        args,nShotsToRead = self._shotSliceToCalibIndices(shotSlice)
        # check read limit
        toRead = nShotsToRead*self.itemsize/1024/1024/1024
        if toRead > config.readlimitGB:
          msg  = "You are trying to read %.2f Gb in one go!!" % toRead
          msg += ",change the readlimit_GB (currently at %.2f Gb)" % \
                 config.readlimitGB
          msg += " in the configuration file if you are sure"
          raise ValueError(msg)
        data = [self.getCalibs(*a,what=what,ravel=True) for a in args]
        if len(data) == 1:
          ret = data[0]
        else:
          ret = np.concatenate( data )
    return ret

  def addFilter(self,name,isOk):
    dfilter = DetectorFilter(self,isOk)
    self.filter._add(name,dfilter)
    

  def defineFilter(self,isOkFilter):
    """ Define filter to use, pass None to not use any filter """
    if isOkFilter is not None:
      self._isOk  = np.argwhere(isOkFilter).ravel()
      self.nShots = self._isOk.shape[0]
    else:
      self._isOk  = None
      self.nShots = self._unFilteredNShots

  def chunks(self,sizeGB=0.5,n=None):
    if n is None:
      itemsize = self.itemsize/1024**3
      n = int(sizeGB/itemsize)
    # round up in multiples of chunks
    if self._hdf5Chunks > 1:
      n = (n // self._hdf5Chunks) * self._hdf5Chunks
    return toolsVarious.chunk(self.nShots,n)

  def _applyFilterToShotSlice(self,shotSlice):
    if self._isOk is not None:
      shotSlice = self._isOk[shotSlice]
    return shotSlice

  def _shotSliceToCalibIndices(self,shotSlice):
    """ helper function to 'partition' shotSlice into calibcycles; 
        input (shotSlice) could be a range, list, booleans, slice """
    shotSlice = toolsVarious.toList(shotSlice,self._unFilteredNShots)
    args = [ [] for i in range(self.nCalib) ]
    nShotsToRead = 0
    for s in shotSlice:
      calib,shotInCalib =  _fromShotIndexToCalibShot(s,self._unFilteredCumLens)
      args[calib].append(shotInCalib)
      nShotsToRead += 1
    # exclude empyy calibs
    args = [ [i,a] for (i,a) in enumerate(args) if len(a)> 0 ]
    # try to put together read of the same calib; this speed up compressed cspad ...
    args = [ (calib,toolsVarious.tryToSlice(listOfShots)) for (calib,listOfShots) in args ]
    return args,nShotsToRead
 
  def __getitem__(self,x):
    return self.getShots(x,what="data")

class getitemWrapper(object):
  def __init__(self,x,what="data"):
    self.x = x
    self.what=what
  def __getitem__(self,x):
    return self.x.getShots(x,what=self.what)

def _fromShotIndexToCalibShot(shotNum,calibsCumSum):
  if shotNum < calibsCumSum[0]:
    calib,shot=0,shotNum
  else:
    calib = int( np.argwhere(shotNum//calibsCumSum==0)[0] )
    shot  = shotNum-calibsCumSum[calib-1]
  return calib,shot

def _interpretSlices(reader,x,nCalib,ravel,shotShape=(1,)):
  """ ok this function is not very elegant ... the goal was to have a consistent
      indexing, if ravel is False it returns (calib,shot,shape); otherwise if 
      ravel is True calib, the shots are combined in a single index; tested for
      cspad and ipm """
#  print("in interpret",x)
  if isinstance(x,tuple) or isinstance(x,list):
    calib,shots = x
  else:
    shots = None
  if isinstance(calib,slice):
    calib = range(*calib.indices(nCalib))
  elif calib is None:
    calib = range(nCalib)
  else:
    calib = toolsVarious.iterfy(calib)
  temp = []
  # read calibs
  for c in calib:
    d = np.atleast_1d( reader(c,shots) )
    if d.ndim>1 and d.ndim == len(shotShape):
      d=d[np.newaxis,:]; # has to have a shot index !
    temp.append(d)
  if ravel and len(temp)==1:
    temp = temp[0]; # avoids useless mamory copy
  elif ravel and len(temp) > 1:
    temp = np.concatenate( temp )
  else:
    temp = np.stack( temp )
  return temp
