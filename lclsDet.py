""" this module defines the LCLS specific detectors and how to map them as abstract 
    detectorsdetectors
"""
import functools
import numpy as np
import time

import x3py.lclsSpecialDet as lclsSpecialDet
import x3py.toolsVarious   as toolsVarious
from   x3py.abstractDet    import Detector

class Hdf5Detector(object):
  def __init__(self,mne,calibs,time_field="/time"):
    self.name = mne
    self.calibs = calibs
    self._time_field = time_field
    self.nCalib = len(calibs)
    self._initialize()

  def _initialize(self):
    """ Understand where the data are ... (/image, /data, etc) """
    test=["data","image","channelValue"]
    keys = self._getCalibPointer(0,what=None)
    for t in test:
      if t in keys:
        self._data_field = t
        break

  def _getCalibPointer(self,calib,what=None):
    h5,path = self.calibs[calib]
    if what is None:
      pass
    elif what == "time":
      path = path + "/" + self._time_field
    elif what == "data":
      path = path + "/" + self._data_field
    return h5[path]

  def getDataPointer(self,calib):
    return self._getCalibPointer(calib,what="data")

  def getTimeStampPointer(self,calib):
    return self._getCalibPointer(calib,what="time")

  def getData(self,calib,shotSlice=None):
    data = self.getDataPointer(calib)
    if shotSlice is None:
      return data[...]
    else:
      return data[shotSlice]

  @functools.lru_cache(maxsize=10000)
  def _getTimeStamp(self,calib):
    """ this function is just to allow caching ... that does not work with
        slices """
    time = self.getTimeStampPointer(calib)
    return time[...]

  def getTimeStamp(self,calib,shotSlice=None):
    time = self._getTimeStamp(calib)
    if shotSlice is None:
      return time[...]
    else:
      return time[shotSlice]

class StructuredArrayDetector(object):
  def __init__(self,mne,calibs,parent=None):
    if parent is None:
      self.parent = Hdf5Detector(mne,calibs)
    else:
      self.parent = parent
    self.name = mne
    self.nCalib = self.parent.nCalib
    data = self.parent.getDataPointer(0)
    for name in data.dtype.names:
      getData = functools.partial(self.getData,what=name)
      self.__dict__[name] = Detector(mne+"."+name,getData,
        self.getTimeStamp,parent=self,dtype=data.dtype)

  @functools.lru_cache(maxsize=10000)
  def _getData(self,calib):
    return self.parent.getData(calib)

  @functools.lru_cache(maxsize=10000)
  def _getTimeStamp(self,calib):
    return self.parent.getTimeStamp(calib)

  def getData(self,calib,shotSlice=None,what=None):
    data = self._getData(calib)
    if what is not None: data = data[what]
    if shotSlice is not None: data=data[shotSlice]
    return data

  def getTimeStamp(self,calib,shotSlice=None):
    time = self._getTimeStamp(calib)
    if shotSlice is not None: time=time[shotSlice]
    return time

def defineDetector(mne,calibs):
  c=toolsVarious.CodeBlock('Defining %s...'%mne)
  t0 = time.time()
  det =  Hdf5Detector(mne,calibs)
  data = det.getDataPointer(0)
  # order is important because eventCode is also a strctureArray
  if mne.find("event")>-1:
    det = lclsSpecialDet.EventCode(mne,det)
    print("..(as eventCode detector, use .eventCode.autoDiscovery() if needed)..",end="")
  elif mne.lower().find("timetool")>-1:
    det = lclsSpecialDet.TimeTool(mne,det)
    print("..(as timetool detector)..",end="")
  elif toolsVarious.isStructuredArray(data):
    det = StructuredArrayDetector(mne,calibs,parent=det)
  else:
    det = Hdf5Detector(mne,calibs)
    dtype = det.getDataPointer(0).dtype
    det = Detector(mne,det.getData,det.getTimeStamp,parent=det,dtype=dtype)
    print("..(as general detector)..",end="")
  c.done()
  return det
 
