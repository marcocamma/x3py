""" this module defines the LCLS specific detectors and how to map them as abstract 
    detectorsdetectors
"""
import functools
import numpy as np
import time

from . import lclsSpecialDet
from . import toolsVarious
from   .abstractDet    import Detector

class Hdf5Detector(object):
  def __init__(self,mne,calibs,time_field="/time"):
    self.name = mne
    self.calibs = calibs
    self._time_field = time_field
    self.nCalib = len(calibs)
    self._initialize()

  def _initialize(self):
    """ Understand where the data are ... (/image, /data, etc) """
    test=["image","data","channelValue","encoder_values"]
    keys = self._getCalibPointer(0,what=None)
    for t in test:
      if t in keys:
        self._data_field = t
        break

  def _getCalibPointer(self,calib,what=None):
    h5,path = self.calibs[calib]
    if what == "time":
      path = path + "/" + self._time_field
    elif what == "data":
      path = path + "/" + self._data_field
    elif what is None:
      pass
    else:
      path = path + "/" + what
    if path in h5:
      return h5[path]
    else:
      None

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
    if parent is None:
      self.fullname = self.name
    else:
      fullname = parent.fullname if hasattr(parent,"fullname") else parent.name
      self.fullname = fullname
    self.nCalib = self.parent.nCalib
    data = self.parent.getDataPointer(0)
    for name in data.dtype.names:
      getData = functools.partial(self.getData,what=name)
      self.__dict__[name] = Detector(name,getData,
        self.getTimeStamp,parent=self,dtype=data.dtype)
    self._kids = data.dtype.names

  def __repr__(self):
    s  = "StructuredArrayDetector %s\n" % self.name
    for k in self._kids:
      d = getattr(self,k)
      s += "  |→ %s\n" % d.__str__()
    return s


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
    # autodiscovery s important for time stamp matching upon loading
    det = lclsSpecialDet.EventCode(mne,det,autoDiscovery=True)
    print("..(as eventCode detector)..",end="")
  elif mne.lower().find("timetool")>-1:
    det = lclsSpecialDet.TimeTool(mne,det)
    print("..(as timetool detector)..",end="")
  elif toolsVarious.isStructuredArray(data):
    det = StructuredArrayDetector(mne,calibs,parent=det)
  else:
    det = Hdf5Detector(mne,calibs)
    dtype = det.getDataPointer(0).dtype
    det = Detector(None,det.getData,det.getTimeStamp,parent=det,dtype=dtype)
    print("..(as general detector)..",end="")
  c.done()
  return det
 
