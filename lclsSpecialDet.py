""" Module to handle specific detectors like eventcodes, timetool, etc """
import functools
import numpy as np

from x3py.abstractDet import Detector

class EventCode(object):
  ""
  def __init__(self,mne,parent,autoDefine=True):
    self.name = mne
    self.parent = parent
    self.nCalib = parent.nCalib
    self._out   = dict()
    for c in range(self.nCalib):
      self._out[c] = dict()
      self._out[c] = dict()
    self._foundCodes = []

    if autoDefine: [self._getData(i) for i in range(self.nCalib)]

  def getDet(self,code):
    """ return detector associated with a given code, defining it if needed """
    return self._defineDetector(code)

  def _getData(self,calib,shotSlice=None,code=140):
    if code not in self._out[calib]:
      data = self.parent.getData(calib)
      nShots = data.shape[0]
      for i in range(nShots):
        codeList = data[i]["fifoEvents"]["eventCode"]
        for code in codeList:
          if code not in self._out[calib]:
            self._out[calib][code] =  np.zeros(nShots,dtype=np.bool)
          self._out[calib][code][i] = True
      # add discovered keys
      for code in self._out[calib].keys():
        if code not in self._foundCodes:
          self._foundCodes.append(code)
          self._defineDetector(code)
    if shotSlice is not None:
      return self._out[calib][code][shotSlice]
    else:
      return self._out[calib][code]

  def _defineDetector(self,code):
    mne = "code_%d" % code
    if mne not in self.__dict__:
      getData      = functools.partial(self._getData,code=code)
      getTimeStamp = self.parent.getTimeStamp
      d = Detector(mne,getData,getTimeStamp,parent=self)
      self.__dict__[mne] = d
    return self.__dict__[mne]
