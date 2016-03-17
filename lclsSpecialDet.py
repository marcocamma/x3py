""" Module to handle specific detectors like eventcodes, timetool, etc """
import functools
import numpy as np
import re

from x3py.abstractDet import Detector
from x3py.toolsLog import log


class EventCode(object):
  ""
  def __init__(self,mne,parent,autoDiscovery=False):
    self.name = mne
    self.parent = parent
    self.nCalib = parent.nCalib
    self._out   = dict()
    for c in range(self.nCalib):
      self._out[c] = dict()
      self._out[c] = dict()
    self._foundCodes = []
    if autoDiscovery: self.autoDiscovery()

  def getDetCode(self,code):
    """ return detector associated with a given code, defining it if needed """
    return self._defineDetector(code)

  def autoDiscovery(self):
    [self._getData(i) for i in range(self.nCalib)]

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
    if code not in self._foundCodes:
      log.warn("Asked for eventcode %d, but it could not be found",code)
    return self.__dict__[mne]

class TimeTool(object):
  ""
  def __init__(self,mne,parent):
    self.name = mne
    self.parent = parent
    self._regex = dict (
      ampl = ":AMPL$",
      pos  = ":FLTPOS$",
      fwhm = ":FLTPOSFWHM$"
    )
    dets = self._regex.keys()
    calibs = dict( [ (det,self._getCalibForSubField(det)) for det in dets ] )
    self._calibs = calibs
    self.nCalib = len(calibs["ampl"])
    for det in dets:
      try:
        self._defineDetector(det,calibs[det])
      except IndexError:
        log.warn("Could not define timetool det %s, found %d calibs",det,len(calibs[det]))
    


  def _getCalibForSubField(self,what="ampl"):
    calibs = self.parent.calibs
    regex  = self._regex[what]
    r = re.compile(regex)
    # calibs are list of tuples, each tuple is (h5handle,key)
    c = [ cc for cc in calibs if r.search(cc[1]) is not None ]
    return c


  def _defineDetector(self,mne,calibs):
    def getData(calib,shotslice=None):
      h,k = calibs[calib]
      if shotslice is None:
        data = h[k]["data"][...]["value"]
      else:
        data = h[k]["data"][shotslice]["value"]
      return data
    def getTimeStamp(calib,shotslice=None):
      h,k = calibs[calib]
      if shotslice is None:
        time = h[k]["time"][...]
      else:
        time = h[k]["time"][shotslice]
      return time
    d = Detector(self.name+"."+mne,getData,getTimeStamp,parent=self)
    self.__dict__[mne] = d
    return self.__dict__[mne]
