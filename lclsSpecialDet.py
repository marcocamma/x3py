""" Module to handle specific detectors like eventcodes, timetool, etc """
import functools
import collections
import os
import numpy as np
import re

from .abstractDet import Detector
from .toolsLog import log
from .toolsConf import config
from . import toolsVarious

class EpicsPV(Detector):
  def __init__(self,mne,datasets):
    self.name = mne
    self._datasets = datasets
    self.nCalib = len(datasets)

  def defineDet(self):
    Detector.__init__(self,self.name,self.getData,self.getTimeStamp,nCalib=self.nCalib)

  def getData(self,calib,shotSlice=None):
    data = self._datasets[calib]["data"]["value"]
    if shotSlice is None:
      data = data[...]
    else:
      data=data[shotSlice]
    return data

  def getTimeStamp(self,calib,shotSlice=None):
    time = self._datasets[calib]["time"]
    if shotSlice is None:
      time = time[...]
    else:
      time=time[shotSlice]
    return time
  

class Epics(object):
  def __init__(self,h5handles,autoDiscovery=False):
    self._h5handles = toolsVarious.iterfy(h5handles)
    if autoDiscovery: self.definePVs()
    
  def definePVs(self):
    """ looking into the _pathlist does not work because the soft links are missing """
    epics_str = config.epics
    _common = "Configure:0000/Run:0000/"
    pv_datasets = collections.defaultdict( list )
    for h in self._h5handles:
      pv_paths_per_handle = []
      calibs = h[_common].keys()
      # remove ndarray_* fields found in some hdf5 files 
      calibs = [c for c in calibs if c.find("CalibCycle") >= 0];
      for c in calibs:
        epicsArchiver = h[ os.path.join(_common,c,epics_str) ].keys()
        for e in epicsArchiver:
          for pvname,path in h[os.path.join(_common,c,epics_str,e)].items():
            pv_datasets[pvname].append(path)
    self.pv_datasets = pv_datasets
    self.pv_names    = list(pv_datasets.keys())
    self.pv_names.sort()
    for pv in self.pv_names:
      mne = pv.replace(":","_").replace(".","_").replace(" ","_")
      setattr(self,mne,EpicsPV(mne,pv_datasets[pv]))
      getattr(self,mne).defineDet()
    return 
        

class EventCode(object):
  def __init__(self,mne,parent,autoDiscovery=False):
    self.name = mne
    self.parent = parent
    self.nCalib = parent.nCalib
    self._out   = dict()
    for c in range(self.nCalib):
      self._out[c] = None
    self._foundCodes = []
    if autoDiscovery: self.autoDiscovery()

  def getDetCode(self,code):
    """ return detector associated with a given code, defining it if needed """
    return self._defineDetector(code)

  def autoDiscovery(self):
    [self._getData(i) for i in range(self.nCalib)]
    for code in self._foundCodes: self._defineDetector(code)
    self._foundCodes.sort()

  def _getDataNewFormat(self,calib):
    matrix = self.parent._getCalibPointer(calib,what="present")
    if matrix is None:
      raise KeyError("New data format not found")
    else:
      log.debug("Reading new eventcodes format (/present) for calib %d"%calib)
      self._out[calib] = matrix[:].astype(np.bool).T; # need code as first index
      # find codes that are non zero at least once
      nCodes = self._out[calib].shape[0]
      codes = [ x for x in range(nCodes) if np.sum(self._out[calib][x]) > 0]
      # add non-zeros codes in list of found codes
      for c in codes:
        if c not in self._foundCodes: self._foundCodes.append(c)

  def _getDataOldFormat(self,calib,code):
    log.debug("Reading old eventcodes format (vlen) for calib %d"%calib)
    data = self.parent.getData(calib)
    nShots = data.shape[0]
    self._out[calib] = np.zeros( (256,nShots), dtype=np.bool)
    for i in range(nShots):
      codeList = data[i]["fifoEvents"]["eventCode"]
      for code in codeList:
        self._out[calib][code,i] = True
        if code not in self._foundCodes: self._foundCodes.append(code)

  def _getData(self,calib,shotSlice=None,code=140):
    # check if data have been cached
    if self._out[calib] is None:
      try:
        self._getDataNewFormat(calib)
      except KeyError:
        self._getDataOldFormat(calib,code)
  
    if shotSlice is not None:
      return self._out[calib][code][shotSlice]
    else:
      return self._out[calib][code]

  def __repr__(self):
    s  = "lclsSpecialDet.EventCode (obj id %s)\n" % (hex(id(self)))
    s += "  non-always-false codes found: %s\n" % str(self._foundCodes)
    s += "  Guessed Frequency (based on first 400 shots):\n"
    for c in self._foundCodes:
      v = self.getDetCode(c)[:400]
      try:
        spacing_true  = np.diff( np.argwhere(  v ).ravel() )[0] 
      except IndexError:
        spacing_true  = None
      try:
        spacing_false  = np.diff( np.argwhere( ~v ).ravel() )[0] 
      except IndexError:
        spacing_false = None
      if spacing_false is None:
        s += "  |→ code %3d, always True\n" % c
      elif spacing_true is None:
        s += "  |→ code %3d, always False\n" % c
      elif (spacing_false == 1) or (spacing_true == spacing_false):
        s += "  |→ code %3d, True every %d\n" % (c,spacing_true)
      elif spacing_true == 1:
        s += "  |→ code %3d, False every %d\n" % (c,spacing_false)
    return s

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
      amp  = ":AMPL$",
      pos  = ":FLTPOS$",
      fwhm = ":FLTPOSFWHM$"
    )
    dets = self._regex.keys()
    calibs = dict( [ (det,self._getCalibForSubField(det)) for det in dets ] )
    self._calibs = calibs
    self.nCalib = len(calibs["amp"])
    for det in dets:
      try:
        self._defineDetector(det,calibs[det])
      except IndexError:
        log.warn("Could not define timetool det %s, found %d calibs",det,len(calibs[det]))
    

  def __repr__(self):
    s  = "lclsSpecialDet.TimeTool (obj id %s)\n" % (hex(id(self)))
    kids = list(self._regex.keys())
    kids.sort()
    for k in kids:
      d = getattr(self,k)
      s += "  |→ %s\n" % d.__str__()
    return s

  def _getCalibForSubField(self,what="amp"):
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
    d = Detector(mne,getData,getTimeStamp,parent=self)
    setattr(self,mne,d)
    return getattr(self,mne)
