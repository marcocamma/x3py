import os
import numpy as np
import functools
from x3py.toolsVarious import iterfy
from x3py import toolsOS
from joblib import Memory


_g_updateTime = {}
_g_content    = {}

def _parseDetectorStr(s):
  d = dict()
  s = s.split("\n")
  for si in s:
    s_strip = si.strip()
    if s_strip == "":
      continue
    else:
      mne,reg = s_strip.split("=")
      d[reg.strip()] = mne.strip()
  return d

def readFile(filename):
  modTime = os.stat(filename)[-2]
  # update file if necessary
  if (filename not in _g_updateTime) or (modTime > _g_updateTime[filename]):
    print("Reading configuration file",filename)
    d = {}
    exec(open(filename).read(),globals(),d)
    # cache result
    globals()["_g_content"][filename] = d
    globals()["_g_updateTime"][filename] = modTime
  return _g_content[filename]
    
def readFiles(fileList):
  fileList=iterfy(fileList)
  d = readFile(fileList[0])
  for f in fileList[1:]:
    d.update( readFile(f) )
  return d

class ConfFile(object):
  lastRead = None
  fileDict = None
  pars     = None
  joblibcache = None
  detectors = dict()

  def __init__(self,confFile):
    """ confFile can be a list of files, will be read (and the dict updated) in
        the orther given """
    self.confFile = iterfy(confFile)
    self._readFiles()
    self._parseDict()

  def _readFiles(self):
    self.fileDict=readFiles(self.confFile)

  def updateFromFile(self,filename):
    d =  readFiles(filename)
    self.updateFromFile(d)

  def updateFromDict(self,d):
    self.fileDict.update(d)
    self._parseDict()

  def _parseDict(self):
    d = self.fileDict
    self.beamline = d["beamline"]
    self.scan     = d["scan"]
    self.scanMon  = d["scanMon"]
    self.cachesizeGB = d["cachesize_GB"]
    self.readlimitGB = d["readlimit_GB"]
    self.epics       = d["epics"]
    for k in ["datapath","cachepath"]:
      v = d[k].replace("$HOME",os.environ["HOME"])
      self.__dict__[k] = v
    if not os.path.isdir(self.cachepath): os.makedirs(self.cachepath)
    self.joblibcache = Memory(cachedir=self.cachepath, verbose=0,compress=True)
    print("Using %s as cache folder, current size %s" % \
      (self.cachepath,toolsOS.du(self.cachepath)))
    d_beamline = _parseDetectorStr(d["detectors_common"])
    d_common   = _parseDetectorStr(d["detectors_" + self.beamline])
    self.detectorsToMatch = d_beamline
    self.detectorsToMatch.update(d_common)

path = os.path.split(__file__)[0]

file_default = path + "/x3py_config"
file_home    = os.path.join(os.environ["HOME"],".x3py.rc")
file_cwd     = os.path.join(os.path.curdir,"x3py_config")
files = [file_default,file_home,file_cwd]
files = [f for f in files if os.path.exists(f)]
config = ConfFile(files) 
