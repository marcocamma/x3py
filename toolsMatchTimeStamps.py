import collections
import numpy as np
from   .toolsLog import log

try:
  import x3py.matchTimeStamps
  matchTwoTimeStamps_cython = x3py.matchTimeStamps.matchTwoTimeStamps
except:
  log.warning("Could not import cythonized version of matchTimeStamps, using numpy one")

def matchTwoTimeStamps_numpy(timestamp1,timestamp2):
  t1 = timestamp1; t2 = timestamp2
  commonTimeStamps = np.intersect1d(t1,t2,assume_unique=True)
  idx = [ np.in1d(t,commonTimeStamps) for t in (t1,t2)]
  return idx

def matchTwoTimeStamps(timestamp1,timestamp2,returnCommonTimeStamp=False):
  """ Match two timestamps vectors; uses cython version if available """
  # check in case they are the same
  if (timestamp1.shape == timestamp2.shape) and np.all(timestamp1==timestamp2):
    idx = np.ones_like(timestamp1,dtype=np.bool)
    idx = idx,idx
  elif "matchTwoTimeStamps_cython" in globals():
    idx = matchTwoTimeStamps_cython(timestamp1,timestamp2)
  else:
    idx = matchTwoTimeStamps_numpy(timestamp1,timestamp2)
  if returnCommonTimeStamp:
    return idx,timestamp1[idx[0]]
  else:
    return idx

def matchTimeStamps(*timestamps,returnCommonTimeStamp=False):
  """ Given the detectors (or directly the timestamps vector) t1,t2,...
      the funciton returns a list of arrays of len(t1),len(t2),...
      each is a boolean array. Elements that are true are shots that are in all
      detectors given as argument """
  # get the timestamps (in case you give a detector)
  timestamps = [ t.getShots(slice(t._unFilteredNShots),what="time",useTimeStampFilter=False) if hasattr(t,"time") else t for t in timestamps]
  if len(timestamps) == 1:
    idx = (np.ones_like(timestamps[0],dtype=bool),)
    if returnCommonTimeStamp:
      return idx,timestamps[0]
    else:
      return idx
  # run a first time to timestamp common to all
  idx,tcommon = matchTwoTimeStamps(timestamps[0],timestamps[1], \
                returnCommonTimeStamp=True)
  idx = list(idx)
  for t in timestamps[2:]:
    temp,tcommon = matchTwoTimeStamps(tcommon,t,returnCommonTimeStamp=True)
  # run a second time (needed if len(timestamps)>2)) otherwise first checked
  # detecotrs might be left with shots filtered out later on;
  if len(timestamps)>2:
    idx = []
    for t in timestamps:
      icom,i = matchTwoTimeStamps(tcommon,t)
      idx.append(i)
  if returnCommonTimeStamp:
    return idx,tcommon
  else:
    return idx

