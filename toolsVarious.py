import collections
import numpy as np
from   x3py.toolsLog import log

def toSliceOrList(sl,n):
  """ Convert array of bools, range, slices to ranges; if input is something
      else raise error """
  if isinstance(sl,slice):
    sl = sl
  elif (type(sl) == np.ndarray) and (sl.shape[0] == n):
    sl = sl.astype(np.bool)
    idx = np.argwhere(sl).ravel()
    sl = list(idx)
  elif isinstance(sl,int) and (sl<n):
    sl = [sl,]
  return sl


def toList(sl,n):
  """ Convert array of bools, range, slices to ranges; if input is something
      else raise error """
  if isinstance(sl,slice):
    sl = list(range(*sl.indices(n)))
  elif isinstance(sl,tuple):
    sl = list(sl)
  elif isinstance(sl,range):
    sl = list(sl)
  elif (type(sl) == np.ndarray) and (sl.shape[0] == n):
    sl = sl.astype(np.bool)
    idx = np.argwhere(sl).ravel()
    sl = list(idx)
  elif isinstance(sl,int) and (sl<n):
    sl = [sl,]
  else:
    raise TypeError("argument has to be int,array of bools, slice or range,\
    found %s (type: %s)"%(str(sl),type(sl)))
  return sl

def sliceArgToRange(sl,n):
  """ Convert array of bools, range, slices to ranges; if input is something
      else raise error """
  if isinstance(sl,slice):
    sl = range(*sl.indices(n))
  elif (type(sl) == np.ndarray) and (sl.shape[0] == n):
    sl = sl.astype(np.bool)
    idx = np.argwhere(sl)
    m,M = int(idx[0]),int(idx[-1])
    # check if contigous
    assert idx.sum() == M-m
    sl = range(m,M)
  elif isinstance(sl,int) and (sl<n):
    sl = range(sl,sl+1)
  else:
    raise TypeError("argument has to be int,array of bools or slices")
  return sl

def matchTwoTimeStamps_numpy(timestamp1,timestamp2):
  t1 = timestamp1; t2 = timestamp2
  commonTimeStamps = np.intersect1d(t1,t2,assume_unique=True)
  idx = [ np.in1d(t,commonTimeStamps) for t in (t1,t2)]
  return idx

try:
  import x3py.matchTimeStamps
  matchTwoTimeStamps_cython = x3py.matchTimeStamps.matchTwoTimeStamps
except:
  log.warning("Could not import cythonized version of matchTimeStamps, using numpy one")

def matchTwoTimeStamps(timestamp1,timestamp2,returnCommonTimeStamp=False):
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
  assert ( len(timestamps) >= 2 )
  # get the timestamps
  idx,tcommon = matchTwoTimeStamps(timestamps[0],timestamps[1], \
                returnCommonTimeStamp=True)
  idx = list(idx)
  for t in timestamps[2:]:
    i,tcommon = matchTwoTimeStamps(tcommon,t,returnCommonTimeStamp=True)
    idx.append(i[1])
  if returnCommonTimeStamp:
    return idx,tcommon
  else:
    return idx

class DropObject(object):
  def __init__(self,name='noname'):
    self._name = name

  def _add(self,name,data):
    self.__dict__[name]=data

  def __repr__(self):
    return "dropObject with fields: "+str(self.__dict__.keys())

  def __getitem__(self,x):
    return self.__dict__[x]

  def __setitem__(self,name,var,setParent=True):
    self._add(name,var)

def listWrapper(f,*args,**kwargs):
  return map(f,*args,**kwargs)

def iterfy(x):
  """ Returns iterables """
  if isinstance(x, collections.Iterable) and not isinstance(x,str):
    return x
  else:
    return (x,)

def isStructuredArray(x):
  return x.dtype.names is not None
