import collections
import numpy as np
import time
import sys
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
  elif (type(sl) == np.ndarray) and (sl.shape[0] == n) and (sl.dtype == bool):
    sl = sl.astype(np.bool)
    idx = np.argwhere(sl).ravel()
    sl = list(idx)
  elif (type(sl) == np.ndarray) and (sl.dtype != bool):
    sl = sl.tolist()
  elif (isinstance(sl,int) or isinstance(sl,np.int64)) and (sl<n):
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
  elif (type(sl) == np.ndarray) and (sl.shape[0] == n) and (sl.dtype == bool):
    sl = sl.astype(np.bool)
    idx = np.argwhere(sl)
    m,M = int(idx[0]),int(idx[-1])
    # check if contigous
    assert idx.sum() == M-m
    sl = range(m,M)
  elif (type(sl) == np.ndarray) and (sl.dtype != bool):
    sl = sl.tolist()
  elif isinstance(sl,int) and (sl<n):
    sl = range(sl,sl+1)
  else:
    raise TypeError("argument has to be int,array of bools or slices")
  return sl

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
  """ Returns iterable """
  if isinstance(x, collections.Iterable) and not isinstance(x,str):
    return x
  else:
    return (x,)

def isStructuredArray(x):
  """ Returns True is the argument is a structured array """
  return x.dtype.names is not None

def bytesToHuman(bytes,units="auto",fmt="%.2f %s"):
  _units = dict( B = 0, KB = 1, MB = 2, GB = 3, TB = 4, PT = 5 )
  _symb  = {v: k for k, v in _units.items()}
  bytes  = float(bytes)
  if units == "auto":
    u = np.log(bytes)//np.log(1024)
    units = _symb[u]
  else:
    u = _units[units]
  value = bytes/1024**u
  return fmt % (value,units)

def flush(what="stdout"):
  """ Flush strasm, default is stdout """
  if what == "stdout": what = sys.__dict__[what]
  what.flush()

class CodeBlock(object):
  def __init__(self,s):
    self.t0 = time.time()
    print(s,end=""); flush()

  def lapse(self,s="..elapsed %.2f s..",end=""):
    dt = time.time()-self.t0
    print(s%dt,end=end); flush()

  def done(self,s="...done %.2f s",end="\n"):
    dt = time.time()-self.t0
    print(s%dt,end=end); flush()
    
