import numpy as np
import pylab as plt
import os
from   x3py.toolsLog import log
from   x3py.toolsConf import config
from   x3py.toolsVarious import iterfy,isStructuredArray,DropObject
from   x3py import abstractDet

class StructuredArrayDetector(object):
  def __init__(self,mne,data,time,parent=None):
    assert data.dtype.names is not None
    self.name,self.fullname = abstractDet.baptize(mne,parent)
    self._kids = data.dtype.names
    self._data = data
    self._time = time
    for name in self._kids:
      setattr(self,name,wrapArray(name,data[name],time,parent=self))
    if parent is not None: setattr(parent,mne,self)

  def __repr__(self):
    s  = "StructuredArrayDetector %s\n" % self.name
    for k in self._kids:
      d = getattr(self,k)
      s += "  |→ %s\n" % d.__str__()
    return s

  def save(self,detname="auto",fname="auto",force=True):
    """ save in cachepath """
    path = config.cachepath
    if fname == "auto":
      fname = path+"/"+os.path.basename(config.h5handles[0].filename)
    if detname == "auto": detname = self.fullname.replace(".","/")
    path = fname+"/"+detname
    if not os.path.exists( path ): os.makedirs(path)
    np.save(path+"/abstractDetector.npy",dict(data=self._data, time=self._time))

def wrapArray(mne,data,time=None,parent=None):
  """ If data is a string, it is interpreted as filename of a
      numpy file containing a dictionary with 'data' and 'time' """
  if isinstance(data,str):
    temp = np.load(data).item()
    data = temp["data"]
    time = temp["time"]
  if time is None and parent is not None: time=parent.time[:]
  if isStructuredArray(data):
    return StructuredArrayDetector(mne,data,time,parent)
  if not isinstance(data,(list,tuple)):
    data = (data,)
    time = (time,)
  nCalib = len(data)
  def getData(calib,shotSlice=None):
    if shotSlice is None:
      return data[calib]
    else:
      return data[calib][shotSlice]
  def getTimeStamp(calib,shotSlice=None):
    if shotSlice is None:
      return time[calib]
    else:
      return time[calib][shotSlice]
  return abstractDet.Detector(mne,getData,getTimeStamp,nCalib=nCalib,parent=parent)

def findDetectorsInsideObj(obj):
  """recursively find detectors """
  # find kids
  if not hasattr(obj,"__dict__"): return []
  kids = obj.__dict__.values()
  dets = {}
  for k in kids:
    if isinstance(k,abstractDet.Detector):
      dets[k.fullname]=k
    else:
      dets.update( findDetectorsInsideObj(k) )
  return dets
 
def corrNonlinGetPar(linearDet,nonLinearDet,order=2,data_0=0,
    correct_0=0,plot=False,returnCorrectedDet=False):
  """ Find parameters for non linear correction
    *linearDet* should be an 1D array of the detector that is linear
    *nonLinearDet* is the detector that is sussposed to be none linear
    *data_0" is an offset to use for the data (used only if plotting)"
    *correct_0* offset of the "linear detector"""
  p =  np.polyfit(nonLinearDet,linearDet,order)
  p[-1] = p[-1]-correct_0
  if plot:
    d = corrNonlin(nonLinearDet,p,data_0=data_0,correct_0=correct_0)
    plt.plot(linearDet,nonLinearDet,".",label="before correction")
    plt.plot(linearDet,d,".",label="after correction")
    poly_lin = np.polyfit(linearDet,d,1)
    xmin = min(linearDet.min(),0)
    xtemp = np.asarray( (xmin,linearDet.max()) )
    plt.plot(xtemp,np.polyval(poly_lin,xtemp),label="linear fit")
    plt.plot(linearDet,d-np.polyval(poly_lin,linearDet),
       ".",label="difference after-linear")
    plt.xlabel("linearDet")
    plt.ylabel("nonLinearDet")
    plt.legend()
  if order>=2 and p[-3]<0:
    log("corrNonlinGetPar: consistency problem, second order coefficient should \
    be > 0, please double check result (plot=True) or try inverting the data and the\
    correct arguments")

  if returnCorrectedDet:
    return corrNonlin(nonLinearDet,p,data_0=data_0,correct_0=correct_0)
  else:
    return p


def corrNonlin(nonLinearDet,polypar,data_0=0,correct_0=0):
  """ unses parameters found by corrNonlinGetPar to correct data;
  example of usage (assuming mon is non linear and loff is
  the laseroff filter
  d = ixppy.dataset("xppc3614-r0102.stripped.h5")
  loff = d.eventCode.code_91.filter(True)
  mon = d.ipm3.sum;
  dio = d.diodeU.channel0;
  poly = ixppy.tools.corrNonlinGetPar(dio*loff,mon*loff,plot=True)
  mon_corr = ixppy.corrNonlin(mon,poly)"""
  m = 1/np.polyval(np.polyder(polypar),data_0)
  return m*(np.polyval(polypar,nonLinearDet)-correct_0) + data_0

def smoothing(x,y,err=None,k=5,s=None,newx=None,derivative_order=0):
  # remove NaNs
  idx = np.isfinite(x) & np.isfinite(y)
  if idx.sum() != len(x): x=x[idx]; y=y[idx]

  # if we don't need to interpolate, use same x as input
  if newx is None: newx=x

  if err is None:
    w=None
  elif err == "auto":
    n=len(x)
    imin = int(max(0,n/2-20))
    imax = imin + 20
    idx = range(imin,imax)
    p = np.polyfit(x[idx],y[idx],2)
    e = np.std( y[idx] - np.polyval(p,x[idx] ) )
    w = np.ones_like(x)/e
  else:
    w=np.ones_like(x)/err
  from scipy.interpolate import UnivariateSpline
  if (s is not None):
    s = len(x)*s
  s = UnivariateSpline(x, y,w=w, k=k,s=s)
  if (derivative_order==0):
    return s(newx)
  else:
    try:
      len(derivative_order)
      return np.asarray([s.derivative(d)(newx) for d in derivative_order])
    except:
      return s.derivative(derivative_order)(newx)

def filterMinMax(values,min,max):
  return (values>=min) & (values<=max)
