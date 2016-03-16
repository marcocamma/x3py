import numpy as np
import pylab as plt
from   x3py.toolsLog import log

def corrNonlinGetPar(linearDet,nonLinearDet,order=2,data_0=0,
    correct_0=0,plot=False,returnCorrectedDet=False):
  """ Find parameters for non linear correction
    *linearDet* should be an 1D array of the detector that is linear
    *nonLinearDet* is the detector that is sussposed to be none linear
    *data_0" is an offset to use for the data (used only if plotting)"
    *correct_0* offset of the "linear detector"""
  p =  np.polyfit(nonLinearDet,linearDet,order)
  if order>=2 and p[-3]<0:
    log("corrNonlinGetPar: consistency problem, second order coefficient should \
    be > 0, please double check result (plot=True) or try inverting the data and the\
    correct arguments")
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

