import cython
import numpy as np
cimport numpy as np
def fib(n):
    """Print the Fibonacci series up to n."""
    a, b = 0, 1
    while b < n:
        print b,
        a, b = b, a + b

cdef packed struct tstruct:
    np.uint32_t s
    np.uint32_t ns
    np.uint32_t t
    np.uint32_t fid
    np.uint32_t con
    np.uint32_t vec

@cython.boundscheck(False)
@cython.wraparound(False)
def matchTwoTimeStamps(tstruct[:] t1, tstruct[:] t2 ):
  """ This is a cythonized version of timestamp matching. it is meant to work 
      with sorted structured arrays like lcls ones.
      it takes ~2ms instead of 250ms for ~30 kShots)"""
  # assume sorted!!
  cdef Py_ssize_t i_t1 = 0
  cdef Py_ssize_t i_t2 = 0
  cdef idx1 = np.zeros( len(t1), dtype=np.bool )
  cdef idx2 = np.zeros( len(t2), dtype=np.bool )
  cdef Py_ssize_t n1 = len(t1)
  cdef Py_ssize_t n2 = len(t2)
  #matchedTime = []
  #cdef Py_ssize_t nMatch = 0
  
  while (i_t1 < n1) and (i_t2 < n2):
    if   (t1[i_t1].s>t2[i_t2].s):
      idx2[i_t2]  = False; i_t2 += 1
    elif (t1[i_t1].s<t2[i_t2].s):
      idx1[i_t1]  = False; i_t1 += 1
    else:
      if   (t1[i_t1].ns>t2[i_t2].ns):
        idx2[i_t2]  = False; i_t2 += 1
      elif (t1[i_t1].ns<t2[i_t2].ns):
        idx1[i_t1]  = False; i_t1 += 1
      else:
        idx1[i_t1] = True; i_t1 += 1
        idx2[i_t2] = True; i_t2 += 1
  return idx1,idx2
