# x3py
Python 3 library for accessing and manipulating LCLS data.

The idea is to create a thin (= minimum overhead) layer that can be used as building block.
For a more feature rich library please use https://github.com/htlemke/ixppy

Basic usage:
```python
from x3py import x3py
d = x3py.Dataset(filename); # by default it checks the timestamp of all detectors and set filters
d.ipm3.sum[:]; # get all shots (will be automatically cached)
d.cspad[:20]; # first 20 shots (no caching)

# event matching
from x3py import toolsMatchTimeStamps

# I know it is not very elegant. will create a nicer interface soon...
# could a detector or a timestamp vector
t=[d.phasec.fCharge1,d.eventCode.code_91.time[:],d.cspad,d.ipm3.sum]
idx=toolsMatchTimeStamps.matchTimeStamps(*t)
print("Total number of shot:",d.ipm3.sum[:].shape)
# set filter
d.ipm3.sum.defineFilter(idx[-1])


# check correlations
def mean(img):  return img.mean(-1).mean(-1).mean(-1)
shots = np.randon.random_integers(0,d.imp3.sum.time[:].shape[0],200)
avImg = np.asarray( [mean(d.cspad.data(i)) for i in idx] )
mon   = d.ipm3.sum[shots]
plt.plot(mon,avImg,"o"); # check that there are no missing shots (i.e. that x and y still correlates"
```

If available a cython version of the eventmatching time stamp will be used (defaulting to a numpy vesion if unavailable).
To compile it: ```python3 setup.py build_ext --inplace```
