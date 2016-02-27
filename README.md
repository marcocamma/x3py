# x3py
Python 3 library for accessing and manipulating LCLS data.

The idea is to create a thin (= minimum overhead) layer that can be used as building block.
For a more feature rich library please use https://github.com/htlemke/ixppy

Basic usage:
```python
from x3py import x3py
d = x3py.Dataset(filename)
d.ipm3.sum[:]; # get all shots (will be automatically cached)
d.cspad[:20]; # first 20 shots (no caching)

# event matching
from x3py import toolsVarious

# I know it is not very elegant. will create a nicer interface soon...
t=[d.phasec.fCharge1.time[:],d.eventCode.code_91.time[:],d.cspad.time[:],d.timeTool.precision.time[:],d.ipm3.sum.time[:]]
idx=toolsVarious.matchTimeStamps(*t)
print("Total number of shot:",d.ipm3.sum[:].shape)
# this is how to include a filter (a proper method will be created soon)
d.ipm3.sum._isOk = np.argwhere(idx[-1]).ravel()
print("Number of shots after filtering",d.ipm3.sum[:].shape)
```

If available a cython version of the eventmatching time stamp will be used (defaulting to a numpy vesion if unavailable).
To compile it: ```python3 setup.py build_ext --inplace```
