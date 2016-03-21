# x3py
Python 3 library for accessing and manipulating LCLS data.

The idea is to create a thin (= minimum overhead) layer that can be used as building block.
For a more feature rich library please use https://github.com/htlemke/ixppy

Basic usage:
```python
from x3py import x3py
d = x3py.Dataset(filenames); # by default it checks the timestamp of all detectors and set filters to have 
d.ipm3.sum[:]; # get all shots (will be automatically cached)
d.cspad[:20]; # first 20 shots (no caching)

# create filters
xoff = x.ipm3.sum[:]<0.01
# add a filter takes only couple of millisecond
d.addFilter("xoff",xoff); # will define a filter for every detector
# access filtered data
d.cspad.filter.xoff[:10]; # take the first 10 xray off images
# the call above might be less convenient than reading a big chunk and selecting shots afterwards depending on the way the file is written, for example if chunking and compression are used

# check correlations
def mean(img):  return img.mean(-1).mean(-1).mean(-1)
shots = np.randon.random_integers(0,d.imp3.sum.time[:].shape[0],200)
avImg = np.asarray( [mean(d.cspad.data(i)) for i in shots] )
mon   = d.ipm3.sum[shots]
plt.plot(mon,avImg,"o"); # check that there are no missing shots (i.e. that x and y still correlates"

# read EPICS
d.epics.definePVs()
d.epics.pvname
# every PV can be transformed into an Abstractact Detector Instance
d.epics.pvname.defineDet()
```

If available a cython version of the eventmatching time stamp will be used (defaulting to a numpy vesion if unavailable).
To compile it: ```python3 setup.py build_ext --inplace```
