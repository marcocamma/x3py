import os
import psutil
from x3py import toolsVarious

def getCMD(cmd,strip=True):
  shell = os.popen(cmd)
  ret = shell.readlines()
  shell.close()
  if (strip):
    ret = [x.strip() for x in ret]
  return ret

def du(path,asHuman=True):
  size = getCMD("du -b %s"%path)[-1].split("\t")[0]
  if asHuman: size = toolsVarious.bytesToHuman(size)
  return size

def memAvailable(asHuman=True):
  m = psutil.virtual_memory()
  m = m.free+m.cached
  if asHuman: m = toolsVarious.bytesToHuman(m)
  return m
