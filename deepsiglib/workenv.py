# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:40:53 2017

@author: cas
"""

import tempfile
import shutil
import os

class TemporaryEnv():
  def __init__(self):
    self.tempdir = os.path.abspath(tempfile.mkdtemp(prefix="job.tmpd.", dir="."))

  def destroy(self):
    if not self.tempdir == None:
      shutil.rmtree(self.tempdir)

  def createFile(self, prefix, suffix):
    outTmpFile = tempfile.NamedTemporaryFile(mode   = 'w',
                                             prefix = prefix,
                                             suffix = suffix,
                                             delete = False,
                                             dir = self.tempdir)
    outTmpFileName = outTmpFile.name
    outTmpFile.close()
    return outTmpFileName

  def createDir(self, prefix):
    outTmpDir = tempfile.mkdtemp(prefix=prefix, dir=self.tempdir)
    return outTmpDir
