#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.environ['DEEPSIG_ROOT'])
import argparse

DESC="DeepSig: Predictor of signal peptides in proteins"

import deepsiglib.deepsigconfig as cfg
from deepsiglib.helpers import printDate, write_gff_output
from deepsiglib.helpers import readdata
from deepsiglib.helpers import detectsp, predictsp, setUpTFCPU
from deepsiglib import workenv

pclasses = {2: 'SignalPeptide', 1: 'Transmembrane', 0: 'Other'}

def main():
  parser = argparse.ArgumentParser(description=DESC)
  parser.add_argument("-f", "--fasta",
                      help = "The input multi-FASTA file name",
                      dest = "fasta", required = True)
  parser.add_argument("-o", "--outf",
                      help = "The output GFF3 file",
                      dest = "outf", required = True)
  parser.add_argument("-k", "--organism",
                      help = "The organism the sequences belongs to",
                      choices=['euk', 'gramp', 'gramn'],
                      dest = "organism", required = True)

  ns = parser.parse_args()
  try:
    we = workenv.TemporaryEnv()
    printDate("Reading input data")
    X, accs, seqs  = readdata(ns.fasta, cfg.NTERM)
    printDate("Read %d protein sequences" % len(accs))
    printDate("Detecting signal peptides")
    Y, Ytm, Ync, cls, Ytm_norm, Ync_norm = detectsp(X, ns.organism)
    printDate("Detected %d signal peptides" % cls.count(2))
    printDate("Predicting cleavage sites")
    cleavage = predictsp(X, cls, ns.organism, we, cpu=1)
    printDate("Writing results to output file")
    ofs = open(ns.outf, 'w')
    for i in range(len(accs)):
      if cls[i] == 2:
        reliability = Y[i]
      elif cls[i] == 1:
        reliability = Ytm_norm[i]
      else:
        reliability = Ync_norm[i]
      write_gff_output(accs[i], seqs[i], ofs, pclasses[cls[i]], reliability, cleavage[i])
      #ofs.write("\t".join([accs[i], pclasses[cls[i]], str(round(reliability,2)), str(cleavage[i])]) + '\n')
    ofs.close()
  except:
    printDate("Errors occured during execution")
    printDate("Leaving outdir unchanged")
    raise
  else:
    we.destroy()
    pass
  sys.exit(0)

if __name__ == "__main__":
  main()
