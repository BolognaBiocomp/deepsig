#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import argparse
import json
from pathlib import Path

DESC = "DeepSig: Predictor of signal peptides in proteins"

import deepsig
import deepsig.deepsigconfig as cfg
from deepsig import workenv
from deepsig.helpers import readdata, printDate, write_gff_output, get_json_output, detectsp, predictsp

pclasses = {2: 'SignalPeptide', 1: 'Transmembrane', 0: 'Other'}

if('DEEPSIG_ROOT' in os.environ):
  try:
      deepsig_root = os.environ['DEEPSIG_ROOT']
      deepsig_root_path = Path(deepsig_root).resolve()
      if(not deepsig_root_path.is_dir()):
          raise IOError()
      elif(not deepsig_root_path.resolve(cfg.DNN_MODEL_DIR).is_dir()):
          raise IOError()
      elif(not deepsig_root_path.resolve(cfg.CRF_MODEL_DIR).is_dir()):
          raise IOError()
      else:
        sys.path.append(deepsig_root)
  except:
      sys.exit(f'ERROR: wrong DeepSig root path! DEEPSIG_ROOT={deepsig_root}')
else:
  sys.exit("ERROR: required environment variable 'DEEPSIG_ROOT' is not set")

def main():
  parser = argparse.ArgumentParser(description=DESC)
  parser.add_argument("-f", "--fasta",
                      help = "The input multi-FASTA file name",
                      dest = "fasta", required = True)
  parser.add_argument("-o", "--outf",
                      help = "The output file",
                      dest = "outf", required = True)
  parser.add_argument("-k", "--organism",
                      help = "The organism the sequences belongs to",
                      choices=['euk', 'gramp', 'gramn'],
                      dest = "organism", required = True)
  parser.add_argument("-m", "--outfmt",
                      help = "The output format: json or gff3 (default)",
                      choices=['json', 'gff3'], required = False, default = "gff3")

  ns = parser.parse_args()
  protein_jsons = []
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
      if ns.outfmt == "gff3":
        write_gff_output(accs[i], seqs[i], ofs, pclasses[cls[i]], reliability, cleavage[i])
      else:
        acc_json = get_json_output(accs[i], seqs[i], pclasses[cls[i]], reliability, cleavage[i])
        protein_jsons.append(acc_json)
      #ofs.write("\t".join([accs[i], pclasses[cls[i]], str(round(reliability,2)), str(cleavage[i])]) + '\n')
    ofs.close()
    if ns.outfmt == "json":
      ofs = open(ns.outf, 'w')
      json.dump(protein_jsons, ofs, indent=5)
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
