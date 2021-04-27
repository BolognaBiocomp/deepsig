import tempfile
import shutil
import os
import subprocess
import numpy
import json


from time import localtime, strftime

from Bio import SeqIO

from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json, Model
from keras import backend as K

from . import deepsigconfig as cfg
from . import crfdecoding as crf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def printDate(msg):
  print("[%s] %s" % (strftime("%a, %d %b %Y %H:%M:%S", localtime()), msg))

def setUpTFCPU(cpus):
  import tensorflow as tf
  from keras.backend.tensorflow_backend import set_session
  config = tf.ConfigProto(intra_op_parallelism_threads=cpus,
                          inter_op_parallelism_threads=cpus,
                          allow_soft_placement=True,
                          device_count = {'CPU': cpus})
  set_session(tf.Session(config=config))

def seq2pssm(sequence, maxlen):
  aaOrder = 'VLIMFWYGAPSTCHRKQEND'
  M = []
  for res in sequence[:maxlen]:
    v = [0.0]*20
    try:
      i = aaOrder.index(res)
      v[i] = 1.0
    except ValueError:
      pass
    M.append(v)
  return M

def readdata(filename, maxlen):
  X, accs, seqs = [], [], []
  for record in SeqIO.parse(filename, 'fasta'):
    accs.append(record.id)
    X.append(seq2pssm(str(record.seq).replace("U", "C"), maxlen))
    seqs.append(str(record.seq))
  X = numpy.array(pad_sequences(X, padding='post', maxlen=maxlen))
  return X, accs, seqs

def detectsp(X, organism):
  Y = []
  Ytm = []
  Ync = []
  for i in range(5):
    for j in range(4):
      modpfx = os.path.join(cfg.DEEPSIG_ROOT, cfg.DNN_MODEL_DIR, organism, "%s.%d.txt" % (cfg.DNN_MODELS[organism][i], j))
      dnn = model_from_json(json.load(open("%s.arch.json" % modpfx)))
      dnn.load_weights("%s.hdf5" % modpfx)
      ypred = dnn.predict(X)
      Y.append(ypred[:,2].reshape((ypred.shape[0],1)))
      Ytm.append(ypred[:,1].reshape((ypred.shape[0],1)))
      Ync.append(ypred[:,0].reshape((ypred.shape[0],1)))

  Y = numpy.mean(numpy.concatenate(Y, axis=1), axis=1)
  Ytm = numpy.mean(numpy.concatenate(Ytm, axis=1), axis=1)
  Ync = numpy.mean(numpy.concatenate(Ync, axis=1), axis=1)
  spc = [int(Y[i]>=cfg.DNN_THS[organism]) for i in range(Y.shape[0])]
  tmc = [int(spc[i] == 0 and Ytm[i]>=Ync[i]) for i in range(Ytm.shape[0])]
  cls = [2*spc[i]+tmc[i] for i in range(Y.shape[0])]
  return Y, Ytm, Ync, cls, Ytm/(Ytm + Ync), 1.0 - Ytm/(Ytm + Ync)

def runCRF(crf_datf, model, window, decoding, we, cpu=1):
  #crf_stdout = getNewTmpFile("crf_stdout", ".log")
  crf_stdout = we.createFile("crf_stdout", ".log")
  #crf_stderr = getNewTmpFile("crf_stderr", ".log")
  crf_stderr = we.createFile("crf_stderr", ".log")
  crf_bin = os.path.join(cfg.DEEPSIG_ROOT, 'tools', 'biocrf-static')
  #crf_outf   = getNewTmpFile("crf.", ".out")
  crf_outf = we.createFile("crf.", ".out")
  subprocess.call([crf_bin, '-test', '-m', model, '-a', str(cpu),
                   '-w', '%d' % ( (window-1)/2), '-d', decoding,
                   '-o', crf_outf,
                   '-q', crf_outf + "_post",
                    crf_datf],

                   stdout=open(crf_stdout, 'w'),
                   stderr=open(crf_stderr, 'w'))

  pred = []
  curr = []
  for line in open(crf_outf).readlines():
    line = line.split()
    if len(line) == 2:
      curr.append(int(line[1] == 'S'))
    else:
      pred.append(curr)
      curr = []
  return pred

def runCRF2(X, model, window, decoding, cpu=1):
  crfmodel = crf.CRF()
  crfmodel.parse(model)
  rawpred = crfmodel.predict(X, algo=decoding)
  pred = []
  posterior = []
  for s in rawpred:
    pred.append([int(x=='S') for x in s[0]])
    posterior.append(s[1])
  return pred, posterior

def relevance(x, model, relout=2):
  weights  = model.get_weights()
  maxlen   = model.layers[0].input_shape[1]
  window   = weights[0].shape[0]
  channels = weights[0].shape[1]
  filters  = weights[0].shape[2]

  hiddens  = weights[3].shape[0]
  out      = weights[-1].shape[0]

  layers = [-1, -3, -4, 1]
  inp = model.input
  outputs = [model.layers[l].output for l in layers]  # all layer outputs
  functor = K.function([inp, K.learning_phase()], outputs)
  layer_outs = functor([x, 1.])
  Rf     = layer_outs[0][0][relout]
  hiddenLayerOut, mergedPoolOut, convLayerOut = layer_outs[1][0], layer_outs[2][0], layer_outs[3][0]
  #Rf     = model.predict(x)[0][relout]

  hiddenLayerOut = Model(inputs=model.input, outputs=model.layers[-3].output).predict(x)[0]
  Rh     = numpy.zeros(hiddens)
  B      = numpy.select([weights[-2][:, relout]>0], [weights[-2][:, relout]])
  Nb     = numpy.sum(numpy.multiply(B, hiddenLayerOut)) + 0.0001
  for i in range(hiddens):
    Rh[i] = Rf * hiddenLayerOut[i] * B[i] / Nb

  #mergedPoolOut  = Model(inputs=model.input, outputs=model.layers[-4].output).predict(x)[0]
  Rp     = numpy.zeros(2 * filters)
  W      = numpy.select([weights[-4] > 0], [weights[-4]])
  assert(W.shape[0] == Rp.shape[0])
  for i in range(2 * filters):
    for j in range(hiddens):
      Rp[i] += Rh[j] * mergedPoolOut[i] * W[i,j] / numpy.dot(mergedPoolOut,W[:,j])
  Rpm = Rp[:filters]
  Rpa = Rp[filters:]

  #convLayerOut  = Model(inputs=model.input, outputs=model.layers[1].output).predict(x)[0]
  Rc            = numpy.zeros((maxlen, filters))

  for i in range(filters):
    Rc[numpy.argmax(convLayerOut[:,i]), i] = Rpm[i]
    N = numpy.sum(convLayerOut[:,i])
    for j in range(maxlen):
      if N != 0.0:
        Rc[j, i] += Rpa[i] * convLayerOut[j,i] / N

  Rx           = numpy.zeros((maxlen, channels))
  F            = []
  for i in range(filters):
    F.append(numpy.zeros((window, channels)))
    for j in range(window):
      for k in range(channels):
        F[i][j,k] = numpy.select([weights[0][j,k,i]>0],[weights[0][j,k,i]])


  for f in range(filters):
    for i in range(maxlen):
      for j in range(max(0, i - window/2), min(maxlen, i + window/2 + 1)):
        dj = -(j - i + window/2 + 1)
        Nj = 0.0 # numpy.zeros(channels)
        for k in range(max(0, j - window/2), min(maxlen, j + window/2 + 1)):
          dk = k - j + window/2
          Nj += numpy.dot(x[0,k], F[f][dk])
        for c in range(channels):
          if Nj != 0:
            Rx[i,c] += (Rc[j,f] * x[0,i,c] * F[f][dj,c] / Nj)
  Rx = numpy.sum(Rx, axis=1)
  return Rx

def predictsp(X, cls, organism, we, cpu=1):
  P = []
  cleavage = []
  for k in range(X.shape[0]):
    if cls[k] == 2:
      x = numpy.array([X[k]])
      rx = numpy.zeros((cfg.NTERM,1))
      P.append(numpy.concatenate((x[0], rx), axis=1))

  if len(P) > 0:
    P = numpy.array(P)
    #crf_datf = getNewTmpFile("crf.", ".dat")
    crf_datf = we.createFile("crf.", ".dat")
    ofs = open(crf_datf, 'w')
    for i in range(P.shape[0]):
      for j in range(P.shape[1]):
        ofs.write(" ".join([str(v) for v in list(P[i][j])] + ['G']))
        ofs.write("\n")
      ofs.write('\n')
    ofs.close()

    C = []
    for i in range(5): #5
      for j in range(4): #4
        crfmodel = os.path.join(cfg.DEEPSIG_ROOT,
                                cfg.CRF_MODEL_DIR, organism,
                                "model.seq.w%d.s%s.l%d.%d.%d" % (cfg.CRF_WINDOWS[organism],
                                                                     cfg.CRF_PARAMS[organism][i]['sigma'],
                                                                     cfg.NTERM,
                                                                     i, j))
        pred = runCRF(crf_datf, crfmodel, cfg.CRF_WINDOWS[organism], cfg.CRF_PARAMS[organism][i]['decoding'], we, cpu=cpu)
        C.append(pred)
    C = numpy.array(C)
    C = numpy.mean(C.T, axis=2).T
    k = 0
    for i in range(X.shape[0]):
      if cls[i] == 2:
        cleavage.append(len([y for y in C[k] if y >= 0.5]))
        k += 1
      else:
        cleavage.append('-')
  else:
    cleavage.extend(["-"]*len(cls))
  return cleavage

def write_gff_output(acc, sequence, output_file, p_class, prob, cleavage):
  l = len(sequence)
  if p_class == "SignalPeptide":
    print(acc, "DeepSig", "Signal peptide", 1, cleavage, round(prob,2), ".", ".",
          "evidence=ECO:0000256",
          sep = "\t", file = output_file)
    print(acc, "DeepSig", "Chain", int(cleavage)+1, l, ".", ".", ".", "evidence=ECO:0000256",
          sep = "\t", file = output_file)
  else:
    print(acc, "DeepSig", "Chain", 1, l, ".", ".", ".", "evidence=ECO:0000256",
          sep = "\t", file = output_file)

def get_json_output(acc, sequence, p_class, prob, cleavage):
  acc_json = {'accession': acc, 'features': []}
  acc_json['sequence'] = {
                            "length": len(sequence),
                            "sequence": sequence
                         }
  start = 1
  score = round(float(prob),2)
  if p_class == "SignalPeptide":
    acc_json['features'].append({
        "type": "SIGNAL",
        "category": "MOLECULE_PROCESSING",
        "description": "",
        "begin": start,
        "end": cleavage,
        "score": score,
        "evidences": [
          {
            "code": "ECO:0000256",
            "source": {
              "name": "SAM",
              "id": "DeepSig",
              "url": "https://deepsig.biocomp.unibo.it"
            }
          }
        ]
      })
    acc_json['features'].append({
        "type": "CHAIN",
        "category": "MOLECULE_PROCESSING",
        "description": "Mature protein",
        "begin": cleavage+1,
        "end": len(sequence),
        "evidences": [
          {
            "code": "ECO:0000256",
            "source": {
              "name": "SAM",
              "id": "DeepSig",
              "url": "https://deepsig.biocomp.unibo.it"
            }
          }
        ]
      })
  return acc_json
