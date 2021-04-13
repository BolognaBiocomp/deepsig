import numpy
import multiprocessing
import copy
from scipy.optimize import fmin_l_bfgs_b


class DuplicateLabelError(Exception):
    def __init__(self):
        pass


class DuplicateStateError(Exception):
    def __init__(self):
        pass


class UnknownStateError(Exception):
    def __init__(self):
        pass


class UnknownLabelError(Exception):
    def __init__(self):
        pass


class CRFError(Exception):
    def __init__(self):
        pass


class CRFDecoderProcess(multiprocessing.Process):
    def __init__(self, queue, outqueue, model):
        multiprocessing.Process.__init__(self)
        self.model = model
        self.queue = queue
        self.outqueue = outqueue

    def run(self):
        while True:
            ex = self.queue.get()
            # self.model.sample = [ex[1]]
            prediction = self.model.doPrediction(ex[1])
            self.outqueue.put((ex[0], prediction))


class CRFLLTrainerProcess(multiprocessing.Process):
    def __init__(self, queue, outqueue, model):
        multiprocessing.Process.__init__(self)
        self.model = model
        self.queue = queue
        self.outqueue = outqueue

    def run(self):
        while True:
            ex = self.queue.get()
            self.model.sample = [ex[0]]
            self.model.doLLComputation(ex[1])
            self.outqueue.put((self.model.likelihood, self.model.gradient))


class CRFState(object):
    def __init__(self, label=None):
        self.weights = {}
        self.trans = {}
        self.label = label
        self.fw = None
        self.bw = None
        self.vt = None
        self.scores = None

    def __str__(self):
        ret = ""
        for t in self.trans:
            ret += " %s:%1.5f" % (t, self.getTransition(t))
        ret += "\n"

        for wo in self.weights:
            ret += "PARAMS " + str(wo)
            for we in self.weights[wo]:
                ret += " %1.15f" % (we,)
            ret += "\n"
        ret += "LABEL " + self.label + "\n"
        return ret

    def setLabel(self, label):
        self.label = label

    def getLabel(self):
        return self.label

    def outgoing(self):
        return self.trans.keys()

    def setTransition(self, t, value=0.0):
        self.trans[t] = value

    def getTransition(self, t):
        return self.trans.get(t, 0.0)

    def hasTransition(self, t):
        return t in self.trans

    def setWeights(self, windowOffset, weights):
        self.weights[windowOffset] = weights

    def getWeights(self, windowOffset, a=None):
        ret = None
        if a == None:
            ret = self.weights.get(windowOffset, None)
        else:
            if a >= 0 and a < len(self.weights[windowOffset]):
                ret = self.weights.get(windowOffset, None)
                if not ret == None:
                    ret = ret[a]
        return ret

    def initViterbi(self, seqlen):
        self.vt = [[0.0, ''] for x in range(seqlen)]

    def initFw(self, seqlen):
        self.fw = [[0.0, 0.0] for x in range(seqlen)]

    def initBw(self, seqlen):
        self.bw = [[0.0, 0.0] for x in range(seqlen)]

    def initScores(self, seqlen):
        self.scores = [1.0] * seqlen

    def updateScore(self, vector, j, o):
        if j >= 0 and j < len(self.scores):
            self.scores[j] *= numpy.exp(numpy.dot(self.weights[o], vector))


class CRF(object):
    '''
    classdocs
    '''

    def __init__(self, windowSize=1):
        '''
        Constructor
        '''
        self.labels = {}
        self.state_obj = {}
        self.tying = {}
        self.numTransitions = 0
        self.transOrder = []
        self.allst = None

        # Training attributes
        self.Zx = 1.0
        self.Zyx = 1.0
        self.likelihood = 0.0
        self.gradient = None
        self.scale = None
        self.sample = None
        self.sigma = None

        # Decoding attributes
        self.algo = 'viterbi'

        self.queue = None
        self.outqueue = None

        self.windowSize = windowSize
        self.probdim = 0
        self.dim = 0

    def parse(self, modelFile):
        try:
            modelFile = open(modelFile).readlines()
        except IOError:
            print("Error: Failed to open/reading crf model file")
            raise
        curr = None

        d = (self.windowSize - 1) / 2
        for line in modelFile:
            line = line.split()
            if line[0] == 'LABELS':
                for lab in line[1:]:
                    #if not self.labels.has_key(lab):
                    if lab not in self.labels:
                        self.labels[lab] = []
                    else:
                        raise DuplicateLabelError
            elif line[0] == 'TRANSITION_ALPHABET':
                self.state_obj['BEGIN'] = CRFState(label='BEGL')
                self.state_obj['END'] = CRFState(label='END')
                self.allst = sorted(line[1:])
                for st in self.allst:
                    #if not self.labels.has_key(st):
                    if st not in self.state_obj:
                        self.state_obj[st] = CRFState()
                        self.tying[st] = st
                    else:
                        raise DuplicateStateError
            elif line[0] == 'NAME':
                if line[1] in self.state_obj:
                    curr = line[1]
                else:
                    raise UnknownStateError
            elif line[0] == 'LABEL':
                if line[1] in self.labels:
                    self.state_obj[curr].setLabel(line[1])
                    self.labels[line[1]].append(curr)
                else:
                    raise UnknownLabelError
            elif line[0] == 'TIED':
                if line[1] in self.state_obj:
                    self.tying[curr] = line[1]
                else:
                    raise UnknownStateError
            elif line[0] == 'TRANS':
                for tr in line[1:]:
                    tr = tr.split(":")
                    if tr[0] in self.state_obj:
                        self.state_obj[curr].setTransition(tr[0], value=float(tr[1]))
                        self.numTransitions += 1
                        self.transOrder.append((curr, tr[0]))
                    else:
                        raise UnknownStateError
            elif line[0] == 'PARAMS':
                winOffset = int(line[1])
                if winOffset > d:
                    d = winOffset
                wgts = []
                for par in line[2:]:
                    wgts.append(float(par))
                if len(wgts) > self.dim:
                    self.dim = len(wgts)
                self.state_obj[curr].setWeights(winOffset, numpy.array(wgts))
            else:
                pass
        self.windowSize = d * 2 + 1

    def __deepcopy__(self, memo):
        clone = CRF(windowSize=self.windowSize)
        clone.state_obj = copy.deepcopy(self.state_obj, memo)
        clone.labels = copy.deepcopy(self.labels, memo)
        clone.allst = copy.deepcopy(self.allst, memo)
        clone.dim = self.dim
        clone.probdim = self.probdim
        clone.numTransitions = self.numTransitions
        clone.algo = self.algo
        clone.transOrder = copy.deepcopy(self.transOrder, memo)
        clone.tying = copy.deepcopy(self.tying, memo)
        clone.sigma = self.sigma
        return clone

    def __len__(self):
        return len(self.allst)

    def __getitem__(self, key):
        return self.state_obj.get(key, None)

    def __str__(self):
        modStr = ''
        modStr += "LABELS"
        for l in self.labels:
            modStr += " " + l
        modStr += "\n"
        modStr += "TRANSITION_ALPHABET"
        for st in self.allst:
            modStr += " " + st
        modStr += "\n"
        modStr += "NAME BEGIN\n"
        modStr += "TRANS"
        for t in self['BEGIN'].outgoing():
            modStr += " %s:%1.15f" % (t, self['BEGIN'].getTransition(t))
        modStr += "\n"

        for st in self.allst:
            modStr += "NAME " + st + "\n"
            modStr += "TRANS"
            modStr += str(self[st])
            if not self.tying[st] == st:
                modStr += "TIED " + self.tying[st] + "\n"

        return modStr

    def writeModel(self, modelFile):
        try:
            of = open(modelFile, 'w')
        except IOError:
            print("Error opening/writing model file.")
            raise
        of.write(str(self))
        of.close()

    def setWindowSize(self, w):
        self.windowSize = w

    def initWeights(self, protocol="uniform"):
        ret = None
        if protocol == "uniform":
            ret = numpy.zeros(self.probdim)
        else:
            pass
        return ret

    def setWeights(self, w):
        for i in range(self.numTransitions):
            s = self.transOrder[i][0]
            t = self.transOrder[i][1]
            self[s].setTransition(t, w[i])

        d = (self.windowSize - 1) / 2
        k = 0
        for st in self.allst:
            for i in range(-d, d + 1):
                s = self.numTransitions + k * self.dim * self.windowSize + (i + d) * self.dim
                e = self.numTransitions + k * self.dim * self.windowSize + (i + d + 1) * self.dim
                self[st].setWeights(i, w[s:e])
            k += 1

    def getRegularization(self, w):
        freg = -1 * numpy.sum(numpy.power(w, 2) / (2 * self.sigma))
        greg = -1 * w / self.sigma
        return freg, greg

    def computeExpectations(self, matrix):
        seqlen = len(matrix)

        d = (self.windowSize - 1) / 2

        Ex = numpy.zeros((len(self), self.dim * self.windowSize))
        Eyx = numpy.zeros((len(self), self.dim * self.windowSize))
        ExpTx = numpy.zeros(self.numTransitions)
        ExpTyx = numpy.zeros(self.numTransitions)
        for j in range(seqlen):
            v = numpy.array([])
            for o in range(-d, d + 1):
                if j + o >= 0 and j + o < seqlen:
                    v = numpy.append(v, matrix[j + o])
                else:
                    v = numpy.append(v, numpy.zeros(self.dim))
            k = 0
            for st in self.allst:
                p = self.prob(st, j)
                p_c = self.prob(st, j, clamped=True)
                Ex[k] = numpy.add(Ex[k], p * v)
                Eyx[k] = numpy.add(Eyx[k], p_c * v)

                if j == 0:
                    pt = self.prob('BEGIN', j, st)
                    pt_c = self.prob('BEGIN', j, st, clamped=True)
                    try:
                        idx = self.transOrder.index(('BEGIN', st))
                    except ValueError:
                        pass
                    else:
                        ExpTx[idx] += pt
                        ExpTyx[idx] += pt_c
                elif j == seqlen - 1:
                    pt = self.prob(st, j, 'END')
                    pt_c = self.prob(st, j, 'END', clamped=True)
                    try:
                        idx = self.transOrder.index((st, 'END'))
                    except ValueError:
                        pass
                    else:
                        ExpTx[idx] += pt
                        ExpTyx[idx] += pt_c
                if j > 0:
                    for t in self.allst:
                        pt = self.prob(st, j, t)
                        pt_c = self.prob(st, j, t, clamped=True)
                        try:
                            idx = self.transOrder.index((st, t))
                        except ValueError:
                            pass
                        else:
                            ExpTx[idx] += pt
                            ExpTyx[idx] += pt_c
                k += 1
        Ex = numpy.append(ExpTx, numpy.ravel(Ex))
        Eyx = numpy.append(ExpTyx, numpy.ravel(Eyx))
        return Eyx, Ex

    def computeLogLikelihood(self, w, processors):

        fr, gr = self.getRegularization(w)
        self.likelihood = fr
        self.gradient = gr

        for seq in self.sample:
            self.queue.put((seq, w))

        results = []
        while len(results) < len(self.sample):
            res = self.outqueue.get()
            results.append(res)
        for res in results:
            self.likelihood += res[0]
            self.gradient += res[1]

        print("Current log-likelihood:", self.likelihood)
        return -self.likelihood, -1 * self.gradient

    def doLLComputation(self, w):
        self.setWeights(w)
        self.likelihood = 0.0
        self.gradient = numpy.zeros(self.probdim)
        for seq in self.sample:
            matrix = seq[0]
            labels = seq[1]
            self.forwardBackward(matrix, labels=labels)
            self.likelihood += numpy.log(self.Zyx) - numpy.log(self.Zx)
            Eyx, Ex = self.computeExpectations(matrix)
            self.gradient = numpy.add(self.gradient, numpy.add(Eyx, -1 * Ex))

    def train(self, sample, iterations=60, eps=0.001, sigma=0.05, processors=1):

        self.sample = sample
        self.sigma = sigma
        self.dim = len(sample[0][0][0])
        self.probdim = len(self) * self.dim * self.windowSize + self.numTransitions

        self.queue = multiprocessing.Queue()
        self.outqueue = multiprocessing.Queue()

        for j in range(processors):
            model = copy.deepcopy(self)
            t = CRFLLTrainerProcess(self.queue, self.outqueue, model)
            t.daemon = True
            t.start()

        start = self.initWeights()
        x, f, d = fmin_l_bfgs_b(self.computeLogLikelihood, start, m=7, args=(processors,), maxfun=iterations, iprint=-1)
        print("Maximum log-likelihood :", -f)
        print("Gradient vector norm   :", numpy.linalg.norm(d['grad']))
        print("Parameter vector norm  :", numpy.linalg.norm(x))
        self.likelihood = f
        self.setWeights(x)

    def _probS(self, s, j, t=None, clamped=False):
        ret = 0.0
        i = int(clamped)
        try:
            if not t:
                ret = self[s].fw[j][i] * self[s].bw[j][i] * self.scale[j][i]
            else:
                trans = 0.0
                if self[s].hasTransition(t):
                    trans = numpy.exp(self[s].getTransition(t))
                if s == 'BEGIN':
                    ret = self['BEGIN'].fw[j][i] * trans * self[t].scores[j] * self[t].bw[j][i]
                elif t == 'END':
                    ret = self[s].fw[j][i] * trans * int(j == len(self[s].fw) - 1)
                else:
                    ret = self[s].fw[j - 1][i] * trans * self[t].scores[j] * self[t].bw[j][i]
        except (IndexError, KeyError):
            pass
        return ret

    def prob(self, s, j, t=None, clamped=False, label=False):
        ret = 0.0
        if label:
            if not t:
                for st in self.labels[s]:
                    ret += self._probS(st, j, clamped=clamped)
            else:
                for st in self.labels[s]:
                    for tt in self.labels[t]:
                        ret += self._probS(st, j, t=tt, clamped=clamped)
        else:
            ret = self._probS(s, j, t=t, clamped=clamped)
        return ret

    def forwardBackward(self, matrix, labels=None):

        clamped = False
        if not labels == None:
            clamped = True

        seqlen = len(matrix)
        d = int((self.windowSize - 1) / 2)

        # initialization
        for st in self.allst:
            self[st].initFw(seqlen)
            self[st].initBw(seqlen)
            self[st].initScores(seqlen)
        self['BEGIN'].initFw(seqlen)
        self.scale = [[0.0, 0.0] for x in range(seqlen)]
        self.Zx = 1.0
        self.Zyx = 1.0
        # compute backward and scaling

        next = ['END']
        next_c = ['END']
        for j in range(seqlen - 1, -1, -1):
            allowed = set([])
            allowed_c = set([])
            if not j == seqlen - 1:
                for n in next:
                    for o in range(-d, d + 1):
                        if j + 1 + o >= 0 and j + 1 + o < seqlen:
                            self[n].updateScore(matrix[j + 1 + o], j, o)

            for st in self.allst:
                if j == 0:
                    if not self['BEGIN'].hasTransition(st):
                        continue
                for n in next:
                    # transition score
                    # if transition is allowed
                    if self[st].hasTransition(n):
                        allowed.add(st)
                        trScore = self[st].getTransition(n)
                        trScore = numpy.exp(trScore)
                        # compute the state score
                        if not j == seqlen - 1:
                            nScore = self[n].scores[j + 1]
                            self[st].bw[j][0] += self[n].bw[j + 1][0] * nScore * trScore
                            if clamped:
                                if n in next_c and self[st].getLabel() == labels[j]:
                                    allowed_c.add(st)
                                    self[st].bw[j][1] += self[n].bw[j + 1][1] * nScore * trScore
                        elif j == seqlen - 1:
                            self[st].bw[j][0] += trScore
                            if clamped:
                                if self[st].getLabel() == labels[j]:
                                    allowed_c.add(st)
                                    self[st].bw[j][1] += trScore
                if j == 0:
                    # compute the score at position 0
                    for o in range(0, d + 1):
                        if j + o < seqlen:
                            self[st].updateScore(matrix[j + o], j, o)

            for st in allowed:
                self.scale[j][0] += self[st].bw[j][0]
                if clamped and st in allowed_c:
                    self.scale[j][1] += self[st].bw[j][1]
            for st in allowed:
                self[st].bw[j][0] /= self.scale[j][0]
                if clamped and st in allowed_c:
                    self[st].bw[j][1] /= self.scale[j][1]
                if j == 0:
                    self['BEGIN'].fw[0][0] += self[st].bw[j][0] * numpy.exp(self['BEGIN'].getTransition(st)) * \
                                              self[st].scores[j]
                    if clamped and st in allowed_c:
                        self['BEGIN'].fw[0][1] += self[st].bw[j][1] * numpy.exp(self['BEGIN'].getTransition(st)) * \
                                                  self[st].scores[j]

            # self.Zx *= self.scale[j][0]
            # if clamped:
            #    self.Zyx *= self.scale[j][1]
            # if j == 0:
            #    self.Zx *= self['BEGIN'].fw[0][0]
            #    if clamped:
            #        self.Zyx *= self['BEGIN'].fw[0][1]
            next = list(allowed)
            next_c = list(allowed_c)

        prevs = ['BEGIN']
        prevs_c = ['BEGIN']
        self['BEGIN'].fw[0][0] = 1.0 / self['BEGIN'].fw[0][0]
        if clamped:
            self['BEGIN'].fw[0][1] = 1.0 / self['BEGIN'].fw[0][1]
        for j in range(seqlen):
            allowed = set([])
            allowed_c = set([])
            for st in self.allst:
                if j == seqlen - 1:
                    if not self[st].hasTransition('END'):
                        continue
                # compute the state score
                stScore = self[st].scores[j]
                for p in prevs:
                    # if transition is allowed
                    if self[p].hasTransition(st):
                        # transition score
                        trScore = self[p].getTransition(st)
                        trScore = numpy.exp(trScore)
                        allowed.add(st)
                        if j > 0:
                            self[st].fw[j][0] += self[p].fw[j - 1][0] * trScore * stScore
                            if clamped:
                                if p in prevs_c and self[st].getLabel() == labels[j]:
                                    allowed_c.add(st)
                                    self[st].fw[j][1] += self[p].fw[j - 1][1] * trScore * stScore
                        else:
                            self[st].fw[j][0] = self['BEGIN'].fw[0][0] * trScore * stScore
                            if clamped:
                                if self[st].getLabel() == labels[j]:
                                    allowed_c.add(st)
                                    self[st].fw[j][1] = self['BEGIN'].fw[0][1] * trScore * stScore
            for st in allowed:
                self[st].fw[j][0] = self[st].fw[j][0] / self.scale[j][0]
                if clamped and st in allowed_c:
                    self[st].fw[j][1] = self[st].fw[j][1] / self.scale[j][1]
            prevs = list(allowed)
            prevs_c = list(allowed_c)

    def predict(self, sample, algo='viterbi', be=True, processors=1):
        self.sample = sample

        self.queue = multiprocessing.Queue()
        self.outqueue = multiprocessing.Queue()

        self.algo = algo
        self.be = be

        for j in range(processors):
            model = copy.deepcopy(self)
            t = CRFDecoderProcess(self.queue, self.outqueue, model)
            t.daemon = True
            t.start()

        for j in range(len(self.sample)):
            self.queue.put((j, self.sample[j]))

        results = []

        while len(results) < len(self.sample):
            res = self.outqueue.get()
            results.append(res)

        results.sort()
        return [x[1] for x in results]

    def doPrediction(self, matrix):
        ret = None
        if self.algo == 'viterbi':
            ret = self.viterbi(matrix)
        elif self.algo == 'posterior-viterbi-sum':
            ret = self.posteriorViterbi(matrix)
        elif self.algo == 'posterior-viterbi-max':
            ret = self.posteriorViterbi(matrix, suml=False)
        else:
            pass
        return ret

    def _backtrace(self, seqlen, prob=False):
        path = []
        maxst = None
        maxval = 0.0
        for st in self.allst:
            if self[st].vt[-1][0] >= maxval:
                maxval = self[st].vt[-1][0]
                maxst = st
        for j in range(seqlen - 1, -1, -1):
            if not prob:
                path.append(self[maxst].getLabel())
            else:
                path.append((self[maxst].getLabel(), self.prob(self[maxst].getLabel(), j, label=True)))
            maxst = self[maxst].vt[j][1]
        path.reverse()
        return path

    def posteriorViterbi(self, matrix, suml=True, prob=False):
        seqlen = len(matrix)
        d = (self.windowSize - 1) / 2
        for st in self.allst:
            self[st].initViterbi(seqlen)

        self.forwardBackward(matrix)

        prevs = ['BEGIN']
        for j in range(seqlen):
            allowed = set([])
            for st in self.allst:
                if j == (seqlen - 1):
                    if not self[st].hasTransition('END'):
                        continue
                maxst = None
                maxval = 0.0
                p = 0.0
                if suml:
                    # p = self.p_l(self[st].getLabel(), j)
                    p = self.prob(self[st].getLabel(), j, label=True)
                else:
                    # p = self.p_s(st, j)
                    p = self.prob(st, j)
                for prev in prevs:
                    if self[prev].hasTransition(st):
                        allowed.add(st)
                        if j > 0:
                            sc = self[prev].vt[j - 1][0] * p
                            if sc >= maxval:
                                maxval = sc
                                maxst = prev
                        else:
                            maxval = p
                            maxst = prev
                if st in allowed:
                    self[st].vt[j] = [maxval, maxst]

            prevs = list(allowed)
            sumScores = 0.0
            for st in allowed:
                sumScores += self[st].vt[j][0]
            for st in allowed:
                self[st].vt[j][0] /= sumScores
        return self._backtrace(seqlen, prob=prob)

    def viterbi(self, matrix):

        seqlen = len(matrix)
        d = (self.windowSize - 1) / 2
        for st in self.allst:
            self[st].initScores(seqlen)
            self[st].initViterbi(seqlen)

        prevs = ['BEGIN']
        # for each position in the input sequence
        for j in range(seqlen):
            allowed = set([])
            # compute the best scores for each state
            for st in self.allst:
                maxval = 0.0
                maxst = None
                if j == (seqlen - 1):
                    if not self[st].hasTransition('END'):
                        continue
                # compute the state score
                for o in range(-d, d + 1):
                    if j + o >= 0 and j + o < seqlen:
                        self[st].updateScore(matrix[j + o], j, o)
                stScore = self[st].scores[j]
                # scan all previous states
                for prev in prevs:
                    # transition score
                    # if transition is allowed
                    if self[prev].hasTransition(st):
                        allowed.add(st)
                        trScore = self[prev].getTransition(st)
                        trScore = numpy.exp(trScore)
                        totScore = 0.0
                        if j > 0:
                            totScore = self[prev].vt[j - 1][0] * trScore * stScore
                            if j == (seqlen - 1):
                                totScore *= numpy.exp(self[st].getTransition('END'))
                        else:
                            totScore = trScore * stScore
                        # update the maximum value and previous state
                        if totScore > maxval:
                            maxval = totScore
                            maxst = prev
                # append the value only if the state st is reachable
                if st in allowed:
                    self[st].vt[j] = [maxval, maxst]
            # Scaling values
            sumScores = 0.0
            for st in allowed:
                sumScores += self[st].vt[j][0]
            for st in allowed:
                self[st].vt[j][0] /= sumScores
            prevs = list(allowed)
        return self._backtrace(seqlen)
