import numpy

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
                self.allst = line[1:] #sorted(line[1:])
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

    def setWindowSize(self, w):
        self.windowSize = w

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

    def predict(self, sample, algo='viterbi', be=True):
        self.algo = algo
        self.be = be
        results = []
        for matrix in sample:
            labelling = self.doPrediction(matrix)
            posterior = []
            for i in range(len(matrix)):
                v = []
                for st in self.allst:
                    v.append(self.prob(st, i))
                posterior.append(v)
            results.append((labelling, posterior))
        return results

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
