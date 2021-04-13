import os

DEEPSIG_ROOT = os.environ.get('DEEPSIG_ROOT')

AAORDER = 'VLIMFWYGAPSTCHRKQEND'

NTERM = 96

DNN_WINDOWS = {'euk': 13, 'gramp': 13, 'gramn': 19}
DNN_REL_WINDOWS = {'euk': 13, 'gramp': 13, 'gramn': 19}

DNN_MODEL_DIR = 'models/dnn'

DNN_THS    = {'euk': 0.725, 'gramp': 0.72, 'gramn': 0.7575 }

DNN_MODELS = {'euk':   {0: 'model.3cl.w13.f16-32-64.k16.l96.0',
                        1: 'model.3cl.w13.f16-32-64.k128.l96.1',
                        2: 'model.3cl.w13.f16-32-64.k16.l96.2',
                        3: 'model.3cl.w13.f16-32-64.k128.l96.3',
                        4: 'model.3cl.w13.f16-32-64.k64.l96.4' },
              'gramp': {0: 'model.3cl.w13.f8-16-32.k16.l96.0',
                        1: 'model.3cl.w13.f16-32-64.k16.l96.1',
                        2: 'model.3cl.w13.f16-32-64.k16.l96.2',
                        3: 'model.3cl.w13.f16-32-64.k16.l96.3',
                        4: 'model.3cl.w13.f16-32-64.k16.l96.4' },
              'gramn': {0: 'model.3cl.w19.f32-64-128.k16.l96.0',
                        1: 'model.3cl.w19.f32-64-128.k16.l96.1',
                        2: 'model.3cl.w19.f32-64-128.k16.l96.2',
                        3: 'model.3cl.w19.f16-32-64.k16.l96.3',
                        4: 'model.3cl.w19.f32-64-128.k128.l96.4' }}

CRF_PARAMS = {'euk':   {0: {'decoding': 'posterior-viterbi-max', 'sigma': '0.009'},
                        1: {'decoding': 'posterior-viterbi-max', 'sigma': '0.009'},
                        2: {'decoding': 'posterior-viterbi-max', 'sigma': '0.009'},
                        3: {'decoding': 'posterior-viterbi-max', 'sigma': '0.009'},
                        4: {'decoding': 'posterior-viterbi-max', 'sigma': '0.009'}},
              'gramp': {0: {'decoding': 'viterbi', 'sigma': '0.009'},
                        1: {'decoding': 'viterbi', 'sigma': '0.005'},
                        2: {'decoding': 'viterbi', 'sigma': '0.009'},
                        3: {'decoding': 'viterbi', 'sigma': '0.009'},
                        4: {'decoding': 'viterbi', 'sigma': '0.009'}},
              'gramn': {0: {'decoding': 'posterior-viterbi-max', 'sigma': 0.05},
                        1: {'decoding': 'viterbi', 'sigma': '0.009'},
                        2: {'decoding': 'viterbi', 'sigma': '0.009'},
                        3: {'decoding': 'posterior-viterbi-max', 'sigma': '0.009'},
                        4: {'decoding': 'posterior-viterbi-max', 'sigma': '0.009'}}}

CRF_MODEL_DIR = 'models/crf'

CRF_WINDOWS = {'euk': 15, 'gramp': 7, 'gramn': 15}
