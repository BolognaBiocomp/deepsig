### DeepSig - Predictor of signal peptides in proteins based on deep learning

#### Installation and configuration

DeepSig is designed to run on Unix/Linux platforms. The software was written using the Python programming language and it was tested under the Python version 2.7.

To obtain DeepSig, clone the repository from GitHub:

```
$ git clone https://github.com/savojard/deepsig.git
```

This will produce a directory “deepsig”. Before running deepsig you need to set and export a variable named DEEPSIG_ROOT to point to the deepsig installation dir:
```
$ export DEEPSIG_ROOT='/path/to/deepsig'
```

Before running the program, you need to install DeepSig dependencies. The following Python libraries are required:

- biopython (version 1.72)
- Keras (version 2.2.4)
- Tensorflow (version 1.5.0)

The best way to install all requirements is using pip. We provide a file "requirements.txt" which can be used to install all DeepSig dependecies using a single command. To do so, run:

```
$ pip install --no-cache-dir -r requirements.txt
```

We suggest to use Conda or Virtualenv to create a Python virtual environment and activate it before running pip. In this way, all dependencies will be installed in the environment. 

Now you are able to use deepsig (see next Section). If you whish, you can copy the “deepsig.py” script to a directory in the users' PATH.

#### Usage

The program accepts three mandatory arguments:

- The full path of the input FASTA file containing protein sequences to be predicted;
- The kingdom the sequences belong to. You must specify "euk" for Eukaryotes, "gramp" for Gram-positive bacteria or "gramn" for Gram-negative bacteria;
- The output file where predictions will be stored.

As an example, run the program on the eukaryotic example FASTA file contained in the folder "testdata":

```
$ ./deepsig.py -f testdata/SPEuk.nr.fasta -k euk -o testdata/SPEuk.nr.out -a 4
```

This will run deepsig on sequences contained in the "testdata/SPEuk.nr.fasta" file, using the Eukaryotes models and storing the output in the "testdata/SPEuk.nr.out" file.

Once the prediction is done, the output should look like the following:

```
$ cat 
G5ED35  SignaPeptide    0.98    20
Q59XX2  SignaPeptide    1.0 21
Q9VMD9  SignaPeptide    0.98    18
Q4V4I9  SignaPeptide    0.98    22
Q8SXL2  SignaPeptide    1.0 18
F1NSM7  SignaPeptide    1.0 18
Q9SUQ8  Transmembrane   0.94    -
P0DKU2  Other   1.0 -
C9K4X8  SignaPeptide    1.0 29
Q9LRC8  SignaPeptide    0.96    25
....
```
The first column is the protein accession/id as reported in the input fasta file. The second column report the result of signal peptide detection. Three different outcomes are possible: 1. SignalPeptide (i.e. a signal sequence is detected at the N-terminus of the protein); 2. Transmembrane (i.e. a transmembrane region is detected at the N-terminus) 3. Other (i.e. no signal sequence nor transmembrane region are detected). The third column is a reliabilty score attached to the previous prediction. Finally, the fourth column reports, for those proteins predicted as having a signal peptide, the predicted position of the cleavage site.

Please, reports bugs to: savojard@biocomp.unibo.it