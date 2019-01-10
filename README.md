## DeepSig - Predictor of signal peptides in proteins based on deep learning

#### Publication

Savojardo C., Martelli P.L., Fariselli P., Casadio R. [DeepSig: deep learning improves signal peptide detection in proteins](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btx818/4769493) *Bioinformatics* (2017) **34**(10): 1690-1696.

### The DeepSig Docker image

Image availbale on DockerHub [https://hub.docker.com/r/bolognabiocomp/deepsig](https://hub.docker.com/r/bolognabiocomp/deepsig)

#### Usage of the image

The first step to run DeepSig Docker container is the pull the container image. To do so, run:

```
$ docker pull bolognabiocomp/deepsig
```

Now the DeepSig Docker image is installed in your local Docker environment and ready to be used. To show DeepSig help page run:

```
$ docker run bolognabiocomp/deepsig -h

Using TensorFlow backend.
usage: deepsig.py [-h] -f FASTA -o OUTF -k {euk,gramp,gramn} [-a CPU]

DeepSig: Predictor of signal peptides in proteins

optional arguments:
  -h, --help            show this help message and exit
  -f FASTA, --fasta FASTA
                        The input multi-FASTA file name
  -o OUTF, --outf OUTF  The output tabular file
  -k {euk,gramp,gramn}, --organism {euk,gramp,gramn}
                        The organism the sequences belongs to
```
The program accepts three mandatory arguments:
- The full path of the input FASTA file containing protein sequences to be predicted;
- The kingdom the sequences belong to. You must specify "euk" for Eukaryotes, "gramp" for Gram-positive bacteria or "gramn" for Gram-negative bacteria;
- The output file where predictions will be stored.

Let's now try a concrete example. First of all, let's downlaod an example sequence from UniProtKB, e.g. the Transthyretin-like protein 52 form Caenorhabditis elegans with accession G5ED35:

```
$ wget http://www.uniprot.org/uniprot/G5ED35.fasta
```

Now, we are ready to predict the signal peptide of our input protein. Run:

```
$ docker run -v $(pwd):/data/ bolognabiocomp/deepsig -f G5ED35.fasta -o G5ED35.out -k euk
```

In the example above, we are mapping the current program working directory ($(pwd)) to the /data/ folder inside the container. This will allow the container to see the external FASTA file G5ED35.fasta.
The file G5ED35.out now contains the DeepSig prediction:
```
$ cat G5ED35.out

sp|G5ED35|TTR52_CAEEL   SignalPeptide   0.98    20
```
The first column is the protein accession/id as reported in the input fasta file. The second column report the result of signal peptide detection. Three different outcomes are possible: SignalPeptide (i.e. a signal sequence is detected at the N-terminus of the protein), Transmembrane (i.e. a transmembrane region is detected at the N-terminus) or Other (i.e. neither a signal sequence nor a transmembrane region are detected). The third column is a reliabilty score attached to the previous prediction. Finally, the fourth column reports, for those proteins predicted as having a signal peptide, the predicted position of the cleavage site.


### Install and use DeepSig from source

Source code available on GitHub at [https://github.com/BolognaBiocomp/deepsig](https://github.com/BolognaBiocomp/deepsig)

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
$ ./deepsig.py -f testdata/SPEuk.nr.fasta -k euk -o testdata/SPEuk.nr.out
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
