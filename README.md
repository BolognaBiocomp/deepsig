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
$ wget https://www.uniprot.org/uniprot/G5ED35.fasta
```

Now, we are ready to predict the signal peptide of our input protein. Run:

```
$ docker run -v $(pwd):/data/ bolognabiocomp/deepsig -f G5ED35.fasta -o G5ED35.out -k euk
```

In the example above, we are mapping the current program working directory ($(pwd)) to the /data/ folder inside the container. This will allow the container to see the external FASTA file G5ED35.fasta.
The file G5ED35.out now contains the DeepSig prediction, in GFF3 format:
```
$ cat G5ED35.out

sp|G5ED35|TTR52_CAEEL	DeepSig	Signal peptide	1	20	0.98	.	.	evidence=ECO:0000256
sp|G5ED35|TTR52_CAEEL	DeepSig	Chain	21	135	.	.	.	evidence=ECO:0000256

```
Columns are as follows:
- Column 1: the protein ID/accession as reported in the FASTA input file;
- Column 2: the name of tool performing the annotation (i.e. DeepSig)
- Column 3: the annotated feature alogn the sequence. Can be "Signal peptide" or "Chain" (indicating the mature protein). When no signal peptide is detected, the entire protein sequence is annotated as "Chain";
- Column 4: start position of the feature;
- Column 5: end position of the feature;
- Column 6: feature annotation score (as assigned by DeepSig);
- Columns 7,8: always empty, reported for compliance with GFF3 format
- Column 9: Description field. Report the evidence code for the annotation (i.e. ECO:0000256, automatic annotation).

### Install and use DeepSig from source

Source code available on GitHub at [https://github.com/BolognaBiocomp/deepsig](https://github.com/BolognaBiocomp/deepsig)

#### Installation and configuration

DeepSig is designed to run on Unix/Linux platforms. The software was written using the Python programming language and it was tested under the Python version 3.

To obtain DeepSig, clone the repository from GitHub:

```
$ git clone https://github.com/BolognaBiocomp/deepsig.git
```

This will produce a directory “deepsig”. Before running deepsig you need to set and export a variable named DEEPSIG_ROOT to point to the deepsig installation dir:
```
$ export DEEPSIG_ROOT='/path/to/deepsig'
```

Before running the program, you need to install DeepSig dependencies. We suggest to use Conda (we suggest [Miniconda3](https://docs.conda.io/en/latest/miniconda.html)) create a Python virtual environment and activate it.

To create a conda env for deepsig:

```
$ conda create -n deepsig
```
To activate the environment:

```
$ conda activate deepsig
```

The following Python libraries are required:

- biopython (version 1.78)
- Keras (version 2.4.3)
- Tensorflow (version 2.2)

To install all requirements:

```
$ conda install --yes nomkl keras==2.4.3 biopython==1.78 tensorflow==2.2.0
```

Now you are able to use deepsig (see next Section). Remember to keep the environment active.
If you whish, you can copy the “deepsig.py” script to a directory in the users' PATH.

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

Once the prediction is done, the GFF3 output should look like the following:

```
$ cat
G5ED35	DeepSig	Signal peptide	1	20	0.98	.	.	evidence=ECO:0000256
G5ED35	DeepSig	Chain	21	135	.	.	.	evidence=ECO:0000256
Q59XX2	DeepSig	Signal peptide	1	21	1.0	.	.	evidence=ECO:0000256
Q59XX2	DeepSig	Chain	22	378	.	.	.	evidence=ECO:0000256
Q9VMD9	DeepSig	Signal peptide	1	18	0.98	.	.	evidence=ECO:0000256
Q9VMD9	DeepSig	Chain	19	2188	.	.	.	evidence=ECO:0000256
Q4V4I9	DeepSig	Signal peptide	1	22	0.98	.	.	evidence=ECO:0000256
Q4V4I9	DeepSig	Chain	23	182	.	.	.	evidence=ECO:0000256
Q8SXL2	DeepSig	Signal peptide	1	18	1.0	.	.	evidence=ECO:0000256
Q8SXL2	DeepSig	Chain	19	136	.	.	.	evidence=ECO:0000256
F1NSM7	DeepSig	Signal peptide	1	18	1.0	.	.	evidence=ECO:0000256
F1NSM7	DeepSig	Chain	19	743	.	.	.	evidence=ECO:0000256
Q9SUQ8	DeepSig	Chain	1	187	.	.	.	evidence=ECO:0000256
P0DKU2	DeepSig	Chain	1	145	.	.	.	evidence=ECO:0000256
C9K4X8	DeepSig	Signal peptide	1	29	1.0	.	.	evidence=ECO:0000256
C9K4X8	DeepSig	Chain	30	116	.	.	.	evidence=ECO:0000256
....
```
Columns are as follows:
- Column 1: the protein ID/accession as reported in the FASTA input file;
- Column 2: the name of tool performing the annotation (i.e. DeepSig)
- Column 3: the annotated feature alogn the sequence. Can be "Signal peptide" or "Chain" (indicating the mature protein). When no signal peptide is detected, the entire protein sequence is annotated as "Chain";
- Column 4: start position of the feature;
- Column 5: end position of the feature;
- Column 6: feature annotation score (as assigned by DeepSig);
- Columns 7,8: always empty, reported for compliance with GFF3 format
- Column 9: Description field. Report the evidence code for the annotation (i.e. ECO:0000256, automatic annotation).

Please, reports bugs to: castrense.savojardo2@unibo.it
