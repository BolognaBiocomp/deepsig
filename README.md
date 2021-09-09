## DeepSig - Predictor of signal peptides in proteins based on deep learning

#### Publication

Savojardo C., Martelli P.L., Fariselli P., Casadio R. [DeepSig: deep learning improves signal peptide detection in proteins](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btx818/4769493) *Bioinformatics* (2017) **34**(10): 1690-1696.

#### Installation using pip

First, install deepsig-biocomp package using pip:
```
pip install deepsig-biocomp
```

Then, clone the deepsig repo from GitHub and export the DEEPSIG_ROOT directory:
```

git clone git@github.com:BolognaBiocomp/deepsig.git
cd deepsig
export DEEPSIG_ROOT=$(pwd)

```

#### Usage
```
$ deepsig -h

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



Please, reports bugs to: castrense.savojardo2@unibo.it
