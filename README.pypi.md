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


Please, reports bugs to: castrense.savojardo2@unibo.it
