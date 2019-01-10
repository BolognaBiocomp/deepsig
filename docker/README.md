### DeepSig - Predictor of signal peptides in proteins based on deep learning

#### Publication

Savojardo C., Martelli P.L., Fariselli P., Casadio R. [DeepSig: deep learning improves signal peptide detection in proteins](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btx818/4769493) *Bioinformatics* (2017) **34**(10): 1690-1696.


#### Usage of the DeepSig Docker image

The first step to run DeepSig Docker container is the pull the container image. To do so, run:

```
$ docker pull bolognabiocomp/deepsig
```

Now the DeepSig Docker image is installed in your local Docker environment and ready to be used. To show DeepSig help page run:

```
$ docker run deepsig -h

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
$ docker run -v $(pwd):/data/ deepsig -f G5ED35.fasta -o G5ED35.out -k euk
```

In the example above, we are mapping the current program working directory ($(pwd)) to the /data/ folder inside the container. This will allow the container to see the external FASTA file G5ED35.fasta.
The file G5ED35.out now contains the DeepSig prediction:
```
$ cat G5ED35.out

sp|G5ED35|TTR52_CAEEL   SignalPeptide   0.98    20
```
The first column is the protein accession/id as reported in the input fasta file. The second column report the result of signal peptide detection. Three different outcomes are possible: SignalPeptide (i.e. a signal sequence is detected at the N-terminus of the protein), Transmembrane (i.e. a transmembrane region is detected at the N-terminus) or Other (i.e. neither a signal sequence nor a transmembrane region are detected). The third column is a reliabilty score attached to the previous prediction. Finally, the fourth column reports, for those proteins predicted as having a signal peptide, the predicted position of the cleavage site.
