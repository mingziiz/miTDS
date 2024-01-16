# miTDS: Uncovering miRNA-mRNA Interactions with Deep Learning for Functional Target Prediction


<br/>

## Abstract
<p style="text-align:justify">
MicroRNAs (miRNAs) are vital in regulating gene expression through binding to specific target sites on messenger RNAs (mRNAs), a process closely tied to cancer pathogenesis. Identifying miRNA functional targets is essential but challenging due to incomplete genome annotation and an emphasis on known miRNA-mRNA interactions, restricting predictions of unknown ones. To address those challenges, we have developed a deep learning model based miRNA functional target identification, named miTDS, to investigate miRNA–mRNA interactions. miTDS first employs a scoring mechanism to eliminate unstable sequence pairs and then utilizes a dynamic word embedding model based on the transformer architecture, enabling a comprehensive analysis of miRNA-mRNA interaction sites by harnessing the global contextual associations of each nucleotide. On this basis, miTDS fuses extended seed alignment representations learned in the multi-scale attention mechanism module with dynamic semantic representations extracted in the RNA-based dual-path module, which can further elucidate and predict miRNA and mRNA functions and interactions. To validate the effectiveness of miTDS, we conducted a thorough comparison with state-of-the-art miRNA-mRNA functional target prediction methods. The evaluation, performed on a dataset cross-referenced with entries from  MirTarbase and Diana-TarBase, revealed that miTDS surpasses current methods in accurately predicting functional targets. In addition, our model exhibited proficiency in identifying A-to-I RNA editing sites, which represents an aberrant interaction that yields valuable insights into the suppression of cancerous processes.
<br/>
</p>
<br/>

## Installation
We recommend creating a conda environment from the <code>miTDS.yml</code> file as:
```
conda env create -f miTDS.yml
```


## Data Format
Extract the <code>data.tar</code> file to obtain the dataset used for the manuscript as
```
tar -zxvf data.rar
```
If you want to use other datasets, please follow the data format described as follows

- The dataset must be in a tab-delimited file with **at least 4 columns**
- The first row must be a header line (thus, will not be processed by the TargetNet algorithm).
- The 1st ~ 4th columns must hold the following information
    - [1st column] miRNA id
    - [2nd column] miRNA sequence
    - [3rd column] mRNA id
    - [4th column] mRNA sequence -- mRNA sequence must be longer than 40 nucleotides
- For the file containing train and validation datasets, it requires **additional 2 columns**
    - [5th column] label -- *0* or *1*
    - [6th column] split -- *train* or *val*
    
Please refer to the provided dataset files for more detailed examples.
<br/><br/>

## Pretrained Models
Due to the capacity limitations of GitHub, we have placed the relevant files (including BERT and pretrained models) on Google Drive. We offer BERT models of various kmer lengths, which you can download and try out as needed to explore the impact of different kmer lengths on the results. Below are the links to each of the models:
<br/>
[3merBERT](https://drive.google.com/drive/folders/18p9U_dTefrEgsRhf3iQgzAJ2pI6giBAS?usp=drive_link)
<br/>
[4merBERT](https://drive.google.com/drive/folders/13FXzXmWBYhiFfNYgIhYeSM8uIrcslAZ7?usp=drive_link)
<br/>
[5merBERT](https://drive.google.com/drive/folders/1vmbF0iRv1iacTCIVew8q07_wyAQmMFy7?usp=drive_link)
<br/>
[6merBERT](https://drive.google.com/drive/folders/18LzDAXAgCYvPqzNWbWZriruEFw4BHlb5?usp=drive_link)
<br/>
[pretrained_model](https://drive.google.com/drive/folders/127Hq_pk7KfCzvNZk9JooQ9L6N7Yqac3y?usp=drive_link)
<br/>

These models can assist you in studying and comparing the effects of different kmer lengths (ranging from 3 to 6) on bioinformatics analysis and predictions. Such comparisons can provide deeper insights for your research or projects.


## How to Run
### Training a TargetNet model
You can use the <code>train_model.py</code> script with the necessary configuration files as
```
CUDA_VISIBLE_DEVICES=0 python train_model.py --data-config config/data/miRAW_train.json --model-config config/model/miTDS.json --run-config config/run/run.json --output-path results/miTDS_training/
```
The script will generate a <code>TargetNet.pt</code> file containing a trained model. <br>
For using other datasets, modify the data paths specified in the <code>miRAW_train.json</code> data-config file.

### Evaluating a TargetNet model
You can use the <code>evaluate_model.py</code> script with the necessary configuration files as
```
CUDA_VISIBLE_DEVICES=0 python evaluate_model.py --data-config config/data/miRAW_eval.json --model-config config/model/miTDS.json --run-config config/run/run.json --checkpoint pretrained_models/miTDS.pt --output-path results/miTDS-evaluation/
```
The script will generate a tab-delimited <code>*_outputs.txt</code> file described as follows

- The output file contains a TargetNet prediction for each miRNA-mRNA set from the given dataset.
    - [1st column] set_idx -- indicates the set index from the given dataset.
    - [2nd column] output -- TargetNet prediction score ranging from 0 to 1.
- For binary classification results, use a threshold of 0.5 to binarize the output scores.

For using other datasets, modify the data paths specified in the <code>miRAW_eval.json</code> data-config file.
<br/><br/><br/>
