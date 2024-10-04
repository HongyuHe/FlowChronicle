
## Pattern Mining (Training)

1. load the dataset as a pandas dataframe
2. ensure timestamps are first column and repreentent as integers
2. Discretize all columns
2. generate a dataset object
```
from dataloader import Dataset
dataset = Dataset(df)
```
5. Run FlowChronicle
```
import search
model = search.search(dataset)
```
model is of type `model.py > Model`\
use `model.get_patterns()` to get patterns discovered by FlowChronicle

### Optional - change domain knowledge
update `domain_knowledge.py` to allow different search kinds of patterns

## Baselines

5 baselines are included in this repository
E-WGAN-GP and NetShare were not included because based on other repository from other authors(reach them if you want access)

The files related to the baseline are :
1. BN_baseline.py (for both SequenceBN and IndependantBN)
2. cteegan.py (for both CTGAN and TVAE)
3. transformer_baseline/transformer_train_and_generate.py (for Transformer)

### Generate new data from our model and baselines

1. for BNs run:
```
python3 BN_baseline.py path_to_train_csv 5 path_to_output_dir
```
with path_to_train_vsc being data/train.csv and path_to_output_dir being data/ by default, and 5 being the length of the sequence (0 for IndependentBN, 5 for SequenceBN)

2. for CTGAN/TVAE run:
```
python3 BN_baseline.py path_to_train_csv ctgan path_to_output_dir
```
with ctgan being the model you want (ctgan for CTGAN, tvae for TVAE)

3. for Transformer and or method
open either our_train_and_generate.py or transformer_baseline_train_and_generate.py and change the variable path to the directory where the training csv is stored (data/ by default)
then symply run the code (warning: high computational demands)

### Getting the results
If you could not run any programs, all the generated data by all the baseline (including NetShare and E-WGAN-GP) that were used for the evaluation (section 7 of the paper) are present in the data/ folder

## Evaluation

Evaluation can be done in notebook/evaluation.ipynb

The TSTR method should be done independantly (high computational ressource) with launching the code metric.py. The result should then be stored in /results.
Results of the paper are arleady given
