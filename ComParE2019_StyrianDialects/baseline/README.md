### Run the augmentation and feature extraction
```
sh run_extraction.sh
```

### Run SVM baseline
```
mkdir predictions
python3 baseline.py
python3 baseline_aug.py
```
### Run attention-based CNN
```
# run with optimal hyperparameters
# training
python3 cnn_att.py -mode train -lr 0.0001 -keep_proba 0.5 -batch_size 256 -save_path ./data/results/model/cnn_att -hidden_dim 32 -df_dim 16 
# testing
python3 cnn_att.py -mode test -save_path ./model/cnn_att_25 -hidden_dim 32 -df_dim 16 
```
### Run model fusion and output submission
```
python3 fusion.py
```
