This experiment uses dropout and layernorm, in the hope of reducing the variance between sets.
Run with ```python3 -m experiments.GCN_Experiment_1.script```

==== Observations ====
Performs quite differently on each fold in K-fold cross validation.
Therefore, next results may wish to use ensembling methods, or maybe dropout.


train_data_path: "data/melting-point/train.csv"
path_to_test: "data/melting-point/test.csv"
output_path: "out/submission.csv"
batch_size: 16
lr: 0.001
weight_decay: 0.000001
seed: 0
model_parameters:
  num_embeddings: 11
  embed_dim: 8
  hidden_channels: 8
  num_layers: 6
logging_directory: "experiments/GCN_Experiment_1/experiment.log"
to_kaggle: True
kaggle_message: "Simple GCN Model."

RESULTS = num layers: 4, avg loss: 6691


Holding all else constant:
but increasing num layers, avg loss: 6668
but increasing inside dimensions: 6596.949918940311