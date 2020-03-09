Supervised learning of a continuous risk score for circulatory failure, using
LightGBM and other models including logistic regression.

### cluster_learning_serial.py
Cluster dispatcher for the supervised learning script.

### learning_serial.py
Supervised learning script, which explores all hyperparameter settings, 
performs prediction on test set patients, and stores the prediction
to the file-system.