Skeleton for data derived from the scripts contained this 
project. The folders are prefixed with numbers indicating
the dependencies between pipeline stages. This data tree 
layout is a suggestion, and a variant of it has been 
used in the circEWS project.

The paths in the other scripts have to be updated to point
to this or another data folder of the user's choice.

### 1_source
Meant for source data downloaded from Physionet

### 2_merged
Merged data with cleaning and artifact deletion performed.

### 3_endpoints
Data frames with state annotations for stability of circulatory 
failure levels.

### 4_imputed
Time series data imputed to a fixed time grid, as well
as imputed static data.

### 5_labels
Labels derived from endpoint data by defining a future prediction
problem.

### 6_ml_input
Non-shapelet features defined for each time point of the fixed grid.

### 7_shapelet_features
Shapelet features derived by discovering shapelets on the training
set and then applying them on the other sets.

### 8_predictions
circEWS prediction scores for each point on the grid, in which 
the model is active.
