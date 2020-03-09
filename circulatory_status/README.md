Scripts and functions for labelling patient timepoints with circulatory failure status.

### make_grid.py
This is basically the wrapper script; it calls on the other files.
It generates a parallel 5-minute grid (or 60, if on MIMIC) with endpoint status labelled at each grid point.

### interpolate_lactate.py
As the definition of circulatory failure depends on lactate, but lactate is not always available, we need to impute its values.

Note that this is separate to the imputation of lactate performed during feature generation; this is purely for labelling circulatory status.

### find_endpoints.py
Logic for finding 'circulatory failure endpoints' for a given patient, and within a given window.

### paths.py
System-specific paths
