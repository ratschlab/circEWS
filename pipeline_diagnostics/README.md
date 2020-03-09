Various diagnostics about the pipeline! Mostly for quality assurance and debugging.j

### check_patient_numbers.py
This is about tracking all the patients throughout the stages of the pipeline, to track who we're losing and when/why.

### get_statistics.py
Compute e.g. demographic statistics about the cohort.

### list_patients_with_endpoints.py
Does what it says on the tin.

### summarise_endpoints.py
"Endpoints" here refers to the circulatory failure endpoint. This script is for collecting statistics, e.g. average duration of periods of circulatory failure, and so on.

### temporal_split_info.py
Collect data about the temporal splits (number of patients, start and end times, etc.)
