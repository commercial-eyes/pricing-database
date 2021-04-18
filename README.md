# pricing-database
Backend code to stitch together ex-man price releases from the PBS.

To use, run 'main' and change BASE_DIR to the desired base directory for the database.
Depends on locally available pricing history, i.e., first time run, will download the entire pricing history and build the database.
Otherwise, this will update and append new data only. Lookups are performed based on the most recent data available in the existing local pricing history database.
