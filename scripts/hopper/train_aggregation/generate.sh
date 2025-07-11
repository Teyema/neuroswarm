#!/bin/bash

# stop on error
set -e

for i in {25..35}
do
  # Create a new file for each value
  new_file="aggregation-seed_${i}.slurm"
  cp aggregation-0n.slurmtemplate $new_file

  # Replace the values in the new file
  # sed -i "s/agg-0n/agg-${i}n/g" $new_file
  # sed -i "s/agg\/0/agg\/${i}/g" $new_file
  sed -i "s/seedint/${i}/g" $new_file
done