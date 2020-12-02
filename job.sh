#!/bin/bash
#SBATCH --partition=gpu          # partition (queue)
#SBATCH --tasks=1               # number of tasks     <---------- this is different to above
#SBATCH --mem=40G                 # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --output=finall.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=finall.err     # filename for STDERR

# start here your MPI program
python3 train.py
