#!/bin/bash
#SBATCH --partition=gpu4          # partition (queue)
#SBATCH --tasks=10               # number of tasks     <---------- this is different to above
#SBATCH --mem=40G                 # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --output=finall.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=finall.err     # filename for STDERR
#SBATCH --time=2-00:00:00      # total runtime of job allocation ((format D-HH:MM:SS; first parts optional)

# start here your MPI program
python3 train.py
