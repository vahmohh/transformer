#!/bin/bash
#SBATCH --partition=any          # partition (queue)
#SBATCH --tasks=80               # number of tasks     <---------- this is different to above
#SBATCH --mem=4G                 # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time=2:00              # total runtime of job allocation ((format D-HH:MM:SS; first parts optional)
#SBATCH --output=slurm.%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=slurm.%j.err     # filename for STDERR

# start here your MPI program
python3 train.py
