#!/bin/bash -l

#####################
# job-array example #
#####################

#SBATCH --job-name=convolution_im2col
#SBATCH --nodes=1
#SBATCH --ntasks=2

# you may not place bash commands before the last SBATCH directive

# define and create a unique scratch directory
# SCRATCH_DIRECTORY=$(pwd)/${SLURM_JOBID}
# mkdir -p ${SCRATCH_DIRECTORY}
# cd ${SCRATCH_DIRECTORY}

# we copy everything we need to the scratch directory
# ${SLURM_SUBMIT_DIR} points to the path where this script was submitted from
# cp ${SLURM_SUBMIT_DIR}/main ${SCRATCH_DIRECTORY}

# we execute the job and time it
# JOB_OUT_DIR=$(pwd)/${SLURM_JOBID}
# mkdir -p ${JOB_OUT_DIR}
# cp ./main ${JOB_OUT_DIR}
# cd ${JOB_OUT_DIR}
#
# echo "now processing task id:: " ${slurm_array_task_id}
# cd ${SLURM_SUBMIT_DIR}
./main -n 5 -v 3 8 16 256 256 3 8 8 
# ./main -n 5 -v 3 8 256 256 127 3 8 8

# rm ./main

# after the job is done we copy our output back to $SLURM_SUBMIT_DIR
# cp ${SCRATCH_DIRECTORY}/output_${SLURM_ARRAY_TASK_ID}.txt 
#  ${SLURM_SUBMIT_DIR}

# we step out of the scratch directory and remove it
# cd ${SLURM_SUBMIT_DIR}
# rm -rf ${SCRATCH_DIRECTORY}

# happy end
exit 0
