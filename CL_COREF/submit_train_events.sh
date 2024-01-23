#!/bin/bash
#SBATCH --job-name=train-cross-encoder-gpu
#SBATCH -t 18:30:00                  # estimated time # TODO: adapt to your needs
#SBATCH -p grete:shared              # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G A100:1                   # take 1 GPU, see https://www.hlrn.de/doc/display/PUB/GPU+Usage for more options
#SBATCH --mem-per-gpu=30G             # setting the right constraints for the splitted gpu partitions
#SBATCH --nodes=1                    # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --cpus-per-task=4            # number cores per task
#SBATCH --mail-type=all              # send mail when job begins and ends
#SBATCH --mail-user=s.gonzalezorjuela@stud.uni-goettingen.de  # TODO: change this to your mailaddress!
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error

source $WORK/.bashrc
loadconda
module load cuda
echo nvidia-smi 

conda info --envs
conda activate cdcr
which python 
echo $PATH

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"


export http_proxy=http://www-cache.gwdg.de:3128
export https_proxy=http://www-cache.gwdg.de:3128

echo "added http proxy"

# For debugging purposes.
python --version
python -m torch.utils.collect_env
nvcc -V

# Run the script:
python -u train.py --job_id=${SLURM_JOB_ID} --mention_type event --train_batch_size 32 --val_batch_size 32 --epochs 100

# Run the script with logger:
#python -u train_with_logger.py -l ~/${SLURM_JOB_NAME}_${SLURM_JOB_ID}  -t True -p True -d True -s True -f True

# Obtain job information for estimated start and end times
job_info=$(scontrol show job $SLURM_JOB_ID)

# Extract and print estimated start and end times
estimated_start_time=$(echo "$job_info" | grep "StartTime" | cut -d'=' -f2)
estimated_end_time=$(echo "$job_info" | grep "EndTime" | cut -d'=' -f2)

echo "Estimated start time: $estimated_start_time"
echo "Estimated end time: $estimated_end_time"

# Calculate and print the time difference
start_timestamp=$(date -d "$estimated_start_time" +"%s")
end_timestamp=$(date -d "$estimated_end_time" +"%s")

time_difference=$((end_timestamp - start_timestamp))

echo "Estimated running time: $((time_difference / 60)) minutes"