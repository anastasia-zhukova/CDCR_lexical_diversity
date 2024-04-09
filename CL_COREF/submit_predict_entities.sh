#!/bin/bash
#SBATCH -A nimcdcr2
#SBATCH --job-name=train-cross-encoder-gpu
#SBATCH -t 02:00:00                  # estimated time # TODO: adapt to your needs
#SBATCH -p grete:shared              # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G V100:1                   # take 1 GPU, see https://www.hlrn.de/doc/display/PUB/GPU+Usage for more options
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
python -u predict.py --model_path models/entity/roberta-large/tokens_E/siamese_model_batch_64_epochs_50_margin_0.8_threshold_0.2_setup_2.pth

# Run the script with logger:
#python -u train_with_logger.py -l ~/${SLURM_JOB_NAME}_${SLURM_JOB_ID}  -t True -p True -d True -s True -f True


# Obtain job information for estimated start and end times
job_info=$(scontrol show job $SLURM_JOB_ID)

# Extract and print estimated start and end times
estimated_start_time=$(echo "$job_info" | grep "StartTime" | cut -d'=' -f2)
estimated_end_time=$(echo "$job_info" | grep "EndTime" | cut -d'=' -f2)

echo "Estimated start time: $estimated_start_time"
echo "Estimated end time: $estimated_end_time"

# Convert date and time strings to a format that 'date' can understand
start_timestamp=$(date -d "$(echo "$estimated_start_time" | cut -d' ' -f1) $(echo "$estimated_start_time" | cut -d' ' -f2)" +"%s")
end_timestamp=$(date -d "$(echo "$estimated_end_time" | cut -d' ' -f1) $(echo "$estimated_end_time" | cut -d' ' -f2)" +"%s")

# Calculate and print the time difference
time_difference=$((end_timestamp - start_timestamp))
echo "Estimated running time: $((time_difference / 60)) minutes"