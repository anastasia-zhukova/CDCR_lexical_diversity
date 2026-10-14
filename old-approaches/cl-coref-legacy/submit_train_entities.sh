#!/bin/bash
#SBATCH -A nimcdcr2
#SBATCH --job-name=train-cross-encoder-gpu
#SBATCH -t 23:00:00                  # estimated time # TODO: adapt to your needs
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

#Common arguments
MENTION_TYPE=entity
TRAIN_BATCH_SIZE=64
NUM_EPOCHS=50
MARGIN=0.8
THRESHOLD=0.2
LEARNING_RATE=2e-6
DATA_SETUP=2
METRIC=pd_lea
LANGUAGE_MODEL=roberta-large
#LANGUAGE_MODEL=deberta-large
#LANGUAGE_MODEL=bart-large
#LANGUAGE_MODEL=flan-t5-large
# LANGUAGE_MODEL=conv-bert-base
# LANGUAGE_MODEL=deberta-v2-xlarge
#LANGUAGE_MODEL=da-roberta-large
# LANGUAGE_MODEL=roberta-allen-dapt5000

python -u train.py --job_id=${SLURM_JOB_ID} --mention_type ${MENTION_TYPE} --train_batch_size ${TRAIN_BATCH_SIZE} --val_batch_size 256 --num_epochs ${NUM_EPOCHS} --margin ${MARGIN} --clustering_threshold ${THRESHOLD} --data_setup ${DATA_SETUP} --language_model "${LANGUAGE_MODEL}" --learning_rate ${LEARNING_RATE} --metric "${METRIC}" --additional_sampling 
#Prediction after model training
MODEL_PATH=models/${MENTION_TYPE}/${LANGUAGE_MODEL}/tokens_E/siamese_model_batch_${TRAIN_BATCH_SIZE}_epochs_${NUM_EPOCHS}_margin_${MARGIN}_threshold_${THRESHOLD}_setup_${DATA_SETUP}.pth

echo "Model path: ${MODEL_PATH}"
python -u predict.py --model_path ${MODEL_PATH}



# Run the script with logger:
#python -u train_with_logger.py -l ~/${SLURM_JOB_NAME}_${SLURM_JOB_ID}  -t True -p True -d True -s True -f True

# Obtain job information for estimated start and end times
job_info=$(scontrol show job $SLURM_JOB_ID)
echo "Attempt to print elapsed time:"
elapsed_time=$(echo "$job_info" | grep -oP 'Elapsed: \K[0-9]+ min' | awk '{print $1}')
echo "Current node: ${elapsed_time}"