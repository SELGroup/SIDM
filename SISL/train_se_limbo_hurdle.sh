PARENT_DIR="exp_result_se"  # New directory for experiment results
mkdir -p ${PARENT_DIR}  # Create parent directory if it doesn't exist

for ENV_NAME in BiskGaps-v1 BiskLimbo-v1
do
    DIR_NAME="${PARENT_DIR}/walker_${ENV_NAME}"
    CHECKPOINT_PATH="${DIR_NAME}/checkpoint.pt"
    mkdir -p ${DIR_NAME}  # Create directory if it doesn't exist
    #export CUDA_VISIBLE_DEVICES=1

    nohup  python train.py -cn walker_se  checkpoint_path=${CHECKPOINT_PATH} visdom.logfile=${DIR_NAME}/visdom.log env.name=${ENV_NAME} > ${DIR_NAME}/output.log 2>&1 &
done
