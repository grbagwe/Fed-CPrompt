# bash experiments/imagenet-r.sh
# experiment settings
SPLIT=10
DATASET=CIFAR100
N_CLASS=100

# save directory
DATE=ICCV
OUTDIR=_outputs/${DATE}/${DATASET}_centralized/${SPLIT}-task

# hard coded inputs
GPUID='0 1 2 3 4 5 6 7'
#CONFIG_VIT=configs/imnet-r_vit.yaml
CONFIG_VIT_CODA=configs/coda_cifar100debug_centralized.yaml
#CONFIG_VIT_P=configs/imnet-r_vit_prompt.yaml
REPEAT=1
MEMORY=0
OVERWRITE=1
DEBUG=0

###############################################################

# process inputs
if [ $DEBUG -eq 1 ] 
then   
    MAXTASK=3
    SCHEDULE="2"
fi
mkdir -p $OUTDIR

# NOTE - final results are found in _outputs/EXP_NAME/results-acc/global.yaml

# CODA-P
# 
# NOTE - coda-p is currently implemented within the "DualPrompt" class, with a different model file (see configs) and prompting inputs
POOL=100
LEN=8
MU_ORTHO=0.1
python -u run_centralized.py --config $CONFIG_VIT_CODA --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type prompt --learner_name DualPrompt \
    --prompt_param $POOL $LEN 0 0 0 --mu $MU_ORTHO \
    --log_dir ${OUTDIR}/vit/coda-p_cifarcentralized

# DualPrompt
#
# NOTE - for implementation issues, set general prompt length to 6 instead of 5 so that it can be split into half for prefix tuning
#python -u run.py --config $CONFIG_VIT_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
#    --learner_type prompt --learner_name DualPrompt \
#    --prompt_param 10 20 6 \
#    --log_dir ${OUTDIR}/vit/dual-prompt
