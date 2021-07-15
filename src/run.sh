#!/bin/bash
tag="autoaug"

# Generate a random ID for the run if no tag is specified
uuid=$(python -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${uuid}
fi


expdir=exp/train_dprnntasnet_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"

python train.py 


