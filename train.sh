config_file=$1
GPUS=$2

python scripts/train.py $config_file $GPUS && \
python scripts/test.py  $config_file $GPUS && \
python scripts/post.py  $config_file