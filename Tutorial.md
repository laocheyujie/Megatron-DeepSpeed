## 安装
```bash
mkdir github
git clone https://github.com/laocheyujie/Megatron-DeepSpeed.git

mkdir datasets
cd datasets
wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
xz -d oscar-1GB.jsonl.xz

docker pull nvcr.io/nvidia/pytorch:23.12-py3

docker run -dt --gpus all --privileged --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name megatron-deepspeed -v ./Megatron-DeepSpeed:/workspace/Megatron-DeepSpeed -v ./datasets:/workspace/datasets -v /etc/localtime:/etc/localtime -v /root/.ssh:/root/.ssh nvcr.io/nvidia/pytorch:23.12-py3
```

## 修改代码
```bash
vi megatron/optimizer/clip_grads.py +19
# 改成: from torch import inf
vi megatron/arguments.py +738
# 改成: group.add_argument('--local-rank', '--local_rank', type=int, default=None,
vi megatron/model/fused_softmax.py +191
# 注释掉原来这行：# assert mask is None, "Mask is silently ignored due to the use of a custom kernel"
```

## 进入容器
```bash
docker exec -it megatron-deepspeed bash

cd Megatron-DeepSpeed
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install nvitop

# 如果需要单独安装 DeepSpeed 的话
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
pip install .

pip install mpi4py
```

## 数据预处理
```bash
python3 tools/preprocess_data.py \
    --input oscar-1GB.jsonl \
    --output-prefix meg-gpt2 \
    --vocab gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file gpt2-merges.txt \
    --append-eod \
    --workers 8
```

```bash
mkdir data
mv meg-gpt2* ./data
mv ../datasets/gpt2* ./data
```

## 预训练
```bash
vim pretrain_gpt2.sh
```
### 单机单卡
```bash
#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=data/meg-gpt2_text_document
CHECKPOINT_PATH=checkpoints/gpt2

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 2 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 8 \
       --global-batch-size 16 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 5000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file data/gpt2-vocab.json \
       --merge-file data/gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 10 \
       --save-interval 500 \
       --eval-interval 100 \
       --eval-iters 10 \
       --fp16
```

### 单机多卡
$$\text{data\_parallel\_size}=\frac{\text{Number of GPUs}}{\text{tensor-model-parallel-size} \times \text{pipeline-model-parallel-size}}$$

$$\text{global-batch-size}=\text{micro-batch-size}\times \text{data\_parallel\_size}$$

```bash
#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=data/meg-gpt2_text_document
CHECKPOINT_PATH=checkpoints/gpt2

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 2 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 8 \
       --global-batch-size 16 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 5000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file data/gpt2-vocab.json \
       --merge-file data/gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 10 \
       --save-interval 500 \
       --eval-interval 100 \
       --eval-iters 10 \
       --fp16
```

### 多机多卡
```bash
#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=192.168.0.1
MASTER_PORT=6000
NNODES=2
NODE_RANK=$1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=data/meg-gpt2_text_document
CHECKPOINT_PATH=checkpoints/gpt2

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 4 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 8 \
       --global-batch-size 16 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 5000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file data/gpt2-vocab.json \
       --merge-file data/gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 10 \
       --save-interval 500 \
       --eval-interval 100 \
       --eval-iters 10 \
       --fp16
```

## 生成
### 自动式生成文本
```bash
vim generate_text.sh
```
```bash
#!/bin/bash

CHECKPOINT_PATH=checkpoints/gpt2
VOCAB_FILE=data/gpt2-vocab.json
MERGE_FILE=data/gpt2-merges.txt

python tools/generate_samples_gpt.py \
       --tensor-model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 1024 \
       --load $CHECKPOINT_PATH \
       --num-attention-heads 16 \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --fp16 \
       --micro-batch-size 2 \
       --seq-length 1024 \
       --out-seq-length 1024 \
       --temperature 1.0 \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --genfile unconditional_samples.json \
       --num-samples 2 \
       --top_p 0.9 \
       --recompute
```
```bash
sh ./generate_text.sh
```
```bash
cat unconditional_samples.json
```

### 交互式对话模式
```bash
vim interactive_text.sh
```
```bash
#!/bin/bash

CHECKPOINT_PATH=/workspace/Megatron-DeepSpeed/checkpoints/gpt2_345m
VOCAB_FILE=/workspace/Megatron-DeepSpeed/data/gpt2-vocab.json
MERGE_FILE=/workspace/Megatron-DeepSpeed/data/gpt2-merges.txt

deepspeed /workspace/Megatron-DeepSpeed/tools/generate_samples_gpt.py \
       --tensor-model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 1024 \
       --load $CHECKPOINT_PATH \
       --num-attention-heads 16 \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --fp16 \
       --micro-batch-size 2 \
       --seq-length 1024 \
       --out-seq-length 1024 \
       --temperature 1.0 \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --genfile unconditional_samples.json \
       --num-samples 0 \
       --top_p 0.9 \
       --recompute
```
```bash
bash interactive_text.sh
```