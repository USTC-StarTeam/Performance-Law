# README for Predictive Modeling in Sequential Recommendation: Bridging Performance Laws with Data Quality Insights:General Transformer Sequential Recommendation

This README file provides instructions on how to use the `main.py` script for training a sequential recommendation model using a general transformer architecture. The script supports distributed data parallel (DDP) training and is capable of training and evaluating two types of models: `LLaMA2_SASRec` and `HSTU_SASRec`.

## Requirements

- Python 3.7 or later
- PyTorch 1.8 or later
- CUDA (if using GPU)
- `torch.distributed` for distributed training
- WandB for logging (optional)

## Usage

### Training

To start training with distributed data parallelism on multiple GPUs:

```bash
torchrun --standalone --nproc_per_node=<num_gpus> main.py [options]
```

Replace `<num_gpus>` with the number of GPUs you wish to use.

### Arguments

- `--eval_only`: If True, the script will evaluate the model and exit. Default is `False`.
- `--ckpt_name`: The name of the checkpoint file to load if resuming training. Default is `"epoch_0.pth"`.
- `--out_dir`: Directory where models and logs will be saved. Default is `"/root/autodl-tmp/model/sasrec8"`.
- `--n_layers`: Number of layers in the transformer model. Default is `8`.
- `--n_heads`: Number of heads in the multi-head attention mechanism of the transformer. Default is `8`.
- `--batch_size`: Batch size for training. Default is `32`.
- `--model_name`: Selects the model architecture; options are `"llama"` or `"hstu"`. Default is `"llama"`.
- `--local_rank`: Used for distributed training. Default is `0`.
- `--deepspeed`: Path to the DeepSpeed configuration file. Default is `"ds_config.json"`.

### Logs & Checkpoints

- Logs are saved to `[out_dir]/train/log.log`.
- Model checkpoints are saved periodically during training under the `out_dir` directory.

### Evaluation

To perform evaluation only, use the `--eval_only` flag. Ensure the relevant checkpoint file is specified using the `--ckpt_name` option.

### Model Training Details

- Supports cosine learning rate decay with a warm-up phase.
- Training loop supports gradient accumulation.
- Includes functionality to use mixed precision training for enhanced performance on compatible hardware.

### Distributed Training

To perform distributed training, ensure to set the environment variables appropriately or use PyTorch's `torchrun` utility as described above.

## Results

During training and evaluation, various performance metrics are reported including:

- NDCG@10, 50, 200
- HR@10, 50, 200
- MRR

These metrics help evaluate the recommendation quality of the model.

## Notes

- Make sure to adjust the data paths in `data_path_list` to point to your dataset.
- When using DDP, remember to configure the environment variables related to rank and local rank.

## Troubleshooting

- Ensure CUDA is properly configured if using GPU acceleration.
- Follow the WandB setup instructions if using for logging.
- Check data loading paths and formats if encountering dataset related issues.

This repository aims to provide a robust baseline for sequential recommendation in a distributed environment. Adjust model parameters and configurations based on your specific data and requirements.