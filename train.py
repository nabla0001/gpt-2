"""
TODO
0. clean logging
1. checkpoint saving/resuming
2. learning rate schedule
3. evaluation
4. wandb/tensorboard
5. clean config
"""

import torch
from torchinfo import summary

from data import OpenWebTextData
from gpt import GPT, GPTConfig
from utils import save_checkpoint, load_checkpoint

import argparse
from attr import dataclass
from datetime import datetime
from pathlib import Path


# hyperparameters
@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 2.5e-3
    # beta1: float = 0.9
    # beta2: float = 0.95
    n_batches: int = 10
    checkpoint_interval: int = 2

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='trains GPT-1 on OpenWebText')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--checkpoint', type=str, help='path to .ckpt')
    parser.add_argument('--exp-name', type=str, required=True, help='short experiment name, used for subfolder')
    parser.add_argument('--exp-dir', type=str, default='experiments', help='root directory for experiments')
    parser.add_argument('--data-dir', type=str, default='data', help='directory for OpenWebText .bin files')
    args = parser.parse_args()
    print(args)

    # experiment & checkpoint tracking
    exp_path = Path(args.exp_dir) / args.exp_name
    exp_path.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = exp_path / (timestamp + '.ckpt.pt')

    print(f'experiment: {args.exp_name}')
    print(f'exp dir: {exp_path}')

    # device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'device: {device}')

    # data
    data = OpenWebTextData(args.data_dir)

    # model

    # resume training if specified
    if args.resume:
        print(f'resuming training from checkpoint {args.checkpoint}')
        checkpoint = load_checkpoint(args.checkpoint, device=device)

        model_args = checkpoint.get('model_args')
        config = GPTConfig(**model_args)

        model = GPT(config)
        state_dict = checkpoint.get('model')
        model.load_state_dict(state_dict)

        start_batch = checkpoint.get('batch_num')
    else:
        print(f'training model from scratch')
        model_args = dict() # TODO get from command line or config
        config = GPTConfig(**model_args)
        model = GPT(config)

        start_batch = 0

    model.to(device)
    model.train()

    # hyperparameters
    train_config = TrainingConfig()
    print(train_config)

    # print model summary
    input_data = torch.randint(0, config.vocab_size,
                               size=(train_config.batch_size, config.context_size),
                               device=device)
    print(summary(model, input_data=input_data))

    # optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)

    # learning rate schedule
    # TODO

    losses = []

    for batch_num in range(start_batch, train_config.n_batches):

        input_tokens, targets = data.get_batch('train', train_config.batch_size, config.context_size)
        input_tokens = input_tokens.to(device)
        targets = targets.to(device)

        logits = model(input_tokens)

        targets = targets.view(-1) # N*CONTEXT_SIZE,
        logits = logits.view(-1, logits.size(-1)) # N*CONTEXT_SIZE, VOCAB_SIZE

        loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        print(f'[iter {batch_num:06d}/{train_config.n_batches-1:6d}]\tloss: {loss.item():.2f}')

        # save checkpoint
        if batch_num % train_config.checkpoint_interval == 0:
            print(f'saving checkpoint to {checkpoint_path}')
            kwargs = dict(model_args=model_args,
                          train_config=train_config,
                          batch_num=batch_num)
            save_checkpoint(checkpoint_path, model, optimizer, **kwargs)