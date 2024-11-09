import torch
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

from data import OpenWebTextData
from gpt import GPT, GPTConfig
from utils import save_checkpoint, load_checkpoint

import argparse
from attr import dataclass
import time
import datetime
from pathlib import Path


# hyperparameters
@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 2.5e-3
    # beta1: float = 0.9
    # beta2: float = 0.95
    n_batches: int = 10
    log_interval: int = 1
    checkpoint_interval: int = 100

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='trains GPT-1 on OpenWebText')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--checkpoint', type=str, help='path to .ckpt')
    parser.add_argument('--exp-name', type=str, required=True, help='short experiment name, used for subfolder')
    parser.add_argument('--exp-dir', type=str, default='experiments', help='root directory for experiments')
    parser.add_argument('--data-dir', type=str, default='data', help='directory for OpenWebText .bin files')
    parser.add_argument('--log-dir', type=str, default='logs', help='Tensorboard log directory')
    args = parser.parse_args()
    print(args)

    # experiment & checkpoint tracking
    exp_path = Path(args.exp_dir) / args.exp_name
    exp_path.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = exp_path / (timestamp + '.ckpt.pt')

    print(f'experiment: {args.exp_name}')
    print(f'exp dir: {exp_path}')

    # tensorboard
    # Path(args.log_dir).mkdir(exist_ok=True)
    log_dir = Path(args.log_dir) / args.exp_name
    writer = SummaryWriter(log_dir)

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
    def get_lr(it):
        import math
        warmup_iters = 2000
        lr_decay_iters = train_config.n_batches
        min_lr = 6e-5
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return train_config.learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (train_config.learning_rate - min_lr)

    start_time = time.time()

    for batch_num in range(start_batch, train_config.n_batches):

        # update learning rate
        lr = get_lr(batch_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        input_tokens, targets = data.get_batch('train', train_config.batch_size, config.context_size)
        input_tokens = input_tokens.to(device)
        targets = targets.to(device)

        logits = model(input_tokens)

        targets = targets.view(-1) # N*CONTEXT_SIZE,
        logits = logits.view(-1, logits.size(-1)) # N*CONTEXT_SIZE, VOCAB_SIZE

        loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1)

        prepare_time = time.time() - start_time

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        process_time = time.time() - prepare_time - start_time
        start_time = time.time()

        if batch_num % train_config.log_interval == 0:
            compute_efficiency = process_time/(process_time+prepare_time)
            print(f'[iter {batch_num:06d}/{train_config.n_batches-1:06d}]\tloss: {loss.item():<.2f}\t'
                  f'compute efficiency {compute_efficiency:.2f}\tprep time {prepare_time:.2f}s\tprocess time {process_time:.2f}s')
            writer.add_scalar('loss', loss.item(), batch_num)
            writer.add_scalar(f'learning rate', optimizer.param_groups[0]['lr'], batch_num) if len(optimizer.param_groups) == 1 else None
            writer.add_scalar('compute efficiency', compute_efficiency, batch_num)
            writer.add_scalar('prep time', prepare_time, batch_num)
            writer.add_scalar('process time', process_time, batch_num)

        # save checkpoint
        if batch_num % train_config.checkpoint_interval == 0 and batch_num > 0:
            print(f'saving checkpoint to {checkpoint_path}')
            kwargs = dict(model_args=model_args,
                          train_config=train_config,
                          batch_num=batch_num)
            save_checkpoint(checkpoint_path, model, optimizer, **kwargs)

    # clean up
    del data

    writer.flush()
    writer.close()