import torch
from torchinfo import summary

from data import OpenWebTextData
from gpt import GPT, GPTConfig

import argparse
from attr import dataclass
from datetime import datetime
from pathlib import Path

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='trains GPT-1 on OpenWebText')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--exp-dir', type=str, default='experiments')
    parser.add_argument('--data-dir', type=str, default='data')
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

    # hyperparameters
    @dataclass
    class TrainingConfig:
        batch_size: int = 32
        learning_rate: float = 2.5e-3
        # beta1: float = 0.9
        # beta2: float = 0.95
        num_updates: int = 50000
        checkpoint_interval: int = 5000

    training_config = TrainingConfig()

    # data
    data = OpenWebTextData(args.data_dir)

    # model
    config = GPTConfig()
    model = GPT(config)
    print(config)

    # resume training if specified
    if args.checkpoint is not None:
        # TODO
        model.train()
        print(f'Resuming training from checkpoint {args.checkpoint}')

    input_data = torch.randint(0, config.vocab_size, size=(training_config.batch_size, config.context_size))
    print(summary(model, input_data=input_data))
    model.to(device)

    # loss & optimiser
    objective = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)

    # learning rate schedule
    # TODO

    update_num = 0
    losses = []

    for n_update in range(training_config.num_updates):

        input_tokens, targets = data.get_batch('train', training_config.batch_size, config.context_size)
        input_tokens = input_tokens.to(device)
        targets = targets.to(device)

        logits = model(input_tokens)

        targets = targets.view(-1) # N*CONTEXT_SIZE,
        logits = logits.view(-1, logits.size(-1)) # N*CONTEXT_SIZE, VOCAB_SIZE

        loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        update_num += 1
        losses.append(loss.item())

        print(f'[iter {n_update:06d}/{training_config.num_updates:6d}]\tloss: {loss.item():.2f}')

        # save checkpoint
        if n_update % training_config.checkpoint_interval == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_config': config,
                'training_config': training_config,
                'iter_num': update_num,
                'loss': losses
            }
            print(f'saving checkpoint to {checkpoint_path}')
            torch.save(checkpoint, checkpoint_path)