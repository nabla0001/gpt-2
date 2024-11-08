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
    experiment_path = exp_path / (timestamp + '.pkl')
    checkpoint_path = exp_path / (timestamp + '.ckpt')

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
        learning_rate: float = 0.01
        num_updates: int = 10

    training_config = TrainingConfig()

    # data
    data = OpenWebTextData(args.data_dir)

    # model
    config = GPTConfig()
    model = GPT(config)

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

        print(f'[iter {n_update:04f}/{training_config.num_updates}]\tloss: {loss.item():.2f}')