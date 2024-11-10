import torch
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

from data import OpenWebTextData
from gpt import GPT
from config import Config, GPTConfig
from utils import save_checkpoint, load_checkpoint, get_learning_rate

import argparse
import time
import datetime
from pathlib import Path


# hyperparameters


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

        config = checkpoint.get('config')

        # load model
        model = GPT(config.gpt)
        model.to(device)

        state_dict = checkpoint.get('model')
        model.load_state_dict(state_dict)

        # TODO: load optimizer
        optimizer = checkpoint.get('optimizer_class')
        optimizer = optimizer(model.parameters(), lr=config.learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(optimizer)

        start_batch = checkpoint.get('batch_num')
        checkpoint = None
    else:
        print(f'training model from scratch')

        config = Config()
        # TODO: modify from command line

        model = GPT(config.gpt)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        start_batch = 0

    model.train()
    print(config)

    # print model summary
    input_data = torch.randint(0, config.gpt.vocab_size,
                               size=(config.batch_size, config.gpt.context_size),
                               device=device)
    print(summary(model, input_data=input_data))

    start_time = time.time()

    for batch_num in range(start_batch, config.n_batches):

        # update learning rate
        lr = get_learning_rate(batch_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        input_tokens, targets = data.get_batch('train', config.batch_size, config.gpt.context_size)
        input_tokens = input_tokens.to(device)
        targets = targets.to(device)

        _, loss = model(input_tokens, targets)

        prepare_time = time.time() - start_time

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        process_time = time.time() - prepare_time - start_time
        start_time = time.time()

        if batch_num % config.log_interval == 0:
            compute_efficiency = process_time/(process_time+prepare_time)
            print(f'[iter {batch_num:06d}/{config.n_batches-1:06d}]\tloss: {loss.item():<.2f}\t'
                  f'compute efficiency {compute_efficiency:.2f}\tprep time {prepare_time:.2f}s\tprocess time {process_time:.2f}s'
                  f'\ttotal batch time: {process_time+prepare_time:.2f}s')
            writer.add_scalar('loss', loss.item(), batch_num)
            writer.add_scalar(f'learning rate', optimizer.param_groups[0]['lr'], batch_num) if len(optimizer.param_groups) == 1 else None
            writer.add_scalar('compute efficiency [%]', compute_efficiency, batch_num)
            writer.add_scalar('prep time [s]', prepare_time, batch_num)
            writer.add_scalar('process time [s]', process_time, batch_num)
            writer.add_scalar('total batch time [s]', process_time+prepare_time, batch_num)

        # evaluate test loss
        # if batch_num % config.eval_interval == 0 and batch_num > 0:
        #     with torch.no_grad():
        #         val_loss, train_loss = evaluate_loss()

        # save checkpoint
        if batch_num % config.checkpoint_interval == 0 and batch_num > 0:
            print(f'saving checkpoint to {checkpoint_path}')
            kwargs = dict(config=config,
                          optimizer_class=optimizer.__class__,
                          batch_num=batch_num)
            save_checkpoint(checkpoint_path, model, optimizer, **kwargs)

    # clean up
    del data

    writer.flush()
    writer.close()