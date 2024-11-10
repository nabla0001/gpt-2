import torch
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

from data import OpenWebTextData
from gpt import GPT
from config import Config, GPTConfig
from utils import save_checkpoint, load_checkpoint, get_learning_rate, evaluate_loss

import argparse
import time
import datetime
from pathlib import Path


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

        optimizer = checkpoint.get('optimizer_class')
        optimizer = optimizer(model.parameters(), lr=config.learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'reloading optimizer')
        print(optimizer)

        start_batch = checkpoint.get('batch_num')
        best_val_loss = checkpoint.get('best_val_loss')

        # to restore gradient scaler state for mixed precision
        grad_scaler_state_dict = checkpoint.get('grad_scaler')

        checkpoint = None
    else:
        print(f'training model from scratch')

        config = Config()
        # TODO: modify from command line

        model = GPT(config.gpt)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        start_batch = 0
        best_val_loss = float('inf')
        grad_scaler_state_dict = None

    # gradient scaler for mixed precision
    mixed_precision = config.dtype == torch.float16
    print(f'training with mixed precision: {mixed_precision} dtype={config.dtype}')
    # requires running train.py with PYTORCH_ENABLE_MPS_FALLBACK=1 for MPS since some ops are not implemented yet :(
    grad_scaler = torch.amp.GradScaler(device=device, enabled=mixed_precision)

    if grad_scaler_state_dict is not None:
        print(f'found grad scaler object. loading its state dict.')
        print(grad_scaler_state_dict)
        grad_scaler.load_state_dict(grad_scaler_state_dict)

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

        with torch.amp.autocast(device_type=config.device, dtype=config.dtype):
            _, loss = model(input_tokens, targets)

        prepare_time = time.time() - start_time

        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)

        grad_scaler.update()

        process_time = time.time() - prepare_time - start_time
        start_time = time.time()

        if batch_num % config.log_interval == 0:
            compute_efficiency = process_time/(process_time+prepare_time)
            print(f'[iter {batch_num:06d}/{config.n_batches-1:06d}]\tloss: {loss.item():<.2f}\t'
                  f'compute efficiency {compute_efficiency:.2f}\tprep time {prepare_time:.2f}s\tprocess time {process_time:.2f}s'
                  f'\ttotal batch time: {process_time+prepare_time:.2f}s')
            writer.add_scalar('loss/batch', loss.item(), batch_num)
            writer.add_scalar(f'learning rate', optimizer.param_groups[0]['lr'], batch_num) if len(optimizer.param_groups) == 1 else None
            writer.add_scalar('compute efficiency [%]', compute_efficiency, batch_num)
            writer.add_scalar('prep time [s]', prepare_time, batch_num)
            writer.add_scalar('process time [s]', process_time, batch_num)
            writer.add_scalar('total batch time [s]', process_time+prepare_time, batch_num)

        # evaluate test loss
        if batch_num % config.eval_interval == 0 and batch_num > 0:
            losses = evaluate_loss(model, data, device,
                                   n_batches=config.n_eval_batches,
                                   batch_size=config.batch_size,
                                   seq_len=config.gpt.context_size)
            print(f"Losses\ttrain: {losses['train']:.2f}\ttest: {losses['test']:.2f}")
            writer.add_scalars('loss', losses, batch_num)
            writer.flush()

            if losses['test'] < best_val_loss:
                best_val_loss = losses['test']
                kwargs = dict(config=config,
                              optimizer_class=optimizer.__class__,
                              grad_scaler=grad_scaler.state_dict(),
                              batch_num=batch_num,
                              best_val_loss=best_val_loss)
                save_checkpoint(checkpoint_path, model, optimizer, **kwargs)
                print(f'saved checkpoint to {checkpoint_path}')

        # save checkpoint
        if batch_num % config.checkpoint_interval == 0 and batch_num > 0:
            kwargs = dict(config=config,
                          optimizer_class=optimizer.__class__,
                          grad_scaler=grad_scaler.state_dict(),
                          batch_num=batch_num,
                          best_val_loss=best_val_loss)
            save_checkpoint(checkpoint_path, model, optimizer, **kwargs)
            print(f'saved checkpoint to {checkpoint_path}')

    # clean up
    del data

    writer.flush()
    writer.close()