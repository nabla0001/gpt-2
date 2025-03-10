import argparse
import time
from pathlib import Path
import structlog
from attr import asdict

import torch
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

from data import OpenWebTextData
from gpt import GPT
from config import Config, GPTConfig
from utils import save_checkpoint, load_checkpoint, get_learning_rate, evaluate_loss


log = structlog.get_logger(__name__)


def train(args: argparse.Namespace) -> None:
    """Main training loop. Model/training configuration is defined in config.py

    Args:
        args: additional parameters which can be controlled via command line
                --exp-name
                --exp-path
                --data-dir
                --log-dir
    """

    # experiment & checkpoint tracking
    exp_path = Path(args.exp_dir) / args.exp_name
    exp_path.mkdir(exist_ok=True, parents=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    checkpoint_path = exp_path / (timestamp + '.ckpt.pt')

    log.info('experiment', exp_name=args.exp_name, exp_dir=exp_path)

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
    log.info('device', device=device)

    # data
    data = OpenWebTextData(args.data_dir)

    # resume training if specified
    if args.checkpoint is not None:
        checkpoint = load_checkpoint(args.checkpoint, device=device)
        log.info('resuming training from checkpoint', checkpoint=args.checkpoint)

        config_args = checkpoint.get('config')
        config = Config(**config_args)
        config.gpt = GPTConfig(**config.gpt)

        # load model
        model = GPT(config.gpt)
        model.to(device)

        state_dict = checkpoint.get('model')
        model.load_state_dict(state_dict)

        optimizer = checkpoint.get('optimizer_class')
        optimizer = optimizer(model.parameters(), lr=config.learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer'])
        log.info('reloading optimizer', optimizer=optimizer)

        token_total = checkpoint.get('token_total', 0)
        start_step = checkpoint.get('step')
        best_val_loss = checkpoint.get('best_val_loss')

        # gradient scaler state for mixed precision
        grad_scaler_state_dict = checkpoint.get('grad_scaler')

        checkpoint = None
    else:
        log.info('training model from scratch')

        config = Config()
        # TODO: modify from command line

        model = GPT(config.gpt)
        model.to(device)

        # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=config.learning_rate,
                                      betas=(config.beta1, config.beta2))

        token_total = 0
        start_step = 0
        best_val_loss = float('inf')
        grad_scaler_state_dict = None

    # gradient scaler for mixed precision
    mixed_precision = config.dtype == torch.float16
    log.info('training with mixed precision', dtype=config.dtype)
    # requires running train.py with PYTORCH_ENABLE_MPS_FALLBACK=1 for MPS since some ops are not implemented yet :(
    grad_scaler = torch.amp.GradScaler(device=device.type, enabled=mixed_precision)

    if grad_scaler_state_dict is not None:
        log.info('found grad scaler in checkpoint', **grad_scaler_state_dict)
        grad_scaler.load_state_dict(grad_scaler_state_dict)
        log.info('grad scaler state after re-loading its state', **grad_scaler.state_dict())

    model.train()
    log.info('experiment config', **asdict(config))

    # print model summary
    input_data = torch.randint(0, config.gpt.vocab_size,
                               size=(config.batch_size, config.gpt.context_size),
                               device=device)
    log.info('model summary')
    log.info(summary(model, input_data=input_data))

    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step, config.n_steps):
        t0 = time.time()

        # update learning rate
        lr = get_learning_rate(step, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # accumulate gradients
        loss_accum = 0.0
        for _ in range(config.gradient_accumulation_steps):

            input_tokens, targets = data.get_batch('train', config.batch_size, config.gpt.context_size)
            input_tokens = input_tokens.to(device)
            targets = targets.to(device)

            with torch.amp.autocast(device_type=device.type, dtype=config.dtype):
                _, loss = model(input_tokens, targets)
                loss = loss / config.gradient_accumulation_steps # divide to account for gradient accumulation
                loss_accum += loss.detach()

            # scale gradients for backward pass
            grad_scaler.scale(loss).backward()

        # clip the gradient
        if config.grad_clip > 0.0:
            grad_scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        else:
            grad_norm = -1. # dummy value for logging if we are not clipping
        # parameter update
        grad_scaler.step(optimizer)
        grad_scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        batch_time_ms = (t1 - t0) * 1000
        tokens_per_sec = (config.batch_size * config.gpt.context_size * config.gradient_accumulation_steps) / (t1 - t0)

        token_total += (config.batch_size * config.gpt.context_size * config.gradient_accumulation_steps) / (1000 ** 2)  # in millions

        if step % config.log_interval == 0:
            log.info(f'step [{step:07d}/{config.n_steps:07d}]\tloss={loss_accum:.2f}',
                     tokens_per_sec=f'{tokens_per_sec:.1f}',
                     batch_time_ms=f'{batch_time_ms:.1f}ms',
                     grad_norm=f'{grad_norm:.1f}',
                     token_total=f'{token_total:.3f}M')

            writer.add_scalar('loss/batch', loss_accum, step)
            writer.add_scalar(f'learning rate', optimizer.param_groups[0]['lr'], step) if len(optimizer.param_groups) == 1 else None
            writer.add_scalar('tokens per second', tokens_per_sec, step)
            writer.add_scalar('total batch time [ms]', batch_time_ms, step)

        # evaluate test loss
        if step % config.eval_interval == 0 and step > 0:
            losses = evaluate_loss(model, data, device,
                                   n_batches=config.n_eval_batches,
                                   batch_size=config.batch_size,
                                   seq_len=config.gpt.context_size)
            log.info(f'batch [{step:07d}/{config.n_steps:07d}] evaluation',
                     train_loss=f'{losses['train']:.2f}',
                     test_loss=f'{losses['test']:.2f}',
                     n=config.n_eval_batches*config.batch_size)
            writer.add_scalars('loss', losses, step)
            writer.flush()

            if losses['test'] < best_val_loss:
                best_val_loss = losses['test']
                kwargs = dict(config=asdict(config),
                              optimizer_class=optimizer.__class__,
                              grad_scaler=grad_scaler.state_dict(),
                              step=step,
                              best_val_loss=best_val_loss,
                              token_total=token_total)
                save_checkpoint(checkpoint_path, model, optimizer, **kwargs)
                log.info(f'saved checkpoint to {checkpoint_path}')

    # clean up
    data = None

    writer.flush()
    writer.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='trains GPT-1 on OpenWebText')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to .ckpt')
    parser.add_argument('--exp-name', type=str, required=True, help='short experiment name, used for subfolder')
    parser.add_argument('--exp-dir', type=str, default='experiments', help='root directory for experiments')
    parser.add_argument('--data-dir', type=str, default='data', help='directory for OpenWebText .bin files')
    parser.add_argument('--log-dir', type=str, default='logs', help='Tensorboard log directory')
    args = parser.parse_args()

    train(args)