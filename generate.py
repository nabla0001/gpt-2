import argparse
from pathlib import Path
import structlog
import tiktoken
from attr import asdict

import torch

from gpt import GPT
from config import Config, GPTConfig
from utils import load_checkpoint


log = structlog.get_logger(__name__)


def generate(args: argparse.Namespace, tokenizer) -> None:
    """Generates new text given a GPT checkpoint and input text file.

    Args:
        args: additional parameters which can be controlled via command line
                --checkpoint
                --input-file
                --temperature
                --top-k
    """

    # load model
    checkpoint = load_checkpoint(args.checkpoint, device=device)

    config_args = checkpoint.get('config')
    config = Config(**config_args)
    config.gpt = GPTConfig(**config.gpt)

    model = GPT(config.gpt)
    model.to(device)

    log.info('model config', **asdict(config))

    # load input data
    text = Path(args.input_file).read_text(encoding='utf-8')
    log.info('read input text', input_file=args.input_file)
    tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    log.info('tokenized input text', tokens=tokens)

    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    tokens = tokens.unsqueeze(0)

    log.info('parameters for sampling',
             max_new_tokens=args.max_new_tokens,
             temperature=args.temperature,
             top_k=args.top_k)

    # generate
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=config.dtype):
            for _ in range(args.num_samples):
                output = model.generate(tokens,
                                        max_new_tokens=args.max_new_tokens,
                                        temperature=args.temperature,
                                        top_k=args.top_k)

                output = output[0].tolist()
                log.info('output', output=output)
                # decode
                output_text = tokenizer.decode(output)

                log.info('output text', output_text=output_text)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GPT text generation')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to .ckpt')
    parser.add_argument('--input-file', type=str, required=True, help='file containing input text')
    parser.add_argument('--num-samples', type=int, default=10, help='number of tokens GPT will generate.')
    parser.add_argument('--max-new-tokens', type=int, default=10, help='number of tokens GPT will generate.')
    parser.add_argument('--temperature', type=float, default=1.0, help='parameter for dividing logits; the closer to 0 the greedier sampling becomes.')
    parser.add_argument('--top-k', type=int, default=None, help='top-k sampling')

    args = parser.parse_args()

    # device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    log.info('device', device=device)

    # tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')

    generate(args, tokenizer)