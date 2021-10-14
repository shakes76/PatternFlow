import torch
import argparse
# Hi
print("Hello, world")
print(torch.cuda.is_available())



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true',
                        help='use lr scheduling')
    parser.add_argument('--init_size', default=8, type=int,
                        help='initial image size')
    parser.add_argument('--max_size', default=1024, type=int,
                        help='max image size')
    parser.add_argument(
        '--ckpt', default=None, type=str, help='load from previous checkpoints'
    )
    parser.add_argument(
        '--no_from_rgb_activate',
        action='store_true',
        help='use activate in from_rgb (original implementation)',
    )