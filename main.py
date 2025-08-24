import tensorflow as tf
import numpy as np


def config_parser():

    import configargparse
    parser = None
    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()
    
    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)

    print('e')




if __name__ == '__main__':
    train()