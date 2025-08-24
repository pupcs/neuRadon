import tensorflow as tf
import numpy as np
import os
import time

def create_nerf(args):
    """Instantiate NeRF's MLP model."""
    models = {'model': None}
    return models

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    return parser


def train():
    #TODO parser does nothing
    parser = config_parser()
    args = parser.parse_args()

    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)

### Pre training stuff ###

    start = 0
    basedir = args.basedir
    expname = args.expname

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    ## Load data
    # "camera" parameters, and CT images 


    ## Create nerf model
    models = create_nerf(args)

    ## Create optimizer
    lrate = args.lrate
    if args.lrate_decay > 0:
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(lrate,
                                                               decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(lrate)
    models['optimizer'] = optimizer

    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step.assign(start)

    ## Short circuit if rendering from a trained model




### Training ### 
    N_iters = 1000000
    print('Begin')

    # Summary writers
    writer = tf.contrib.summary.create_file_writer(
        os.path.join(basedir, 'summaries', expname))
    writer.set_as_default()

    for i in range(start, N_iters):
        time0 = time.time()



### Post Training stuff ###








    print('hello world')




if __name__ == '__main__':
    train()