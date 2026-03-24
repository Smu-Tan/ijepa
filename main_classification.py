import argparse
import multiprocessing as mp
import os
import pprint

import yaml

from src.classification.train import main as app_main
from src.utils.distributed import init_distributed


parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname',
    type=str,
    default='configs/classification.yaml',
    help='name of config file to load',
)
parser.add_argument(
    '--devices',
    type=str,
    nargs='+',
    default=['cuda:0'],
    help='which devices to use on local machine',
)


def process_main(rank, fname, world_size, devices):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    import logging

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    logger.info('called-params %s', fname)

    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pprint.PrettyPrinter(indent=4).pprint(params)

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info('Running... (rank: %s/%s)', rank, world_size)
    app_main(args=params)


if __name__ == '__main__':
    args = parser.parse_args()
    num_gpus = len(args.devices)
    mp.set_start_method('spawn')

    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices),
        ).start()
