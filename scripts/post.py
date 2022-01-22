import argparse
import os
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import lib.dataset as dataset
from lib.utils.util import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMN")
    parser.add_argument("config", metavar="FILE", type=str)  # path to config file
    args = parser.parse_args()

    # load settings
    cfg = load_config(args.config)

    # post processing according dataset
    ## proposal evaluation
    prop_process = eval("dataset.prop_%s" % cfg.DATASET.name)
    prop_process(cfg)

    ## detection evaluation
    det_process = eval("dataset.det_%s" % cfg.DATASET.name)
    det_process(cfg)
