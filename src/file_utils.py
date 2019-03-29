import json
import pathlib
from datetime import datetime
import os

def build_checkpoint_dir_name(name="Checkpoint"):
    """Builds a dir name"""
    return datetime.now().strftime(name+"-%Y%m%d-%H%M%S")


def build_checkpoint_file_name(dir, descriptor):
    """Builds a relative filename to save the checkpoint at"""
    pathlib.Path("{}/{}/".format(dir, descriptor)).mkdir(exist_ok=True)
    return "{}/{}/params.ckpt".format(dir, descriptor)

def get_dir(name="Checkpoint"):
    dir_name = build_checkpoint_dir_name(name)
    root_dir = os.path.join(os.getcwd(), dir_name)
    pathlib.Path(root_dir).mkdir(exist_ok=True)
    return root_dir