import argparse
from argparse import Namespace
from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional

class Config(BaseModel):
    """
    Class defining the train/eval config data in Lmeye model.
    """
    
    seed: int                           = 3407
    '''Random seed'''
    debug: bool                         = False
    '''If debug = True, use a small dataset to test program'''
    output_dir: str                     = './output'
    '''The path to save models'''
    llm_path: str                       = "/root/data/model/blip2-flan-t5-xl"
    '''The llm path need to load'''
    checkpoint: str                     = "/root/data/model/LMEye/checkpoint_with_epoch1.pth"
    '''The checkpoint path need to load'''
    dataset: str                        = "/root/dataset/LLaVA-CC3M-Pretrain-595K/chat.json"
    '''The train dataset path need to load'''
    mme_dataset: Optional[str]          = None # "./dataset/MME"
    '''MME eval dataset'''
    mmbench_dataset: Optional[str]      = None # "./dataset/MMBench/mmbench_dev_20230712.tsv"
    '''MMBench eval dataset'''
    decoder_only: bool                  = False
    '''LLM type: encoder-decoder or deco+der-only'''

    num_train_epochs: int               = 20
    '''The number of train epochs'''
    batch_size: int                     = 2
    '''Train/eval batch size'''
    gradient_accumulation_steps: int    = 16
    '''Train gradient accumulation steps'''
    valid_steps: int                    = 30
    '''Valid steps'''
    save_steps: int                     = 100
    '''Save steps'''

    learning_rate: float                = 1e-4
    '''Learning rate'''
    scheduler: str                      = "constant"
    '''Which learning rate scheduler to use'''

    max_grad_norm: float                = 1.0
    '''Params grad clip'''


def update_config_values(base_config: Config, args_config: Namespace) -> None:
    """
    Map the data obtained from the command line using argparse to the config.

    Args:
        base_config [`Config`]:
            base_config is an object of the [`Config`] class base on [`BaseModel`] class.
        args_config [`Namespace`]:
            argparse data.
    """
    for key, value in vars(args_config).items():
        if value is not None:
            setattr(base_config, key, value)
    
    for key, value in base_config:
        print("{}: {}".format(key, value))


def parse_arguments() -> Config:
    """
    To construct argparse command-line instructions

    Returns:
        base_config [`Config`]:
            base_config is an object of the [`Config`] class base on [`BaseModel`] class.
    """
    base_config = Config()
    parser = argparse.ArgumentParser(description = 'Config data')

    for key, value in base_config:
        if value is not None:
            eval("parser.add_argument('--{args_name}', type = {args_type})"
                .format(args_name = key, args_type = value.__class__.__name__))
        else:
            eval("parser.add_argument('--{args_name}', default = None)"
                .format(args_name = key))

    args = parser.parse_args()
    update_config_values(base_config, args)
    return base_config

config = parse_arguments()