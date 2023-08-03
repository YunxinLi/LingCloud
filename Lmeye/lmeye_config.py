import argparse
from argparse import Namespace
from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional

model_type_dict = {
    "FLAN-T5": "encoder-decoder",
    "GLMv2": "decoder-only",
}

IGNORE_INDEX = -100
IMG_INDEX = -101
IMG_Q_INDEX = -102
IMG_D_INDEX = -103

class Config(BaseModel):
    """
    Class defining the train/eval config data in Lmeye model.
    """
    
    # Model config
    seed: int                           = 3407
    ''' Random seed '''
    debug: bool                         = False
    ''' If debug = True, use a small dataset to test program '''
    decoder_only: bool                  = True
    ''' LLM type: encoder-decoder or decoder-only '''
    model_type: str                     = "GLMv2"
    ''' Support model type ["FLAN-T5", "GLMv2"], more models will be supported in the future🤗.'''
    imgq_number: int                    = 1
    ''' The number of <img-q>, recommand 1'''
    imgd_number: int                    = 5
    ''' The number of <img-d>'''

    # Path config: model path, dataset path and checkpoint path
    output_dir: str                     = './output'
    ''' The path to save models '''
    llm_path: str                       = "/root/data/model/blip2-GLM"  #"/root/data/model/blip2-flan-t5-xl"
    ''' The llm path need to load '''
    checkpoint: Optional[str]           = None
    ''' The checkpoint path need to load '''
    dataset: str                        = "/root/data/mutimodel_dataset/data_split/OCR-VQA/"
    ''' The train dataset path need to load '''
    mme_dataset: Optional[str]          = "./dataset/MME"
    ''' MME eval dataset '''
    mmbench_dataset: Optional[str]      = None  # "./dataset/MMBench/mmbench_dev_20230712.tsv"
    ''' MMBench eval dataset '''

    # Training config                 
    num_samples: int                    = 20000
    '''The number of data sampled from the training set in one training epoch'''
    num_train_epochs: int               = 20
    '''The number of train epochs'''
    padding: int                        = 256
    '''The max padding number'''
    batch_size: int                     = 8
    '''Train/eval batch size'''
    gradient_accumulation_steps: int    = 4
    '''Train gradient accumulation steps'''
    logging_steps: int                  = 20
    '''Logging steps'''
    save_steps: int                     = 1000
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

base_config = parse_arguments()