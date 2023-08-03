from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
import datetime

class TrainParams(BaseModel):
    ''' Params used in training '''
    epoch: int

class DataloaderSet():
    ''' Dataloader name and type, you can add new dataloader in this class. '''
    def __init__(self, dataloader_dict: Dict):
        self.train_dataloader: DataLoader = dataloader_dict['train']
        self.mme_dataloader: Optional[DataLoader] = dataloader_dict['MME']
        self.mmbench_dataloader: Optional[DataLoader] = dataloader_dict['MMBench']