"""
Recode from https://github.com/YunxinLi/LingCloud/blob/main/LMEye/run_llm_instruction.py
"""

import os
import torch
import random
import numpy as np
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, AutoModel
from Lmeye.clip.clip_model import CLIPModel
from transformers import CLIPTokenizer

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.nn import SyncBatchNorm
from torch.optim import AdamW

from Lmeye.lmeye_model import Blip2InstructionQueryModel
from Lmeye.lmeye_processor import Blip2Processor
from Lmeye.lmeye_dataset import TrainDataset
from Lmeye.lmeye_config import *

from utils.glm_dataloader import *
from utils.mme_eval import *
from utils.mmbench_eval import *
from utils.train_manager import TrainManager
from utils.type_manager import DataloaderSet, TrainParams

def train(
        dataloader_set: DataloaderSet,
        model: Blip2InstructionQueryModel,
        lmeye_processor: Blip2Processor,
        clip_tokenizer: CLIPTokenizer
    ) -> None:

    train_manager = TrainManager(
        model = model,
        processor = lmeye_processor,
        clip_tokenizer = clip_tokenizer,
        dataloader_set = dataloader_set,
    )

    for epoch in range(base_config.num_train_epochs):
        train_manager.run([
            train_manager.train,
            train_manager.mme_eval,
        ], TrainParams(epoch = epoch))

def main():
    seed = base_config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)

    if base_config.model_type not in model_type_dict:
        raise "model type not in {}".format(model_type_dict)
    base_config.decoder_only = True if model_type_dict[base_config.model_type] == "decoder-only" else False
    
    # 加载 Lmeye 框架，载入 LLM 模型
    llm_processor = Blip2Processor.from_pretrained(base_config.llm_path, trust_remote_code = True)
    llm_model = Blip2InstructionQueryModel.from_pretrained(base_config.llm_path, trust_remote_code = True)
    if base_config.model_type == "GLMv2":
        # need to fix, the GLMv2 modeling index can not used in Blip2InstructionQueryModel.from_pretrained, this is a compromise solution.
        llm_model.language_model = AutoModel.from_pretrained("/root/data/model/ChatGLM2-6B/ChatGLM2-6B", trust_remote_code = True)
 
    # 加载数据集
    train_dataset = TrainDataset(base_config, llm_processor)
    train_sampler = RandomSampler(train_dataset, num_samples = base_config.num_samples)
    train_loader = DataLoader(train_dataset, batch_size = base_config.batch_size, sampler = train_sampler, num_workers = 8)
    
    # MME 测试集
    if base_config.mme_dataset is not None:
        mme_eval_dataset = MMEvalDataset(base_config, llm_processor)
        mme_eval_loader = DataLoader(mme_eval_dataset, batch_size = 16)
    else: mme_eval_loader = None

    # MMBench 测试集
    if base_config.mmbench_dataset is not None:
        mmbench_eval_dataset = MMBenchDataset(base_config, llm_processor)
        mmbench_eval_loader = DataLoader(mmbench_eval_dataset, batch_size = 6)
    else: mmbench_eval_loader = None

    # 载入 CLIP 模型
    clip_path = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_path)
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_path)
    llm_model.load_clip(clip_model)
    llm_model = llm_model.cuda()

    # 冻结层
    # llm_model.query_tokens.requires_grad = False
    # for name, parameter in llm_model.language_projection.named_parameters():
    #    parameter.requires_grad = False

    for name, parameter in llm_model.language_model.named_parameters():
        parameter.requires_grad = False

    for name, parameter in clip_model.named_parameters():
        parameter.requires_grad = False
        
    for name, parameter in llm_model.qformer.named_parameters():
        parameter.requires_grad = False

    for name, parameter in llm_model.vision_model.named_parameters():
        parameter.requires_grad = False
 
    if base_config.checkpoint is not None:
        params = torch.load(base_config.checkpoint, map_location = 'cuda:0')['net']
        llm_model.load_state_dict(params, strict = False)

    for name, para in llm_model.named_parameters():
        if para.requires_grad is True:
            print(name)

    model = SyncBatchNorm.convert_sync_batchnorm(llm_model)

    train(
        dataloader_set = DataloaderSet({
            "train": train_loader,
            "MME": mme_eval_loader,
            "MMBench": mmbench_eval_loader,
        }),
        model = model,
        lmeye_processor = llm_processor,
        clip_tokenizer = clip_tokenizer
    )


if __name__ == "__main__":
    main()
