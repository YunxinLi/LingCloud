"""
Recode from https://github.com/YunxinLi/LingCloud/blob/main/LMEye/run_llm_instruction.py
"""

import os
import torch
import random
import numpy as np
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
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

from utils.mme_eval import *
from utils.mmbench_eval import *
import torch.nn.functional as F

def train(
        train_dataloader: DataLoader,
        eval_dataloader_dict: Dict,
        model: Blip2InstructionQueryModel,
        lmeye_processor: Blip2Processor,
        clip_tokenizer: CLIPTokenizer
    ) -> None:
    
    t_total = len(train_dataloader) // config.gradient_accumulation_steps * config.num_train_epochs
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = config.learning_rate)

    if config.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = t_total)
    elif config.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps = 0)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = t_total)

    global_loss = 0.0
    new_step = 0
    global_step = 0

    model.zero_grad()
    model.train()
        #shfks
    for epoch in range(config.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            text = ["! " * 37] * batch["inputs"].size(0)
            clip_text = clip_tokenizer(text, return_tensors = "pt")
            clip_text_ids = clip_text["input_ids"].cuda()

            model_output = model.forward(
                labels = batch['target'].squeeze().cuda(),
                input_ids = batch["inputs"].squeeze().cuda(),
                pixel_values =  batch["image"].cuda(),
                imgd_token_id = 32100,
                imgq_token_id = 32101,
                clip_text_input = clip_text_ids,
                attention_mask = batch["attention_mask"].squeeze().cuda(),
            )
            loss: torch.FloatTensor = model_output.loss
            '''
            with torch.no_grad():
                text = ["! " * 39] * batch["inputs"].size(0)
                clip_text = clip_tokenizer(text, return_tensors = "pt")
                clip_text_ids = clip_text["input_ids"].cuda()
                
                generate_ids = model.generate(
                    input_ids = batch["inputs"].squeeze().cuda(),
                    pixel_values =  batch["image"].cuda(),
                    attention_mask = batch["attention_mask"].squeeze().cuda(),
                    imgd_token_id = 32100,
                    imgq_token_id = 32101,
                    num_beams = 5,
                    temperature = 0.2,
                    top_p = 1,
                    top_k = 3,
                    clip_text_input = clip_text_ids,
                    max_new_tokens = 128,
                )

                output = lmeye_processor.batch_decode(generate_ids, skip_special_tokens = True, clean_up_tokenization_spaces = False)
                #print(batch["input_text"])
                #print(batch["ground_truth"])
                #print(output)
                #print(loss)
            '''
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            global_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                new_step += 1
                global_step += 1
                optimizer.step()
                optimizer.zero_grad()

                scheduler.step()
                model.zero_grad()

                if global_step % config.valid_steps == 0:
                    print("when epoch {} and total step {},  the total loss is {}".format(epoch, global_step, global_loss / global_step))

                save_name_list = [
                    'instruction_embedding_imgd',
                    'instruction_embedding_imgq',
                    'language_projection',
                    'instruction_blip22clip',
                    'instruction_linear',
                    'instruction_imgdLinear'
                ]

                if global_step % config.save_steps == 0 and global_step > 0:
                    to_save_param = {}
                    for name, params in model.named_parameters():
                        for save_name in save_name_list:
                            if save_name in name:
                                to_save_param[name] = params
                    state = {'net': to_save_param, 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                    torch.save(
                        state,
                        config.output_dir + "/" + "checkpoint_with_step{}".format(global_step) + ".pth",
                        _use_new_zipfile_serialization = False
                    )

        if "MME" in eval_dataloader_dict and eval_dataloader_dict["MME"] is not None:
            mme_metrics = MMECalculateMetrics(save_path = "/home/Lmeye/output/eval_data/MME")
            mme_metrics.del_data()
            model.eval()

            for step, batch in enumerate(eval_dataloader_dict["MME"]):
                with torch.no_grad():
                    text = ["! " * 37] * batch["inputs"].size(0)
                    clip_text = clip_tokenizer(text, return_tensors = "pt")
                    clip_text_ids = clip_text["input_ids"].cuda()
                    
                    generate_ids = model.generate(
                        input_ids = batch["inputs"].squeeze().cuda(),
                        pixel_values =  batch["image"].cuda(),
                        attention_mask = batch["attention_mask"].squeeze().cuda(),
                        imgd_token_id = 32100,
                        imgq_token_id = 32101,
                        num_beams = 5,
                        temperature = 0.2,
                        top_p = 1,
                        top_k = 3,
                        clip_text_input = clip_text_ids,
                        max_new_tokens = 16,
                    )

                    output = lmeye_processor.batch_decode(generate_ids, skip_special_tokens = True, clean_up_tokenization_spaces = False)
                    mme_metrics.save_data(output, batch)
            torch.cuda.empty_cache()
            mme_metrics.process_result()

        if "MMBench" in eval_dataloader_dict and eval_dataloader_dict["MMBench"] is not None:
            mmbench_metrics = MMBenchCalculateMetrics(save_path = "/home/Lmeye/output/eval_data/MMBench")
            mmbench_metrics.del_data()
            model.eval()

            for step, batch in enumerate(eval_dataloader_dict["MMBench"]):
                with torch.no_grad():
                    text = ["! " * 37] * batch["inputs"].size(0)
                    clip_text = clip_tokenizer(text, return_tensors = "pt")
                    clip_text_ids = clip_text["input_ids"].cuda()
                    
                    generate_ids = model.generate(
                        input_ids = batch["inputs"].squeeze().cuda(),
                        pixel_values =  batch["image"].cuda(),
                        attention_mask = batch["attention_mask"].squeeze().cuda(),
                        imgd_token_id = 32100,
                        imgq_token_id = 32101,
                        num_beams = 5,
                        temperature = 0.2,
                        top_p = 1,
                        top_k = 3,
                        clip_text_input = clip_text_ids,
                        max_new_tokens = 16,
                    )

                    output = lmeye_processor.batch_decode(generate_ids, skip_special_tokens = True, clean_up_tokenization_spaces = False)
                    mmbench_metrics.save_data(output, batch)
            torch.cuda.empty_cache()

def main():
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
 
    # 加载 Lmeye 框架，载入 LLM 模型
    llm_model = Blip2InstructionQueryModel.from_pretrained(config.llm_path)
    llm_processor = Blip2Processor.from_pretrained(config.llm_path)

    # 加载数据集
    train_dataset = TrainDataset(config.dataset, llm_processor)
    train_sampler = RandomSampler(train_dataset, num_samples = 10000)
    train_loader = DataLoader(train_dataset, batch_size = config.batch_size, sampler = train_sampler)
    
    # MME 测试集
    if config.mme_dataset is not None:
        mme_eval_dataset = MMEvalDataset("/home/Lmeye/dataset/MME", llm_processor)
        mme_eval_loader = DataLoader(mme_eval_dataset, batch_size = 8)
    else: mme_eval_loader = None

    # MMBench 测试集
    if config.mmbench_dataset is not None:
        mmbench_eval_dataset = MMBenchDataset("/home/Lmeye/dataset/MMBench/mmbench_dev_20230712.tsv", llm_processor)
        mmbench_eval_loader = DataLoader(mmbench_eval_dataset, batch_size = 6)
    else: mmbench_eval_loader = None

    # 载入 CLIP 模型
    clip_path = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_path)
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_path)
    llm_model.load_clip(clip_model)
    llm_model = llm_model.cuda()

    # 冻结层
    llm_model.query_tokens.requires_grad = False
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
 
    params = torch.load(config.checkpoint, map_location = 'cuda:0')['net']
    llm_model.load_state_dict(params, strict = False)

    for name, para in llm_model.named_parameters():
        if para.requires_grad is True:
            print(name)

    model = SyncBatchNorm.convert_sync_batchnorm(llm_model)

    train(
        train_dataloader = train_loader,
        eval_dataloader_dict = {
            "MME": mme_eval_loader,
            "MMBench": mmbench_eval_loader,
        },
        model = model,
        lmeye_processor = llm_processor,
        clip_tokenizer = clip_tokenizer
    )


if __name__ == "__main__":
    main()
