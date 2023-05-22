# 将LLM换成OPT模型
# 同时还需要更改预训练任务，不再是MLM

import pickle
import random
import sys
import argparse
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

# from local_transformers.transformers_new import GPT2Tokenizer
from transformers import GPT2Tokenizer, AutoTokenizer
from local_transformers.transformers_new.models.opt import OPTConfig
from local_transformers.transformers_new.models.opt.modeling_opt import OPTForCausalLM
from local_transformers.transformers_new.models.clip import CLIPModel
from local_transformers.transformers_new import DataCollatorForLanguageModeling, PreTrainedTokenizer, BloomConfig, \
    BloomForCausalLM
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from local_transformers.transformers_new.models.llama import LlamaConfig, LlamaForCausalLM
from local_transformers.transformers_new.models.llama.tokenization_llama import LlamaTokenizer
from Data.mapping_dataset import llm_extra_dataset, llm_extra_dataset_ddp
from modeling.modeling_mapping import single_mapping_model_extra_pretrain_linear

from utils.logger import setup_logger
from progressbar import ProgressBar
from utils.misc import (mkdir, set_seed,
                        load_from_yaml_file, find_file_path_in_yaml, padding)
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel
from torch.nn import SyncBatchNorm
from torch.cuda.amp import GradScaler, autocast


def train(args, train_dataloader, model, tokenizer):
    t_total = len(train_dataloader) // args.gradient_accumulation_steps \
              * args.num_train_epochs
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=0, num_training_steps=t_total)
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=0)
    # scaler = GradScaler()
    if args.local_rank == 0:
        logger.info("***** Running training *****")
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Batch size per GPU = %d", args.batch_size)
        logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                    args.batch_size * args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
    if args.parallel is True:
        model = DistributedDataParallel(model, gradient_as_bucket_view=False)
    global_loss = 0.0
    global_loss1 = 0.0
    global_loss2 = 0.0
    new_step = 0
    global_step = 0
    best_acc = 0

    model.zero_grad()
    model.train()

    pbar_len = len(train_dataloader) // args.gradient_accumulation_steps

    for epoch in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_dataloader) // args.gradient_accumulation_steps, desc='training')
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"]
            inputs = tokenizer(input_ids, truncation=True, max_length=args.max_length - 1, add_special_tokens=False)

            input_ids, attention_mask = padding(inputs, eos_token_id=tokenizer.eos_token_id,
                                                pad_token_id=tokenizer.pad_token_id, max_length=args.max_length)
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            image_id = batch["image_id"]
            if not isinstance(image_id, list):
                image_id = image_id.cpu().numpy()

            target = input_ids.clone()
            target[target == tokenizer.pad_token_id] = -100

            img_feat = []

            for i in image_id:
                if "coco" in i:
                    feat = clip_img_feat_coco[i.split("-")[-1]]
                elif "cc3m" in i:
                    feat = clip_img_feat_cc3m[i.split("-")[-1]]
                elif "flickr" in i:
                    feat = clip_img_feat_flickr[i.split("-")[-1]]
                else:
                    feat = clip_img_feat_laion[str(int(i))]

                if not isinstance(feat, torch.Tensor):
                    feat = torch.from_numpy(feat)
                if feat.size(0) != 1:
                    feat = feat.unsqueeze(dim=0)

                img_feat.append(feat)

            img_feat = torch.stack(img_feat).cuda().squeeze(dim=1)
            # with autocast():
            loss = model(llm_text_input=input_ids, llm_attention_mask=attention_mask, labels=target,
                         clip_img_input=img_feat)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # scaler.scale(loss).backward()
            loss.backward()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            global_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                new_step += 1
                global_step += 1

                # scaler.step(optimizer)
                # scaler.update()
                #
                # scheduler.step()
                # model.zero_grad()
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        p.data = p.data.float()
                        if p.grad is not None:
                            p.grad.data = p.grad.data.float()
                optimizer.step()
                optimizer.zero_grad()

                scheduler.step()
                model.zero_grad()

                for name, p in model.named_parameters():
                    if p.requires_grad:
                        p.data = p.data.half()

                if args.local_rank == 0:
                    pbar(step=new_step % pbar_len,
                         info={'Epoch': epoch, 'loss': global_loss / new_step})

                    if global_step % args.valid_steps == 0:
                        logger.info(
                            "when epoch {} and total step {},  the total loss is {}".format(
                                epoch, global_step,
                                global_loss / global_step))

                if global_step % args.save_steps == 0 and global_step > 0:
                    if args.local_rank == 0:
                        to_save_param = {}
                        for name, params in model.named_parameters():
                            if "textLinear" in name or "new_embedding" in name or "imageLinear" in name or "extraLinear" in name:
                                to_save_param[name] = params
                        state = {'net': to_save_param, 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                        torch.save(state,
                                   args.output_dir + "/" + "checkpoint_with_step{}".format(global_step) + ".pth",
                                   _use_new_zipfile_serialization=False)

        if epoch >= args.epoch_begin:
            to_save_param = {}
            if args.local_rank == 0:
                for name, params in model.named_parameters():
                    if "textLinear" in name or "new_embedding" in name or "imageLinear" in name or "extraLinear" in name:
                        to_save_param[name] = params
                state = {'net': to_save_param, 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state,
                           args.output_dir + "/" + "checkpoint_with_epoch{}".format(epoch + 1) + ".pth",
                           _use_new_zipfile_serialization=False)


def init():
    # Load the clip image features extracted in advance
    global clip_img_feat_coco, clip_img_feat_cc3m, clip_img_feat_flickr, clip_img_feat_laion
    print("loading image feature...")
    # clip_img_feat_coco = pickle.load(
        # open("dataset/image_feat/coco_img_large.pkl", "rb"))
    # clip_img_feat_cc3m = pickle.load(open("/data/share/dataset/MutilModalDataset/CC3M/cc3m_img_train.pkl", "rb"))
    # clip_img_feat_flickr = pickle.load(
        # open("dataset/image_feat/img_feat_flickr30k.pkl", "rb"))
    clip_img_feat_laion = pickle.load(open("dataset/image_feat/laion_0to120.pkl", "rb"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_data",
                        default="dataset/laion_0to120.json",
                        type=str)
    parser.add_argument("--pretrain_data1",
                        default="dataset/laion_0to120.json",
                        type=str)

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--epoch_begin", default=0, type=int)
    parser.add_argument("--batch_size", default=96, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--llm_model", default="opt", type=str)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--load_model_dir",
                        default="output/bloomz-7b1/checkpoint_with_epoch1.pth")
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--max_length", default=50, type=float)
    parser.add_argument("--valid_steps", default=100, type=int)
    parser.add_argument("--save_steps", default=5000, type=int)
    parser.add_argument("--output_dir", default="output/bloomz-7b1-continue/", type=str)
    parser.add_argument("--resume", action='store_true')

    args = parser.parse_args()
    # print(args.parallel)
    if args.parallel == True:
        dist.init_process_group(backend='nccl')
    print(args.local_rank)
    torch.cuda.set_device(args.local_rank)

    global logger

    mkdir(args.output_dir)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if args.local_rank == 0:
        logger = setup_logger("mvp_finetune_pmr", args.output_dir, 0)

    # -------------------------加载OPT-iml模型-----------------------
    if args.llm_model == "opt":
        opt_path = "local_transformers/opt-iml-max-1.3b/"
        llm_tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-1.3b")
        llm_config = OPTConfig.from_pretrained(opt_path + "config.json")
        llm_model = OPTForCausalLM.from_pretrained(opt_path, config=llm_config)
    elif args.llm_model == "llama-13b":
        llama_path = "local_transformers/llama-13b-hf/"
        llm_tokenizer = LlamaTokenizer.from_pretrained(llama_path, add_bos_token=False, add_eos_token=True)
        llm_config = LlamaConfig.from_pretrained(llama_path + "config.json")
        llm_model = LlamaForCausalLM.from_pretrained(llama_path, config=llm_config)
        llm_tokenizer.pad_token_id = 0
    elif args.llm_model == "llama-7b":
        llama_path = "local_transformers/llama-13b-hf/"
        llm_tokenizer = LlamaTokenizer.from_pretrained(llama_path, add_bos_token=False, add_eos_token=True)
        llm_config = LlamaConfig.from_pretrained(llama_path + "config.json")
        llm_model = LlamaForCausalLM.from_pretrained(llama_path, config=llm_config)
        llm_tokenizer.pad_token_id = 0
    else:
        bloomz_path = "local_transformers/bloomz-7b1/"
        llm_tokenizer = AutoTokenizer.from_pretrained(bloomz_path)
        llm_config = BloomConfig.from_pretrained(bloomz_path + "config.json")
        llm_model = BloomForCausalLM.from_pretrained(bloomz_path, config=llm_config)

    # special token add
    llm_tokenizer.add_special_tokens({"additional_special_tokens": ["<img>"]})
    llm_model.resize_token_embeddings(len(llm_tokenizer))

    for name, parameter in llm_model.named_parameters():
        parameter.requires_grad = False

    # ---------------------------------------------------

    model = single_mapping_model_extra_pretrain_linear(llm_model=llm_model, clip_size=768,
                                                       llm_size=llm_config.hidden_size)

    model = model.half()

    pretrain_params = torch.load(args.load_model_dir, map_location='cpu')['net']
    imgLinear_params = {}

    for name, params in pretrain_params.items():
        # params = params.float()
        if "imageLinear" in name:
            imgLinear_params[name.split(".")[-1]] = params
        if "new_embedding" in name:
            model.new_embedding = nn.Parameter(params)
    model.imageLinear.load_state_dict(imgLinear_params)

    if args.local_rank == 0:
        for name, para in model.named_parameters():
            if para.requires_grad is True:
                print(name)

    model = model.cuda()
    if args.parallel is True:
        model = SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            logger.info("Training/evaluation parameters %s", args)

    dataset = llm_extra_dataset_ddp(args.pretrain_data, args, isSmall=False,  task="pretrain")
    if args.parallel is False:
        train_sampler = torch.utils.data.RandomSampler(dataset)
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)

    init()
    train(args, train_loader, model, llm_tokenizer)


if __name__ == "__main__":
    main()
