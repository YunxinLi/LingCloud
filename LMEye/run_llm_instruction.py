
import json
import pickle
import sys
import argparse
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import random
import numpy as np
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

# from local_transformers.transformers import GPT2Tokenizer
from transformers import GPT2Tokenizer, AutoTokenizer
from local_transformers.transformers_new.models.opt import OPTConfig
from local_transformers.transformers_new.models.opt.modeling_opt import OPTForCausalLM
from local_transformers.transformers_new.models.clip import CLIPModel, CLIPTokenizer
from local_transformers.transformers_new import DataCollatorForLanguageModeling, PreTrainedTokenizer, BloomConfig, \
    BloomForCausalLM
from local_transformers.transformers_new.models.llama import LlamaConfig, LlamaForCausalLM
from local_transformers.transformers_new.models.llama.tokenization_llama import LlamaTokenizer
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, \
    get_cosine_schedule_with_warmup
from Data.mapping_dataset import llm_extra_dataset, llm_extra_dataset_caption, llm_extra_dataset_instruction
from modeling.modeling_mapping import promot_model_instruction, promot_model_instruction_remake

from utils.logger import setup_logger
from progressbar import ProgressBar
from utils.misc import (mkdir, set_seed,
                        load_from_yaml_file, find_file_path_in_yaml, padding)
from torch.nn.parallel import DistributedDataParallel
from torch.nn import SyncBatchNorm
from torch.cuda.amp import GradScaler, autocast

# DEVICE = 0
# device = torch.device("cuda:{}".format(DEVICE))

def train(args, train_dataloader, model, tokenizer, clip_tokenizer):
    t_total = len(train_dataloader) // args.gradient_accumulation_steps \
              * args.num_train_epochs

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    if args.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=t_total)
    elif args.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=0
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=t_total)
    # scaler = GradScaler()

    if args.parallel is True:
        model = DistributedDataParallel(model, gradient_as_bucket_view=False)

    # scheduler = ConstantLRSchedule(optimizer)
    if args.local_rank == 0:
        logger.info("***** Running training *****")
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Batch size per GPU = %d", args.batch_size)
        logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                    args.batch_size * args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

    global_loss = 0.0
    global_dis_loss = 0.0
    global_lm_loss = 0.0
    new_step = 0
    global_step = 0
    best_acc = 0

    model.zero_grad()
    model.train()

    pbar_len = len(train_dataloader) // args.gradient_accumulation_steps

    for epoch in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_dataloader) // args.gradient_accumulation_steps, desc='training')
        for step, batch in enumerate(train_dataloader):
            pre_input_ids = batch["pre_input_ids"]
            pre_inputs = tokenizer(pre_input_ids, add_special_tokens=False)

            pre_input_ids, pre_attention_mask = padding(pre_inputs, eos_token_id=tokenizer.eos_token_id,
                                                pad_token_id=tokenizer.pad_token_id, max_length=args.max_pre_length, bos_token_id=tokenizer.bos_token_id, add_bos=args.add_bos)

            pre_input_ids = pre_input_ids.cuda(non_blocking=True)
            pre_attention_mask = pre_attention_mask.cuda(non_blocking=True)

            input_ids = batch["input_ids"]

            inputs = tokenizer(input_ids, add_special_tokens=False)

            input_ids, attention_mask = padding(inputs, eos_token_id=tokenizer.eos_token_id,
                                                pad_token_id=tokenizer.pad_token_id, max_length=args.max_length, bos_token_id=tokenizer.bos_token_id, add_bos=args.add_bos)

            input_ids = input_ids.cuda(non_blocking=True)
            attention_mask = attention_mask.cuda(non_blocking=True)
            image_id = batch["image_id"]
            if not isinstance(image_id, list):
                image_id = image_id.cpu().numpy()


            target = input_ids.clone()
            target[target == tokenizer.pad_token_id] = -100
            # padding to -100 for the prompt
            for idx, _ in enumerate(target):
                answer_ids = tokenizer("Answer", add_special_tokens=False).input_ids[0]
                pos = inputs["input_ids"][idx].index(answer_ids)
                target[idx, :pos+2] = -100

            text = ["! " * (args.prompt_len + 1)] * input_ids.size(0)
            img_feat = []
            for i in image_id:
                i = str(i)
                if "coco" in i:
                    feat = clip_img_feat_coco[i.split("-")[-1]]
                # elif "cc3m" in i:
                #     feat = clip_img_feat_cc3m[i.split("-")[-1]]
                elif "flickr" in i:
                    feat = clip_img_feat_flickr[i.split("-")[-1]]
                else:
                    feat = clip_img_feat_semart[i[7:]]
                if not isinstance(feat, torch.Tensor):
                    feat = torch.from_numpy(feat)
                if feat.size(0) == 768:
                    feat = feat.unsqueeze(dim=0)
                img_feat.append(feat)

            clip_text = clip_tokenizer(text, padding=True, return_tensors="pt", truncation=True)
            clip_text_ids = clip_text["input_ids"].cuda(non_blocking=True)
            clip_attention_mask = clip_text["attention_mask"].cuda(non_blocking=True)
            img_feat = torch.stack(img_feat).cuda(non_blocking=True).squeeze(dim=1)

            # with autocast():
            lm_loss = model(llm_text_input=input_ids, llm_attention_mask=attention_mask, labels=target,
                            pre_llm_text_input=pre_input_ids, pre_llm_attention_mask=pre_attention_mask,
                            clip_text_input=clip_text_ids, clip_attention_mask=clip_attention_mask,
                            clip_img_input=img_feat, img_token_id=img_token_id, imgd_token_id=imgd_token_id)

            loss = lm_loss
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            global_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                new_step += 1
                global_step += 1
                # scaler.step(optimizer)
                # scaler.update()
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
                    if args.local_rank == 0:
                        logger.info(
                            "when epoch {} and total step {},  the total loss is {}".format(
                                epoch, global_step,
                                global_loss / global_step, ))

                if global_step % args.save_steps == 0 and global_step > 0:
                    if args.local_rank == 0:
                        to_save_param = {}
                        for name, params in model.named_parameters():
                            if "imgLinear" in name or "imgdLinear" in name or "embedding_img" in name or "embedding_imgd" in name or "promptMlp" in name or "llm2clip" in name:
                                to_save_param[name] = params
                        state = {'net': to_save_param, 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                        torch.save(state,
                                   args.output_dir + "/" + "checkpoint_with_step{}".format(global_step) + ".pth",
                                   _use_new_zipfile_serialization=False)


        if epoch >= args.epoch_begin:
            if args.local_rank == 0:
                to_save_param = {}
                for name, params in model.named_parameters():
                    if "imgLinear" in name or "imgdLinear" in name or "embedding_img" in name or "embedding_imgd" in name or "promptMlp" in name or "llm2clip" in name:
                        to_save_param[name] = params
                state = {'net': to_save_param, 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state,
                           args.output_dir + "/" + "checkpoint_with_epoch{}".format(epoch) + ".pth",
                           _use_new_zipfile_serialization=False)



def init(args):
    global clip_img_feat_coco, clip_img_feat_cc3m, clip_img_feat_flickr, clip_img_feat_semart
    print("loading image features...")
    # clip_img_feat = pickle.load(open("coco_data/img_feat/coco_img_large.pkl", "rb"))
    # clip_img_feat_cc3m = pickle.load(open("dataset/CC3M/cc3m_img_train.pkl", "rb"))

    clip_img_feat_coco = pickle.load(open("dataset/image_feat/coco_img_large.pkl", "rb"))
    # clip_img_feat_cc3m = pickle.load(open("dataset/image_feat/cc3m_img_train.pkl", "rb"))
    clip_img_feat_flickr = pickle.load(open("dataset/image_feat/img_feat_flickr30k.pkl", "rb"))
    clip_img_feat_semart = pickle.load(open("dataset/image_feat/semart_img_train.pkl", "rb"))

    print("image features load success!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_data",
                        default="dataset/instruction_data_balanced_m.json",
                        # default="dataset.json",
                        type=str)


    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--local_rank", default=0, type=int)


    parser.add_argument("--epoch_begin", default=0, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--adam_epsilon", default=1e-5, type=float)
    parser.add_argument("--prompt_len", default=5, type=int)
    parser.add_argument("--max_pre_length", default=-1, type=int)
    parser.add_argument("--max_length", default=300, type=int)
    parser.add_argument("--llm_model", default="opt", type=str)
    parser.add_argument("--scheduler", default="linear", type=str)
    parser.add_argument("--unfreeze_imageLinear", action="store_true")

    parser.add_argument("--pretrain_ckpt_dir_llama_7b", type=str,
                        default='output/llama-imageLinear/checkpoint_with_epoch1.pth')
    parser.add_argument("--pretrain_ckpt_dir_llama_13b", type=str,
                        default='output/llama-imageLinear-13b/checkpoint_with_epoch1.pth')
    parser.add_argument("--pretrain_ckpt_dir_opt", type=str,
                        default='output/opt-imageLinear/checkpoint_with_epoch1.pth')
    parser.add_argument("--pretrain_ckpt_dir_bloomz", type=str,
                        default="output/bloomz-7b1/checkpoint_with_step15000.pth")
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--valid_steps", default=50, type=int)
    parser.add_argument("--query_len", default=1, type=int)
    parser.add_argument("--save_steps", default=3000, type=int)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--output_dir", default="output/tmp", type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_ckpt_dir", default="output/bloomz-instruction/checkpoint_with_step6000.pth")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--add_bos",action="store_true")
    parser.add_argument("--do_predict", action="store_true")

    args = parser.parse_args()
    # print(args.pretrain_ckpt_dir_llama)
    # _ = torch.load(args.pretrain_ckpt_dir_llama, map_location='cpu')['net']
    print(args.local_rank)
    if args.parallel:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    global logger

    mkdir(args.output_dir)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



    if args.local_rank == 0:
        print("do training......")
        logger = setup_logger("mvp_finetune_pmr", args.output_dir, 0)
        logger.warning("Device: %s, n_gpu: %s", torch.cuda.current_device(), torch.cuda.device_count())

    # ------------------------------------------------
    if args.llm_model == "opt":
        pretrain_ckpt_dir = args.pretrain_ckpt_dir_opt

        opt_path = "local_transformers/opt-iml-max-1.3b/"
        llm_tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-1.3b")
        llm_config = OPTConfig.from_pretrained(opt_path + "config.json")
        llm_model = OPTForCausalLM.from_pretrained(opt_path, config=llm_config)
    elif args.llm_model == "llama-7b":
        pretrain_ckpt_dir = args.pretrain_ckpt_dir_llama_7b

        llama_path = "local_transformers/llama-7b-hf/"
        llm_tokenizer = LlamaTokenizer.from_pretrained(llama_path, add_bos_token=False, add_eos_token=True)
        llm_config = LlamaConfig.from_pretrained(llama_path + "config.json")
        llm_model = LlamaForCausalLM.from_pretrained(llama_path, config=llm_config)
        llm_tokenizer.pad_token_id = 0
    elif args.llm_model == "llama-13b":
        pretrain_ckpt_dir = args.pretrain_ckpt_dir_llama_13b

        llama_path = "local_transformers/llama-13b-hf/"
        llm_tokenizer = LlamaTokenizer.from_pretrained(llama_path, add_bos_token=False, add_eos_token=True)
        llm_config = LlamaConfig.from_pretrained(llama_path + "config.json")
        llm_model = LlamaForCausalLM.from_pretrained(llama_path, config=llm_config)
        llm_tokenizer.pad_token_id = 0
    else:
        pretrain_ckpt_dir = args.pretrain_ckpt_dir_bloomz
        bloomz_path = "local_transformers/bloomz-7b1/"
        llm_tokenizer = AutoTokenizer.from_pretrained(bloomz_path)
        llm_config = BloomConfig.from_pretrained(bloomz_path + "config.json")
        llm_model = BloomForCausalLM.from_pretrained(bloomz_path, config=llm_config)

    # special token add
    llm_tokenizer.add_special_tokens({"additional_special_tokens": ["<img>"]})
    llm_tokenizer.add_special_tokens({"additional_special_tokens": ["<img-d>"]})
    global img_token_id, imgd_token_id
    img_token_id = llm_tokenizer.vocab_size
    imgd_token_id = img_token_id + 1

    llm_model.resize_token_embeddings(len(llm_tokenizer))

    for name, parameter in llm_model.named_parameters():
        parameter.requires_grad = False

    # --------------------------加载CLIP模型-------------------------
    clip_path = "local_transformers/clip-vit-large-patch14"

    clip_model = CLIPModel.from_pretrained(clip_path)
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_path)

    for name, parameter in clip_model.named_parameters():
        parameter.requires_grad = False

    model = promot_model_instruction_remake(llm_model=llm_model, clip_model=clip_model, clip_size=768, llm_size=llm_config.hidden_size,
                                prompt_len=args.prompt_len, query_len=args.query_len)

    # loading the first-stage pretraining parameter
    pretrain_params = torch.load(pretrain_ckpt_dir, map_location='cpu')['net']
    imgLinear_params = {}

    for name, params in pretrain_params.items():
        params = params.float()
        if "imageLinear" in name:
            imgLinear_params[name.split(".")[-1]] = params
        if "new_embedding" in name:
            model.embedding_img = nn.Parameter(params)
    model.imgLinear.load_state_dict(imgLinear_params)

    if args.resume:
        imgLinear_params = {}
        imgdLinear_params = {}
        promptMlp_params = {}
        llm2clip_params = {}
        resume_params = torch.load(args.resume_ckpt_dir, map_location='cpu')['net']
        for name, params in resume_params.items():
            if "imgLinear" in name:
                imgLinear_params[name.split(".")[-1]] = params
            if "imgdLinear" in name:
                imgdLinear_params[name.split(".")[-1]] = params
            if "embedding_img" in name:
                model.embedding_img = nn.Parameter(params)
            if "embedding_imgd" in name:
                model.embedding_imgd = nn.Parameter(params)
            if "promptMlp" in name:
                promptMlp_params[name.split("p.")[-1]] = params
            if "llm2clip" in name:
                llm2clip_params[name.split(".")[-1]] = params
        model.imgLinear.load_state_dict(imgLinear_params)
        model.imgdLinear.load_state_dict(imgdLinear_params)
        model.promptMlp.load_state_dict(promptMlp_params)
        model.llm2clip.load_state_dict(llm2clip_params)

    if not args.unfreeze_imageLinear:
        model.embedding_img.requires_grad = False
        for name, params in model.imgLinear.named_parameters():
            params.requires_grad = False


    model = model.half()
    model = model.cuda()

    for name, para in model.named_parameters():
        if para.requires_grad is True:
            print(name)

    if args.local_rank == 0:
        logger.info("Training/evaluation parameters %s", args)

    dataset = llm_extra_dataset_instruction(args.pretrain_data, llm_tokenizer, args, isSmall=False)

    if args.parallel:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
    model = SyncBatchNorm.convert_sync_batchnorm(model)

    init(args)
    train(args, train_loader, model, llm_tokenizer, clip_tokenizer)



if __name__ == "__main__":
    main()
