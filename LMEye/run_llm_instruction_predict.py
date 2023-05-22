import json
import pickle
import sys
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import clip
from PIL import Image


import torch
import random
import numpy as np
import torch.distributed as dist
from torch import nn

from torch.utils.data import DataLoader
from torch.optim import AdamW

# from local_transformers.transformers import GPT2Tokenizer
from transformers import GPT2Tokenizer
from local_transformers.transformers_new.models.opt import OPTConfig
from local_transformers.transformers_new.models.opt.modeling_opt import OPTForCausalLM
from local_transformers.transformers_new.models.clip import CLIPModel, CLIPTokenizer, CLIPProcessor, CLIPImageProcessor
from local_transformers.transformers_new import DataCollatorForLanguageModeling, PreTrainedTokenizer, AutoTokenizer, \
    BloomConfig, BloomForCausalLM
from local_transformers.transformers_new.models.llama import LlamaConfig, LlamaForCausalLM
from local_transformers.transformers_new.models.llama.tokenization_llama import LlamaTokenizer
from Data.mapping_dataset import llm_extra_dataset, llm_extra_dataset_caption, llm_extra_dataset_ddp, \
    llm_extra_dataset_vqa, llm_extra_dataset_predict, llm_extra_dataset_instruction_predict, \
    llm_extra_dataset_instruction_predict_llama
from modeling.modeling_mapping import promot_model_instruction_remake

from utils.logger import setup_logger
from progressbar import ProgressBar
from utils.misc import (mkdir, set_seed,
                        load_from_yaml_file, find_file_path_in_yaml, padding)



# DEVICE = 0
# device = torch.device("cuda:{}".format(DEVICE))
def generate(args, dataloader, model, tokenizer, clip_tokenizer, clip_img_feat):
    model.eval()
    clip_processor = CLIPImageProcessor.from_pretrained("local_transformers/clip-vit-large-patch14")

    pbar = ProgressBar(n_total=len(dataloader), desc='testing')
    outputs = []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            input_text = batch["input_ids"]
            inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            if args.add_bos:
                input_ids = torch.concat([torch.tensor([[1]]), input_ids], dim=1)
                attention_mask = torch.concat([torch.tensor([[1]]), attention_mask], dim=1)

            input_ids = input_ids.cuda(non_blocking=True)
            attention_mask = attention_mask.cuda(non_blocking=True)


            text = ["! " * (args.prompt_len + 1)] * input_ids.size(0)

            clip_text = clip_tokenizer(text, max_length=77, padding=True, return_tensors="pt", truncation=True)
            clip_text_ids = clip_text["input_ids"].cuda()
            clip_attention_mask = clip_text["attention_mask"].cuda()

            if args.generate_with_image:
                image_id = batch["image_id"]
                image = Image.open(image_id[0]).convert('RGB')
                image = torch.from_numpy(clip_processor(image)['pixel_values'][0]).cuda().half()
                generate_ids = model.generate_with_image(llm_text_input=input_ids,
                                              pre_llm_text_input=input_ids, pre_llm_attention_mask=attention_mask,
                                              image=image, clip_text_input=clip_text_ids,
                                              clip_attention_mask=clip_attention_mask, img_token_id=img_token_id,
                                              imgd_token_id=imgd_token_id, task=args.predict_task)
            else:
                image_id = batch["image_id"]
                if not isinstance(image_id, list):
                    image_id = image_id.cpu().numpy()
                img_feat = []
                id = batch['id']
                for i in image_id:
                    if args.predict_task == "pmr" or "vcr" in args.predict_task:
                        img_feat.append(clip_img_feat["val-" + str(i)])
                    else:
                        feat = clip_img_feat[str(i)]
                        if not isinstance(feat, torch.Tensor):
                            feat = torch.from_numpy(feat)
                        if feat.size(0)!=1:
                            feat = feat.unsqueeze(dim=0)
                        img_feat.append(feat)

                img_feat = torch.stack(img_feat).cuda().squeeze(dim=1)
                generate_ids = model.generate(llm_text_input=input_ids, llm_attention_mask=attention_mask,
                                              pre_llm_text_input=input_ids, pre_llm_attention_mask=attention_mask,
                                              clip_img_input=img_feat, clip_text_input=clip_text_ids,
                                              clip_attention_mask=clip_attention_mask, img_token_id=img_token_id,
                                              imgd_token_id=imgd_token_id, task=args.predict_task)

            output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
                0]
            if args.predict_task == "coco-caption" or "detail" in args.predict_task:
                tmp = {"image_id": int(image_id[0]), "caption":output}
            elif "vqa" in args.predict_task:
                tmp = {"question_id": int(id), "answer": output}
            elif args.predict_task == "gqa":
                tmp = {"questionId": str(id), "prediction":output}
            else:
                tmp = {"output": output}

            outputs.append(tmp)
            pbar(step=step)

        if args.predict_task == "cc3m-caption":
            with open(task2output[args.predict_task], "w") as f:
                for line in outputs:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    f.flush()
        else:
            with open(task2output[args.predict_task], "w") as f:
                json.dump(outputs, f)



def init(args):
    # loading the pre-extracted image features
    # imgid2caption is used for VQA tasks, and the corresponding generated caption is obtained according to image_id
    global clip_img_feat, clip_text, imgid2caption
    clip_img_feat = None
    if args.predict_task == "cc3m-caption":
        clip_img_feat = pickle.load(open("/raid/cxy/ReasoningTask/coco_data/img_feat/coco_img_large.pkl", "rb"))
        clip_text = pickle.load(open("/raid/cxy/ReasoningTask/coco_data/id2text.pkl", "rb"))
    elif args.predict_task == "coco":
        clip_img_feat = pickle.load(open("/raid/cxy/ReasoningTask/dataset/image_feat/coco_img_large.pkl", "rb"))
    elif "vqa" in args.predict_task:
        clip_img_feat = pickle.load(open("/raid/cxy/ReasoningTask/dataset/image_feat/coco_img_large.pkl", "rb"))
        # clip_img_feat = pickle.load(open("/data/share/dataset/MutilModalDataset/coco_data/img_feat/coco_img_large.pkl", "rb"))
    elif args.predict_task == "pmr":
        clip_img_feat = pickle.load(open("/raid/cxy/ReasoningTask/dataset/PMR/img_feat_val.pkl", "rb"))
    elif "vcr" in args.predict_task:
        clip_img_feat = pickle.load(open("/raid/cxy/ReasoningTask/dataset/VCR/img_feat_val.pkl", "rb"))
        # clip_img_feat = pickle.load(open("/data/share/dataset/MutilModalDataset/VCR/img_feat_val.pkl", "rb"))
    elif args.predict_task == "esnlive":
        clip_img_feat = pickle.load(open("/raid/cxy/ReasoningTask/dataset/e-ViL/img_feat_test.pkl", "rb"))
    elif args.predict_task == "nlvr":
        clip_img_feat = pickle.load(open("/raid/cxy/ReasoningTask/dataset/nlvr2/data/img_feat_test2.pkl", "rb"))
    elif args.predict_task == "gqa":
        clip_img_feat = pickle.load(open("/raid/cxy/ReasoningTask/dataset/GQA/img_feat_test.pkl", "rb"))
    elif "detail" in args.predict_task:
        clip_img_feat = pickle.load(open("/raid/cxy/ReasoningTask/dataset/cc_sbu_align/img_feat_detailed_instruction.pkl", "rb"))
    elif args.predict_task == "semart":
        clip_img_feat = pickle.load(open("/raid/cxy/ReasoningTask/dataset/SemArt/img_feat_semart_test.pkl", "rb"))


task2data = {
    "pmr": "/raid/cxy/ReasoningTask/dataset/PMR/val-ori_re.json",
    "coco": "/raid/cxy/ReasoningTask/coco_data/coco_test_image.json",
    "vqa": "/raid/cxy/ReasoningTask/dataset/VQA2/val_vqav2_10000.json",
    # "vqa": "/data/share/dataset/MutilModalDataset/VQA2/val_vqav2_10000.json",
    "vcr": "/raid/cxy/ReasoningTask/dataset/VCR/val_re.json",
    # "vcr": "/data/share/dataset/MutilModalDataset/VCR/val_re.json",
    "esnlive": "/raid/cxy/ReasoningTask/dataset/e-ViL/esnlive_test.json",
    "ok-vqa": "/raid/cxy/ReasoningTask/dataset/ok-vqa/ok-vqa_val.json",
    "nlvr": "/raid/cxy/ReasoningTask/dataset/nlvr2/data/nlvr2_test2.json",
    "vcrqa2r": "/raid/cxy/ReasoningTask/dataset/VCR/val_qa2r.json",
    "gqa": "/raid/cxy/ReasoningTask/dataset/GQA/gqa_test.json",
    "detail-c": "/raid/cxy/ReasoningTask/dataset/cc_sbu_align/detail_caption.json",
    "test": "coco_data/coco_test_image.json",
    "detail-q": "/raid/cxy/ReasoningTask/dataset/cc_sbu_align/detail_vqa.json",
    "semart": "/raid/cxy/ReasoningTask/dataset/SemArt/semart_test.json"
}

task2output = {
    "pmr": "result/pmr/bloomz-7b1-instruction-b-epoch2.json",
    "vqa": "result/vqa/other/llama-13b-instruction-epoch5.json",
    "vcr": "result/vcr/llama-13b-instruction-epoch2.json",
    "esnlive": "result/esnlive/bloomz-7b1-instruction-b-epoch5-smal.json",
    # "ok-vqa": "result/ok-vqa/bloomz-7b1-instruction-laion-1epoch-unbalanced-0.9epoch-balanced-3epoch-query5-re.json",
    "ok-vqa": "result/ok-vqa/llama-13b-instruction-epoch2.json",
    "nlvr": "result/nlvr/opt-imageLinear-test2.json",
    "vcrqa2r": "result/vcrqa2r/bloomz-7b1-instruction-laion-1epoch-unbalanced-0.9epoch-balanced-5epoch-query5-re.json",
    "gqa": "result/gqa/opt-imageLinear.json",
    "detail-c": "result/detail-c/llama-13b-instruction-epoch2-laion.json",
    "detail-q": "result/detail-q/llama-13b-instruction-epoch2-laion.json",
    "semart":"result/semart/llama-7b-instruction-epoch2.json"
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--epoch_begin", default=0, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--adam_epsilon", default=1e-5, type=float)
    parser.add_argument("--to_train_embed", default=False, type=bool)
    parser.add_argument("--prompt_len", default=5, type=int)
    parser.add_argument("--query_len", default=1, type=int)
    parser.add_argument("--llm_model", default="opt", type=str)
    parser.add_argument("--special_vqa", action="store_true")
    parser.add_argument("--add_bos",action="store_true")
    parser.add_argument("--prompt_type", default=0, type=int)

    parser.add_argument("--eval_model_dir", type=str,
                        default='',
                        help="Model directory for evaluation.")
    parser.add_argument("--stage1_linear_dir", type=str,
                        default='')
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--generate_with_image", action="store_true")
    parser.add_argument("--output_dir", default="output/tmp", type=str)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--predict_task", default="cc3m-caption", type=str)
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--predict_model_dir", type=str,
                        # default='/raid/cxy/ReasoningTask/output/llama-unbalanced-coco-7b-laion/checkpoint_with_epoch1.pth')
                        default='/raid/cxy/ReasoningTask/output/llama-unbalanced-coco-7b/checkpoint_with_epoch1.pth')
                        # default='/raid/cxy/ReasoningTask/output/bloomz-instruction-query5/checkpoint_with_step30000.pth')
                        # default='/raid/cxy/ReasoningTask/output/bloomz-instruction-balanced/test1/checkpoint_with_epoch1.pth')
    parser.add_argument("--resume", action='store_true')

    args = parser.parse_args()

    global logger

    print("Using Language Model: {}".format(args.llm_model.upper()))
    print("Doing Task: {}".format(args.predict_task.upper()))

    mkdir(args.output_dir)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    init(args)

    logger = setup_logger("mvp_finetune_pmr", args.output_dir, 0)
    # logger.warning("Device: %s, n_gpu: %s", torch.cuda.current_device(), torch.cuda.device_count())

    # -------------------------加载OPT-iml模型-----------------------
    if args.llm_model == "opt":
        opt_path = "local_transformers/opt-iml-max-1.3b/"
        llm_tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-1.3b")
        llm_config = OPTConfig.from_pretrained(opt_path + "config.json")
        llm_model = OPTForCausalLM.from_pretrained(opt_path, config=llm_config)
    elif args.llm_model == "llama-7b":
        llama_path = "local_transformers/llama-7b-hf/"
        llm_tokenizer = LlamaTokenizer.from_pretrained(llama_path, add_bos_token=False, add_eos_token=True)
        llm_config = LlamaConfig.from_pretrained(llama_path + "config.json")
        llm_model = LlamaForCausalLM.from_pretrained(llama_path, config=llm_config)
        llm_tokenizer.pad_token_id = 0
    elif args.llm_model == "bloomz-7b1":
        bloomz_path = "local_transformers/bloomz-7b1/"
        llm_tokenizer = AutoTokenizer.from_pretrained(bloomz_path, add_bos_token=True, add_eos_token=False)
        llm_config = BloomConfig.from_pretrained(bloomz_path + "config.json")
        llm_model = BloomForCausalLM.from_pretrained(bloomz_path, config=llm_config)
    else:
        llama_path = "local_transformers/llama-13b-hf/"
        llm_tokenizer = LlamaTokenizer.from_pretrained(llama_path, add_bos_token=True, add_eos_token=False)
        llm_config = LlamaConfig.from_pretrained(llama_path + "config.json")
        llm_model = LlamaForCausalLM.from_pretrained(llama_path, config=llm_config)
        llm_tokenizer.pad_token_id = 0



    # special token add
    llm_tokenizer.add_tokens("<img>")
    llm_tokenizer.add_tokens("<img-d>")
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



    stage2_params = torch.load(args.predict_model_dir, map_location='cpu')['net']
    imgLinear_params = {}
    imgdLinear_params = {}
    promptMlp_params = {}
    llm2clip_params = {}

    for name, params in stage2_params.items():
        if not args.fp16:
            params = params.float()
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
    if args.fp16:
        model = model.half()
    model = model.cuda()

    if args.llm_model == "opt":
        dataset = llm_extra_dataset_instruction_predict(task2data[args.predict_task], llm_tokenizer, task=args.predict_task, prompt_type=args.prompt_type)
    else:
        dataset = llm_extra_dataset_instruction_predict_llama(task2data[args.predict_task], llm_tokenizer, task=args.predict_task, prompt_type=args.prompt_type, query_len=args.query_len, isSmall=args.debug)
    data_sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=data_sampler)
    generate(args, dataloader, model, llm_tokenizer, clip_tokenizer, clip_img_feat)


if __name__ == "__main__":
    main()
