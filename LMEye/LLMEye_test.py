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
def generate(args, input_text, model, tokenizer, clip_tokenizer, image_dir):
    model.eval()
    clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    print("Start Generating")
    with torch.no_grad():
        if args.llm_model == "opt":
            input_text = "Image: <img>\n " + input_text + " <img-d>\nAnswer:"
        else:
            if "qa" in args.predict_task:
                input_text = "Image as evidence: <img>\n Question: " + input_text + " <img-d> <img-d> <img-d> <img-d> <img-d>\nAnswer:"
            else:
                input_text = "Image: <img>\n " + input_text + " <img-d> <img-d> <img-d> <img-d> <img-d>\nAnswer:"
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
        image = Image.open(image_dir).convert('RGB')
        image = torch.from_numpy(clip_processor(image)['pixel_values'][0]).cuda().half()
        generate_ids = model.generate_with_image(llm_text_input=input_ids,
                                      pre_llm_text_input=input_ids, pre_llm_attention_mask=attention_mask,
                                      image=image, clip_text_input=clip_text_ids,
                                      clip_attention_mask=clip_attention_mask, img_token_id=img_token_id,
                                      imgd_token_id=imgd_token_id, task=args.predict_task)

        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return output




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
    parser.add_argument("--query_len", default=5, type=int)
    parser.add_argument("--llm_model", default="llama-7b", type=str)
    parser.add_argument("--special_vqa", action="store_true")
    parser.add_argument("--add_bos", action="store_true")
    parser.add_argument("--prompt_type", default=0, type=int)
    parser.add_argument("--input_text", default=" Give a detailed description of this image. ", type=str)
    parser.add_argument("--image_dir", default="./example_images/example_1.jpg", type=str)
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
    parser.add_argument("--predict_task", default="Description", type=str) # ["qa", "description"]
    parser.add_argument("--fp16", default=True, type=bool)
    parser.add_argument("--predict_model_dir", type=str,
                        #default='./output/llama-7b-instruction-laion/checkpoint_with_epoch4.pth'
                        default='./output/bloomz-instruction/checkpoint_with_epoch4.pth'
                        )
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

    logger = setup_logger("mvp_finetune_pmr", args.output_dir, 0)
    # logger.warning("Device: %s, n_gpu: %s", torch.cuda.current_device(), torch.cuda.device_count())

    # -------------------------LLMs-----------------------
    if args.llm_model == "opt":
        opt_path = "facebook/opt-iml-max-1.3b"
        llm_tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-1.3b")
        llm_config = OPTConfig.from_pretrained(opt_path)
        llm_model = OPTForCausalLM.from_pretrained(opt_path, config=llm_config)
    elif args.llm_model == "llama-7b":
        llama_path = "decapoda-research/llama-7b-hf"
        llm_tokenizer = LlamaTokenizer.from_pretrained(llama_path, add_bos_token=False, add_eos_token=True)
        llm_config = LlamaConfig.from_pretrained(llama_path)
        llm_model = LlamaForCausalLM.from_pretrained(llama_path, config=llm_config)
        llm_tokenizer.pad_token_id = 0
    elif args.llm_model == "bloomz-7b1":
        bloomz_path = "bigscience/bloomz-7b1"
        llm_tokenizer = AutoTokenizer.from_pretrained(bloomz_path, add_bos_token=False, add_eos_token=True)
        llm_config = BloomConfig.from_pretrained(bloomz_path)
        llm_model = BloomForCausalLM.from_pretrained(bloomz_path, config=llm_config)
    else:
        llama_path = "decapoda-research/llama-13b-hf/"
        llm_tokenizer = LlamaTokenizer.from_pretrained(llama_path, add_bos_token=False, add_eos_token=True)
        llm_config = LlamaConfig.from_pretrained(llama_path)
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

    # --------------------------Loading CLIP Model-------------------------
    clip_path = "openai/clip-vit-large-patch14"

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
    output = generate(args, args.input_text, model, llm_tokenizer, clip_tokenizer, args.image_dir)
    print("Input: ", args.input_text)
    print("Output: ")
    print(output)

if __name__ == "__main__":
    main()
