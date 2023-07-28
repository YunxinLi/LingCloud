import json
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

from Lmeye.lmeye_config import Config
from Lmeye.lmeye_processor import Blip2Processor

class TrainDataset(Dataset):
    def __init__(self, config: Config, llm_processor: Blip2Processor):
        with open(config.dataset, 'r') as f:
            chat_json = f.read()
        self.json_data = json.loads(chat_json)
        self.llm_processor = llm_processor
        if config.debug:
            self.json_data = self.json_data[:1000]

        # image_prompt = "<img-q> <img-d> <img-d> <img-d> <img-d> <img-d>"

        self.all_inputs = []
        self.all_target = []
        self.all_images = []
        self.all_attention_mask = []
        self.all_input_text = []
        self.all_ground_truth = []

        for data in tqdm(self.json_data):
            image_path = "/root/dataset/LLaVA-CC3M-Pretrain-595K/{}".format(data["image"])
            query = data["conversations"][0]['value'].replace('<image>', '').replace('\n', '')
            answer = data["conversations"][1]['value']

            if config.decoder_only:
                query_prompt = "[Round 0] \n\n问：{query}\n\n答："
                input_data = query_prompt.format(query = query)
                output_data = answer

                token_input = llm_processor.tokenizer(input_data, add_special_tokens = False, return_tensors = "pt")
                input_ids = token_input["input_ids"][0].tolist()
                input_attention_mask = token_input["attention_mask"][0].tolist()

                token_output = llm_processor.tokenizer(output_data, add_special_tokens = False, return_tensors = "pt")
                output_ids = token_output["input_ids"][0].tolist()
                output_attention_mask = token_output["attention_mask"][0].tolist()

                context_length = len(input_ids) + 2 # gmask and sop tokens
                all_input_ids = llm_processor.tokenizer.build_inputs_with_special_tokens(input_ids, output_ids)
                output_ids = all_input_ids[context_length: ]
                
                target = [-100] * context_length + output_ids
                padding_length = 512 - len(target)
                target = target + [-100] * padding_length
                all_input_ids = all_input_ids + [0] * padding_length
                attention_mask = input_attention_mask + [1] * 2 + output_attention_mask + [1] + [0] * padding_length

                # labels_output = llm_processor.tokenizer._pad({"input_ids": input_ids}, max_length = 512, padding_strategy = "max_length")
                # input_ids = labels_output["input_ids"]
                '''
                token_input = llm_processor.tokenizer([input_data], add_special_tokens = False, return_tensors = "pt")
                input_ids = token_input["input_ids"]
                input_attention_mask = token_input["attention_mask"]
                
                token_output = llm_processor.tokenizer([output_data], add_special_tokens = False, return_tensors = "pt")
                output_ids = token_output["input_ids"]
                output_attention_mask = token_output["attention_mask"]

                target = torch.cat((torch.full_like((input_ids), -100), output_ids), dim = 1)
                padding_length = 512 - target.shape[1]

                all_input_ids = torch.cat((input_ids, output_ids), dim = 1)
                all_input_ids = torch.cat((all_input_ids, torch.full((all_input_ids.shape[0], padding_length), 0)), dim = 1)

                attention_mask = torch.cat((input_attention_mask, output_attention_mask), dim = 1)
                attention_mask = torch.cat((attention_mask, torch.full((attention_mask.shape[0], padding_length), 0)), dim = 1)

                target = torch.cat((target, torch.full((target.shape[0], padding_length), -100)), dim = 1)
                '''
                image = Image.open(image_path)

                self.all_inputs.append(torch.tensor(all_input_ids))
                self.all_target.append(torch.tensor(target))
                self.all_images.append(image)
                self.all_attention_mask.append(torch.tensor(attention_mask))
                self.all_ground_truth.append(answer)
                self.all_input_text.append(query)
            else:
                query_prompt = "Human: {query} Assistant: "
                input_data = query_prompt.format(query = query)
                output_data = answer

                token_input = llm_processor.tokenizer([input_data], add_special_tokens = True, padding = 'max_length', max_length = 512, return_tensors = "pt")
                input_ids = token_input["input_ids"]
                input_attention_mask = token_input["attention_mask"]
                
                token_output = llm_processor.tokenizer([output_data], add_special_tokens = True, return_tensors = "pt")
                output_ids = token_output["input_ids"]
                padding_length = 512 - output_ids.shape[1]
                target = torch.cat((output_ids, torch.full((output_ids.shape[0], padding_length), -100)), dim = 1)

                image = Image.open(image_path)

                self.all_inputs.append(input_ids)
                self.all_target.append(target)
                self.all_images.append(image)
                self.all_attention_mask.append(input_attention_mask)
                self.all_ground_truth.append(answer)
                self.all_input_text.append(query)

    def __len__(self):
        return len(self.all_inputs)

    def __getitem__(self, i):
        return {
            "inputs": self.all_inputs[i],
            "target": self.all_target[i],
            "image": self.llm_processor(images = self.all_images[i], return_tensors = "pt", padding = True)['pixel_values'][0],
            "attention_mask": self.all_attention_mask[i],
            "input_text": self.all_input_text[i],
            "ground_truth": self.all_ground_truth[i],
        }