import io
import os
import json
import torch
import zipfile
from PIL import Image
from torch.utils.data import Dataset

from Lmeye.lmeye_config import Config, IGNORE_INDEX, IMG_INDEX, IMG_D_INDEX, IMG_Q_INDEX
from Lmeye.lmeye_processor import Blip2Processor


def extract_image_from_zip(zip_path, image_to_extract):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open(image_to_extract) as image_file:
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes))
    return image


class TrainDataset(Dataset):
    def __init__(self, config: Config, llm_processor: Blip2Processor):
        dataset_path = config.dataset
        self.llm_processor = llm_processor
        self.config = config

        with open(os.path.join(dataset_path, 'json/data.json'), 'r') as f:
            chat_json = f.read()
        self.json_data = json.loads(chat_json)

        self.query = []
        self.answer = []
        self.image_path = []
        self.chunk_path = []

        if config.debug:
            self.json_data = self.json_data[:1000]
        
        for data in self.json_data:
            self.query.append(data['question'])
            self.answer.append(data['answer'])
            self.image_path.append(data['image_path'])
            self.chunk_path.append(os.path.join(dataset_path, 'image', data['chunk_belong']))

    def __len__(self):
        return len(self.query)

    def __getitem__(self, index):
        qformer_length = 32
        imgq_token_number = self.config.imgq_number
        imgd_token_number = self.config.imgd_number

        query = self.query[index]
        answer = self.answer[index]
        image_path = self.image_path[index]
        chunk_path = self.chunk_path[index]

        extracted_image = extract_image_from_zip(chunk_path, image_path)

        if self.config.decoder_only:
            query_prompt = "[Round 0] \n\n问：{query}\n\n答："
            input_data = query_prompt.format(query = query)
            output_data = answer

            token_input = self.llm_processor.tokenizer(input_data, add_special_tokens = False, return_tensors = "pt")
            input_ids = token_input["input_ids"]
            input_attention_mask = token_input["attention_mask"]

            token_output = self.llm_processor.tokenizer(output_data, add_special_tokens = False, return_tensors = "pt")
            output_ids = token_output["input_ids"]
            output_attention_mask = token_output["attention_mask"]

            all_input_ids = torch.cat([
                input_ids,
                torch.full((1, qformer_length), IMG_INDEX),
                torch.full((1, imgq_token_number), IMG_Q_INDEX),
                torch.full((1, imgd_token_number), IMG_D_INDEX),
            ], dim = 1)[0].tolist()

            input_attention_mask = torch.cat([
                input_attention_mask,
                torch.full((1, qformer_length + imgq_token_number + imgd_token_number), 1),
            ], dim = 1)[0].tolist()

            context_length = len(all_input_ids) + 2 # gmask and sop tokens
            new_all_input_ids = self.llm_processor.tokenizer.build_inputs_with_special_tokens(all_input_ids, output_ids[0].tolist())
            output_ids = new_all_input_ids[context_length: ]
            
            # padding_side left
            padding_length = self.config.padding - len(new_all_input_ids)
            target = [IGNORE_INDEX] * (context_length + padding_length) + output_ids
            new_all_input_ids = [0] * padding_length + new_all_input_ids
            # [1] * 3 is the characters occupied by special tokens in ChatGLM.
            input_attention_mask = [0] * padding_length + input_attention_mask + output_attention_mask[0].tolist() + [1] * 3

            return {
                "inputs": torch.tensor(new_all_input_ids),
                "target": torch.tensor(target),
                "image": self.llm_processor(images = extracted_image, return_tensors = "pt", padding = True)['pixel_values'][0],
                "attention_mask": torch.tensor(input_attention_mask),
            }
        else:
            query_prompt = "Human: {query} Assistant: "
            input_data = query_prompt.format(query = query)
            output_data = answer

            token_input = self.llm_processor.tokenizer([input_data], add_special_tokens = True, return_tensors = "pt")
            input_ids = token_input["input_ids"]
            input_attention_mask = token_input["attention_mask"]

            token_output = self.llm_processor.tokenizer([output_data], add_special_tokens = True, return_tensors = "pt")
            output_ids = token_output["input_ids"]
            padding_length = self.config.padding - output_ids.shape[-1]

            all_input_ids = torch.cat([
                input_ids[:, :-1],
                torch.full((1, qformer_length), IMG_INDEX),
                torch.full((1, imgq_token_number), IMG_Q_INDEX),
                torch.full((1, imgd_token_number), IMG_D_INDEX),
                input_ids[:, -1].unsqueeze(-1),
            ], dim = 1)
            
            input_attention_mask = torch.cat([
                input_attention_mask,
                torch.full((1, qformer_length + imgq_token_number + imgd_token_number), 1),
            ], dim = 1)

            # padding_side right
            all_input_ids = torch.cat([
                all_input_ids,
                torch.full((1, self.config.padding - all_input_ids.shape[-1]), 0),
            ], dim = 1)

            input_attention_mask = torch.cat([
                input_attention_mask,
                torch.full((1, self.config.padding - input_attention_mask.shape[-1]), 0),
            ], dim = 1)

            target = torch.cat([
                output_ids,
                torch.full((output_ids.shape[0], self.config.padding - output_ids.shape[-1]), IGNORE_INDEX)
            ], dim = 1)

            return {
                "inputs": all_input_ids,
                "target": target,
                "image": self.llm_processor(images = extracted_image, return_tensors = "pt", padding = True)['pixel_values'][0],
                "attention_mask": input_attention_mask,
            }