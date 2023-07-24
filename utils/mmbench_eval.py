"""
Recode from https://github.com/InternLM/opencompass/blob/mm/docs/en/MMBench.md
2023-07-21:
-> (1001)       Add 'mme_eval_dataset(Dataset)' to allow the Lmeye model to read the MME dataset.
"""

import base64
import io
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

class MMBenchDataset(Dataset):
    def __init__(self, input_path, llm_processor):
        self.llm_processor = llm_processor
        self.df = pd.read_csv(input_path, sep = '\t')

        self.all_inputs= []
        self.all_images = []
        self.all_attention_mask = []
        self.all_ground_truth = []
        self.all_input_text = []
        self.all_type = []
        
        img_human_prompt = "Human: <img>\n"
        imgd_assistant_prompt = "\n <img-q> <img-d> <img-d> <img-d> <img-d> <img-d>\nAssistant:"

        for idx, _ in enumerate(self.df.iloc):
            image = self.df.iloc[idx]['image']
            query = self.df.iloc[idx]['question']
            answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[0].keys() else None
            catetory = self.df.iloc[idx]['category']
            l2_catetory = self.df.iloc[idx]['l2-category']

            option_candidate = ['A', 'B', 'C', 'D', 'E']
            options = {
                cand: self.load_from_df(idx, cand)
                for cand in option_candidate
                if self.load_from_df(idx, cand) is not None
            }
            options_prompt = '\n'
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'

            input_data = img_human_prompt + query + options_prompt + imgd_assistant_prompt
            token_input = llm_processor.tokenizer([input_data], add_special_tokens = False, padding = 'max_length', max_length = 1024, return_tensors = "pt")
            input_ids = token_input["input_ids"]
            input_attention_mask = token_input["attention_mask"]

            self.all_inputs.append(input_ids)
            self.all_images.append(decode_base64_to_image(image))
            self.all_attention_mask.append(input_attention_mask)
            self.all_ground_truth.append(answer)
            self.all_input_text.append(query)
            self.all_type.append((catetory, l2_catetory))

    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None
        
    def __len__(self):
        return len(self.all_inputs)

    def __getitem__(self, i):
        return {
            "input_text": self.all_input_text[i],
            "ground_truth": self.all_ground_truth[i],
            "inputs": self.all_inputs[i],
            "image": self.llm_processor(images = self.all_images[i], return_tensors = "pt", padding = True)['pixel_values'][0],
            "attention_mask": self.all_attention_mask[i],
            "type": self.all_type[i],
        }

class MMBenchCalculateMetrics():
    def __init__(self, save_path):
        self.save_path = save_path

    def save_data(self, output, batch):
        for index, data in enumerate(output):
            with open(os.path.join(self.save_path, "data.txt"), "a") as f:
                f.write(
                    batch["input_text"][index] + '\t' +
                    batch["ground_truth"][index] + '\t' +
                    output[index] + '\n'
                )

    def del_data(self):
        try:
            os.remove(os.path.join(self.save_path, "data.txt"))
        except: pass