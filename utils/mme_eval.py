"""
Recode from https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation/tools
2023-07-17:
-> (1001)       Add 'mme_eval_dataset(Dataset)' to allow the Lmeye model to read the MME dataset.
2023-07-18:
-> (1002)       Add 'save_data' and 'del_data' in 'calculate_metrics()'
"""
eval_type_dict = {
    "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
    "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
}

from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


class MMEvalDataset(Dataset):
    def __init__(self, input_path, llm_processor):
        self.eval_type_list = eval_type_dict["Perception"] + eval_type_dict["Cognition"]
        self.llm_processor = llm_processor
        
        self.all_inputs= []
        self.all_target = []
        self.all_images = []
        self.all_attention_mask = []
        self.all_image_id = []
        self.all_ground_truth = []
        self.all_input_text = []
        self.all_type = []

        img_human_prompt = "Human: <img>\n"
        imgd_assistant_prompt = "\n <img-q> <img-d> <img-d> <img-d> <img-d> <img-d>\nAssistant:"

        for eval_type in self.eval_type_list:
            full_path = os.path.join(input_path, eval_type, "images")
            for image_name in os.listdir(full_path):
                image_path = os.path.join(full_path, image_name)
                image = Image.open(image_path)

                text_path = os.path.join(full_path.replace('images', 'questions_answers_YN'), image_name.replace('.jpg', '.txt').replace('.png', '.txt'))
                with open(text_path, 'r', encoding = 'utf-8') as f:
                    text_data = f.read().split('\n')

                for text in text_data:
                    if text == '': continue
                    query, answer = text.split('\t')[:2]
                    input_data = img_human_prompt + query + imgd_assistant_prompt

                    token_input = llm_processor.tokenizer([input_data], add_special_tokens = True, padding = 'max_length', max_length = 512, return_tensors = "pt")
                    input_ids = token_input["input_ids"]
                    input_attention_mask = token_input["attention_mask"]

                    self.all_inputs.append(input_ids)
                    self.all_images.append(image)
                    self.all_attention_mask.append(input_attention_mask)
                    self.all_image_id.append(image_name)
                    self.all_ground_truth.append(answer)
                    self.all_input_text.append(query)
                    self.all_type.append(eval_type)
        
    def __len__(self):
        return len(self.all_inputs)

    def __getitem__(self, i):
        return {
            "input_text": self.all_input_text[i],
            "ground_truth": self.all_ground_truth[i],
            "image_id": self.all_image_id[i],
            "inputs": self.all_inputs[i],
            "image": self.llm_processor(images = self.all_images[i], return_tensors = "pt", padding = True)['pixel_values'][0],
            "attention_mask": self.all_attention_mask[i],
            "type": self.all_type[i],
        }

class MMECalculateMetrics():
    def __init__(self, save_path):
        self.save_path = save_path

    def save_data(self, output, batch):
        for index, data in enumerate(output):
            with open(os.path.join(self.save_path, batch["type"][index] + ".txt"), "a") as f:
                f.write(
                    batch["image_id"][index] + '\t' +
                    batch["input_text"][index] + '\t' +
                    batch["ground_truth"][index] + '\t' +
                    output[index] + '\n'
                )

    def del_data(self):
        for type_name in os.listdir(self.save_path):
            try:
                os.remove(os.path.join(self.save_path, type_name))
            except: pass


    def divide_chunks(self, l, n = 2):
        # looping till length l
        for i in range(0, len(l), n): 
            yield l[i:i + n]
        
        return 

    def parse_pred_ans(self, pred_ans):
        pred_label = None
        if pred_ans in ["yes", "no"]:
            pred_label = pred_ans
        else:
            prefix_pred_ans = pred_ans[:4]

            if "yes" in prefix_pred_ans:
                pred_label = "yes"
            elif "no" in prefix_pred_ans:
                pred_label = "no"
            else:
                pred_label = "other"

        return pred_label


    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)

        label_map = {
            "yes": 1,
            "no": 0,
            "other": -1,
        }
        
        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]

        acc = accuracy_score(gts, preds) 

        clean_gts = []
        clean_preds = []
        other_num = 0 
        for gt, pred in zip(gts, preds):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)
        

        conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1,0])
        precision = precision_score(clean_gts, clean_preds, average='binary')
        recall = recall_score(clean_gts, clean_preds, average='binary')
        tp, fn = conf_mat[0]
        fp, tn = conf_mat[1]

        metric_dict = dict()
        metric_dict = {
            "TP": tp,
            "FN": fn,
            "TN": tn,
            "FP": fp,
            "precision": precision,
            "recall": recall,
            "other_num": other_num,
            "acc": acc,
        }

        return metric_dict


    def process_result(self):

        results_dir = self.save_path
        model_score_dict = dict()
        for eval_type, task_name_list in eval_type_dict.items():
            print("===========", eval_type, "===========")
           
            scores = 0
            task_score_dict = dict()

            for task_name in task_name_list:

                task_txt = os.path.join(results_dir, task_name + ".txt")
                lines = open(task_txt, 'r').readlines()
                chunk_lines = list(self.divide_chunks(lines)) # one image corresponds to two questions
                
                img_num = len(chunk_lines)
                task_other_ans_num = 0
                task_score = 0
                acc_plus_correct_num = 0
                gts = []
                preds = []

                for img_items in chunk_lines:
                    assert len(img_items) == 2
                    img_correct_num = 0

                    for img_item in img_items:
                        img_name, question, gt_ans, pred_ans = img_item.split("\t")

                        gt_ans = gt_ans.lower()
                        pred_ans = pred_ans.lower()

                        assert gt_ans in ["yes", "no"] # gt can only be yes or no.

                        pred_ans = self.parse_pred_ans(pred_ans)
                        assert pred_ans in ["yes", "no", "other"]

                        gts.append(gt_ans)
                        preds.append(pred_ans)
                        
                        if gt_ans == pred_ans:
                            img_correct_num += 1
                        
                        if pred_ans not in ["yes", "no"]:
                            task_other_ans_num += 1

                    if img_correct_num == 2:
                        acc_plus_correct_num += 1

                # cal TP precision acc, etc.
                metric_dict = self.compute_metric(gts, preds)
                acc_plus = acc_plus_correct_num / img_num
                metric_dict["acc_plus"] = acc_plus
                
                
                for k, v in metric_dict.items():
                    if k in ["acc", "acc_plus"]:
                        task_score += v*100
                
                task_score_dict[task_name] = task_score
                
                scores += task_score

            print("total score:", scores, "\n")
            for task_name, score in task_score_dict.items():
                print("\t", task_name, " score:", score)
            print("\n")
        
        return 