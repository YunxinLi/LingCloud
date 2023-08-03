import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import CLIPTokenizer

from Lmeye.lmeye_model import Blip2InstructionQueryModel
from Lmeye.lmeye_processor import Blip2Processor
from Lmeye.lmeye_config import *

from utils.mme_eval import *
from utils.mmbench_eval import *
from utils.type_manager import DataloaderSet, TrainParams

import inspect
from typing import List, Callable, Type

class TrainManager():
    ''' Manage training/eval/test function. '''

    def __init__(self,
        model: Blip2InstructionQueryModel,
        processor: Blip2Processor,
        clip_tokenizer: CLIPTokenizer,
        dataloader_set: DataloaderSet,
    ):
        self.model = model
        self.processor = processor
        self.clip_tokenizer = clip_tokenizer
        self.dataloader_set = dataloader_set

        # Init optimizer and scheduler
        self.optimizer = AdamW(filter(lambda params: params.requires_grad, self.model.parameters()), lr = base_config.learning_rate)
        t_total = len(dataloader_set.train_dataloader) // base_config.gradient_accumulation_steps * base_config.num_train_epochs
        if base_config.scheduler == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps = 0, num_training_steps = t_total)
        elif base_config.scheduler == "constant":
            self.scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps = 0)
        else:
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps = 0, num_training_steps = t_total)
        
        # Model save list
        self.save_name_list = [
            'instruction_embedding_imgd',
            'instruction_embedding_imgq',
            'language_projection',
            'instruction_blip22clip',
            'instruction_linear',
            'instruction_imgdLinear',
        ]

    def mme_first_stage_eval(self, params: TrainParams) -> bool:
        ''' fisrt stage eval in MME dataset, similar to blip2.
        `(fisrt stage: only qformer + LLM -> answer)`
        '''
        dataloader = self.dataloader_set.mme_dataloader
        if dataloader is None: return False

        mme_metrics = MMECalculateMetrics(save_path = "/home/Lmeye/output/eval_data/MME")
        mme_metrics.del_data()
        self.model.eval()

        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                generate_ids = self.model.first_stage_generate(
                    input_ids = batch["inputs"].squeeze().cuda(),
                    pixel_values =  batch["image"].cuda(),
                    attention_mask = batch["attention_mask"].squeeze().cuda(),
                    num_beams = 5,
                    temperature = 0.2,
                    top_p = 1,
                    top_k = 3,
                    max_new_tokens = 16,
                )

                output = self.processor.batch_decode(generate_ids, skip_special_tokens = True, clean_up_tokenization_spaces = False)
                mme_metrics.save_data(output, batch)
        
        torch.cuda.empty_cache()
        mme_metrics.process_result()
        return True
    
    def mme_eval(self, params: TrainParams) -> bool:
        ''' All stage eval in MME dataset.
        `(all stage: qformer + LLM -> hidden_query + CLIP + LLM -> answer)`
        '''
        dataloader = self.dataloader_set.mme_dataloader
        if dataloader is None: return False

        mme_metrics = MMECalculateMetrics(save_path = "/home/Lmeye/output/eval_data/MME")
        mme_metrics.del_data()
        self.model.eval()

        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                text = ["! " * 37] * batch["inputs"].size(0)
                clip_text = self.clip_tokenizer(text, return_tensors = "pt")
                clip_text_ids = clip_text["input_ids"].cuda()
                
                generate_ids = self.model.generate(
                    input_ids = batch["inputs"].squeeze().cuda(),
                    pixel_values =  batch["image"].cuda(),
                    attention_mask = batch["attention_mask"].squeeze().cuda(),
                    num_beams = 5,
                    temperature = 0.2,
                    top_p = 1,
                    top_k = 3,
                    clip_text_input = clip_text_ids,
                    max_new_tokens = 16,
                )

                output = self.processor.batch_decode(generate_ids, skip_special_tokens = True, clean_up_tokenization_spaces = False)
                mme_metrics.save_data(output, batch)
        
        torch.cuda.empty_cache()
        mme_metrics.process_result()
        return True

    def mmbench_first_stage_eval(self, params: TrainParams) -> bool:
        ''' Fisrt stage eval in MME dataset, similar to blip2.
        `(fisrt stage: only qformer + LLM -> answer)`
        '''
        dataloader = self.dataloader_set.mmbench_dataloader
        if dataloader is None: return False

        mmbench_metrics = MMBenchCalculateMetrics(save_path = "/home/Lmeye/output/eval_data/MMBench")
        mmbench_metrics.del_data()
        self.model.eval()

        for step, batch in enumerate(dataloader):
            with torch.no_grad():
                text = ["! " * 37] * batch["inputs"].size(0)
                clip_text = self.clip_tokenizer(text, return_tensors = "pt")
                clip_text_ids = clip_text["input_ids"].cuda()
                
                generate_ids = self.model.first_stage_generate(
                    input_ids = batch["inputs"].squeeze().cuda(),
                    pixel_values =  batch["image"].cuda(),
                    attention_mask = batch["attention_mask"].squeeze().cuda(),
                    num_beams = 5,
                    temperature = 0.2,
                    top_p = 1,
                    top_k = 3,
                    max_new_tokens = 16,
                )

                output = self.processor.batch_decode(generate_ids, skip_special_tokens = True, clean_up_tokenization_spaces = False)
                mmbench_metrics.save_data(output, batch)

        torch.cuda.empty_cache()
        return True

    def mmbench_eval(self, params: TrainParams) -> bool:
        ''' All stage eval in MMBench dataset.
        `(all stage: qformer + LLM -> hidden_query + CLIP + LLM -> answer)`
        '''
        dataloader = self.dataloader_set.mmbench_dataloader
        if dataloader is None: return False

        mmbench_metrics = MMBenchCalculateMetrics(save_path = "/home/Lmeye/output/eval_data/MMBench")
        mmbench_metrics.del_data()
        self.model.eval()

        for step, batch in enumerate(dataloader):
            with torch.no_grad():
                text = ["! " * 37] * batch["inputs"].size(0)
                clip_text = self.clip_tokenizer(text, return_tensors = "pt")
                clip_text_ids = clip_text["input_ids"].cuda()
                
                generate_ids = self.model.generate(
                    input_ids = batch["inputs"].squeeze().cuda(),
                    pixel_values =  batch["image"].cuda(),
                    attention_mask = batch["attention_mask"].squeeze().cuda(),
                    num_beams = 5,
                    temperature = 0.2,
                    top_p = 1,
                    top_k = 3,
                    clip_text_input = clip_text_ids,
                    max_new_tokens = 16,
                )

                output = self.processor.batch_decode(generate_ids, skip_special_tokens = True, clean_up_tokenization_spaces = False)
                mmbench_metrics.save_data(output, batch)

        torch.cuda.empty_cache()
        return True

    def train(self, params: TrainParams) -> bool:
        ''' All stage training in train dataset.
        `(all stage: qformer + LLM -> hidden_query + CLIP + LLM -> answer)`
        '''
        epoch = params.epoch
        total_loss = 0.0
        total_step = 0
        dataloader = self.dataloader_set.train_dataloader

        self.model.train()
        for step, batch in tqdm(enumerate(dataloader)):
            text = ["! " * 37] * batch["inputs"].size(0)
            clip_text = self.clip_tokenizer(text, return_tensors = "pt")
            clip_text_ids = clip_text["input_ids"].cuda()

            model_output = self.model.forward(
                labels = batch['target'].squeeze().cuda(),
                input_ids = batch["inputs"].squeeze().cuda(),
                pixel_values =  batch["image"].cuda(),
                clip_text_input = clip_text_ids,
                attention_mask = batch["attention_mask"].squeeze().cuda(),
            )
            loss: torch.FloatTensor = model_output.loss
            
            if base_config.gradient_accumulation_steps > 1:
                loss = loss / base_config.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), base_config.max_grad_norm)
            total_loss += loss.item()

            if (step + 1) % base_config.gradient_accumulation_steps == 0:
                total_step += 1
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.scheduler.step()
                self.model.zero_grad()

                if total_step % base_config.logging_steps == 0:
                    print("[Epoch {} | Step {}] average loss: {}".format(epoch, total_step, total_loss / total_step))

                if total_step % base_config.save_steps == 0 and total_step > 0:
                    to_save_param = {}
                    for name, params in self.model.named_parameters():
                        for save_name in self.save_name_list:
                            if save_name in name:
                                to_save_param[name] = params
                    state = { 'net': to_save_param, 'optimizer': self.optimizer.state_dict(), 'epoch': epoch }
                    torch.save(
                        state,
                        os.path.join(base_config.output_dir, "checkpoint_with_step{}.pth".format(total_step)),
                        _use_new_zipfile_serialization = False
                    )
        return True

    def first_stage_train(self, params: TrainParams) -> bool:
        ''' Fisrt stage trainning in train dataset, similar to blip2.
        `(fisrt stage: only qformer + LLM -> answer)`
        '''
        epoch = params.epoch
        total_loss = 0.0
        total_step = 0
        dataloader = self.dataloader_set.train_dataloader

        self.model.train()
        for step, batch in enumerate(dataloader):
            model_output = self.model.first_stage_forward(
                labels = batch['target'].squeeze().cuda(),
                input_ids = batch["inputs"].squeeze().cuda(),
                pixel_values = batch["image"].cuda(),
                attention_mask = batch["attention_mask"].squeeze().cuda(),
            )   
            loss: torch.FloatTensor = model_output.loss
            
            if base_config.gradient_accumulation_steps > 1:
                loss = loss / base_config.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), base_config.max_grad_norm)
            total_loss += loss.item()
            
            if (step + 1) % base_config.gradient_accumulation_steps == 0:
                total_step += 1
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.scheduler.step()
                self.model.zero_grad()

                if total_step % base_config.logging_steps == 0:
                    print("[Epoch {} | Step {}] average loss: {}".format(epoch, total_step, total_loss / total_step))

                if total_step % base_config.save_steps == 0 and total_step > 0:
                    to_save_param = {}
                    for name, params in self.model.named_parameters():
                        for save_name in self.save_name_list:
                            if save_name in name:
                                to_save_param[name] = params
                    state = { 'net': to_save_param, 'optimizer': self.optimizer.state_dict(), 'epoch': epoch }
                    torch.save(
                        state,
                        os.path.join(base_config.output_dir, "checkpoint_with_step{}.pth".format(total_step)),
                        _use_new_zipfile_serialization = False
                    )
        return True

    def run(self, run_list: List[Callable], func_params: TrainParams):
        ''' Run the code! '''
        for run_function in run_list:
            run_function(func_params)