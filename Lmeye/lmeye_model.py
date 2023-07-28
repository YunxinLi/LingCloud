import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoConfig, AutoModel
from transformers import (
    Blip2Config,
    Blip2PreTrainedModel,
    Blip2VisionModel,
    Blip2QFormerModel,
)

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
from Lmeye.lmeye_config import config
base_config = config
@dataclass
class Blip2ForConditionalGenerationModelOutput(ModelOutput):
    """
    Class defining the outputs of [`Blip2ForConditionalGeneration`].

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        vision_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.
        qformer_outputs (`BaseModelOutputWithPoolingAndCrossAttentions`):
            Outputs of the Q-Former (Querying Transformer).
        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
    """

    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    vision_outputs: Optional[torch.FloatTensor] = None
    qformer_outputs: Optional[Tuple[torch.FloatTensor]] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["vision_outputs", "qformer_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )

class Blip2InstructionQueryModel(Blip2PreTrainedModel):
    config_class = Blip2Config
    main_input_name = "pixel_values"

    def __init__(self, data_path):
        if base_config.model_type == "GLM":
            blip2_model_config = Blip2Config.from_pretrained("/root/data/model/blip2-GLM")
            language_model_config = AutoConfig.from_pretrained("/root/data/model/ChatGLM2-6B/ChatGLM2-6B", trust_remote_code = True)
            blip2_model_config.text_config = language_model_config
            print(blip2_model_config.vision_config)
            super().__init__(blip2_model_config)


            self.vision_model = Blip2VisionModel(blip2_model_config.vision_config)
            self.query_tokens = nn.Parameter(torch.zeros(1, blip2_model_config.num_query_tokens, blip2_model_config.qformer_config.hidden_size))
            self.qformer = Blip2QFormerModel(blip2_model_config.qformer_config)

            self.language_model = AutoModel.from_pretrained("/root/data/model/ChatGLM2-6B/ChatGLM2-6B", trust_remote_code = True)
            self.language_projection = nn.Linear(blip2_model_config.qformer_config.hidden_size, language_model_config.hidden_size)  
            self.instruction_embedding_imgd = nn.Parameter(torch.randn(1, language_model_config.hidden_size))
            self.instruction_embedding_imgq = nn.Parameter(torch.randn(1, language_model_config.hidden_size))
            self.instruction_blip22clip = nn.Linear(language_model_config.hidden_size, 5 * 768)
            self.instruction_linear = nn.Linear(768, 768)
            self.instruction_imgdLinear = nn.Linear(768, language_model_config.hidden_size)

        if base_config.model_type == "FLAN-T5":
            config = Blip2Config.from_pretrained("/root/data/model/blip2-flan-t5-xl")
            super().__init__(config)
            self.vision_model = Blip2VisionModel(config.vision_config)
            self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
            self.qformer = Blip2QFormerModel(config.qformer_config)
            #language_model = AutoModel.from_pretrained("/root/data/model/ChatGLM2-6B/ChatGLM2-6B", trust_remote_code=True).cuda()
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)
            config.text_config = config.text_config
            self.language_model = language_model

            print(config.text_config.hidden_size)
            self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)  
            self.instruction_embedding_imgd = nn.Parameter(torch.randn(1, config.text_config.hidden_size))
            self.instruction_embedding_imgq = nn.Parameter(torch.randn(1, config.text_config.hidden_size))
            self.instruction_blip22clip = nn.Linear(config.text_config.hidden_size, 5 * 768)
            self.instruction_linear = nn.Linear(768, 768)
            self.instruction_imgdLinear = nn.Linear(768, config.text_config.hidden_size)

            # Initialize weights and apply final processing
            self.post_init()

    def load_clip(self, clip_model):
        self.clip = clip_model

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.FloatTensor,
            clip_text_input: torch.Tensor,
            attention_mask: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            labels: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Blip2ForConditionalGenerationModelOutput:
        r"""
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size) = (batch_size * 257 * 1408)
        vision_outputs = self.vision_model(
            pixel_values = pixel_values,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype = torch.long, device = device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds = query_tokens,
            encoder_hidden_states = image_embeds,
            encoder_attention_mask = image_attention_mask,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype = torch.long, device = device
        )
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # step 4: cat [image_qformer_query, 1*img-q + 5*img-d, text_query]
        qformer_length = language_model_inputs.shape[1]                                     # The qformer output length
        imgq_token_number = 1                                                               # The number of imgq
        imgd_token_number = 5                                                               # The number of imgd
        all_image_query_length = qformer_length + imgq_token_number + imgd_token_number     # Total length of image tokens before text tokens

        inputs_embeds = torch.cat([
            language_model_inputs,
            torch.zeros((batch_size, imgq_token_number + imgd_token_number, inputs_embeds.shape[2]), device = device),
            inputs_embeds,
        ], dim = 1)

        inputs_embeds[:, qformer_length] += self.instruction_embedding_imgq
        inputs_embeds[:, qformer_length + imgq_token_number: all_image_query_length] += self.instruction_embedding_imgd

        language_model_attention_mask = torch.cat([
            language_model_attention_mask,
            torch.ones((batch_size, imgq_token_number + imgd_token_number), device = device),
        ], dim = 1)

        tmp_pos_ids = torch.concat([
            torch.zeros((batch_size, all_image_query_length), device = device),
            input_ids
        ], dim = 1)

        if attention_mask is None:
            attention_mask = torch.ones_like(tmp_pos_ids)
        attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(device)], dim=1)

        if config.decoder_only:
            # step 5: first stage to get imgq embedding
            labels = labels.squeeze()
            labels = torch.concat([torch.full((batch_size, all_image_query_length), -100).to(device), labels], dim = 1)

            encoder_hidden_state = self.language_model.transformer(
                input_ids = None,
                inputs_embeds = inputs_embeds.half(),
            )[0].transpose(0, 1)

            pre_query = encoder_hidden_state[:, qformer_length].view(batch_size, -1)
            pre_query = self.instruction_blip22clip(pre_query.to(torch.float32)).view(batch_size, imgd_token_number, -1)

            # step 6: get imgd embedding
            clip_frozon_embed = self.clip.text_model.embeddings
            clip_base_embeds = clip_frozon_embed(clip_text_input)

            query_output = self.instruction_linear(query_output)

            clip_hidden_states = torch.concat([
                clip_base_embeds[:, 0, :].unsqueeze(dim = 1),
                query_output, pre_query,
                clip_base_embeds[:, -1, :].unsqueeze(dim = 1)
            ], dim = 1)

            clip_pooled_output = self.clip.text_model(input_ids = clip_text_input, hidden_states = clip_hidden_states)[0]
            clip_pooled_output = clip_pooled_output[:, -imgd_token_number:, :]
            clip_text_feature = self.clip.text_projection(clip_pooled_output)
            imgd_prompt = self.instruction_imgdLinear(clip_text_feature)

            inputs_embeds[:, qformer_length + imgq_token_number: all_image_query_length] += imgd_prompt

            # step 7: second stage to get answer
            outputs = self.language_model(
                input_ids = None,
                inputs_embeds = inputs_embeds.half(),
                attention_mask = attention_mask,
                return_dict = return_dict,
                labels = labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

            if not return_dict:
                output = (logits, vision_outputs, query_outputs, outputs)
                return ((loss,) + output) if loss is not None else output

            return Blip2ForConditionalGenerationModelOutput(
                loss = loss,
                logits = logits,
                vision_outputs = vision_outputs,
                qformer_outputs = query_outputs,
                language_model_outputs = outputs,
            )
        else:
            encoder_hidden_state = self.language_model.encoder(
                inputs_embeds = inputs_embeds,
                attention_mask = attention_mask
            )[0]
            
            pre_query = encoder_hidden_state[:, qformer_length].view(batch_size, -1)
            pre_query = self.instruction_blip22clip(pre_query.to(torch.float32)).view(batch_size, 5, -1)

            clip_frozon_embed = self.clip.text_model.embeddings
            clip_base_embeds = clip_frozon_embed(clip_text_input)

            query_output = self.instruction_linear(query_output)

            clip_hidden_states = torch.concat(
                [clip_base_embeds[:, 0, :].unsqueeze(dim=1), query_output, pre_query,
                 clip_base_embeds[:, -1, :].unsqueeze(dim=1)],
            dim = 1)

            clip_pooled_output = self.clip.text_model(input_ids = clip_text_input, hidden_states = clip_hidden_states)[0]
            clip_pooled_output = clip_pooled_output[:, -5:, :]
            clip_text_feature = self.clip.text_projection(clip_pooled_output)
            imgd_prompt = self.instruction_imgdLinear(clip_text_feature)
            inputs_embeds[:, qformer_length + imgq_token_number: all_image_query_length] += imgd_prompt

            outputs = self.language_model(
                inputs_embeds = inputs_embeds,
                attention_mask = attention_mask,
                decoder_input_ids = decoder_input_ids,
                decoder_attention_mask = decoder_attention_mask,
                output_attentions = output_attentions,
                output_hidden_states = output_hidden_states,
                return_dict = return_dict,
                labels = labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return Blip2ForConditionalGenerationModelOutput(
            loss = loss,
            logits = logits,
            vision_outputs = vision_outputs,
            qformer_outputs = query_outputs,
            language_model_outputs = outputs,
        )

    def generate(
        self,
        pixel_values: torch.FloatTensor,
        clip_text_input: torch.Tensor,
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size) = (batch_size * 257 * 1408)
        image_embeds = self.vision_model(pixel_values, return_dict = True).last_hidden_state

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype = torch.long, device = device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds = query_tokens,
            encoder_hidden_states = image_embeds,
            encoder_attention_mask = image_attention_mask
        )
        query_output = query_outputs.last_hidden_state

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype = torch.long, device = device
        )
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # step 4: cat [image_qformer_query, 1*img-q + 5*img-d, text_query]
        qformer_length = language_model_inputs.shape[1]                                     # The qformer output length
        imgq_token_number = 1                                                               # The number of imgq
        imgd_token_number = 5                                                               # The number of imgd
        all_image_query_length = qformer_length + imgq_token_number + imgd_token_number     # Total length of image tokens before text tokens

        inputs_embeds = torch.cat([
            language_model_inputs,
            torch.zeros((batch_size, imgq_token_number + imgd_token_number, inputs_embeds.shape[2]), device = device),
            inputs_embeds,
        ], dim = 1)

        inputs_embeds[:, qformer_length] += self.instruction_embedding_imgq
        inputs_embeds[:, qformer_length + imgq_token_number: all_image_query_length] += self.instruction_embedding_imgd

        language_model_attention_mask = torch.cat([
            language_model_attention_mask,
            torch.ones((batch_size, imgq_token_number + imgd_token_number), device = device),
        ], dim = 1)

        tmp_pos_ids = torch.concat([
            torch.zeros((batch_size, all_image_query_length), device = device),
            input_ids
        ], dim = 1)

        if attention_mask is None:
            attention_mask = torch.ones_like(tmp_pos_ids)
        attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(device)], dim=1)

        if config.decoder_only:
            # step 5: first stage to get imgq embedding
            encoder_hidden_state = self.language_model.transformer(
                input_ids = None,
                inputs_embeds = inputs_embeds.half(),
                attention_mask = attention_mask,
            )[0].transpose(0, 1)

            pre_query = encoder_hidden_state[:, qformer_length].view(batch_size, -1)
            pre_query = self.instruction_blip22clip(pre_query.to(torch.float32)).view(batch_size, imgd_token_number, -1)

            # step 6: get imgd embedding
            clip_frozon_embed = self.clip.text_model.embeddings
            clip_base_embeds = clip_frozon_embed(clip_text_input)

            query_output = self.instruction_linear(query_output)

            clip_hidden_states = torch.concat([
                clip_base_embeds[:, 0, :].unsqueeze(dim = 1),
                query_output, pre_query,
                clip_base_embeds[:, -1, :].unsqueeze(dim = 1)
            ], dim = 1)

            clip_pooled_output = self.clip.text_model(input_ids = clip_text_input, hidden_states = clip_hidden_states)[0]
            clip_pooled_output = clip_pooled_output[:, -imgd_token_number:, :]
            clip_text_feature = self.clip.text_projection(clip_pooled_output)
            imgd_prompt = self.instruction_imgdLinear(clip_text_feature)

            inputs_embeds[:, qformer_length + imgq_token_number: all_image_query_length] += imgd_prompt

            outputs = self.language_model.generate(
                inputs_embeds = inputs_embeds.half(),
                attention_mask = attention_mask,
                bos_token_id = 0,
                **generate_kwargs,
            )
            return outputs
        else:
            # step 5: first stage to get imgq embedding
            encoder_hidden_state = self.language_model.encoder(
                inputs_embeds = inputs_embeds,
                attention_mask = attention_mask,
            )[0]

            pre_query = encoder_hidden_state[:, qformer_length].view(batch_size, -1)
            pre_query = self.instruction_blip22clip(pre_query.to(torch.float32)).view(batch_size, imgd_token_number, -1)

            # step 6: get imgd embedding
            clip_frozon_embed = self.clip.text_model.embeddings
            clip_base_embeds = clip_frozon_embed(clip_text_input)

            query_output = self.instruction_linear(query_output)

            clip_hidden_states = torch.concat([
                clip_base_embeds[:, 0, :].unsqueeze(dim = 1),
                query_output, pre_query,
                clip_base_embeds[:, -1, :].unsqueeze(dim = 1)
            ], dim = 1)

            clip_pooled_output = self.clip.text_model(input_ids = clip_text_input, hidden_states = clip_hidden_states)[0]
            clip_pooled_output = clip_pooled_output[:, -imgd_token_number:, :]
            clip_text_feature = self.clip.text_projection(clip_pooled_output)
            imgd_prompt = self.instruction_imgdLinear(clip_text_feature)

            inputs_embeds[:, qformer_length + imgq_token_number: all_image_query_length] += imgd_prompt

            outputs = self.language_model.generate(
                inputs_embeds = inputs_embeds,
                attention_mask = attention_mask,
                **generate_kwargs,
            )
            return outputs