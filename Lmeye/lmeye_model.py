import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel
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

    def __init__(self, config: Blip2Config):
        super().__init__(config)

        self.clip = None
        self.vision_model = Blip2VisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_pretrained("/root/data/model/ChatGLM2-6B/ChatGLM2-6B", trust_remote_code = True)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)
        #language_model = AutoModel.from_pretrained("/root/data/model/ChatGLM2-6B/ChatGLM2-6B", trust_remote_code=True).cuda()
        self.language_model = language_model
 
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

    def get_text_features(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.Tensor] = None,
            decoder_attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:
            text_outputs (`CausalLMOutputWithPast`, or `tuple(torch.FloatTensor)` if `return_dict=False`):
                The language model outputs. If `return_dict=True`, the output is a [`CausalLMOutputWithPast`] that
                contains the language model logits, the past key values and the hidden states if
                `output_hidden_states=True`.
        Examples:
        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, Blip2Model

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"

        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt").to(device)
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.use_decoder_only_language_model:
            text_outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

            text_outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )

        return text_outputs

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.FloatTensor,
            imgd_token_id: int,
            imgq_token_id: int,
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

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import Blip2Processor, Blip2Model
        >>> import torch

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"

        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> prompt = "Question: how many cats are there? Answer:"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

        >>> outputs = model(**inputs)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        bsz = input_ids.size(0)

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 图像特征，维度为bz * 257 * 1408
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        # inputs_embeds[input_ids == imgd_token_pos] += self.instruction_embedding_imgd

        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds], dim=1)


        tmp_ids = torch.zeros((bsz, 32)).to('cuda:0')
        tmp_pos_ids = torch.concat([tmp_ids, input_ids], dim=1)

        inputs_embeds[tmp_pos_ids == imgd_token_id] += self.instruction_embedding_imgd
        inputs_embeds[tmp_pos_ids == imgq_token_id] += self.instruction_embedding_imgq


        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        expected_device = language_model_attention_mask.device
        attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)

        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            # we compute the loss here since we need to take into account the sequence length of the query embeds
            if labels is not None:
                logits = logits[:, -labels.size(1):, :]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)

                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction="mean")

                loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
        else:
            encoder_hidden_state = self.language_model.encoder(
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask)[0]

            pre_query = encoder_hidden_state[tmp_pos_ids == imgq_token_id].view(bsz, -1)

            pre_query = self.instruction_blip22clip(pre_query).view(bsz, 5, -1)

            clip_frozon_embed = self.clip.text_model.embeddings
            clip_base_embeds = clip_frozon_embed(clip_text_input)


            query_output = self.instruction_linear(query_output)

            clip_hidden_states = torch.concat(
                [clip_base_embeds[:, 0, :].unsqueeze(dim=1), query_output, pre_query,
                 clip_base_embeds[:, -1, :].unsqueeze(dim=1)], dim=1)

            clip_pooled_output = self.clip.text_model(input_ids = clip_text_input, hidden_states = clip_hidden_states)[0]

            clip_pooled_output = clip_pooled_output[:, -5:, :]

            clip_text_feature = self.clip.text_projection(clip_pooled_output)

            imgd_prompt = self.instruction_imgdLinear(clip_text_feature).view(-1, self.config.text_config.hidden_size)

            inputs_embeds[tmp_pos_ids == imgd_token_id] += imgd_prompt

            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels.squeeze(),
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
            imgd_token_id: int,
            imgq_token_id: int,
            clip_text_input: torch.Tensor,
            input_ids: Optional[torch.LongTensor],
            attention_mask: Optional[torch.LongTensor] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        bsz = input_ids.size(0)

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask
        )
        query_output = query_outputs.last_hidden_state

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds], dim=1)

        tmp_ids = torch.zeros((bsz, 32)).to(inputs_embeds.device)
        tmp_pos_ids = torch.concat([tmp_ids, input_ids], dim=1)
        inputs_embeds[tmp_pos_ids == imgd_token_id] += self.instruction_embedding_imgd
        inputs_embeds[tmp_pos_ids == imgq_token_id] += self.instruction_embedding_imgq

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        expected_device = language_model_attention_mask.device
        attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)

        # 接下来我们需要从encoder中获取结果
        encoder_hidden_state = self.language_model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask)[0]

        pre_query = encoder_hidden_state[tmp_pos_ids == imgq_token_id].view(bsz, -1)

        pre_query = self.instruction_blip22clip(pre_query).view(bsz, 5, -1)

        clip_frozon_embed = self.clip.text_model.embeddings
        clip_base_embeds = clip_frozon_embed(clip_text_input)

        # 将qformer输出的维度进行转化
        query_output = self.instruction_linear(query_output)

        clip_hidden_states = torch.concat(
            [clip_base_embeds[:, 0, :].unsqueeze(dim=1), query_output, pre_query,
             clip_base_embeds[:, -1, :].unsqueeze(dim=1)], dim=1)

        clip_pooled_output = self.clip.text_model(input_ids=clip_text_input, hidden_states=clip_hidden_states)[0]

        clip_pooled_output = clip_pooled_output[:, -5:, :]

        clip_text_feature = self.clip.text_projection(clip_pooled_output)
        
        imgd_prompt = self.instruction_imgdLinear(clip_text_feature).view(-1, self.config.text_config.hidden_size)

        inputs_embeds[tmp_pos_ids == imgd_token_id] += imgd_prompt

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
        return outputs