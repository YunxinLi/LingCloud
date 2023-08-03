import torch
from torch import nn
import torch.utils.checkpoint
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import (
    Blip2PreTrainedModel,
    Blip2VisionModel,
    Blip2QFormerModel,
)
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer

from dataclasses import dataclass
from typing import Any, Optional, Tuple, List

from Lmeye.lmeye_config import base_config, IGNORE_INDEX, IMG_INDEX, IMG_D_INDEX, IMG_Q_INDEX
from Lmeye.blip2.blip2_config import Blip2Config

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

    def __init__(self, blip2_config: Blip2Config):
        super().__init__(blip2_config)

        self.clip = None
        self.vision_model = Blip2VisionModel(blip2_config.vision_config)

        self.qformer = Blip2QFormerModel(blip2_config.qformer_config)
        self.query_tokens = nn.Parameter(torch.zeros(1, blip2_config.num_query_tokens, blip2_config.qformer_config.hidden_size))

        self.language_projection = nn.Linear(blip2_config.qformer_config.hidden_size, blip2_config.text_config.hidden_size)
        if blip2_config.use_decoder_only_language_model:
            language_model = None
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(blip2_config.text_config)
        self.language_model = language_model

        self.instruction_embedding_imgd = nn.Parameter(torch.randn(1, blip2_config.text_config.hidden_size))
        self.instruction_embedding_imgq = nn.Parameter(torch.randn(1, blip2_config.text_config.hidden_size))
        self.instruction_blip22clip = nn.Linear(blip2_config.text_config.hidden_size, 5 * 768)
        self.instruction_linear = nn.Linear(768, 768)
        self.instruction_imgdLinear = nn.Linear(768, blip2_config.text_config.hidden_size)

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

    def get_output_embeddings(self) -> Optional[nn.Module]:
        if self.language_model == None:
            return None
        else:
            return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    def build_input(
            self,
            input_ids: torch.FloatTensor,
            pixel_values: torch.FloatTensor,
            attention_mask: torch.LongTensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = True,
        ):
        r""" Build input data with `input_ids(with padding)`, `pixel_values(output of the visual processor)`, `attention_mask(with padding)` """

        device = input_ids.device
        # step 1: forward the images through the vision encoder,
        # to get image_embedding, shape (batch_size, seq_len, hidden_size) = (batch_size * 257 * 1408)
        vision_outputs = self.vision_model(
            pixel_values = pixel_values,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
        )
        image_embedding = vision_outputs.last_hidden_state
        image_attention_mask = torch.ones(image_embedding.size()[:-1], dtype = torch.long, device = device)

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        # to get qformer_embedding, shape (batch_size, qformer_length, hidden_size)
        query_tokens = self.query_tokens.expand(image_embedding.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds = query_tokens,
            encoder_hidden_states = image_embedding,
            encoder_attention_mask = image_attention_mask,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
        )
        query_output = query_outputs.last_hidden_state
        qformer_embedding = self.language_projection(query_output)

        # step 3: use the language model, conditioned on the query outputs and the prompt
        # to get text_embedding, shape (batch_size, seq_length, hidden_size)
        tmp_ids = input_ids.clone()
        input_ids[input_ids < 0] = 0

        input_embedding = self.language_model.get_input_embeddings()(input_ids)
        input_embedding[tmp_ids == IMG_INDEX] += qformer_embedding.view(-1, qformer_embedding.shape[-1])
        input_embedding[tmp_ids == IMG_Q_INDEX] += self.instruction_embedding_imgq
        input_embedding[tmp_ids == IMG_D_INDEX] += self.instruction_embedding_imgd

        return input_embedding, attention_mask, vision_outputs, query_outputs

    def first_stage_forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = True,
    ):
        r"""First stage forward, similar to blip2.
        `(fisrt stage: only qformer + LLM -> answer)`
        """

        inputs_embedding, attention_mask, vision_outputs, query_outputs = self.build_input(
            input_ids = input_ids,
            pixel_values = pixel_values,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            attention_mask = attention_mask,
            return_dict = return_dict
        )

        outputs = self.language_model(
            input_ids = None,
            inputs_embeds = inputs_embedding.half(),
            attention_mask = attention_mask,
            return_dict = return_dict,
            labels = labels,
        )
        
        loss = outputs.loss if return_dict else outputs[0]
        logits = outputs.logits if return_dict else outputs[1]

        return Blip2ForConditionalGenerationModelOutput(
            loss = loss,
            logits = logits,
            vision_outputs = vision_outputs,
            qformer_outputs = query_outputs,
            language_model_outputs = outputs,
        )

    def first_stage_generate(
        self,
        input_ids: Optional[torch.LongTensor],
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ):
        r"""First stage generate, similar to blip2, use in combination with `first_stage_forward` function.
        `(fisrt stage: only qformer + LLM -> answer)`
        """
        inputs_embedding, attention_mask, _, _ = self.build_input(
            input_ids = input_ids,
            pixel_values = pixel_values,
            attention_mask = attention_mask,
        )

        outputs = self.language_model.generate(
            inputs_embeds = inputs_embedding.half(),
            attention_mask = attention_mask,
            bos_token_id = 1,
            **generate_kwargs,
        )

        return outputs

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
        r""" All stage forward.
        `(all stage: qformer + LLM -> hidden_query + CLIP + LLM -> answer)`
        """

        batch_size = input_ids.shape[0]
        tmp_ids = input_ids.clone()

        inputs_embedding, attention_mask, vision_outputs, query_outputs = self.build_input(
            input_ids = input_ids,
            pixel_values = pixel_values,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            attention_mask = attention_mask,
            return_dict = return_dict
        )

        if base_config.decoder_only:
            # step 4: first stage to get imgq embedding
            labels = labels.squeeze()

            encoder_hidden_state = self.language_model.transformer(
                input_ids = None,
                inputs_embeds = inputs_embedding.half(),
                attention_mask = attention_mask,
            )[0].transpose(0, 1)

            pre_query = encoder_hidden_state[tmp_ids == IMG_Q_INDEX].view(batch_size, -1)
            pre_query = self.instruction_blip22clip(pre_query.to(torch.float32)).view(batch_size, base_config.imgd_number, -1)

            # step 5: get imgd embedding
            clip_frozon_embed = self.clip.text_model.embeddings
            clip_base_embeds = clip_frozon_embed(clip_text_input)

            query_output = self.instruction_linear(query_outputs[0])

            clip_hidden_states = torch.concat([
                clip_base_embeds[:, 0, :].unsqueeze(dim = 1),
                query_output, pre_query,
                clip_base_embeds[:, -1, :].unsqueeze(dim = 1)
            ], dim = 1)

            clip_pooled_output = self.clip.text_model(input_ids = clip_text_input, hidden_states = clip_hidden_states)[0]
            clip_pooled_output = clip_pooled_output[:, -base_config.imgd_number:, :]
            clip_text_feature = self.clip.text_projection(clip_pooled_output)
            imgd_prompt = self.instruction_imgdLinear(clip_text_feature)

            inputs_embedding[tmp_ids == IMG_D_INDEX] += imgd_prompt.view(-1, imgd_prompt.shape[-1])

            # step 6: second stage to get answer
            outputs = self.language_model(
                input_ids = None,
                inputs_embeds = inputs_embedding.half(),
                attention_mask = attention_mask,
                return_dict = return_dict,
                labels = labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

            return Blip2ForConditionalGenerationModelOutput(
                loss = loss,
                logits = logits,
                vision_outputs = vision_outputs,
                qformer_outputs = query_outputs,
                language_model_outputs = outputs,
            )
        else:
            encoder_hidden_state = self.language_model.encoder(
                inputs_embeds = inputs_embedding,
                attention_mask = attention_mask
            )[0]
    
            # pre_query = encoder_hidden_state[torch.tensor([[index] for index in range(batch_size)]), imgq_index].view(batch_size, -1)
            pre_query = encoder_hidden_state[tmp_ids == -102].view(batch_size, -1)
            pre_query = self.instruction_blip22clip(pre_query.to(torch.float32)).view(batch_size, 5, -1)

            clip_frozon_embed = self.clip.text_model.embeddings
            clip_base_embeds = clip_frozon_embed(clip_text_input)

            query_output = self.instruction_linear(query_outputs[0])

            clip_hidden_states = torch.concat([
                clip_base_embeds[:, 0, :].unsqueeze(dim = 1),
                query_output,
                pre_query,
                clip_base_embeds[:, -1, :].unsqueeze(dim = 1)
            ], dim = 1)

            clip_pooled_output = self.clip.text_model(input_ids = clip_text_input, hidden_states = clip_hidden_states)[0]
            clip_pooled_output = clip_pooled_output[:, -5:, :]
            clip_text_feature = self.clip.text_projection(clip_pooled_output)
            imgd_prompt = self.instruction_imgdLinear(clip_text_feature)
            inputs_embedding[tmp_ids == -103] += imgd_prompt.view(-1, imgd_prompt.shape[-1])

            outputs = self.language_model(
                inputs_embeds = inputs_embedding,
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

        #if not return_dict:
        #    output = (logits, vision_outputs, query_outputs, outputs)
        #    return ((loss,) + output) if loss is not None else output

            return Blip2ForConditionalGenerationModelOutput(
                loss = loss,
                logits = logits,
                vision_outputs = vision_outputs,
                qformer_outputs = query_outputs,
                language_model_outputs = outputs,
            )

    def generate(
        self,
        clip_text_input: torch.Tensor,
        input_ids: Optional[torch.LongTensor],
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        r""" All stage generate, use in combination with `forward` function.
        `(all stage: qformer + LLM -> hidden_query + CLIP + LLM -> answer)`
        """
        batch_size = input_ids.shape[0]
        tmp_ids = input_ids.clone()

        inputs_embedding, attention_mask, vision_outputs, query_outputs = self.build_input(
            input_ids = input_ids,
            pixel_values = pixel_values,
            attention_mask = attention_mask,
        )

        if base_config.decoder_only:
            # step 4: first stage to get imgq embedding
            encoder_hidden_state = self.language_model.transformer(
                input_ids = None,
                inputs_embeds = inputs_embedding.half(),
                attention_mask = attention_mask,
            )[0].transpose(0, 1)

            pre_query = encoder_hidden_state[tmp_ids == IMG_Q_INDEX].view(batch_size, -1)
            pre_query = self.instruction_blip22clip(pre_query.to(torch.float32)).view(batch_size, base_config.imgd_number, -1)

            # step 5: get imgd embedding
            clip_frozon_embed = self.clip.text_model.embeddings
            clip_base_embeds = clip_frozon_embed(clip_text_input)

            query_output = self.instruction_linear(query_outputs[0])

            clip_hidden_states = torch.concat([
                clip_base_embeds[:, 0, :].unsqueeze(dim = 1),
                query_output, pre_query,
                clip_base_embeds[:, -1, :].unsqueeze(dim = 1)
            ], dim = 1)

            clip_pooled_output = self.clip.text_model(input_ids = clip_text_input, hidden_states = clip_hidden_states)[0]
            clip_pooled_output = clip_pooled_output[:, -base_config.imgd_number:, :]
            clip_text_feature = self.clip.text_projection(clip_pooled_output)
            imgd_prompt = self.instruction_imgdLinear(clip_text_feature)

            inputs_embedding[tmp_ids == IMG_D_INDEX] += imgd_prompt.view(-1, imgd_prompt.shape[-1])

            # step 6: second stage to get answer
            outputs = self.language_model.generate(
                inputs_embeds = inputs_embedding.half(),
                attention_mask = attention_mask,
                bos_token_id = 1,
                **generate_kwargs,
            )
            return outputs
        else:
            # step 4: first stage to get imgq embedding
            encoder_hidden_state = self.language_model.encoder(
                inputs_embeds = inputs_embedding,
                attention_mask = attention_mask,
            )[0]

            pre_query = encoder_hidden_state[tmp_ids == IMG_Q_INDEX].view(batch_size, -1)
            pre_query = self.instruction_blip22clip(pre_query.to(torch.float32)).view(batch_size, base_config.imgd_number, -1)

            # step 5: get imgd embedding
            clip_frozon_embed = self.clip.text_model.embeddings
            clip_base_embeds = clip_frozon_embed(clip_text_input)

            query_output = self.instruction_linear(query_outputs[0])

            clip_hidden_states = torch.concat([
                clip_base_embeds[:, 0, :].unsqueeze(dim = 1),
                query_output, pre_query,
                clip_base_embeds[:, -1, :].unsqueeze(dim = 1)
            ], dim = 1)

            clip_pooled_output = self.clip.text_model(input_ids = clip_text_input, hidden_states = clip_hidden_states)[0]
            clip_pooled_output = clip_pooled_output[:, -base_config.imgd_number:, :]
            clip_text_feature = self.clip.text_projection(clip_pooled_output)
            imgd_prompt = self.instruction_imgdLinear(clip_text_feature)

            inputs_embedding[tmp_ids == IMG_D_INDEX] += imgd_prompt.view(-1, imgd_prompt.shape[-1])

            # step 6: second stage to get answer
            outputs = self.language_model.generate(
                inputs_embeds = inputs_embedding,
                attention_mask = attention_mask,
                **generate_kwargs,
            )
            return outputs