import torch
from torch import nn

class LMEye_model(nn.Module):
    def __init__(self, llm_model, clip_model, clip_size=768, llm_size=1024, prompt_len=5, query_len=5):
        super(LMEye_model, self).__init__()

        self.imgdLinear = nn.Linear(clip_size, llm_size * query_len)
        self.imgLinear = nn.Linear(clip_size, llm_size)
        # self.promptMlp = ImageMLP(clip_size, clip_size, prompt_len, 0.5)
        self.promptMlp = nn.Linear(clip_size, clip_size * prompt_len)

        self.prompt_len = prompt_len
        self.query_len = query_len


        self.embedding_img = nn.Parameter(torch.randn(1, llm_size))

        self.embedding_imgd = nn.Parameter(torch.randn(1, llm_size))
        self.llm2clip = nn.Linear(llm_size, clip_size)

        self.dropout = nn.Dropout(0.5)
        self.llm_size = llm_size
        self.clip_size = clip_size
        self.llm = llm_model
        self.clip = clip_model

    def generate(self, clip_text_input, clip_attention_mask, image, llm_text_input,
                            pre_llm_text_input, pre_llm_attention_mask, img_token_id, imgd_token_id, beam_size, temperature):
        bsz = llm_text_input.size(0)
        frozon_embed = self.llm.get_input_embeddings()

        clip_img_feat = self.clip.get_image_features(image.unsqueeze(dim=0)).to(self.imgLinear.weight.dtype)

        img_prompt = self.imgLinear(clip_img_feat)


        pre_inputs_embeds = frozon_embed(pre_llm_text_input)

        pre_inputs_embeds[pre_llm_text_input == img_token_id] += img_prompt + self.embedding_img.repeat(bsz, 1)

        pre_inputs_embeds[pre_llm_text_input == imgd_token_id] += self.embedding_imgd.repeat(bsz, 1)

        last_hidden_state = \
            self.llm(inputs_embeds=pre_inputs_embeds, attention_mask=pre_llm_attention_mask, return_dict=True)[2]

        pre_query = last_hidden_state[pre_llm_text_input == imgd_token_id]
        pre_query = pre_query[0].unsqueeze(dim=0)

        pre_query = self.llm2clip(pre_query)


        clip_text_soft_prompt = self.promptMlp(clip_img_feat).view(bsz, self.prompt_len, -1)

        clip_frozon_embed = self.clip.text_model.embeddings

        clip_base_embeds = clip_frozon_embed(clip_text_input)
        clip_hidden_states = torch.concat(
            [clip_base_embeds[:, 0, :].unsqueeze(dim=1), clip_text_soft_prompt, pre_query.unsqueeze(dim=1),
             clip_base_embeds[:, self.prompt_len + 2, :].unsqueeze(dim=1)], dim=1)

        clip_pooled_output = self.clip.text_model(input_ids=clip_text_input, attention_mask=clip_attention_mask,
                                                  hidden_states=clip_hidden_states)[1]
        clip_text_feature = self.clip.text_projection(clip_pooled_output)

        imgd_prompt = self.imgdLinear(clip_text_feature).view(-1, self.llm_size)


        inputs_embeds = frozon_embed(llm_text_input)

        inputs_embeds[llm_text_input == img_token_id] += img_prompt + self.embedding_img.repeat(bsz, 1)

        inputs_embeds[llm_text_input == imgd_token_id] += imgd_prompt + self.embedding_imgd.repeat(bsz * self.query_len,
                                                                                                   1)

        output = self.llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=200, num_beams=beam_size, do_sample=True, temperature=temperature)

        return output
