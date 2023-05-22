import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss


# MLP Structure
class ImageMLP(nn.Module):
    def __init__(self, feature_size, hidden_size, prefix_len, hidden_dropout_prob):
        super(ImageMLP, self).__init__()
        hidden_size = hidden_size * prefix_len

        self.dense0 = nn.Linear(feature_size, hidden_size)
        self.dense1 = nn.Linear(hidden_size, hidden_size * 4)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dense0(hidden_states)
        hidden_states = self.relu(hidden_states)
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.relu(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

class single_mapping_model(nn.Module):
    def __init__(self, llm_model, clip_size=512, llm_size=1024):
        super(single_mapping_model, self).__init__()
        self.mlp = ImageMLP(clip_size, llm_size, 1, 0.5)
        self.llm_size = llm_size
        self.llm = llm_model

    def forward(self, clip_img_input, llm_text_input, llm_attention_mask, labels):
        img_feat = clip_img_input.float()
        prompt_ids = self.mlp(img_feat).view(img_feat.size(0), -1, self.llm_size)

        # embedding层的prompt
        output = self.llm(input_ids=llm_text_input, attention_mask=llm_attention_mask,
                          labels=labels, prompt_ids=prompt_ids)
        loss = output[0]
        return loss


class single_mapping_model_extra_vqa(nn.Module):
    def __init__(self, llm_model, clip_size=512, llm_size=1024):
        super(single_mapping_model_extra_vqa, self).__init__()

        self.linear = nn.Linear(clip_size, llm_size)

        self.new_embedding = nn.Parameter(torch.randn(1, llm_size))

        self.llm_size = llm_size
        self.llm = llm_model

    def forward(self, clip_img_input, llm_text_input, llm_attention_mask):
        img_feat = clip_img_input.float()
        prompt_ids = self.linear(img_feat)

        frozon_embed = self.llm.get_input_embeddings()
        inputs_embeds = frozon_embed(llm_text_input)
        extra_embed = self.new_embedding.repeat(prompt_ids.size(0), 1)

        inputs_embeds[:, 1, :] += (prompt_ids + extra_embed)

        img_embeds = inputs_embeds[:, 1, :].unsqueeze(dim=1)
        generated_caption = self.llm.generate(inputs_embeds=img_embeds)

        generated_embeds = frozon_embed(generated_caption)

        inputs_embeds = torch.concat([inputs_embeds[:, :2, :], generated_embeds, inputs_embeds[:, 2:, ]], dim=1)

        output = self.llm.generate(inputs_embeds=inputs_embeds)

        return output


class single_mapping_model_mutilPredict(nn.Module):
    def __init__(self, llm_model, clip_size=512, llm_size=1024, prompt_len=5):
        super(single_mapping_model_mutilPredict, self).__init__()

        self.imageLinear = nn.Linear(clip_size, llm_size * prompt_len)

        self.new_embedding = nn.Parameter(torch.randn(1, llm_size))

        self.llm_size = llm_size
        self.prompt_size = prompt_len
        self.llm = llm_model

    def forward(self, clip_img_input, llm_text_input, llm_attention_mask, task, img_token_id):
        bsz = llm_text_input.size(0)
        img_pos = llm_text_input == img_token_id
        img_pos = img_pos.nonzero()
        indices = [x[1].item() for x in img_pos]

        img_feat = clip_img_input.float()
        prompt_ids = self.imageLinear(img_feat).view(bsz, self.prompt_size, self.llm_size)

        frozon_embed = self.llm.get_input_embeddings()
        inputs_embeds = frozon_embed(llm_text_input)
        extra_embed = self.new_embedding.repeat(bsz * self.prompt_size, 1)
        extra_embed = extra_embed.view(bsz, self.prompt_size, self.llm_size)

        for idx, indice in enumerate(indices):
            inputs_embeds[:, indice, :] += (prompt_ids[:, idx, :] + extra_embed[:, idx, :])

        if "vqa" in task:
            output = self.llm.generate(inputs_embeds=inputs_embeds, num_beams=5, length_penalty=-1)
        else:
            output = self.llm.generate(inputs_embeds=inputs_embeds)

        return output


class single_mapping_model_extra_caption(nn.Module):
    def __init__(self, llm_model, clip_size=512, llm_size=1024):
        super(single_mapping_model_extra_caption, self).__init__()

        self.imageLinear = nn.Linear(clip_size, llm_size)

        self.new_embedding = nn.Parameter(torch.randn(1, llm_size))

        self.llm_size = llm_size
        self.llm = llm_model

    def forward(self, clip_img_input, llm_text_input, llm_attention_mask, task, img_token_id):
        bsz = llm_text_input.size(0)
        img_pos = llm_text_input == img_token_id
        img_pos = img_pos.nonzero()
        indices = [x[1].item() for x in img_pos]

        img_feat = clip_img_input
        prompt_ids = self.imageLinear(img_feat)

        frozon_embed = self.llm.get_input_embeddings()
        inputs_embeds = frozon_embed(llm_text_input)
        extra_embed = self.new_embedding.repeat(prompt_ids.size(0), 1)

        for indice in indices:
            inputs_embeds[:, indice, :] += (prompt_ids + extra_embed)

        attention_mask = llm_attention_mask
        if "vqa" in task:
            output = self.llm.generate(inputs_embeds=inputs_embeds, num_beams=5, length_penalty=-1)
        elif "detail-c" in task:
            output = self.llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=250, num_beams=5, do_sample=True,
                                       early_stopping=True)
        elif "detail-q" in task:
            output = self.llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=50)
        else:
            output = self.llm.generate(inputs_embeds=inputs_embeds)

        return output

    def dialog(self, clip_img_input, llm_text_inputs, token_ids_map):
        llm_output_text_input = torch.concat(
            [token_ids_map["img_prompt"], token_ids_map["question"], llm_text_inputs[0].unsqueeze(dim=0)], dim=1)
        for i in range(len(llm_text_inputs)):

            llm_text_input = llm_output_text_input
            if i > 0:
                llm_text_input = torch.cat(
                    [llm_text_input, token_ids_map["question"], llm_text_inputs[i].unsqueeze(dim=0)], dim=1)
                llm_output_text_input = torch.cat(
                    [llm_output_text_input, token_ids_map["question"], llm_text_inputs[i].unsqueeze(dim=0)], dim=1)

            llm_text_input = torch.cat([llm_text_input, token_ids_map["answer"]], dim=1)
            llm_output_text_input = torch.cat([llm_output_text_input, token_ids_map["answer"]], dim=1)

            clip_img_feat = clip_img_input
            img_prompt = self.imageLinear(clip_img_feat)

            frozon_embed = self.llm.get_input_embeddings()
            inputs_embeds = frozon_embed(llm_text_input)

            inputs_embeds[llm_text_input == token_ids_map["<img>"]] += img_prompt

            output = self.llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=50)

            llm_output_text_input = torch.cat([llm_output_text_input, output, token_ids_map["cl"]], dim=1)

        return llm_output_text_input


class single_mapping_model_text(nn.Module):
    def __init__(self, llm_model, clip_model, clip_size=512, llm_size=1024, is_linear=False):
        super(single_mapping_model_text, self).__init__()

        if is_linear:
            self.mlp = nn.Linear(clip_size, llm_size)
        else:
            self.mlp = ImageMLP(clip_size, llm_size, 1, 0.5)

        self.linear = nn.Linear(clip_size, llm_size)

        self.new_embedding = nn.Parameter(torch.randn(1, llm_size))

        self.llm_size = llm_size
        self.llm = llm_model
        self.clip = clip_model

    def forward(self, clip_text_input, llm_text_input, llm_attention_mask, labels):
        text_feat = self.clip.get_text_features(clip_text_input).float()

        prompt_ids = self.mlp(text_feat).view(text_feat.size(0), self.llm_size)

        prompt_ids = prompt_ids + self.linear(text_feat).view(text_feat.size(0), self.llm_size)

        frozon_embed = self.llm.get_input_embeddings()

        inputs_embeds = frozon_embed(llm_text_input)

        extra_embed = self.new_embedding.repeat(prompt_ids.size(0), 1)


        inputs_embeds[:, 0, :] += (prompt_ids + extra_embed)

        attention_mask = llm_attention_mask
        loss = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)[0]

        return loss


class single_mapping_model_extra_text_caption(nn.Module):
    def __init__(self, llm_model, clip_model, clip_size=512, llm_size=1024, is_linear=False):
        super(single_mapping_model_extra_text_caption, self).__init__()
        # if is_linear:
        #     self.mlp = nn.Linear(clip_size, llm_size)
        # else:
        #     self.mlp = ImageMLP(clip_size, llm_size, 1, 0.5)
        #
        # self.linear = nn.Linear(clip_size, llm_size)
        self.textLinear = nn.Linear(clip_size, llm_size)
        self.new_embedding = nn.Parameter(torch.randn(1, llm_size))

        self.llm_size = llm_size
        self.llm = llm_model
        self.clip = clip_model

    def forward(self, clip_text_input, llm_text_input, llm_attention_mask, task="caption"):
        text_feat = self.clip.get_text_features(clip_text_input).float()

        prompt_ids = self.textLinear(text_feat).view(text_feat.size(0), self.llm_size)

        frozon_embed = self.llm.get_input_embeddings()

        inputs_embeds = frozon_embed(llm_text_input)

        extra_embed = self.new_embedding.repeat(prompt_ids.size(0), 1)

        # 在预训练时，我们的<img>标识一定位于句首，所以我们直接在句首进行添加即可
        inputs_embeds[:, 0, :] += (prompt_ids + extra_embed)

        attention_mask = llm_attention_mask
        if task == "caption":
            output = self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            output = self.llm.generate(inputs_embeds=inputs_embeds, num_beams=5,
                                       length_penalty=-1)

        return output


class single_mapping_model_extra_text_vqa(nn.Module):
    def __init__(self, llm_model, clip_model, clip_size=512, llm_size=1024, is_linear=False):
        super(single_mapping_model_extra_text_vqa, self).__init__()
        if is_linear:
            self.mlp = nn.Linear(clip_size, llm_size)
        else:
            self.mlp = ImageMLP(clip_size, llm_size, 1, 0.5)

        self.linear = nn.Linear(clip_size, llm_size)

        self.new_embedding = nn.Parameter(torch.randn(1, llm_size))

        self.llm_size = llm_size
        self.llm = llm_model
        self.clip = clip_model

    def forward(self, clip_text_input, llm_text_input, llm_attention_mask):
        text_feat = self.clip.get_text_features(clip_text_input).float()

        prompt_ids = self.mlp(text_feat).view(text_feat.size(0), self.llm_size)

        prompt_ids = prompt_ids + self.linear(text_feat).view(text_feat.size(0), self.llm_size)

        frozon_embed = self.llm.get_input_embeddings()

        inputs_embeds = frozon_embed(llm_text_input)

        extra_embed = self.new_embedding.repeat(prompt_ids.size(0), 1)

        # 在预训练时，我们的<img>标识一定位于句首，所以我们直接在句首进行添加即可
        inputs_embeds[:, 0, :] += (prompt_ids + extra_embed)

        attention_mask = llm_attention_mask
        output = self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, num_beams=5)

        return output


class single_mapping_model_i2t(nn.Module):
    def __init__(self, llm_model, clip_model, clip_size=512, llm_size=1024, is_linear=False):
        super(single_mapping_model_i2t, self).__init__()
        if not is_linear:
            self.mlp1 = ImageMLP(clip_size, clip_size, 1, 0.5)
            self.mlp2 = ImageMLP(clip_size, llm_size, 1, 0.5)
        else:
            self.mlp1 = nn.Linear(clip_size, clip_size)
            self.mlp2 = nn.Linear(clip_size, llm_size)

        self.linear = nn.Linear(clip_size, llm_size)

        self.new_embedding = nn.Parameter(torch.randn(1, llm_size))

        # self.clipi2t = nn.Parameter(torch.rand(1))

        self.llm_size = llm_size
        self.llm = llm_model
        self.clip = clip_model

    def forward(self, clip_img_input, clip_text_input, llm_text_input, llm_attention_mask, labels):
        img_feat = clip_img_input.float()

        m_text_feat = self.mlp1(img_feat)
        # text_feat = clip_text_input.float()
        text_feat = self.clip.get_text_features(clip_text_input).float()
        sim1 = m_text_feat / m_text_feat.norm(dim=-1, keepdim=True)
        sim2 = text_feat / text_feat.norm(dim=-1, keepdim=True)
        logits = 100 * (sim1 @ sim2.T)
        targets = torch.arange(img_feat.size(0)).to(logits.device)
        loss_fct = CrossEntropyLoss()
        loss_1_1_1 = loss_fct(logits, targets)
        loss_1_1_2 = loss_fct(logits.T, targets)
        loss1_1 = (loss_1_1_1 + loss_1_1_2) / 2

        # diff = text_feat - m_text_feat
        # loss1_2 = torch.sum(torch.sum(torch.pow(diff, 2), dim=1)/diff.size(1)) / diff.size(0)
        loss_fct = MSELoss()
        loss1_2 = loss_fct(text_feat, m_text_feat) * 5

        # loss1 = loss1_2 + loss1_1 * self.clipi2t - loss1_2 * self.clipi2t
        loss1 = (loss1_1 + loss1_2) / 2

        prompt_ids = self.mlp2(m_text_feat).view(m_text_feat.size(0), self.llm_size)

        prompt_ids = prompt_ids + self.linear(m_text_feat).view(m_text_feat.size(0), self.llm_size)

        frozon_embed = self.llm.get_input_embeddings()

        inputs_embeds = frozon_embed(llm_text_input)

        extra_embed = self.new_embedding.repeat(prompt_ids.size(0), 1)

        # 在预训练时，我们的<img>标识一定位于句首，所以我们直接在句首进行添加即可
        inputs_embeds[:, 0, :] += (prompt_ids + extra_embed)

        attention_mask = llm_attention_mask
        loss2 = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)[0]

        return loss1, loss2, loss1_1, loss1_2


class single_mapping_model_text_stage1(nn.Module):
    def __init__(self, llm_model, clip_model, clip_size=512, llm_size=1024):
        super(single_mapping_model_text_stage1, self).__init__()

        self.textLinear = nn.Linear(clip_size, llm_size)
        # self.textLinear = ImageMLP(clip_size, llm_size, 1, 0.5)

        self.new_embedding = nn.Parameter(torch.randn(1, llm_size))

        self.llm_size = llm_size
        self.llm = llm_model
        self.clip = clip_model

    def forward(self, clip_text_input, llm_text_input, llm_attention_mask, labels):
        text_feat = self.clip.get_text_features(clip_text_input)

        prompt_ids = self.textLinear(text_feat).view(text_feat.size(0), self.llm_size)

        frozon_embed = self.llm.get_input_embeddings()

        inputs_embeds = frozon_embed(llm_text_input)

        extra_embed = self.new_embedding.repeat(prompt_ids.size(0), 1)

        # 在预训练时，我们的<img>标识一定位于句首，所以我们直接在句首进行添加即可
        inputs_embeds[:, 0, :] += (prompt_ids + extra_embed)

        attention_mask = llm_attention_mask
        loss = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)[0]

        return loss


class single_mapping_model_image_stage1(nn.Module):
    def __init__(self, llm_model, clip_size=512, llm_size=1024):
        super(single_mapping_model_image_stage1, self).__init__()

        # self.imgLinear = nn.Linear(clip_size, llm_size)
        self.imageLinear = nn.Linear(clip_size, llm_size)
        self.new_embedding = nn.Parameter(torch.randn(1, llm_size))

        self.llm_size = llm_size
        self.llm = llm_model

    def forward(self, clip_img_input, llm_text_input, llm_attention_mask, labels):
        prompt_ids = self.imageLinear(clip_img_input).view(clip_img_input.size(0), self.llm_size)

        frozon_embed = self.llm.get_input_embeddings()

        inputs_embeds = frozon_embed(llm_text_input)

        extra_embed = self.new_embedding.repeat(prompt_ids.size(0), 1)

        # 在预训练时，我们的<img>标识一定位于句首，所以我们直接在句首进行添加即可
        inputs_embeds[:, 0, :] += (prompt_ids + extra_embed)

        attention_mask = llm_attention_mask
        loss = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)[0]

        return loss


class single_mapping_model_text_stage2(nn.Module):
    def __init__(self, llm_model, clip_model, clip_size=512, llm_size=1024, stage=2):
        super(single_mapping_model_text_stage2, self).__init__()

        self.textLinear = nn.Linear(clip_size, llm_size)
        self.imageLinear = nn.Linear(clip_size, clip_size)

        self.new_embedding = nn.Parameter(torch.randn(1, llm_size))

        self.stage = stage
        if stage == 3:
            self.extraLinear = nn.Linear(clip_size, llm_size)

        self.loss_fct = MSELoss()
        self.llm_size = llm_size
        self.llm = llm_model
        self.clip = clip_model

    def forward(self, clip_text_input, clip_img_input, llm_text_input, llm_attention_mask, labels):
        bsz = llm_text_input.size(0)
        frozon_embed = self.llm.get_input_embeddings()
        inputs_embeds = frozon_embed(llm_text_input)

        text_feat = self.clip.get_text_features(clip_text_input).float()

        img_token_embed = self.new_embedding.repeat(bsz, 1)

        img_feat = clip_img_input.float()
        transformed_text_feat = self.imageLinear(img_feat)

        prompt_ids = self.textLinear(transformed_text_feat).view(bsz, self.llm_size)

        dis_loss = self.loss_fct(transformed_text_feat, text_feat)

        inputs_embeds[:, 0, :] += (prompt_ids + img_token_embed)

        if self.stage == 3:
            inputs_embeds[:, 0, :] += self.extraLinear(img_feat)

        lm_loss = self.llm(inputs_embeds=inputs_embeds, attention_mask=llm_attention_mask, labels=labels)[0]

        return dis_loss, lm_loss

    def generate(self, clip_img_input, llm_text_input, llm_attention_mask):
        bsz = llm_text_input.size(0)

        # 查找<img> token所在的位置
        img_pos = llm_text_input == 50265
        img_pos = img_pos.nonzero()
        indices = [x[1].item() for x in img_pos]

        frozon_embed = self.llm.get_input_embeddings()
        inputs_embeds = frozon_embed(llm_text_input)

        img_token_embed = self.new_embedding.repeat(bsz, 1)

        img_feat = clip_img_input.float()
        transformed_text_feat = self.imageLinear(img_feat)

        prompt_ids = self.textLinear(transformed_text_feat).view(bsz, self.llm_size)

        for indice in indices:
            inputs_embeds[:, indice, :] += (prompt_ids + img_token_embed)

            if self.stage == 3:
                inputs_embeds[:, indice, :] += self.extraLinear(img_feat)

        output = self.llm.generate(inputs_embeds=inputs_embeds)

        return output


class promot_model_instruction(nn.Module):
    def __init__(self, llm_model, clip_model, clip_size=512, llm_size=1024, prompt_len=5):
        super(promot_model_instruction, self).__init__()

        self.imgdLinear = nn.Linear(clip_size, llm_size)
        self.imgLinear = nn.Linear(clip_size, llm_size)
        self.promptMlp = ImageMLP(clip_size, clip_size, prompt_len, 0.5)

        self.prompt_len = prompt_len


        self.embedding_img = nn.Parameter(torch.randn(1, llm_size))

        self.embedding_imgd = nn.Parameter(torch.randn(1, llm_size))
        self.llm2clip = nn.Linear(llm_size, clip_size)

        self.loss_fct = MSELoss()
        self.llm_size = llm_size
        self.clip_size = clip_size
        self.llm = llm_model
        self.clip = clip_model

    def forward(self, clip_text_input, clip_attention_mask, clip_img_input, llm_text_input, llm_attention_mask,
                pre_llm_text_input, pre_llm_attention_mask, labels, img_token_id, imgd_token_id):
        bsz = llm_text_input.size(0)
        frozon_embed = self.llm.get_input_embeddings()

        # 先找<img>的位置
        img_token_pos = llm_text_input == img_token_id
        img_token_pos = img_token_pos.nonzero().cpu().numpy().tolist()

        # 再找<img-d>的位置
        imgd_token_pos = llm_text_input == imgd_token_id
        imgd_token_pos = imgd_token_pos.nonzero().cpu().numpy().tolist()

        # --------------------模型的上半部分----------------------
        clip_img_feat = clip_img_input
        # 需要加到<img>上
        img_prompt = self.imgLinear(clip_img_feat)

        # --------------------模型交互--------------------
        pre_inputs_embeds = frozon_embed(pre_llm_text_input)

        for pos in img_token_pos:
            pre_inputs_embeds[pos[0], pos[1], :] += img_prompt[pos[0], :] + self.embedding_img[0]

        for pos in imgd_token_pos:
            pre_inputs_embeds[pos[0], pos[1], :] += self.embedding_imgd[0]

        last_hidden_state = \
            self.llm(inputs_embeds=pre_inputs_embeds, attention_mask=pre_llm_attention_mask, return_dict=True)[2]

        pre_query = []
        for pos in imgd_token_pos:
            pre_query.append(last_hidden_state[pos[0], pos[1], :])
        pre_query = torch.stack(pre_query).to(img_prompt.device)

        pre_query = self.llm2clip(pre_query)

        # --------------------模型的下半部分-----------------------
        clip_text_soft_prompt = self.promptMlp(clip_img_feat).view(bsz, self.prompt_len, -1)

        clip_frozon_embed = self.clip.text_model.embeddings

        clip_base_embeds = clip_frozon_embed(clip_text_input)
        clip_hidden_states = torch.concat(
            [clip_base_embeds[:, 0, :].unsqueeze(dim=1), clip_text_soft_prompt, pre_query.unsqueeze(dim=1),
             clip_base_embeds[:, self.prompt_len + 2, :].unsqueeze(dim=1)], dim=1)

        clip_pooled_output = self.clip.text_model(input_ids=clip_text_input, attention_mask=clip_attention_mask,
                                                  hidden_states=clip_hidden_states)[1]
        clip_text_feature = self.clip.text_projection(clip_pooled_output)
        # 需要加到<img-d>上
        imgd_prompt = self.imgdLinear(clip_text_feature)

        # --------------------构造模型最终embedding输入--------------------
        inputs_embeds = frozon_embed(llm_text_input)

        for pos in img_token_pos:
            inputs_embeds[pos[0], pos[1], :] += img_prompt[pos[0], :] + self.embedding_img[0]

        for pos in imgd_token_pos:
            inputs_embeds[pos[0], pos[1], :] += imgd_prompt[pos[0], :] + self.embedding_imgd[0]

        lm_loss = self.llm(inputs_embeds=inputs_embeds, attention_mask=llm_attention_mask, labels=labels)[0]
        return lm_loss

    def generate(self, clip_text_input, clip_attention_mask, clip_img_input, llm_text_input, llm_attention_mask,
                 img_token_id, task):
        bsz = llm_text_input.size(0)

        img_pos = llm_text_input == img_token_id
        img_pos = img_pos.nonzero()
        indices = [x[1].item() for x in img_pos]

        frozon_embed = self.llm.get_input_embeddings()
        inputs_embeds = frozon_embed(llm_text_input)

        img_token_embed = self.new_embedding.repeat(bsz, 1)

        # soft prompt part....
        clip_img_feat = clip_img_input.float()

        # 考虑多张图片的情况
        # 不过其实我们生成时默认batch_size = 1
        # 2 * 5 * 768
        clip_text_soft_prompt = self.promptMlp(clip_img_feat).view(-1, self.prompt_len, self.clip_size)

        clip_frozon_embed = self.clip.text_model.embeddings

        # 1 * seq_len * 768
        clip_base_embeds = clip_frozon_embed(clip_text_input)

        clip_text_input = clip_text_input.repeat(clip_text_soft_prompt.size(0), 1)
        clip_base_embeds = clip_base_embeds.repeat(clip_text_soft_prompt.size(0), 1, 1)
        clip_attention_mask = clip_attention_mask.repeat(clip_text_soft_prompt.size(0), 1)

        clip_hidden_states = torch.concat([clip_base_embeds[:, 0, :].unsqueeze(dim=1), clip_text_soft_prompt,
                                           clip_base_embeds[:, self.prompt_len + 1:, :]], dim=1)

        clip_pooled_output = self.clip.text_model(input_ids=clip_text_input, attention_mask=clip_attention_mask,
                                                  hidden_states=clip_hidden_states)[1]
        clip_text_feature = self.clip.text_projection(clip_pooled_output)

        text_prompt = self.textLinear(clip_text_feature)
        image_prompt = self.imageLinear(clip_img_feat).squeeze(dim=0)
        if self.isfusion:
            prompt = self.fusion(torch.concat([image_prompt, text_prompt], dim=-1))
        else:
            prompt = image_prompt + text_prompt

        for idx, indice in enumerate(indices):
            inputs_embeds[:, indice, :] += (prompt[idx, :] + img_token_embed)

        if task == "vqa":
            output = self.llm.generate(inputs_embeds=inputs_embeds, num_beams=5, length_penalty=-1)
        else:
            output = self.llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=20)

        return output


class single_mapping_model_extra_pretrain_linear(nn.Module):
    def __init__(self, llm_model, clip_size=512, llm_size=1024):
        super(single_mapping_model_extra_pretrain_linear, self).__init__()
        # the linear layer in the feature alignment
        self.imageLinear = nn.Linear(clip_size, llm_size)
        # the embedding of <img>
        self.new_embedding = nn.Parameter(torch.randn(1, llm_size))

        self.llm_size = llm_size
        self.llm = llm_model

    def forward(self, clip_img_input, llm_text_input, llm_attention_mask, labels):
        img_feat = clip_img_input
        prompt_ids = self.imageLinear(img_feat)

        frozon_embed = self.llm.get_input_embeddings()
        inputs_embeds = frozon_embed(llm_text_input)
        extra_embed = self.new_embedding.repeat(prompt_ids.size(0), 1)

        inputs_embeds[:, 0, :] += (prompt_ids + extra_embed)

        attention_mask = llm_attention_mask
        loss = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)[0]

        return loss


class promot_model_instruction_remake(nn.Module):
    def __init__(self, llm_model, clip_model, clip_size=512, llm_size=1024, prompt_len=5, query_len=1):
        super(promot_model_instruction_remake, self).__init__()


        # the embdding of <img> and <img-d>
        self.embedding_img = nn.Parameter(torch.randn(1, llm_size))
        self.embedding_imgd = nn.Parameter(torch.randn(1, llm_size))

        # feature alignment
        self.imgLinear = nn.Linear(clip_size, llm_size)
        # length
        self.prompt_len = prompt_len
        self.query_len = query_len

        # the visual information decomposition.
        self.promptMlp = nn.Linear(clip_size, clip_size * prompt_len)
        # request acquire
        self.llm2clip = nn.Linear(llm_size, clip_size)
        # the linear transmitting layer
        self.imgdLinear = nn.Linear(clip_size, llm_size * query_len)\

        self.dropout = nn.Dropout(0.5)
        self.loss_fct = MSELoss()
        self.llm_size = llm_size
        self.clip_size = clip_size
        self.llm = llm_model

        # the frozen textual encoder of clip is used to perform request-based visual information interaction
        self.clip = clip_model

    def forward(self, clip_text_input, clip_attention_mask, clip_img_input, llm_text_input, llm_attention_mask,
                pre_llm_text_input, pre_llm_attention_mask, labels, img_token_id, imgd_token_id):
        bsz = llm_text_input.size(0)
        frozon_embed = self.llm.get_input_embeddings()

        # --------------------During training, the image features are pre-processed----------------------
        clip_img_feat = clip_img_input
        img_prompt = self.imgLinear(clip_img_feat)

        # --------------------first input to LLMs--------------------
        pre_inputs_embeds = frozon_embed(pre_llm_text_input)

        pre_inputs_embeds[pre_llm_text_input == img_token_id] += img_prompt + self.embedding_img.repeat(bsz, 1)

        pre_inputs_embeds[pre_llm_text_input == imgd_token_id] += self.embedding_imgd.repeat(bsz, 1)

        last_hidden_state = \
            self.llm(inputs_embeds=pre_inputs_embeds, attention_mask=pre_llm_attention_mask, return_dict=True)[2]

        pre_query = last_hidden_state[pre_llm_text_input == imgd_token_id]

        pre_query = self.llm2clip(pre_query)

        # --------------------perform visual information interaction----------------------
        clip_text_soft_prompt = self.promptMlp(clip_img_feat).view(bsz, self.prompt_len, -1)

        clip_text_soft_prompt = self.dropout(clip_text_soft_prompt)

        clip_frozon_embed = self.clip.text_model.embeddings

        clip_base_embeds = clip_frozon_embed(clip_text_input)
        clip_hidden_states = torch.concat(
            [clip_base_embeds[:, 0, :].unsqueeze(dim=1), clip_text_soft_prompt, pre_query.unsqueeze(dim=1),
             clip_base_embeds[:, self.prompt_len + 2, :].unsqueeze(dim=1)], dim=1)

        clip_pooled_output = self.clip.text_model(input_ids=clip_text_input, attention_mask=clip_attention_mask,
                                                  hidden_states=clip_hidden_states)[1]
        clip_text_feature = self.clip.text_projection(clip_pooled_output)

        imgd_prompt = self.imgdLinear(clip_text_feature).view(-1, self.llm_size)

        # --------------------Second input to LLMs--------------------
        inputs_embeds = frozon_embed(llm_text_input)

        inputs_embeds[llm_text_input == img_token_id] += img_prompt + self.embedding_img.repeat(bsz, 1)

        inputs_embeds[llm_text_input == imgd_token_id] += imgd_prompt + self.embedding_imgd.repeat(bsz * self.query_len,
                                                                                                   1)

        lm_loss = self.llm(inputs_embeds=inputs_embeds, attention_mask=llm_attention_mask, labels=labels)[0]
        return lm_loss

    def generate(self, clip_text_input, clip_attention_mask, clip_img_input, llm_text_input, llm_attention_mask,
                 pre_llm_text_input, pre_llm_attention_mask, img_token_id, imgd_token_id, task):
        bsz = llm_text_input.size(0)
        frozon_embed = self.llm.get_input_embeddings()


        clip_img_feat = clip_img_input.to(self.imgLinear.weight.dtype)
        img_prompt = self.imgLinear(clip_img_feat)

        # --------------------the first input to LLMs--------------------
        pre_inputs_embeds = frozon_embed(pre_llm_text_input)

        pre_inputs_embeds[pre_llm_text_input == img_token_id] += img_prompt + self.embedding_img.repeat(bsz, 1)

        pre_inputs_embeds[pre_llm_text_input == imgd_token_id] += self.embedding_imgd.repeat(bsz, 1)

        last_hidden_state = \
            self.llm(inputs_embeds=pre_inputs_embeds, attention_mask=pre_llm_attention_mask, return_dict=True)[2]

        pre_query = last_hidden_state[pre_llm_text_input == imgd_token_id]
        pre_query = pre_query[0].unsqueeze(dim=0)

        pre_query = self.llm2clip(pre_query)

        # --------------------Visual Information Interaction-----------------------
        clip_text_soft_prompt = self.promptMlp(clip_img_feat).view(bsz, self.prompt_len, -1)

        clip_frozon_embed = self.clip.text_model.embeddings

        clip_base_embeds = clip_frozon_embed(clip_text_input)
        clip_hidden_states = torch.concat(
            [clip_base_embeds[:, 0, :].unsqueeze(dim=1), clip_text_soft_prompt, pre_query.unsqueeze(dim=1),
             clip_base_embeds[:, self.prompt_len + 2, :].unsqueeze(dim=1)], dim=1)

        clip_pooled_output = self.clip.text_model(input_ids=clip_text_input, attention_mask=clip_attention_mask,
                                                  hidden_states=clip_hidden_states)[1]
        clip_text_feature = self.clip.text_projection(clip_pooled_output)
        # add to the <img-d>
        imgd_prompt = self.imgdLinear(clip_text_feature).view(-1, self.llm_size)

        # --------------------the final embedding fed to LLMs--------------------
        inputs_embeds = frozon_embed(llm_text_input)

        inputs_embeds[llm_text_input == img_token_id] += img_prompt + self.embedding_img.repeat(bsz, 1)

        inputs_embeds[llm_text_input == imgd_token_id] += imgd_prompt + self.embedding_imgd.repeat(bsz * self.query_len, 1)

        # if task == "vqa":
        if "vqa" in task:
            output = self.llm.generate(inputs_embeds=inputs_embeds, num_beams=5, length_penalty=-1, max_new_tokens=50)
        # elif "detail-c" in task:
        #     output = self.llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=300, num_beams=1, do_sample=True,
        #                                temperature=0.2)
        elif "detail-c" in task or "semart" in task or "coco" in task:
            output = self.llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=250, num_beams=5, do_sample=True)
            # output = self.llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=200, do_sample=True, top_p=0.92,
            #                            top_k=0)
        elif "detail-q" in task:
            output = self.llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=80)
        elif "test" in task:
            # output = self.llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=200, num_beams=5, do_sample=True)
            output = self.llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=300, num_beams=1, do_sample=True,
                                            temperature=0.2)
        else:
            output = self.llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=20)

        return output

    def generate_with_image(self, clip_text_input, clip_attention_mask, image, llm_text_input,
                            pre_llm_text_input, pre_llm_attention_mask, img_token_id, imgd_token_id, task):

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
        pre_query = pre_query.reshape(bsz, self.query_len, pre_query.size(-1))
        pre_query = pre_query[:, 0, :]

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

        inputs_embeds[llm_text_input == imgd_token_id] += imgd_prompt + self.embedding_imgd.repeat(bsz * self.query_len, 1)

        if "qa" in task:
            output = self.llm.generate(inputs_embeds=inputs_embeds, num_beams=5, max_new_tokens=100)
        else:
            #output = self.llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=300, do_sample=True, top_p=0.92)
            output = self.llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=300, do_sample=True, num_beams=4)

        return output

    def generate_few_shot(self, clip_text_input, clip_attention_mask, clip_img_input, llm_text_input,
                          llm_attention_mask, pre_llm_text_input, pre_llm_attention_mask, img_token_id, imgd_token_id, task):

        bsz = llm_text_input.size(0) * 3

        frozon_embed = self.llm.get_input_embeddings()


        clip_img_feat = clip_img_input

        img_prompt = self.imgLinear(clip_img_feat)


        pre_inputs_embeds = frozon_embed(pre_llm_text_input)

        pre_inputs_embeds[pre_llm_text_input == img_token_id] += img_prompt + self.embedding_img.repeat(bsz, 1)

        pre_inputs_embeds[pre_llm_text_input == imgd_token_id] += self.embedding_imgd.repeat(bsz, 1)

        last_hidden_state = \
            self.llm(inputs_embeds=pre_inputs_embeds, attention_mask=pre_llm_attention_mask, return_dict=True)[2]

        pre_query = last_hidden_state[pre_llm_text_input == imgd_token_id]

        pre_query = self.llm2clip(pre_query)


        clip_text_soft_prompt = self.promptMlp(clip_img_feat).view(bsz, self.prompt_len, -1)

        clip_frozon_embed = self.clip.text_model.embeddings
        clip_text_input = clip_text_input.repeat(3, 1)
        clip_attention_mask = clip_attention_mask.repeat(3, 1)
        clip_base_embeds = clip_frozon_embed(clip_text_input)
        clip_hidden_states = torch.concat(
            [clip_base_embeds[:, 0, :].unsqueeze(dim=1), clip_text_soft_prompt, pre_query.unsqueeze(dim=1),
             clip_base_embeds[:, self.prompt_len + 2, :].unsqueeze(dim=1)], dim=1)

        clip_pooled_output = self.clip.text_model(input_ids=clip_text_input, attention_mask=clip_attention_mask,
                                                  hidden_states=clip_hidden_states)[1]
        clip_text_feature = self.clip.text_projection(clip_pooled_output)

        imgd_prompt = self.imgdLinear(clip_text_feature)


        inputs_embeds = frozon_embed(llm_text_input)

        inputs_embeds[llm_text_input == img_token_id] += img_prompt + self.embedding_img.repeat(bsz, 1)

        inputs_embeds[llm_text_input == imgd_token_id] += imgd_prompt + self.embedding_imgd.repeat(bsz, 1)

        if "vqa" in task:
            output = self.llm.generate(inputs_embeds=inputs_embeds, num_beams=5, length_penalty=-1)
        else:
            output = self.llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=20)

        return output

    def dialog(self, clip_text_input, clip_attention_mask, clip_img_input, llm_text_inputs,
               img_token_id, imgd_token_id, answer_prompt_ids, answer_ori_prompt_ids):

        llm_ori_text_input = llm_text_inputs[0].unsqueeze(dim=0)
        for i in range(len(llm_text_inputs)):
            # llm_text_inputs =
            if i > 0:
                llm_ori_text_input = torch.cat([llm_ori_text_input, llm_text_inputs[i].unsqueeze(dim=0)], dim=1)

            llm_text_input = torch.cat([llm_ori_text_input, answer_prompt_ids], dim=1)
            llm_ori_text_input = torch.cat([llm_ori_text_input, answer_prompt_ids], dim=1)
            pre_llm_text_input = llm_text_input

            bsz = llm_text_input.size(0)
            frozon_embed = self.llm.get_input_embeddings()


            clip_img_feat = clip_img_input

            img_prompt = self.imgLinear(clip_img_feat)

            # --------------------模型交互--------------------
            pre_inputs_embeds = frozon_embed(pre_llm_text_input)

            pre_inputs_embeds[pre_llm_text_input == img_token_id] += img_prompt + self.embedding_img.repeat(bsz, 1)

            pre_inputs_embeds[pre_llm_text_input == imgd_token_id] += self.embedding_imgd.repeat(bsz, 1)

            last_hidden_state = self.llm(inputs_embeds=pre_inputs_embeds, return_dict=True)[2]

            pre_query = last_hidden_state[pre_llm_text_input == imgd_token_id]

            if pre_query.size(0) > 1:
                pre_query = pre_query[-1].unsqueeze(dim=0)

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

            imgd_prompt = self.imgdLinear(clip_text_feature)


            inputs_embeds = frozon_embed(llm_text_input)

            inputs_embeds[llm_text_input == img_token_id] += img_prompt + self.embedding_img.repeat(bsz, 1)

            inputs_embeds[llm_text_input == imgd_token_id] += imgd_prompt + self.embedding_imgd.repeat(bsz, 1)

            output = self.llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=50, num_beams=5, do_sample=True,
                                       length_penalty=-0.5)

            llm_ori_text_input = torch.cat([llm_ori_text_input, output], dim=1)

        return llm_ori_text_input



