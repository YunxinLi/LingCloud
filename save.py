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
            torch.zeros((batch_size, imgq_token_number + imgd_token_number), device = device),
        ], dim = 1)

        tmp_pos_ids = torch.concat([
            torch.zeros((batch_size, all_image_query_length), device = device),
            input_ids
        ], dim = 1)

        if attention_mask is None:
            attention_mask = torch.ones_like(tmp_pos_ids)
        attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(device)], dim=1)

        if self.config.use_decoder_only_language_model:
            # step 5: first stage to get imgq embedding
            labels = labels.squeeze()
            labels = torch.concat([torch.full((batch_size, all_image_query_length), -100).to(device), labels], dim = 1)

            encoder_hidden_state = self.language_model.transformer(
                input_ids = tmp_pos_ids,
                inputs_embeds = inputs_embeds.transpose(0, 1).half()
            )[0].transpose(0, 1)

            pre_query = encoder_hidden_state[:, qformer_length + imgq_token_number].view(batch_size, -1)
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

            inputs_embeds[:, qformer_length + imgd_token_number: all_image_query_length] += imgd_prompt

            # step 7: second stage to get answer
            outputs = self.language_model(
                input_ids = tmp_pos_ids,
                inputs_embeds = inputs_embeds.transpose(0, 1).half(),
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