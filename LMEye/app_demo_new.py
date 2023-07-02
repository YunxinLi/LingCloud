import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import gradio as gr
import torch
import argparse
from local_transformers.transformers_l.models.blip_2.modeling_blip_2 import Blip2InstructionQueryModel
from local_transformers.transformers_l import DataCollatorForLanguageModeling, PreTrainedTokenizer, Blip2Processor
from local_transformers.transformers_l.models.clip import CLIPModel, CLIPTokenizer, CLIPImageProcessor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--llm_model", default="blip2-flant5-xl")
    parser.add_argument("--predict_model_dir", default="./blip2-instruction-lmeye/checkpoint_with_epoch1.pth")
    args = parser.parse_args()

    title = "Welcome to LMEye"
    description = "<center><font size=5>Attaching Human-like Eyes to Large Language Models</font></center>"


    prompt_input = gr.Textbox(label="Prompt:", placeholder="Give your prompt here.", lines=2)
    image_input = gr.Image(type="pil")
    beam_size = gr.Slider(
        minimum=1,
        maximum=10,
        value=5,
        step=1,
        interactive=True,
        label="Beam Size",
    )

    temperature = gr.Slider(
        minimum=0.1,
        maximum=1,
        value=1,
        step=0.1,
        interactive=True,
        label="Temperature"
    )

    top_p = gr.Slider(
        minimum=0.1,
        maximum=1,
        value=1,
        step=0.1,
        interactive=True,
        label="Top P"
    )

    top_k = gr.Slider(
        minimum=1,
        maximum=5,
        value=3,
        step=1,
        interactive=True,
        label="Top K"
    )
    # device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # print(device)

    print("Loading model...")



    # loading clip-vit
    #clip_path = "local_transformers/clip-vit-large-patch14"
    clip_path = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_path)
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_path)

    # loading blip-2
    ckpt_dir = "Salesforce/blip2-flan-t5-xl"
    #ckpt_dir = "local_transformers/blip2-flan-t5-xl/"
    model = Blip2InstructionQueryModel.from_pretrained(ckpt_dir)
    processor = Blip2Processor.from_pretrained(ckpt_dir)
    model.load_clip(clip_model)

    params = torch.load(args.predict_model_dir, map_location='cpu')['net']

    params_dict = {}
    for name, param in model.named_parameters():
        if "language_model.encoder.embed_tokens.weight" in name:
            a = 1
        t = "module." + name
        if t in params:
            param = params[t]
        params_dict[name] = param
    params_dict["language_model.encoder.embed_tokens.weight"] = model.language_model.encoder.embed_tokens.weight
    params_dict["language_model.decoder.embed_tokens.weight"] = model.language_model.decoder.embed_tokens.weight
    params_dict["clip.text_model.embeddings.position_ids"] = model.clip.text_model.embeddings.position_ids
    params_dict["clip.vision_model.embeddings.position_ids"] = model.clip.vision_model.embeddings.position_ids
    model.load_state_dict(params_dict)

    model = model.bfloat16()
    model = model.cuda()

    model.eval()

    print("Model loading success!")

    def predict(prompt_input, image_input, beam_size, temperature, top_p, top_k):
        with torch.no_grad():
            prompt_input = "Human: <img>\n" + prompt_input + "\n <img-q> <img-d> <img-d> <img-d> <img-d> <img-d>\nAssistant:"
            prompt_input = [prompt_input]
            image = [image_input.convert('RGB')]

            inputs = processor(image, prompt_input, return_tensors="pt", max_length=400, padding=True, truncation=True)

            inputs["input_ids"] = inputs["input_ids"].cuda()
            inputs["pixel_values"] = inputs["pixel_values"].bfloat16().cuda()
            inputs["attention_mask"] = inputs["attention_mask"].cuda()
            inputs["imgd_token_id"] = 32100
            inputs["imgq_token_id"] = 32101
            text = ["! " * 37] * inputs["input_ids"].size(0)

            clip_text = clip_tokenizer(text, return_tensors="pt")
            clip_text_ids = clip_text["input_ids"].cuda()
            inputs["clip_text_input"] = clip_text_ids
            generate_ids = model.generate(**inputs, num_beams=beam_size, temperature=temperature, top_p=top_p, top_k=top_k,
                                          do_sample=True, max_new_tokens=400, no_repeat_ngram_size=5)

            output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
                0]
            return output

    gr.Interface(
        fn=predict,
        inputs=[prompt_input, image_input, beam_size, temperature, top_p, top_k],
        outputs="text",
        title=title,
        description=description,
        allow_flagging="never",
        examples=[["Describe the following image in detail."], ["Give an elaborate explanation of the image you see."],
                  ["Render a thorough depiction of this chart."], ["Narrate the contents of the image with precision."],
                  ["Illustrate the image through a descriptive explanation."], ["Introduce me this painting in detail."],
                  ["Provide an elaborate account of this painting."], ["Outline a detailed portrayal of this diagram."],
                  ["Provide an elaborate account of this chart."], ["Give detailed answer for this question. Question: "]]
    ).launch(share=True, server_port=8800)
