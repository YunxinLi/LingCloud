import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import gradio as gr
import torch
import argparse
from local_transformers.transformers_new.models.llama import LlamaConfig, LlamaForCausalLM
from local_transformers.transformers_new.models.llama.tokenization_llama import LlamaTokenizer
from local_transformers.transformers_new import DataCollatorForLanguageModeling, PreTrainedTokenizer, AutoTokenizer, \
    BloomConfig, BloomForCausalLM
from local_transformers.transformers_new.models.clip import CLIPModel, CLIPTokenizer, CLIPImageProcessor
from modeling.modeling_LMEye import LMEye_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LMEye Demo")
    parser.add_argument("--llm_model", default="bloomz-7b1") # ["llama-7b", "bloomz-7b1", "llama-13b"]
    parser.add_argument("--model_path", default="./output/bloomz-instruction/checkpoint_with_epoch4.pth", type=str)
    args = parser.parse_args()

    title = "Welcome to LMEye" + " ("+ args.llm_model +")"
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

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    print(device)
    print("Loading model...")

    if args.llm_model == "llama-7b":
        llama_path = "decapoda-research/llama-7b-hf/"
        llm_tokenizer = LlamaTokenizer.from_pretrained(llama_path, add_bos_token=False, add_eos_token=True)
        llm_config = LlamaConfig.from_pretrained(llama_path)
        llm_model = LlamaForCausalLM.from_pretrained(llama_path, config=llm_config)
        llm_tokenizer.pad_token_id = 0
    elif args.llm_model == "llama-13b":
        llama_path = "decapoda-research/llama-13b-hf/"
        llm_tokenizer = LlamaTokenizer.from_pretrained(llama_path, add_bos_token=False, add_eos_token=True)
        llm_config = LlamaConfig.from_pretrained(llama_path )
        llm_model = LlamaForCausalLM.from_pretrained(llama_path, config=llm_config)
        llm_tokenizer.pad_token_id = 0
    else:
        bloomz_path = "bigscience/bloomz-7b1"
        llm_tokenizer = AutoTokenizer.from_pretrained(bloomz_path, add_bos_token=False, add_eos_token=False)
        llm_config = BloomConfig.from_pretrained(bloomz_path)
        llm_model = BloomForCausalLM.from_pretrained(bloomz_path, config=llm_config)

    llm_tokenizer.add_tokens("<img>")
    llm_tokenizer.add_tokens("<img-d>")
    img_token_id = llm_tokenizer.vocab_size
    imgd_token_id = img_token_id + 1
    llm_model.resize_token_embeddings(len(llm_tokenizer))

    clip_path = "openai/clip-vit-large-patch14"

    clip_model = CLIPModel.from_pretrained(clip_path)
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_path)
    clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = LMEye_model(llm_model=llm_model, clip_model=clip_model, clip_size=768, llm_size=llm_config.hidden_size)
    # checkpoint of the input path.
    params = torch.load(args.model_path, map_location='cpu')['net']
    imgLinear_params = {}
    imgdLinear_params = {}
    promptMlp_params = {}
    llm2clip_params = {}

    for name, params in params.items():
        params = params.float()
        if "imgLinear" in name:
            imgLinear_params[name.split(".")[-1]] = params
        if "imgdLinear" in name:
            imgdLinear_params[name.split(".")[-1]] = params
        if "embedding_img" in name:
            model.embedding_img = torch.nn.Parameter(params)
        if "embedding_imgd" in name:
            model.embedding_imgd = torch.nn.Parameter(params)
        if "promptMlp" in name:
            promptMlp_params[name.split("p.")[-1]] = params
        if "llm2clip" in name:
            llm2clip_params[name.split(".")[-1]] = params
    model.imgLinear.load_state_dict(imgLinear_params)
    model.imgdLinear.load_state_dict(imgdLinear_params)
    model.promptMlp.load_state_dict(promptMlp_params)
    model.llm2clip.load_state_dict(llm2clip_params)

    model = model.half()
    model = model.to(device)

    print("Model loading success!")

    def predict(prompt_input, image_input, beam_size, temperature):
        prompt_input = "Image is <img>\n " + prompt_input + " <img-d> <img-d> <img-d> <img-d> <img-d>\nAnswer:"
        prompt_input = [prompt_input]
        inputs = llm_tokenizer(prompt_input, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].cuda(non_blocking=True)
        attention_mask = inputs["attention_mask"].cuda(non_blocking=True)

        text = ["! " * (5 + 1)] * input_ids.size(0)
        clip_text = clip_tokenizer(text, max_length=77, padding=True, return_tensors="pt", truncation=True)
        clip_text_ids = clip_text["input_ids"].cuda()
        image = image_input.convert('RGB')
        image = torch.from_numpy(clip_processor(image)['pixel_values'][0]).cuda().half()
        clip_attention_mask = clip_text["attention_mask"].cuda()
        generate_ids = model.generate(llm_text_input=input_ids,
                                     pre_llm_text_input=input_ids, pre_llm_attention_mask=attention_mask,
                                     image=image, clip_text_input=clip_text_ids,
                                     clip_attention_mask=clip_attention_mask, img_token_id=img_token_id,
                                     imgd_token_id=imgd_token_id, beam_size=beam_size, temperature=temperature)

        output = llm_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return output

    gr.Interface(
        fn=predict,
        inputs=[prompt_input, image_input, beam_size, temperature],
        outputs="text",
        title=title,
        description=description,
        allow_flagging="never",
        examples=[["Please give a detailed description of this image."]]
    ).launch(share=True)