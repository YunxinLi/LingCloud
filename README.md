# LingCloud

The LingCloud project seeks to enhance the large language model's capability by incorporating human-like eyes. 
<p> 
  <a href="https://github.com/YunxinLi/LingCloud/"> <img src="https://img.shields.io/badge/LingCloud-LMEye-brightgreen" height="18px" alt="LingCloud">
  <a href="https://scholar.google.com/citations?user=U98QY0QAAAAJ&hl=en"><img src="https://img.shields.io/badge/scholar-4385FE.svg?&style=plastic&logo=google-scholar&logoColor=white" alt="Google Scholar" height="18px"> </a>
  <a href="https://twitter.com/LyxTg"> <img src="https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" height="18px" alt="Yunxin Li">
</p> 

I would like to express my sincere gratitude to all co-authors: my advisors, Prof. [Baotian Hu](http://faculty.hitsz.edu.cn/hubaotian), [Lin Ma](https://forestlinma.com/), and [Min Zhang](http://faculty.hitsz.edu.cn/MinZhang), and team members, Xinyu Chen and Wanqi Zhong, for their tremendous supports.

Currently, GPT-4 has achieved unparalleled proficiency in image comprehension. Given our limited computational resources and financial supports, we also need to develop a model that can perform various tasks akin to GPT-4. The aim of this project is to connect visual information to the large language model (brain), thus increasing its ability to comprehend the external world's infinite-granularity visual content. As a result, we present the first version of LingCloud, [LMEye(IPN)](https://arxiv.org/abs/2305.03701), which will be continuously improved to achieve the robust and efficient interaction between LLMs and the external world.

If you have any question, please feel free to contact me by e-mail: liyunxin987@163.com, Twitter: [@LyxTg](https://twitter.com/LyxTg), or submit your issue in the repository.

## :fire: News

[07.02] A new verision of LMEye-IPN-FlanT5-xl are released. Please follow [here](https://github.com/YunxinLi/LingCloud/tree/main/LMEye) to check it. 

[07.02] The online demo is closed for fully upgrading. We will continually provide the newest local demo with powerful LMEye variant.

[06.24] An online demo of LMEye(IPN-Bloomz-7b1): http://model.hitwds.cn:7080/. 

[06.12] We release more diverse and high-quality Multimodal Instruction-following Data (V2), termed LMEyeMID, Please see here https://huggingface.co/datasets/YunxinLi/Multimodal_Insturction_Data_V2.
    
[05.25] We provide a file to deploy a simple demo.

[05.22]  We release the codes of LMEye and the tuned [checkpoints](https://huggingface.co/YunxinLi/) of LLaMA-7b/13b and Bloomz-7b1.
    
[05.05] We present the paper [LMEye: An Interactive Perception Network for Large Language Models](https://arxiv.org/abs/2305.03701)
    
[05.04] We release the evaluation dataset(/data/multimodal_data_all_generation.txt) construted by GPT-3.5-turbo based on about 3.5k images from [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4). [Here](https://huggingface.co/datasets/YunxinLi/Multimodal_Instruction_data_v1/blob/main/image.zip), you can also download these images and put them into the path /data/image/.


## :rocket: Architecture

[Here](https://arxiv.org/pdf/2305.03701.pdf), you can see the detailed architecture and some experimental analyses of LingCloud 1.0, LMEye.

![](https://github.com/YunxinLi/LingCloud/blob/main/images/model.png)

## :sparkles: Presentation

You can deploy a simple LMEye demo using the following command:

 ```
 python /LMEye/app_demo.py --llm_model [model name] --model_path [model checkpoint]
 ```
 
![](https://github.com/YunxinLi/LingCloud/blob/main/images/demo.png)
    
    
[Here](https://arxiv.org/pdf/2305.03701.pdf), we present some cases in the experimental part and Appendix.

## :rocket: How to run
All codes are shown in the file directory LMEye.
    
### Environment
1. You can follow the basic conda environment file LMEye_environment.yml to install the environment. 

2. In /LMEye/, you should unzip the utils.zip and local transformers.zip to the current path /LMEye/.
    
### Train
1. If you want to train a similar model from scratch, you could use the run_llm_pretrain.py to perform the first-stage multimodal pretraining.

    Prepare the pretraining image-text pairs from the released corpus, and use the frozen CLIP-ViT-L/14 to extract the image feature.

    Download the checkpoints of corresponding LLMs and modify the path in run_llm_pretrain.py

2. The second-stage instruction-tuning: run_llm_instruction.py.

    [Here](https://huggingface.co/datasets/YunxinLi/Multimodal_Instruction_data_v1), You can download the first version of Multimodal Instruction data, about 80w samples.
    The Image source contains the COCO Caption, Flick30k, and the released multimodal instruction data from [LLaVA](https://github.com/haotian-liu/LLaVA).

### Test

We release the checkpoints of instruction version for LLaMA-7b/13b and Bloomz-7b1. You can download them from the repository in [Huggingface Hub](https://huggingface.co/YunxinLi).
    
The checkpoints could be put in the /LMEye/output/ or anywhere you like. You can starting the LMEye quickly by running the following command.

For Bloomz-7b1:

`python /LMEye/LLMEye_test.py --llm_model bloomz-7b1 --predict_model_dir: your path of the checkpoint`

For LLaMA-7b:

`python /LMEye/LLMEye_test.py --llm_model llama-7b --predict_model_dir: your path of the checkpoint`

For LLaMA-13b:

`python /LMEye/LLMEye_test.py --llm_model llama-13b --predict_model_dir: your path of the checkpoint`

More LMEye variants are coming soon!

## :rotating_light: Discussion

1. Finetune the LLMs with multimodal insturction data may decrease their performances on NLP. In this paper, we find that text instruction-following tuning LLMs have better generalization on performing multimodal interaction.
For future, could we jointly finetune LLMs with multimodal instruction data and text-only instruction-tuning data? How could we alleviate this bias?<br>
2. Hallucination. 
3. Text-only Insturction tuned LLMs perform better than pure LLMs for image understanding in downstream tasks.
4. Self-instructed multimodal instruction-following data is diverse. Yet the quality of data has a big room to improve. 
5. How to perform image-text semantic alignment under this paradigm.

## Acknowledge
Thanks everyone for your contributions.

If you're using LMEye in your research or applications, please cite our work.
```
@article{li2023lmeye,
    title={LMEye: An Interactive Perception Network for Large Language Models},
    author={Li, Yunxin and Hu, Baotian and Chen, Xinyu and Ma, Lin and Zhang, Min},
    journal={arXiv preprint arXiv:2305.03701},
    year={2023}
}
```
## License
This repository respects to Apache license 2.0.




