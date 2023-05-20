# LingCloud

The LingCloud project seeks to enhance the large language model's capability by incorporating human-like eyes. 

I would like to express my sincere gratitude to all co-authors: my advisors, Prof. [Baotian Hu](http://faculty.hitsz.edu.cn/hubaotian), [Lin Ma](https://forestlinma.com/), and [Min Zhang](http://faculty.hitsz.edu.cn/MinZhang), and team members, Xinyu Chen and Wanqi Zhong, for their tremendous supports.

Currently, GPT-4 has achieved unparalleled proficiency in image comprehension. Given our limited computational resources and financial supports, we also need to develop a model that can perform various tasks akin to GPT-4. The aim of this project is to connect visual information to the large language model (brain), thus increasing its ability to comprehend the external world's infinite-granularity visual content. As a result, we present the first version of LingCloud, [LMEye(IPN)](https://arxiv.org/abs/2305.03701), which will be continuously improved to achieve the robust and efficient interaction between LLMs and the external world.

If you have any question, please feel free to contact me by e-mail: liyunxin987@163.com, Twitter: [@LyxTg](https://twitter.com/LyxTg), or submit your issue in the repository.

## News
We first release the evaluation dataset(/data/multimodal_data_all_generation.txt) construted by GPT-3.5-turbo based on about 3.5k images from [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4).


## Architecture

[Here](https://arxiv.org/abs/2305.03701), you can see the detailed architecture and some experimental analyses of LingCloud 1.0, LMEye.

![](https://github.com/YunxinLi/LingCloud/blob/main/images/model.png)

## Presentation

Demo will come soon.

[Here](https://arxiv.org/abs/2305.03701), we present some cases in the experimental part and Appendix.

## How to run

1. The codes and more powerful LMEye will come soon.

2. First version checkpoint: LMEye([OPT-IML-1.3B](https://huggingface.co/facebook/opt-iml-1.3b)) and LMEye([Bloomz-7b1](https://huggingface.co/bigscience/bloomz)).

3. The second verision checkpoint: LMEye([BLIP-2](https://huggingface.co/docs/transformers/model_doc/blip-2)), LMEye(various [Vicuna](https://huggingface.co/lmsys/vicuna-13b-delta-v0) variants) and LMEye(various [LLaMA](https://huggingface.co/docs/transformers/main/model_doc/llama) variants).

4. Third version checkpoint: LMEye([MPT](https://huggingface.co/mosaicml/mpt-7b)).


## Discussion

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




