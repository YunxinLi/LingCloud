# Runing Demo

1. unzip the local_transformer.zip to the current path.

2. app_demo_new.py support the most powerful LMEye-IPN-FlanT5-xl. It is trained with our collected [various multimodal instruction data](https://huggingface.co/datasets/YunxinLi/Multimodal_Insturction_Data_V2). Compared to previous LMEye-IPN versions, to gain more powerful visual perception, we adopt frozen Q-former from BLIP-2 as the first stage and the trained checkpoint are presented in the huggingface repository: [YunxinLi/LMEye_IPN_FlanT5-XL](https://huggingface.co/YunxinLi/LMEye_IPN_FlanT5-XL).

3. ```python app_demo_new.py```

## Evaluation.
We mainly evaluate our model and InstructBLIP to show the performance of Interactive Perception Network.

Flick-30k Images with long description are constructed by GPT-3.5-turbo. 


![](https://github.com/YunxinLi/LingCloud/blob/main/LMEye/example_images/merge_from_ofoct.png)


Evaluation Result on [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)

![](https://github.com/YunxinLi/LingCloud/blob/main/LMEye/example_images/evaluation_MME.png)
