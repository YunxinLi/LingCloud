# Runing Demo

1. unzip the local_transformer.zip to the current path.

2. app_demo_new.py support the most powerful LMEye-IPN-FlanT5-xl. It is trained with our collected [various multimodal instruction data](https://huggingface.co/datasets/YunxinLi/Multimodal_Insturction_Data_V2). Compared to previous LMEye-IPN versions, we adopt Q-former from BLIP-2 as the first stage and the tuned parameters are presented in the huggingface repository: [YunxinLi/LMEye_IPN_FlanT5-XL](https://huggingface.co/YunxinLi/LMEye_IPN_FlanT5-XL).

3. ```python app_demo_new.py```

## Downstream Evaluation.

Flick-30k Images with long description are constructed by GPT-3.5-turbo. 


![](https://github.com/YunxinLi/LingCloud/blob/main/LMEye/example_images/merge_from_ofoct.png)
