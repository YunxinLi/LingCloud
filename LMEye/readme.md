# Runing Demo

1. unzip the local_transformer.zip to the current path.

2. app_demo_new.py support the most powerful LMEye-IPN-FlanT5-xl. It is trained with our collected [various multimodal instruction data](https://huggingface.co/datasets/YunxinLi/Multimodal_Insturction_Data_V2). Compared to previous LMEye-IPN versions, to gain more powerful visual perception, we adopt frozen Q-former from BLIP-2 as the first stage and the trained checkpoint are presented in the huggingface repository: [YunxinLi/LMEye_IPN_FlanT5-XL](https://huggingface.co/YunxinLi/LMEye_IPN_FlanT5-XL).

3. ```python app_demo_new.py```

## Evaluation.
### Evaluation Result on [MMBench](https://opencompass.org.cn/leaderboard-multimodal)

![](https://github.com/YunxinLi/LingCloud/MMbench)



