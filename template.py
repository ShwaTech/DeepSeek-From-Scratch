import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


chapter_01 = "Chapter01"
chapter_02 = "Chapter02"
chapter_03 = "Chapter03"
chapter_04 = "Chapter04"
chapter_05 = "Chapter05"


list_of_files = [
    "requirements.txt",
    
    f"{chapter_01}/01_Tokenization.ipynb",
    f"{chapter_01}/02_Byte_Pair_Encoding.ipynb",
    f"{chapter_01}/03_Data_Loader.ipynb",
    f"{chapter_01}/04_Embeddings.ipynb",
    f"{chapter_01}/05_Self_Attention.ipynb",
    f"{chapter_01}/06_Causal_Attention.ipynb",
    f"{chapter_01}/07_Multi-Head_Attention(MHA).ipynb",
    f"{chapter_01}/08_Multi-Head_Attention_Visualized.ipynb",
    
    f"{chapter_02}/01_KV_Cache.ipynb",
    f"{chapter_02}/02_Multi-Query_Attention(MQA).ipynb",
    f"{chapter_02}/03_Grouped-Query_Attention(GQA).ipynb",

    f"{chapter_03}/01_Multi-Head_Latent_Attention(MLA).ipynb",
    f"{chapter_03}/02_MLA_with_Decoupled_RoPE(DSA).ipynb",
    f"{chapter_03}/Bonus/MHA_vs_MQA_vs_GQA_vs_MLA.ipynb",

    f"{chapter_04}/01_Mixture_of_Experts_(MoE)_from_Scratch.ipynb",
    f"{chapter_04}/02_DeepSeek_Mixture_of_Experts(DeepSeekMoE).ipynb",
    f"{chapter_04}/Bonus/DeepSeek_MoE_Comparison.ipynb",

    f"{chapter_05}/01_Multi_Token_Prediction_(MTP)_From_Scratch.ipynb",
    f"{chapter_05}/02_DeepSeek_Multi_Token_Prediction(DeepSeekMTP).ipynb",
    f"{chapter_05}/03_DeepSeek_Quantization.ipynb",
]



for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    
    else:
        logging.info(f"{filename} is already exists")
