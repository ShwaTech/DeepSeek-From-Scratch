import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


chapter_1 = "Ch01_Attention_Mechanism"


list_of_files = [
    "requirements.txt",
    
    f"{chapter_1}/01_Tokenization.ipynb",
    f"{chapter_1}/02_Byte_Pair_Encoding.ipynb",
    f"{chapter_1}/03_Data_Loader.ipynb",
    f"{chapter_1}/04_Embeddings.ipynb",
    f"{chapter_1}/05_Self_Attention.ipynb",
    f"{chapter_1}/06_Causal_Attention.ipynb",
    f"{chapter_1}/07_Multihead_Attention.ipynb",
    
    
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
