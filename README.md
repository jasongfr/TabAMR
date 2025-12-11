<img width="1176" height="650" alt="image" src="https://github.com/user-attachments/assets/5ba32936-5518-480b-9138-e5d993508aba" />
The folders "tabamr-zsee" and "roberta-large" contain large .bin files. Please be cautious when downloading them.

# Required Dependencies
Python 3.9  

transformers==4.30.2  

torch==1.13.1+cu117  

en-core-web-sm==3.7.1  

tqdm  

spicy  

pydantic  

six

# Model Inference
To run model inference, execute:
```bash
bash ./scripts/infer_ace.sh
```
The model inference results are saved in the Infer/zsee folder.
