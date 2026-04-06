HuggingFace Token
PI0.5 depends on google/paligemma-3b-pt-224 as a vision backbone. Google requires you to accept the model license before the weights can be downloaded.


step 0 uv sync

step 1 hf token to get access to gated model and optionally push checkpoints to hf
Navigate to the model page and accept the license.
Generate an access token at huggingface.co/settings/tokens

put in .env
HF_TOKEN=

step 2: wandb token to do experiment tracking
put in .env
WANDB_TOKEN=


ankith notes for you 
see inference time between models as well trained on same dataset


physical ai leader board 
https://phail.ai/