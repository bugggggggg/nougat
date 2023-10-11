



if [ ! -e model_small ]; then
    mkdir model_small
fi


wget -P model_small https://github.com/facebookresearch/nougat/releases/download/0.1.0-small/config.json
wget -P model_small https://github.com/facebookresearch/nougat/releases/download/0.1.0-small/pytorch_model.bin
wget -P model_small https://github.com/facebookresearch/nougat/releases/download/0.1.0-small/special_tokens_map.json
wget -P model_small https://github.com/facebookresearch/nougat/releases/download/0.1.0-small/tokenizer.json
wget -P model_small https://github.com/facebookresearch/nougat/releases/download/0.1.0-small/tokenizer_config.json
