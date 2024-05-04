import huggingface_hub # 予めpip install huggingface-hub

model_id = "intfloat/multilingual-e5-large" # 落とすモデル名
local_dir = "./hf_model" # 保存先フォルダ名
huggingface_hub.snapshot_download(model_id, local_dir=local_dir, local_dir_use_symlinks=False)
