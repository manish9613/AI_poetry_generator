import os
import json
import numpy as np
import requests
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from tqdm import tqdm

tf.disable_eager_execution()

def download_and_load_gpt2(model_size="1558M", models_dir="models"):
    """
    Downloads GPT-2 model files and loads settings and parameters.
    Returns: (settings, params)
    """
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size must be one of {allowed_sizes}")

    model_dir = os.path.join(models_dir, model_size)
    base_url = f"https://openaipublic.blob.core.windows.net/gpt-2/models/{model_size}"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    os.makedirs(model_dir, exist_ok=True)

    # Download files
    for fname in filenames:
        url = f"{base_url}/{fname}"
        local_path = os.path.join(model_dir, fname)
        if not os.path.exists(local_path):
            print(f"📥 Downloading: {fname}")
            download_file(url, local_path)
        else:
            print(f"✅ Already exists: {fname}")

    # Load hparams
    settings_path = os.path.join(model_dir, "hparams.json")
    with open(settings_path, "r") as f:
        settings = json.load(f)

    # Load parameters from TensorFlow checkpoint
    ckpt_path = tf.train.latest_checkpoint(model_dir)
    params = load_gpt2_params_from_ckpt(ckpt_path, settings)

    print("✅ Settings and parameters loaded.")
    return settings, params

def download_file(url, destination):
    try:
        with requests.get(url, stream=True, verify=False) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(destination, "wb") as f, tqdm(
                desc=os.path.basename(destination),
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")

def load_gpt2_params_from_ckpt(ckpt_path, settings):
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}
    for name, _ in tf.train.list_variables(ckpt_path):
        data = np.squeeze(tf.train.load_variable(ckpt_path, name))
        parts = name.split("/")[1:]  # Skip "model"
        target = params
        if parts[0].startswith("h"):
            layer = int(parts[0][1:])
            target = params["blocks"][layer]
            parts = parts[1:]
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = data
    return params
