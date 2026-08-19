---
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
language:
- en
---

## Model Summary

**HistAug** is a lightweight transformer-based generator for **controllable latent-space augmentations** in the feature space of the [CONCH foundation model](https://www.nature.com/articles/s41591-024-02856-4). Instead of applying costly image-space augmentations on millions of WSI patches, HistAug operates **directly on patch embeddings** extracted from a given foundation model(here CONCH). By conditioning on explicit transformation parameters (e.g., hue shift, erosion, HED color transform), HistAug generates realistic augmented embeddings while preserving semantic content. In practice, the CONCH variant of HistAug can reconstruct the corresponding ground-truth augmented embeddings with an average cosine similarity of **about 93%** at **10X, 20X, and 40X magnification**.

This enables training of Multiple Instance Learning (MIL) models with:  
- ⚡ **Fast augmentation** 
- 🧠 **Low memory usage** (up to 200k patches in parallel on a single V100 32GB GPU)  
- 🎛 **Controllable and WSI-consistent augmentations** (bag-wise or patch-wise)

Need HistAug for a different foundation model? Explore the full collection: [**HistAug models collection**](https://huggingface.co/collections/sofieneb/histaug-models-68a334437f71d35c7037a54e).


📄 **Paper**: [*Controllable Latent Space Augmentation for Digital Pathology* (Boutaj *et al.*, 2025)](https://arxiv.org/abs/2508.14588)




---

## Usage

You can load the model from the Hub with Hugging Face’s `transformers`:

```python
import torch
from transformers import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load HistAug (CONCH latent augmentation model)
model_id = "sofieneb/histaug-conch"
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)

# Example: patch embeddings from CONCH
num_patches = 50000
embedding_dim = 512
patch_embeddings = torch.randn((num_patches, embedding_dim), device=device)

# Sample augmentation parameters
# mode="wsi_wise" applies the same transformation across the whole slide
# mode="instance_wise" applies different transformations per patch
aug_params = model.sample_aug_params(
    batch_size=num_patches,
    device=patch_embeddings.device,
    mode="wsi_wise"
)

# Apply augmentation in latent space
augmented_embeddings = model(patch_embeddings, aug_params)

print(augmented_embeddings.shape)  # (num_patches, embedding_dim)
```

## Default Transform Configuration

The original transform configuration (shipped in the model config) is:

```json
{
  "transforms": {
    "parameters": {
      "brightness": [-0.5, 0.5],
      "contrast": [-0.5, 0.5],
      "crop": 0.75,
      "dilation": 0.75,
      "erosion": 0.75,
      "gamma": [-0.5, 0.5],
      "gaussian_blur": 0.75,
      "h_flip": 0.75,
      "hed": [-0.5, 0.5],
      "hue": [-0.5, 0.5],
      "rotation": 0.75,
      "saturation": [-0.5, 0.5],
      "v_flip": 0.75
    }
  }
}
```

* **Continuous transforms** (e.g., `brightness`, `hue`, `hed`, `gamma`, `saturation`) use an **interval** `[min, max]` from which parameters are sampled.
* **Discrete/binary transforms** (e.g., `h_flip`, `v_flip`, `dilation`, `erosion`, `rotation`, `gaussian_blur`, `crop`) use a **probability** (e.g., `0.75`) indicating how likely the transform is applied during sampling.

> You can access and modify this at runtime via:
>
> ```python
> print(model.histaug.transforms_parameters)
> ```

---

## Controlling Transformations

You can **inspect, modify, or delete** transformations at runtime via `model.histaug.transforms_parameters`.  
- To **remove** a transform, simply `pop` the key; during sampling it will appear with parameter **`0`** (effectively disabled).  
- You can also narrow a transform’s interval or change a transform’s probability, then re-sample to observe the effects.  
- Sampling mode: `mode="wsi_wise"` (same parameters for all patches) or `mode="instance_wise"` (per-patch parameters).

```python
## Controlling Transformations — pop vs. change params (continuous & discrete)

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
num_to_sample = 5

# start: sample once and inspect current config
sample_1 = model.sample_aug_params(batch_size=num_to_sample, device=device, mode="wsi_wise")
print("initial sample:\n", sample_1, "\n")

print("initial transforms_parameters:\n", model.histaug.transforms_parameters, "\n")

# pop examples
# pop a continuous transform: remove "hue" (interval transform)
model.histaug.transforms_parameters.pop("hue", None)

# pop a discrete transform: remove "rotation" (probability-based)
model.histaug.transforms_parameters.pop("rotation", None)

sample_2 = model.sample_aug_params(batch_size=num_to_sample, device=device, mode="wsi_wise")
print("after popping 'hue' (continuous) and 'rotation' (discrete):\n", sample_2, "\n")

# change param examples
# change a continuous transform interval: narrow 'brightness' from [-0.5, 0.5] to [-0.25, 0.25]
model.histaug.transforms_parameters["brightness"] = [-0.25, 0.25]

# change a discrete transform probability: lower 'h_flip' from 0.75 to 0.10
model.histaug.transforms_parameters["h_flip"] = 0.10

sample_3 = model.sample_aug_params(batch_size=num_to_sample, device=device, mode="wsi_wise")
print("after changing 'brightness' interval and 'h_flip' probability:\n", sample_3, "\n")

````
---

## During MIL

You can apply latent-space augmentation **during MIL training** with a probability (e.g., **60%**). We generally recommend applying augmentation with a non-trivial probability (e.g., 0.3–0.7) rather than always-on.

```python
import torch

# histaug: the loaded HistAug model (CONCH variant)
# mil_model: your MIL aggregator (e.g., ABMIL/CLAM/TransMIL head)
# criterion, optimizer, loader already defined

device = "cuda" if torch.cuda.is_available() else "cpu"
histaug = histaug.to(device).eval()   # histaug generator is frozen during MIL training
for p in histaug.parameters():
    p.requires_grad_(False)

def maybe_augment_bag(bag_features: torch.Tensor,
                      p_apply: float = 0.60,
                      mode: str = "wsi_wise") -> torch.Tensor:
    """
    bag_features: (num_patches, embed_dim) on device
    p_apply: probability to apply augmentation
    mode: "wsi_wise" (same params for all patches) or "instance_wise"
    """
    if torch.rand(()) >= p_apply:
        return bag_features
    with torch.no_grad():
        aug_params = histaug.sample_aug_params(
            batch_size=bag_features.size(0),
            device=bag_features.device,
            mode=mode  # "wsi_wise" or "instance_wise"
        )
        bag_features = histaug(bag_features, aug_params)
    return bag_features

# --- single-bag training example ---
for bag_features, label in loader:  # bag_features: (num_patches, embed_dim)
    bag_features = bag_features.to(device)

    # apply augmentation with 60% probability (WSI-wise by default)
    bag_features = maybe_augment_bag(bag_features, p_apply=0.60, mode="wsi_wise") # output : (num_patches, embed_dim)

    logits = mil_model(bag_features)              # forward through your MIL head
    loss = criterion(logits, label.to(device))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

```

---

## Offline usage (HPC clusters without internet)

If compute nodes don’t have internet, **always** run jobs with the offline flags to **prevent unnecessary network calls** and force local loads:

```bash
# On your compute job (no internet):
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Prepare the model **in advance** on a front-end/login node (with internet), then choose **either** approach below.

### Option — Warm the cache (simplest)

```bash
# On the front-end/login node (with internet):
python -c "from transformers import AutoModel; AutoModel.from_pretrained('sofieneb/histaug-conch', trust_remote_code=True)"
```

Then in your offline job/script:

```python
from transformers import AutoModel
model = AutoModel.from_pretrained(
    "sofieneb/histaug-conch",
    trust_remote_code=True,
    local_files_only=True,  # uses local cache only
)
```

### Option — Download to a local folder with `hf download`

```bash
# On the front-end/login node (with internet):
hf download sofieneb/histaug-conch --local-dir ./histaug-conch
```

Then in your offline job/script:

```python
from transformers import AutoModel
cross_transformer = AutoModel.from_pretrained(
    "./histaug-conch",   # local path instead of hub ID
    trust_remote_code=True,
    local_files_only=True,  # uses local files only
)
```

---
## Citation
If our work contributes to your research, or if you incorporate part of this code, please consider citing our paper:

```bibtex
@misc{boutaj2025controllablelatentspaceaugmentation,
      title={Controllable Latent Space Augmentation for Digital Pathology}, 
      author={Sofiène Boutaj and Marin Scalbert and Pierre Marza and Florent Couzinie-Devy and Maria Vakalopoulou and Stergios Christodoulidis},
      year={2025},
      eprint={2508.14588},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.14588}, 
}
```

