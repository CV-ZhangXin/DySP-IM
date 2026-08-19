from transformers import AutoModel
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(
    "/home/shihuazhan/code/newWsi/histaug-conch",   # local path instead of hub ID
    trust_remote_code=True,
    local_files_only=True,  # uses local files only
).to(device)

num_patches = 1
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

print(augmented_embeddings[0])  # (num_patches, embedding_dim)
 