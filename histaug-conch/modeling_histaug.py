import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from .configuration_histaug import HistaugConfig
from .histaug_model import HistaugModel

class HistaugPretrainedModel(PreTrainedModel):
    config_class = HistaugConfig

    def __init__(self, config: HistaugConfig, *model_args, **model_kwargs):
        super().__init__(config)

        # instantiate your core model using values from the config
        self.histaug = HistaugModel(
            input_dim=config.input_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            use_transform_pos_embeddings=config.use_transform_pos_embeddings,
            positional_encoding_type=config.positional_encoding_type,
            final_activation=config.final_activation,
            embedding_type=config.embedding_type,
            chunk_size=config.chunk_size,
            transforms=config.transforms,
            **model_kwargs,
        )

        self.post_init()

        self.histaug.eval()
        for p in self.histaug.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, aug_params, **kwargs) -> torch.Tensor:
        """
        Forward pass through the histaug model.
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            aug_params: Augmentation parameters dict as expected by HistaugModel
        """
        return self.histaug(x, aug_params, **kwargs)

    def sample_aug_params(
        self,
        batch_size: int,
        device: torch.device = None,
        mode: str = "wsi_wise",
    ):
        """
        Proxy to HistaugModel.sample_aug_params
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.histaug.sample_aug_params(batch_size=batch_size, device=device, mode=mode)

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save the model and configuration to the directory.
        """
        super().save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ):
        """
        Load a model from a pretrained checkpoint.
        """
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
    
    def _init_weights(self, module):
        """
        Initialize the weights. This is required by HuggingFace's PreTrainedModel.
        You can customize this if you have specific initialization logic.
        For now, we leave it as a no-op since the model is loaded from a checkpoint.
        """
        pass