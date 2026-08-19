from transformers import PretrainedConfig

class HistaugConfig(PretrainedConfig):
    model_type = "histaug"

    def __init__(
        self,
        input_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        use_transform_pos_embeddings: bool = True,
        positional_encoding_type: str = "learnable",
        final_activation: str = "Identity",
        embedding_type: str = "linear",
        chunk_size: int = 16,
        transforms: dict = None,
        **kwargs,
    ):
        # your model hyperparameters
        self.input_dim = input_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_transform_pos_embeddings = use_transform_pos_embeddings
        self.positional_encoding_type = positional_encoding_type
        self.final_activation = final_activation
        self.embedding_type = embedding_type
        self.chunk_size = chunk_size

        self.transforms = transforms or {"parameters": {}}

        super().__init__(**kwargs)