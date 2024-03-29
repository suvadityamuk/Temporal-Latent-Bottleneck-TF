config = {
    "mixed_precision": True,
    "dataset_name": "cifar10",
    "train_slice": 40_000,
    "batch_size": 1024,
    "buffer_size": 1024 * 2,
    "input_shape": [32, 32, 3],
    "image_size": 48,
    "num_classes": 10,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 100,
    "epochs": 1,
    "patch_size": 4,
    "embed_dim": 128,
    "chunk_size": 8,
    "r": 2,
    "num_layers": 6,
    "ffn_drop": 0.2,
    "attn_drop": 0.2,
    "num_heads": 1,
}
