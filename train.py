# USAGE:
# python train.py


from configs.config_cifar import config
from tlb import CustomRecurrentCell, ModelTrainer, PatchEmbed

# PATCH
patch_layer = PatchEmbed(
    image_size=(config["image_size"], config["image_size"]),
    patch_size=(config["patch_size"], config["patch_size"]),
    embed_dim=config["embed_dim"],
    chunk_size=config["chunk_size"],
)

# RECURRENCE
custom_rnn_cell = CustomRecurrentCell(
    chunk_size=config["chunk_size"],
    r=config["r"],
    num_layers=config["num_layers"],
    ffn_dims=config["embed_dim"],
    ffn_dropout=config["ffn_drop"],
    num_heads=config["num_heads"],
    key_dim=config["embed_dim"],
    attn_dropout=config["attn_drop"],
)


model = ModelTrainer(patch_layer=patch_layer, custom_cell=custom_rnn_cell)
