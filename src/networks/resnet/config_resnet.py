"""
Config file for the experiment:
    - model: resnet
"""

config = {
    "seed": 2024,
    "trainer": {
        "max_epochs": 10,
        "accelerator": "gpu",
        "devices": "auto",
        "default_root_dir": None,
        "log_every_n_steps": 79,
    },
    "data": {
        "data_dir_path": None,
        "dataset_name": "Mag_Inv_POC",
        "img_size": (224, 224),
        "batch_size": 128,
        "num_workers": 9,
        "train_data_csv": None,
        "val_data_csv": None,
        "test_data_csv": None,
        "pixel_size": None,
        "num_patches": 10000,
        "downstream_task": None,
        "exclude_background_in_classification_targets": True,
    },
    "model": {
        "resnet_type": None,
        "image_channels": 3,
        "num_classes": 1000,
        "variant": "base",
        "num_conditions": None, 
        "classification_type": None,
        "class_weights_dict": None,
        "optimizer_init": {
            "class_path": "torch.optim.AdamW",
            "init_args": {
                "lr": 0.001,
            },
        },
        "lr_scheduler_init": {
            "class_path": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "init_args": {"T_max": 200, "eta_min": 1e-5, "verbose": True},
            "interval": "epoch",
        },
    },
}
