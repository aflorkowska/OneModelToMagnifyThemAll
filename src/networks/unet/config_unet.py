"""
Config file for the experiment:
    - model: unet2D
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
        "exclude_background_in_classification_targets": False,
    },
    "model": {
        "n_channels": 3,
        "n_classes": 1000,
        "segmentation_type": None,
        "features_start": 64,
        "bilinear": False,
        "variant": "base",  # 'base', 'conditional_normalization'
        "num_conditions": None,  # Required for conditional variants
        "class_weights_dict": None,
        "optimizer_init": {
            "class_path": "torch.optim.AdamW",
            "init_args": {
                "lr": 0.001,
                # 'weight_decay': 0.01 # L2 regulariation, worthy to add later when fine-tuning
            },
        },
        # For other schedule please adapt also the configure_optimizers method
        #'lr_scheduler_init': None
        "lr_scheduler_init": {
            "class_path": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "init_args": {"T_max": 200, "eta_min": 1e-5, "verbose": True},
            "interval": "epoch",  # scheduler.step() co epokę
        },
        # 'lr_scheduler_init': {
        #     'class_path': 'torch.optim.lr_scheduler.ReduceLROnPlateau',
        #     'init_args': {
        #         'mode': 'min',
        #         'factor': 0.5,
        #         'patience': 5,
        #         'verbose': True
        #     },
        #     'monitor': 'valid_loss',
        #     'interval': 'epoch'
        # },
    },
}
