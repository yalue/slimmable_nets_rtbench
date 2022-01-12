flags_dict = {
    "num_gpus_per_job": 8,
    "num_cpus_per_job": 12,
    "memory_per_job": 1024,
    "gpu_type": "radeon-vii",
    "dataset": "imagenet1k",
    "data_transforms": "imagenet1k_mobile",
    "data_loader": "imagenet1k_basic",
    "dataset_dir": "/storage/other/imagenet1k/Data/CLS-LOC",
    "data_loader_workers": 1,
    "num_classes": 1000,
    "image_size": 224,
    "topk": [
        1,
        5
    ],
    "num_epochs": 250,
    "optimizer": "sgd",
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "nesterov": True,
    "lr": 0.5,
    "lr_scheduler": "linear_decaying",
    "multistep_lr_milestones": [
        30,
        60,
        90
    ],
    "multistep_lr_gamma": 0.1,
    "profiling": [
        "gpu"
    ],
    "pretrained": "us_mobilenet_v1_calibrated.pt",
    "resume": "",
    "test_only": True,
    "random_seed": 1995,
    "batch_size": 64,
    "model": "us_mobilenet_v1",
    "reset_parameters": True,
    "log_dir": "logs/",
    "slimmable_training": True,
    "bn_cal_batch_num": 5,
    "calibrate_bn": True,
    "cumulative_bn_stats": True,
    "num_sample_training": 4,
    "universally_slimmable_training": True,
    "soft_target": True,
    "width_mult": 1.0,
    "width_mult_list": [
        0.25,
        0.275,
        0.3,
        0.325,
        0.35,
        0.375,
        0.4,
        0.425,
        0.45,
        0.475,
        0.5,
        0.525,
        0.55,
        0.575,
        0.6,
        0.625,
        0.65,
        0.675,
        0.7,
        0.725,
        0.75,
        0.775,
        0.8,
        0.825,
        0.85,
        0.875,
        0.9,
        0.925,
        0.95,
        0.975,
        1.0
    ],
    "width_mult_range": [
        0.25,
        1.0
    ],
    "drop_last": True
}

# Copied from https://stackoverflow.com/a/1305682
class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)

FLAGS = obj(flags_dict)

