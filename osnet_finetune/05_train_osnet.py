"""05_train_osnet.py - Torchreid fine-tuning for osnet_x0_25.

Module-level MarketEyeD class is required for Windows multiprocessing
(DataLoader workers use spawn and need to pickle the dataset class).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


def _ensure_torchreid():
    try:
        import torchreid  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "torchreid is required. See README install guide."
        ) from e


# Module-level dataset class (picklable across Windows multiprocessing workers).
try:
    from torchreid.data.datasets.image.market1501 import Market1501

    class MarketEyeD(Market1501):  # type: ignore[misc]
        """EYE-D fine-tuning dataset (Market-1501 compatible layout)."""
        dataset_dir = "market1501"
        dataset_url = None

        def __init__(self, root="", **kwargs):
            super().__init__(root=root, **kwargs)

except ImportError:
    MarketEyeD = None  # type: ignore[assignment]


def _register_dataset(data_root: str):
    if MarketEyeD is None:
        raise RuntimeError("torchreid not installed.")
    from torchreid.data.datasets import register_image_dataset
    register_image_dataset("market1501_eyed", MarketEyeD)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_config.yaml")
    p.add_argument("--evaluate", action="store_true")
    args = p.parse_args()

    _ensure_torchreid()
    from torchreid import data, models, optim, engine
    from torchreid.utils import set_random_seed, mkdir_if_missing

    cfg = yaml.safe_load(Path(args.config).read_text("utf-8"))

    set_random_seed(cfg.get("seed", 1))

    log_dir = Path(cfg["project"]["log_dir"]) / cfg["project"]["name"]
    mkdir_if_missing(str(log_dir))
    print(f"log dir: {log_dir.resolve()}")

    data_root = cfg["data"]["root"]
    _register_dataset(data_root)

    dm = data.ImageDataManager(
        root=data_root,
        sources=cfg["data"]["sources"],
        targets=cfg["data"]["targets"],
        height=cfg["data"]["height"],
        width=cfg["data"]["width"],
        batch_size_train=cfg["train"]["batch_size"],
        batch_size_test=cfg["test"]["batch_size"],
        transforms=cfg["data"]["transforms"],
        norm_mean=cfg["data"]["norm_mean"],
        norm_std=cfg["data"]["norm_std"],
        workers=cfg["data"]["workers"],
        train_sampler=cfg["sampler"]["train_sampler"],
        num_instances=cfg["sampler"]["num_instances"],
        combineall=cfg["data"].get("combineall", False),
    )

    model = models.build_model(
        name=cfg["model"]["name"],
        num_classes=dm.num_train_pids,
        loss=cfg["loss"]["name"],
        pretrained=cfg["model"]["pretrained"],
    )
    model = model.cuda()

    optimizer = optim.build_optimizer(
        model,
        optim=cfg["train"]["optim"],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = optim.build_lr_scheduler(
        optimizer,
        lr_scheduler=cfg["train"]["lr_scheduler"],
        max_epoch=cfg["train"]["max_epoch"],
    )

    loss_name = cfg["loss"]["name"]
    if loss_name == "softmax":
        eng = engine.ImageSoftmaxEngine(
            dm, model, optimizer=optimizer, scheduler=scheduler,
            label_smooth=cfg["loss"]["softmax"].get("label_smooth", True),
        )
    elif loss_name == "triplet":
        eng = engine.ImageTripletEngine(
            dm, model, optimizer=optimizer, scheduler=scheduler,
            margin=cfg["loss"]["triplet"]["margin"],
            weight_t=cfg["loss"]["triplet"].get("weight_t", 1.0),
            weight_x=cfg["loss"]["triplet"].get("weight_x", 1.0),
            label_smooth=cfg["loss"]["softmax"].get("label_smooth", True),
        )
    else:
        raise ValueError(f"unsupported loss: {loss_name}")

    eng.run(
        save_dir=str(log_dir),
        max_epoch=cfg["train"]["max_epoch"],
        start_epoch=cfg["train"].get("start_epoch", 0),
        print_freq=cfg["train"].get("print_freq", 20),
        fixbase_epoch=cfg["train"].get("fixbase_epoch", 0),
        open_layers=cfg["train"].get("open_layers", None),
        eval_freq=cfg["test"].get("eval_freq", 10),
        test_only=args.evaluate or cfg["test"].get("evaluate", False),
        dist_metric=cfg["test"].get("dist_metric", "cosine"),
        normalize_feature=cfg["test"].get("normalize_feature", True),
        rerank=cfg["test"].get("rerank", False),
        visrank=cfg["test"].get("visrank", False),
    )

    print("training done.")
    print(f"latest checkpoint dir: {log_dir / 'model'}")


if __name__ == "__main__":
    main()
