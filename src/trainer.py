import os
import math
import logging
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format= "%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_matryoshka_model(
    model: SentenceTransformer,
    train_dataloader: DataLoader,
    config: Dict[str, Any],
    output_path: str
) -> SentenceTransformer:
    """
    Fine-tunes the embedding model using MatryoshkaLoss nested arounf MultipleNegativeRankingLoss.
    """
    train_cfg = config.get("training", {})
    mrl_cfg = config.get("matryoshka", {})

    epochs = train_cfg.get("epochs", 2)
    lr = float(train_cfg.get("learning_rate", 2.0e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(train_cfg.get("warmup_ratio",0.1))
    use_fp16 = train_cfg.get("fp16", True)

    mrl_dims: List[int] = mrl_cfg.get("training_dimensions", [768, 512, 256, 128])
    mrl_weights: List[float] = mrl_cfg.get("dimension_weights", [1.0, 1.0, 1.0, 1.0])

    logger.info(f"Configuring Base Loss: MultipleNegativeRankingLoss (MNRL)...")
    base_loss = losses.MultipleNegativeRankingLoss(model=model)

    logger.info(f"Wrapping with MatryoshkaLoss over sub-dimensions: {mrl_dims} with weights: {mrl_weights}...")
    train_loss = losses.MatryoshkaLoss(
        model=model,
        loss=base_loss,
        matryoshka_weights=mrl_weights,
        matryoshka_dims = mrl_dims
    )

    total_steps = len(train_dataloader) * epochs
    warmup_steps = math.ceil(total_steps * warmup_ratio)
    logger.info(f"Total training steps: {total_steps} | Warmup steps: {warmup_steps} | Epochs: {epochs}")

    os.makedirs(output_path, exist_ok=True)

    logger.info("Initializing model.fit()...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps = warmup_steps,
        optimizer_params={"lr": lr, "weight_decay": weight_decay},
        output_path=output_path,
        show_progress_bar=True,
        use_amp=use_fp16 # Automated Mixed precision (FP16) on GPU
    )

    logger.info(f"Training complete! Model saved to: {output_path}")
    return model
