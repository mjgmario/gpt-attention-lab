"""Training utilities for GPT model."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from attention_lab.model import GPT


@dataclass
class TrainerConfig:
    """Configuration for the Trainer.

    Attributes:
        learning_rate: Learning rate for AdamW optimizer.
        weight_decay: Weight decay for regularization.
        betas: Adam optimizer betas.
        grad_clip: Maximum gradient norm for clipping.
        warmup_steps: Number of warmup steps for learning rate.
        max_steps: Maximum number of training steps.
        eval_interval: Steps between evaluations.
        eval_steps: Number of steps for evaluation.
        checkpoint_dir: Directory to save checkpoints.
        log_interval: Steps between logging.
        compile: Whether to use torch.compile for faster training.
    """

    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 5000
    eval_interval: int = 500
    eval_steps: int = 50
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10
    compile: bool = False


class Trainer:
    """Trainer for GPT model with logging and checkpointing."""

    def __init__(
        self,
        model: GPT,
        train_loader: DataLoader,
        config: TrainerConfig | None = None,
        val_loader: DataLoader | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: GPT model to train.
            train_loader: DataLoader for training data.
            config: Trainer configuration.
            val_loader: Optional DataLoader for validation data.
            device: Device to train on. If None, auto-detect.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainerConfig()

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model.to(device)

        # Setup optimizer
        self.optimizer = self._configure_optimizer()

        # torch.compile for faster training
        if self.config.compile:
            try:
                self.model = torch.compile(self.model)  # type: ignore[assignment]
                print("Model compiled with torch.compile")
            except Exception as e:
                print(f"torch.compile not available, skipping: {e}")

        # Training state
        self.step = 0
        self.best_val_loss = float("inf")

        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        """Configure AdamW optimizer with weight decay."""
        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "bias" in name or "ln_" in name or "wpe" in name or "wte" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            optim_groups, lr=self.config.learning_rate, betas=self.config.betas
        )

    def _get_lr(self) -> float:
        """Get learning rate with warmup and cosine decay."""
        if self.step < self.config.warmup_steps:
            return self.config.learning_rate * self.step / self.config.warmup_steps
        return self.config.learning_rate

    def _update_lr(self) -> None:
        """Update learning rate."""
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate model on validation set.

        Returns:
            Dictionary with 'val_loss' and 'val_perplexity'.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for i, (x, y) in enumerate(self.val_loader):
            if i >= self.config.eval_steps:
                break

            x, y = x.to(self.device), y.to(self.device)
            _, loss, _ = self.model(x, targets=y)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.model.train()

        return {"val_loss": avg_loss, "val_perplexity": 2.718**avg_loss}

    def save_checkpoint(self, filename: str = "checkpoint.pt") -> str:
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename.

        Returns:
            Path to saved checkpoint.
        """
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step": self.step,
                "config": self.model.config,
                "best_val_loss": self.best_val_loss,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint["step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    def train(self) -> dict[str, list[float]]:
        """Run training loop.

        Returns:
            Dictionary with training history ('train_loss', 'val_loss', 'steps').
        """
        self.model.train()
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "steps": []}

        train_iter = iter(self.train_loader)
        pbar = tqdm(total=self.config.max_steps, desc="Training")

        while self.step < self.config.max_steps:
            # Get next batch
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                x, y = next(train_iter)

            x, y = x.to(self.device), y.to(self.device)

            # Update learning rate
            self._update_lr()

            # Forward pass
            _, loss, _ = self.model(x, targets=y)

            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.optimizer.step()
            self.step += 1

            # Logging
            if self.step % self.config.log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{self._get_lr():.2e}"})
                history["train_loss"].append(loss.item())
                history["steps"].append(self.step)

            # Evaluation
            if self.step % self.config.eval_interval == 0:
                metrics = self.evaluate()
                if metrics:
                    val_loss = metrics["val_loss"]
                    history["val_loss"].append(val_loss)
                    tqdm.write(
                        f"Step {self.step}: val_loss={val_loss:.4f}, "
                        f"perplexity={metrics['val_perplexity']:.2f}"
                    )

                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best_model.pt")

                # Save periodic checkpoint
                self.save_checkpoint(f"checkpoint_{self.step}.pt")

            pbar.update(1)

        pbar.close()

        # Save final checkpoint
        self.save_checkpoint("final_model.pt")

        return history
