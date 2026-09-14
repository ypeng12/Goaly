#!/usr/bin/env python3
"""Train dialogue acts using the same mask and executor as real customer chat."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import zipfile

import torch

from backend.harness.customer_training import CustomerPolicyTrainingEnv, scenarios
from backend.rl.featurizer import StateFeaturizer
from backend.rl.ppo_trainer import PPOConfig, PPOTrainer


class CustomerPPOTrainer(PPOTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.source_hashes = self._source_hashes()
        root = Path(__file__).resolve().parent
        self.source_contents = {name: (root / name).read_bytes() for name in self.source_hashes}

    @staticmethod
    def _source_hashes():
        root = Path(__file__).resolve().parent
        files = ("backend/harness/customer_training.py", "backend/harness/customer_policy.py",
                 "backend/harness/customer_service.py", "backend/harness/conversation_context.py",
                 "backend/harness/state_machine.py", "backend/harness/extractor.py",
                 "backend/harness/grounded_data.py", "backend/harness/context_builder.py",
                 "backend/harness/types.py", "backend/harness/dialogue.py", "backend/harness/rl_baseline.py",
                 "backend/engine/mock_engine.py", "backend/rl/ppo_trainer.py",
                 "backend/engine/base.py", "backend/rl/featurizer.py", "backend/rl/models.py",
                 "train_customer_policy.py", "requirements.txt")
        files = list(files) + [str(path.relative_to(root)) for path in sorted((root / "fixtures").glob("*.json"))]
        return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files if (root / name).is_file()}

    def train(self, total_timesteps=None):
        history = super().train(total_timesteps)
        self.training_segments[-1]["source_hashes"] = self.source_hashes
        return history

    def _create_env(self, split="train"):
        return CustomerPolicyTrainingEnv(random.choice(scenarios(split)), max_turns=self.cfg.max_turns)

    def save_checkpoint(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_id = hashlib.sha256(json.dumps(self.source_hashes, sort_keys=True).encode()).hexdigest()[:16]
        archive = destination.parent / f"training_source_{source_id}.zip"
        archive_tmp = archive.with_suffix(f".seed{self.cfg.seed}.tmp")
        with zipfile.ZipFile(archive_tmp, "w", compression=zipfile.ZIP_DEFLATED) as source:
            for name, content in sorted(self.source_contents.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                source.writestr(info, content)
        archive_tmp.replace(archive)
        temporary = destination.with_suffix(".tmp")
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.cfg), "history": self.history,
            "training_segments": self.training_segments,
            "feature_version": StateFeaturizer.VERSION,
            "state_dim": StateFeaturizer.FEATURE_DIM,
            "environment_version": CustomerPolicyTrainingEnv.ENV_VERSION,
            "checkpoint_family": "customer_multi_turn_v1",
            "mask_version": CustomerPolicyTrainingEnv.MASK_VERSION,
            "dataset_revision": CustomerPolicyTrainingEnv.DATASET_REVISION,
            "training_scenarios": [asdict(s) for s in scenarios("train")],
            "source_hashes_at_start": self.source_hashes,
            "source_hashes_at_save": self._source_hashes(),
            "source_archive": {"file": archive.name, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                               "usage": "Overlay on this repository before reproducing the recorded training segments."},
            "limitations": "Finite reactive text protocol with a small fixture set; not human calls or language-model post-training.",
        }, temporary)
        temporary.replace(destination)
        print(f"Customer checkpoint saved to {destination}")

    def load_customer_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if (checkpoint.get("environment_version") != CustomerPolicyTrainingEnv.ENV_VERSION
                or checkpoint.get("checkpoint_family") != "customer_multi_turn_v1"
                or checkpoint.get("feature_version") != StateFeaturizer.VERSION):
            raise ValueError("A compatible customer checkpoint is required")
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
        self.training_segments = checkpoint["training_segments"]
        self._prior_requested_steps = checkpoint["config"]["total_timesteps"]
        self._warm_start_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path")
    parser.add_argument("--warm-start")
    args = parser.parse_args()
    if args.timesteps < 1:
        parser.error("timesteps must be positive")
    torch.set_num_threads(1)
    trainer = CustomerPPOTrainer(PPOConfig(total_timesteps=args.timesteps, seed=args.seed, max_turns=20, device="cpu"))
    if args.warm_start:
        trainer.load_customer_checkpoint(args.warm_start)
    trainer.train(args.timesteps)
    destination = Path(args.save_path or f"artifacts/customer_policy/seed{args.seed}.pt")
    trainer.save_checkpoint(destination)
    destination.with_suffix(".metrics.json").write_text(json.dumps(trainer.history, indent=2) + "\n")


if __name__ == "__main__":
    main()
