"""Phase 2: Decoder Head Pulse — train and evaluate.

Trains decoder pulse injection on Whisper-Tiny with frozen weights.
Evaluates on gapped test set and hallucination tests.

Variants:
  baseline: Frozen Whisper (no injection)
  decoder_pulse: Pulse injection at decoder self-attention K/V
  decoder_pulse_phase: + state-dependent phase shift

Usage:
    uv run python scripts/run_decoder_pulse.py [--config configs/decoder_pulse.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

from pulse_whisper.data.dataset import get_10h_subset_dataloader, get_dataloader
from pulse_whisper.eval.gapped_eval import evaluate_all_gap_levels, evaluate_hallucination
from pulse_whisper.models.pulse_whisper import get_processor
from pulse_whisper.models.pulse_whisper_decoder import build_decoder_pulse_model
from pulse_whisper.training.config import load_config
from pulse_whisper.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def setup_device(device_str: str | None = None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_decoder_pulse(
    config,
    device: torch.device,
    output_dir: Path,
    use_phase_net: bool = False,
    variant_name: str = "decoder_pulse",
) -> dict:
    """Train decoder pulse model."""
    logger.info("=" * 70)
    logger.info(f"TRAINING: {variant_name}")
    logger.info("=" * 70)

    variant_dir = output_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    model = build_decoder_pulse_model(
        whisper_size=config.model.whisper_size,
        n_frequencies=config.model.n_frequencies,
        alpha_init=config.model.alpha_init,
        alpha_max=config.model.alpha_max,
        use_phase_net=use_phase_net,
    )

    trainable = model.trainable_param_count()
    total = model.total_param_count()
    logger.info(f"Model: {total:,} total params, {trainable:,} trainable")

    # Load training data
    logger.info("Loading 10h training subset...")
    train_loader = get_10h_subset_dataloader(
        whisper_size=config.model.whisper_size,
        batch_size=config.training.batch_size,
        gap_augmentation=config.training.gap_augmentation,
        gap_fractions=config.training.gap_fractions,
    )
    logger.info(f"Training samples: {len(train_loader.dataset)}, batches: {len(train_loader)}")

    config.logging.checkpoint_dir = str(variant_dir / "checkpoints")

    start_time = time.time()
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        config=config,
        device=device,
    )
    history = trainer.train()
    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time / 60:.1f} minutes")

    # Log decoder pulse stats
    stats = model.decoder_pulse.get_pulse_stats()
    logger.info("Post-training decoder pulse stats:")
    for s in stats:
        logger.info(f"  Layer {s['layer']} Head {s['head']}: "
                     f"alpha_k={s['alpha_k']:.4f}, alpha_v={s['alpha_v']:.4f}, "
                     f"omega={s['omega_mean']:.2f}±{s['omega_std']:.2f}")

    # Save model
    final_path = variant_dir / "final.pt"
    pulse_state = {
        k: v for k, v in model.state_dict().items()
        if k.startswith("decoder_pulse")
    }
    torch.save({"model_state_dict": pulse_state}, final_path)
    logger.info(f"Model saved to {final_path}")

    return {
        "variant": variant_name,
        "training": {
            "train_loss": history["train_loss"],
            "time_minutes": train_time / 60,
        },
        "trainable_params": trainable,
        "pulse_stats": stats,
    }


def evaluate_decoder_pulse(
    config,
    device: torch.device,
    output_dir: Path,
    use_phase_net: bool = False,
    variant_name: str = "decoder_pulse",
    max_eval_batches: int | None = None,
) -> dict:
    """Evaluate a trained decoder pulse model."""
    logger.info("=" * 70)
    logger.info(f"EVALUATING: {variant_name}")
    logger.info("=" * 70)

    variant_dir = output_dir / variant_name

    model = build_decoder_pulse_model(
        whisper_size=config.model.whisper_size,
        n_frequencies=config.model.n_frequencies,
        alpha_init=config.model.alpha_init,
        alpha_max=config.model.alpha_max,
        use_phase_net=use_phase_net,
    )

    # Load trained weights
    final_path = variant_dir / "final.pt"
    if final_path.exists():
        ckpt = torch.load(final_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info(f"Loaded trained weights from {final_path}")
    else:
        logger.warning(f"No trained weights at {final_path}, using init weights")

    model = model.to(device)
    model.eval()

    processor = get_processor(config.model.whisper_size)

    # Load test data
    test_loader = get_dataloader(
        split="test-clean",
        whisper_size=config.model.whisper_size,
        batch_size=8,
        max_samples=config.eval.max_eval_samples,
    )
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Gapped evaluation
    gap_results = evaluate_all_gap_levels(
        model=model,
        dataloader=test_loader,
        processor=processor,
        device=device,
        max_batches=max_eval_batches,
    )

    logger.info(f"\n{'Gap Level':<15} {'WER':<10} {'CER':<10} {'Samples':<10}")
    logger.info("-" * 45)
    for level, result in gap_results.items():
        logger.info(f"{level:<15} {result.wer:<10.4f} {result.cer:<10.4f} {result.num_samples:<10}")

    # Hallucination test
    halluc_results = evaluate_hallucination(
        model=model,
        processor=processor,
        device=device,
        num_samples=30,
    )

    for input_type, result in halluc_results.items():
        logger.info(f"Hallucination [{input_type}]: rate={result.hallucination_rate:.2%}, "
                     f"avg_length={result.avg_output_length:.1f}")

    # Pulse stats
    pulse_stats = model.decoder_pulse.get_pulse_stats()

    return {
        "variant": variant_name,
        "gap_evaluation": {
            level: {"wer": r.wer, "cer": r.cer, "num_samples": r.num_samples}
            for level, r in gap_results.items()
        },
        "hallucination": {
            input_type: {
                "hallucination_rate": r.hallucination_rate,
                "avg_output_length": r.avg_output_length,
                "num_samples": r.num_samples,
            }
            for input_type, r in halluc_results.items()
        },
        "pulse_stats": pulse_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Decoder Head Pulse")
    parser.add_argument("--config", default="configs/decoder_pulse.yaml", help="Config file")
    parser.add_argument("--device", default=None, help="Device")
    parser.add_argument("--output-dir", default="results/decoder_pulse", help="Output directory")
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline eval")
    parser.add_argument("--variants", default="decoder_pulse,decoder_pulse_phase",
                        help="Comma-separated variants to run")
    args = parser.parse_args()

    device = setup_device(args.device)
    logger.info(f"Using device: {device}")

    config = load_config(args.config)
    if args.max_eval_samples:
        config.eval.max_eval_samples = args.max_eval_samples

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [v.strip() for v in args.variants.split(",")]

    results = {
        "experiment": "decoder_pulse",
        "config": {
            "whisper_size": config.model.whisper_size,
            "max_epochs": config.training.max_epochs,
            "batch_size": config.training.batch_size,
            "lr": config.training.lr,
            "alpha_max": config.model.alpha_max,
            "n_frequencies": config.model.n_frequencies,
        },
        "device": str(device),
        "training": [],
        "evaluations": [],
    }

    # Baseline evaluation (frozen Whisper, no pulse)
    if not args.skip_baseline:
        logger.info("\n" + "#" * 70)
        logger.info("# BASELINE EVALUATION (Frozen Whisper)")
        logger.info("#" * 70)

        from pulse_whisper.models.pulse_whisper import build_variant, Variant
        baseline_model = build_variant(Variant.A, whisper_size=config.model.whisper_size)
        baseline_model = baseline_model.to(device)
        baseline_model.eval()

        processor = get_processor(config.model.whisper_size)
        test_loader = get_dataloader(
            split="test-clean",
            whisper_size=config.model.whisper_size,
            batch_size=8,
            max_samples=config.eval.max_eval_samples,
        )

        gap_results = evaluate_all_gap_levels(
            model=baseline_model, dataloader=test_loader,
            processor=processor, device=device,
            max_batches=args.max_eval_batches,
        )
        halluc_results = evaluate_hallucination(
            model=baseline_model, processor=processor,
            device=device, num_samples=30,
        )

        baseline_eval = {
            "variant": "baseline",
            "gap_evaluation": {
                level: {"wer": r.wer, "cer": r.cer, "num_samples": r.num_samples}
                for level, r in gap_results.items()
            },
            "hallucination": {
                input_type: {
                    "hallucination_rate": r.hallucination_rate,
                    "avg_output_length": r.avg_output_length,
                    "num_samples": r.num_samples,
                }
                for input_type, r in halluc_results.items()
            },
        }
        results["evaluations"].append(baseline_eval)

        logger.info(f"\nBaseline WER: " + ", ".join(
            f"{k}={v['wer']:.4f}" for k, v in baseline_eval["gap_evaluation"].items()
        ))

        del baseline_model
        torch.cuda.empty_cache() if device.type == "cuda" else None

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Train and evaluate each variant
    for variant_name in variants:
        use_phase = "phase" in variant_name

        if not args.skip_training:
            logger.info("\n" + "#" * 70)
            logger.info(f"# TRAINING: {variant_name}")
            logger.info("#" * 70)

            train_result = train_decoder_pulse(
                config, device, output_dir,
                use_phase_net=use_phase,
                variant_name=variant_name,
            )
            results["training"].append(train_result)

            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)

        logger.info("\n" + "#" * 70)
        logger.info(f"# EVALUATING: {variant_name}")
        logger.info("#" * 70)

        eval_result = evaluate_decoder_pulse(
            config, device, output_dir,
            use_phase_net=use_phase,
            variant_name=variant_name,
            max_eval_batches=args.max_eval_batches,
        )
        results["evaluations"].append(eval_result)

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Variant':<25} {'Clean WER':<12} {'Multi-gap WER':<15} {'Halluc (silence)':<18}")
    logger.info("-" * 70)
    for ev in results["evaluations"]:
        clean = ev["gap_evaluation"].get("gap_0", {}).get("wer", float("nan"))
        multi = ev["gap_evaluation"].get("multi_gap", {}).get("wer", float("nan"))
        halluc = ev.get("hallucination", {}).get("silence", {}).get("hallucination_rate", float("nan"))
        logger.info(f"{ev['variant']:<25} {clean:<12.4f} {multi:<15.4f} {halluc:<18.2%}")

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nAll results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
