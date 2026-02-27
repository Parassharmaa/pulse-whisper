"""Phase 1.2: Zero-shot pulse injection test.

Injects fixed-parameter PulseLayer (no training) between Whisper encoder layers.
Re-runs gapped eval, compares against Phase 1.1 baseline.
Looks for ANY signal (WER or hallucination reduction).

Usage:
    uv run python scripts/run_phase1_zeroshot.py [--max-samples N] [--device DEVICE]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import WhisperProcessor

from pulse_whisper.data.dataset import get_dataloader
from pulse_whisper.data.gapped_audio import GapLevel
from pulse_whisper.eval.gapped_eval import evaluate_all_gap_levels, evaluate_hallucination
from pulse_whisper.models.pulse_whisper import PulseWhisperEncoder, Variant, build_variant
from pulse_whisper.analysis.alpha_analysis import extract_pulse_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 1.2: Zero-shot pulse injection")
    parser.add_argument("--whisper-size", default="tiny", help="Whisper model size")
    parser.add_argument("--max-samples", type=int, default=None, help="Max test samples")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches per gap level")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default=None, help="Device")
    parser.add_argument("--output", default="results/phase1_zeroshot.json", help="Output file")
    parser.add_argument("--baseline-results", default="results/phase1_baseline.json", help="Baseline results for comparison")
    parser.add_argument("--hallucination-samples", type=int, default=20, help="Hallucination test samples")
    parser.add_argument("--alpha-init", type=float, default=0.01, help="Initial alpha value")
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{args.whisper_size}")

    # Test variants C (pulse, no phase) and D (pulse + phase)
    variants_to_test = [
        (Variant.A, "Baseline (frozen Whisper)"),
        (Variant.C, "Pulse (fixed phase, no training)"),
        (Variant.D, "Pulse + Phase (no training)"),
    ]

    # Load test data
    logger.info("Loading test-clean dataset...")
    dataloader = get_dataloader(
        split="test-clean",
        whisper_size=args.whisper_size,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
    logger.info(f"Test samples: {len(dataloader.dataset)}")

    all_results = {}

    for variant, description in variants_to_test:
        logger.info("\n" + "=" * 60)
        logger.info(f"VARIANT {variant.name}: {description}")
        logger.info("=" * 60)

        model = build_variant(
            variant=variant,
            whisper_size=args.whisper_size,
            alpha_init=args.alpha_init,
        ).to(device)
        model.eval()

        # Log pulse stats
        stats = extract_pulse_stats(model)
        for s in stats:
            logger.info(
                f"  Layer {s.layer_idx}: α={s.alpha:.6f}, "
                f"|A|={s.amplitude_mean:.4f}, ω_mean={s.omega_mean:.2f}"
            )

        logger.info(f"  Trainable params: {model.trainable_param_count():,}")
        logger.info(f"  Total params: {model.total_param_count():,}")

        # Gapped evaluation
        gap_results = evaluate_all_gap_levels(
            model=model,
            dataloader=dataloader,
            processor=processor,
            device=device,
            max_batches=args.max_batches,
        )

        # Print results
        logger.info(f"\n{'Gap Level':<15} {'WER':<10} {'CER':<10}")
        logger.info("-" * 35)
        for level, result in gap_results.items():
            logger.info(f"{level:<15} {result.wer:<10.4f} {result.cer:<10.4f}")

        # Hallucination test
        halluc_results = evaluate_hallucination(
            model=model,
            processor=processor,
            device=device,
            num_samples=args.hallucination_samples,
        )

        all_results[variant.value] = {
            "description": description,
            "trainable_params": model.trainable_param_count(),
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

        # Clean up to free memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Comparison table
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON: WER across variants and gap levels")
    logger.info("=" * 70)

    gap_levels = list(next(iter(all_results.values()))["gap_evaluation"].keys())
    header = f"{'Variant':<20}" + "".join(f"{gl:<12}" for gl in gap_levels)
    logger.info(header)
    logger.info("-" * len(header))
    for variant_name, result in all_results.items():
        row = f"{variant_name:<20}"
        for gl in gap_levels:
            wer = result["gap_evaluation"][gl]["wer"]
            row += f"{wer:<12.4f}"
        logger.info(row)

    # Load baseline for delta comparison
    baseline_path = Path(args.baseline_results)
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        logger.info("\nDelta vs Phase 1.1 baseline:")
        for variant_name, result in all_results.items():
            if variant_name == "baseline":
                continue
            for gl in gap_levels:
                if gl in baseline.get("gap_evaluation", {}):
                    base_wer = baseline["gap_evaluation"][gl]["wer"]
                    curr_wer = result["gap_evaluation"][gl]["wer"]
                    delta = curr_wer - base_wer
                    logger.info(f"  {variant_name} @ {gl}: {delta:+.4f} ({'better' if delta < 0 else 'worse'})")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "experiment": "phase1_zeroshot",
            "whisper_size": args.whisper_size,
            "alpha_init": args.alpha_init,
            "device": str(device),
            "variants": all_results,
        }, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
