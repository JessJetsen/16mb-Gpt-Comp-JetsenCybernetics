#!/usr/bin/env python3
"""
Probe adjacent-token BigramHash behavior against a SentencePiece tokenizer.

This utility is aligned with the current BigramHash runtime in `train_gpt.py`:
- the first token in each sequence is assigned to the reserved final bucket
- observed bigrams are hashed with:
    xor(prev_id * PREV_MULT, curr_id * CURR_MULT) % (num_buckets - 1)

Examples:

  # Dump vocab only
  python scripts/bigramhash_probe.py \
      --spm data/tokenizers/fineweb_1024_bpe.model \
      --dump-vocab probe.vocab.tsv

  # Tokenize raw text and analyze bigram buckets
  python scripts/bigramhash_probe.py \
      --spm data/tokenizers/fineweb_1024_bpe.model \
      --text corpus.txt \
      --num-buckets 4096 \
      --top-pairs 5000 \
      --top-buckets 500 \
      --out-prefix probe

  # Analyze pre-tokenized ids (space-separated, one sequence per line)
  python scripts/bigramhash_probe.py \
      --spm data/tokenizers/fineweb_1024_bpe.model \
      --token-ids token_ids.txt \
      --out-prefix probe

  # Probe trainer-format FineWeb shards directly
  python scripts/bigramhash_probe.py \
      --spm data/tokenizers/fineweb_1024_bpe.model \
      --shard-glob 'data/datasets/fineweb10B_sp1024/fineweb_train_*.bin' \
      --shard-limit 2 \
      --max-tokens 2000000 \
      --out-prefix probe
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

try:
    import sentencepiece as spm
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: sentencepiece\n"
        "Install with: pip install sentencepiece"
    ) from exc


Pair = tuple[int, int]
PREV_MULT_DEFAULT = 27191
CURR_MULT_DEFAULT = 36313


@dataclass(frozen=True)
class ProbeStats:
    sequence_count: int
    token_count: int
    pair_count: int
    unique_pair_count: int
    non_empty_buckets: int
    reserved_bucket_count: int
    max_bucket_total: int
    mean_bucket_total: float
    mean_pairs_per_non_empty_bucket: float


@dataclass(frozen=True)
class ProbeResult:
    num_buckets: int
    buckets: dict[int, list[tuple[Pair, int]]]
    stats: ProbeStats


def env_or_default(env_name: str, default: str) -> str:
    return os.environ.get(env_name, default)


def parse_optional_int_csv(value: str | None) -> list[int]:
    if value is None:
        return []
    items = [item.strip() for item in value.split(",")]
    return [int(item) for item in items if item]


def load_spm(model_path: str) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    ok = sp.load(model_path)
    if not ok:
        raise RuntimeError(f"Failed to load SentencePiece model: {model_path}")
    return sp


def visible_token_text(text: str) -> str:
    # Keep whitespace-bearing tokens legible in plain-text and TSV reports.
    return (
        text.replace("\\", "\\\\")
        .replace("\t", "\\t")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace(" ", "␠")
    )


def dump_vocab(sp: spm.SentencePieceProcessor, out_path: str) -> None:
    vocab_size = sp.get_piece_size()
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["token_id", "piece", "piece_visible"])
        for token_id in range(vocab_size):
            piece = sp.id_to_piece(token_id)
            writer.writerow([token_id, piece, visible_token_text(piece)])


def iter_tokenized_from_text(
    sp: spm.SentencePieceProcessor,
    text_path: str,
    chunk_chars: int,
) -> Iterator[list[int]]:
    """
    Stream a text file in chunks and tokenize each chunk independently.

    This is intentionally simple and exploratory. It can split semantic spans
    across chunk boundaries, so it is best used for rough corpus profiling.
    """
    with open(text_path, "r", encoding="utf-8", errors="replace") as handle:
        while True:
            chunk = handle.read(chunk_chars)
            if not chunk:
                break
            token_ids = sp.encode(chunk, out_type=int)
            if token_ids:
                yield token_ids


def iter_token_ids_from_file(token_ids_path: str) -> Iterator[list[int]]:
    with open(token_ids_path, "r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                token_ids = [int(piece) for piece in line.split()]
            except ValueError as exc:
                raise ValueError(
                    f"Invalid token id on line {line_no} of {token_ids_path}"
                ) from exc
            if token_ids:
                yield token_ids


def load_data_shard_tokens(file: Path) -> np.ndarray:
    # Match the trainer shard format exactly so offline probes mirror training.
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    token_ids = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if token_ids.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return token_ids


def resolve_shard_files(
    shard_glob: str | None,
    shard_list: Sequence[str] | None,
) -> list[Path]:
    if shard_glob and shard_list:
        raise ValueError("Use either --shard-glob or --shard, not both")
    if shard_list:
        files = [Path(item) for item in shard_list]
    elif shard_glob:
        files = [Path(item) for item in sorted(glob.glob(shard_glob))]
    else:
        return []
    if not files:
        raise FileNotFoundError("No shard files matched the requested input")
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing shard file(s): {', '.join(missing)}")
    return files


def iter_token_ids_from_shards(
    files: Sequence[Path],
    shard_limit: int | None,
    max_tokens: int | None,
) -> Iterator[list[int]]:
    tokens_remaining = max_tokens
    selected_files = list(files[:shard_limit]) if shard_limit is not None else list(files)
    for file in selected_files:
        if tokens_remaining is not None and tokens_remaining <= 0:
            break
        token_ids = load_data_shard_tokens(file)
        if tokens_remaining is not None and token_ids.size > tokens_remaining:
            token_ids = token_ids[:tokens_remaining]
        if token_ids.size > 0:
            yield token_ids.astype(np.int64, copy=False).tolist()
            if tokens_remaining is not None:
                tokens_remaining -= int(token_ids.size)


def count_bigrams(
    sequences: Iterable[Sequence[int]],
) -> tuple[collections.Counter[Pair], int, int]:
    counts: collections.Counter[Pair] = collections.Counter()
    sequence_count = 0
    token_count = 0
    for token_ids in sequences:
        if not token_ids:
            continue
        sequence_count += 1
        token_count += len(token_ids)
        if len(token_ids) < 2:
            continue
        for prev_id, curr_id in zip(token_ids[:-1], token_ids[1:]):
            counts[(prev_id, curr_id)] += 1
    return counts, sequence_count, token_count


def model_bigram_bucket(
    prev_id: int,
    curr_id: int,
    num_buckets: int,
    prev_mult: int,
    curr_mult: int,
) -> int:
    if num_buckets < 2:
        raise ValueError("num_buckets must be at least 2 to reserve the final bucket")
    usable_bucket_count = num_buckets - 1
    return ((curr_mult * curr_id) ^ (prev_mult * prev_id)) % usable_bucket_count


def invert_to_buckets(
    pair_counts: collections.Counter[Pair],
    num_buckets: int,
    prev_mult: int,
    curr_mult: int,
) -> dict[int, list[tuple[Pair, int]]]:
    buckets: dict[int, list[tuple[Pair, int]]] = collections.defaultdict(list)
    for pair, count in pair_counts.items():
        bucket = model_bigram_bucket(
            prev_id=pair[0],
            curr_id=pair[1],
            num_buckets=num_buckets,
            prev_mult=prev_mult,
            curr_mult=curr_mult,
        )
        buckets[bucket].append((pair, count))
    for bucket_id in buckets:
        buckets[bucket_id].sort(key=lambda item: item[1], reverse=True)
    return dict(buckets)


def pair_to_text(
    sp: spm.SentencePieceProcessor,
    pair: Pair,
) -> tuple[str, str]:
    return sp.id_to_piece(pair[0]), sp.id_to_piece(pair[1])


def compute_probe_stats(
    pair_counts: collections.Counter[Pair],
    buckets: dict[int, list[tuple[Pair, int]]],
    sequence_count: int,
    token_count: int,
    num_buckets: int,
) -> ProbeStats:
    bucket_totals = [sum(count for _, count in items) for items in buckets.values()]
    pair_count = sum(pair_counts.values())
    non_empty_buckets = len(buckets)
    max_bucket_total = max(bucket_totals, default=0)
    mean_bucket_total = pair_count / non_empty_buckets if non_empty_buckets else 0.0
    mean_pairs_per_non_empty_bucket = (
        len(pair_counts) / non_empty_buckets if non_empty_buckets else 0.0
    )
    reserved_bucket_count = sequence_count
    if num_buckets < 2:
        raise ValueError("num_buckets must be at least 2")
    return ProbeStats(
        sequence_count=sequence_count,
        token_count=token_count,
        pair_count=pair_count,
        unique_pair_count=len(pair_counts),
        non_empty_buckets=non_empty_buckets,
        reserved_bucket_count=reserved_bucket_count,
        max_bucket_total=max_bucket_total,
        mean_bucket_total=mean_bucket_total,
        mean_pairs_per_non_empty_bucket=mean_pairs_per_non_empty_bucket,
    )


def build_probe_result(
    pair_counts: collections.Counter[Pair],
    sequence_count: int,
    token_count: int,
    num_buckets: int,
    prev_mult: int,
    curr_mult: int,
) -> ProbeResult:
    buckets = invert_to_buckets(
        pair_counts=pair_counts,
        num_buckets=num_buckets,
        prev_mult=prev_mult,
        curr_mult=curr_mult,
    )
    stats = compute_probe_stats(
        pair_counts=pair_counts,
        buckets=buckets,
        sequence_count=sequence_count,
        token_count=token_count,
        num_buckets=num_buckets,
    )
    return ProbeResult(
        num_buckets=num_buckets,
        buckets=buckets,
        stats=stats,
    )


def write_summary_report(
    stats: ProbeStats,
    num_buckets: int,
    out_path: str,
) -> None:
    usable_bucket_count = num_buckets - 1
    load_factor = (
        stats.unique_pair_count / usable_bucket_count if usable_bucket_count > 0 else 0.0
    )
    occupancy = (
        stats.non_empty_buckets / usable_bucket_count if usable_bucket_count > 0 else 0.0
    )
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("bigramhash_probe_summary\n")
        handle.write(f"  sequence_count: {stats.sequence_count}\n")
        handle.write(f"  token_count: {stats.token_count}\n")
        handle.write(f"  observed_pair_count: {stats.pair_count}\n")
        handle.write(f"  unique_pair_count: {stats.unique_pair_count}\n")
        handle.write(f"  num_buckets_total: {num_buckets}\n")
        handle.write(f"  num_buckets_usable: {usable_bucket_count}\n")
        handle.write(f"  non_empty_usable_buckets: {stats.non_empty_buckets}\n")
        handle.write(f"  reserved_bucket_count: {stats.reserved_bucket_count}\n")
        handle.write(f"  usable_bucket_occupancy: {occupancy:.6f}\n")
        handle.write(f"  unique_pair_load_factor: {load_factor:.6f}\n")
        handle.write(f"  max_bucket_total_count: {stats.max_bucket_total}\n")
        handle.write(f"  mean_bucket_total_count: {stats.mean_bucket_total:.3f}\n")
        handle.write(
            f"  mean_unique_pairs_per_non_empty_bucket: "
            f"{stats.mean_pairs_per_non_empty_bucket:.3f}\n"
        )


def write_top_pairs_report(
    sp: spm.SentencePieceProcessor,
    pair_counts: collections.Counter[Pair],
    num_buckets: int,
    prev_mult: int,
    curr_mult: int,
    out_path: str,
    top_pairs: int,
) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "count",
                "bucket",
                "prev_id",
                "curr_id",
                "prev_piece",
                "curr_piece",
                "prev_visible",
                "curr_visible",
            ]
        )
        for (prev_id, curr_id), count in pair_counts.most_common(top_pairs):
            prev_piece, curr_piece = pair_to_text(sp, (prev_id, curr_id))
            bucket = model_bigram_bucket(
                prev_id=prev_id,
                curr_id=curr_id,
                num_buckets=num_buckets,
                prev_mult=prev_mult,
                curr_mult=curr_mult,
            )
            writer.writerow(
                [
                    count,
                    bucket,
                    prev_id,
                    curr_id,
                    prev_piece,
                    curr_piece,
                    visible_token_text(prev_piece),
                    visible_token_text(curr_piece),
                ]
            )


def write_bucket_report(
    sp: spm.SentencePieceProcessor,
    buckets: dict[int, list[tuple[Pair, int]]],
    out_path: str,
    top_buckets: int,
    top_pairs_per_bucket: int,
) -> None:
    ranked = sorted(
        buckets.items(),
        key=lambda item: sum(count for _, count in item[1]),
        reverse=True,
    )
    with open(out_path, "w", encoding="utf-8") as handle:
        for bucket_id, items in ranked[:top_buckets]:
            total = sum(count for _, count in items)
            unique_pairs = len(items)
            handle.write(
                f"bucket {bucket_id}\n"
                f"  total_count: {total}\n"
                f"  unique_pairs_observed: {unique_pairs}\n"
            )
            for (prev_id, curr_id), count in items[:top_pairs_per_bucket]:
                prev_piece, curr_piece = pair_to_text(sp, (prev_id, curr_id))
                handle.write(
                    f"  - count={count:>8}  pair=({prev_id},{curr_id})  "
                    f"text=({visible_token_text(prev_piece)} , "
                    f"{visible_token_text(curr_piece)})\n"
                )
            handle.write("\n")


def write_bucket_summary_tsv(
    buckets: dict[int, list[tuple[Pair, int]]],
    out_path: str,
) -> None:
    ranked = sorted(
        buckets.items(),
        key=lambda item: sum(count for _, count in item[1]),
        reverse=True,
    )
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["bucket", "total_count", "unique_pairs_observed"])
        for bucket_id, items in ranked:
            total = sum(count for _, count in items)
            writer.writerow([bucket_id, total, len(items)])


def write_sweep_summary_tsv(
    results: Sequence[ProbeResult],
    out_path: str,
) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "num_buckets_total",
                "num_buckets_usable",
                "sequence_count",
                "token_count",
                "observed_pair_count",
                "unique_pair_count",
                "non_empty_usable_buckets",
                "usable_bucket_occupancy",
                "unique_pair_load_factor",
                "max_bucket_total_count",
                "mean_bucket_total_count",
                "mean_unique_pairs_per_non_empty_bucket",
                "reserved_bucket_count",
                "reserved_bucket_id",
            ]
        )
        for result in results:
            usable_bucket_count = result.num_buckets - 1
            occupancy = (
                result.stats.non_empty_buckets / usable_bucket_count
                if usable_bucket_count > 0
                else 0.0
            )
            load_factor = (
                result.stats.unique_pair_count / usable_bucket_count
                if usable_bucket_count > 0
                else 0.0
            )
            writer.writerow(
                [
                    result.num_buckets,
                    usable_bucket_count,
                    result.stats.sequence_count,
                    result.stats.token_count,
                    result.stats.pair_count,
                    result.stats.unique_pair_count,
                    result.stats.non_empty_buckets,
                    f"{occupancy:.6f}",
                    f"{load_factor:.6f}",
                    result.stats.max_bucket_total,
                    f"{result.stats.mean_bucket_total:.3f}",
                    f"{result.stats.mean_pairs_per_non_empty_bucket:.3f}",
                    result.stats.reserved_bucket_count,
                    result.num_buckets - 1,
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe SentencePiece adjacent-token bigrams and hash-bucket collisions."
    )
    parser.add_argument(
        "--spm",
        default=os.environ.get("TOKENIZER_PATH"),
        help="Path to SentencePiece .model. Defaults to TOKENIZER_PATH if set.",
    )
    parser.add_argument(
        "--dump-vocab",
        help="Write vocab TSV. If no corpus input is given, the script exits after dumping.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Raw text file to tokenize and analyze")
    source.add_argument(
        "--token-ids",
        help="Pre-tokenized ids file, one sequence per line, ids separated by spaces",
    )
    source.add_argument(
        "--shard-glob",
        help="Glob for trainer-format .bin shards, for example data/datasets/.../fineweb_train_*.bin",
    )
    source.add_argument(
        "--shard",
        action="append",
        default=None,
        help="Explicit trainer-format .bin shard path. Repeat to provide multiple shards.",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=1_000_000,
        help="Characters per chunk when tokenizing raw text.",
    )
    parser.add_argument(
        "--prev-mult",
        type=int,
        default=int(env_or_default("BIGRAMHASH_PREV_MULT", str(PREV_MULT_DEFAULT))),
        help="Multiplier for the previous token id in the trainer-aligned xor hash. "
        "Defaults to BIGRAMHASH_PREV_MULT if set.",
    )
    parser.add_argument(
        "--curr-mult",
        type=int,
        default=int(env_or_default("BIGRAMHASH_CURR_MULT", str(CURR_MULT_DEFAULT))),
        help="Multiplier for the current token id in the trainer-aligned xor hash. "
        "Defaults to BIGRAMHASH_CURR_MULT if set.",
    )
    parser.add_argument(
        "--num-buckets",
        type=int,
        default=int(env_or_default("BIGRAM_VOCAB_SIZE", "4096")),
        help="Total BigramHash bucket count, including the reserved first-token bucket. "
        "Defaults to BIGRAM_VOCAB_SIZE if set.",
    )
    parser.add_argument(
        "--bucket-sweep",
        default=env_or_default("BIGRAM_VOCAB_SWEEP", ""),
        help="Comma-separated bucket counts to evaluate in one pass. "
        "Defaults to BIGRAM_VOCAB_SWEEP if set.",
    )
    parser.add_argument(
        "--top-pairs",
        type=int,
        default=5000,
        help="Rows in the top-pairs TSV report.",
    )
    parser.add_argument(
        "--top-buckets",
        type=int,
        default=500,
        help="Buckets to include in the human-readable bucket report.",
    )
    parser.add_argument(
        "--top-pairs-per-bucket",
        type=int,
        default=25,
        help="Observed pairs to show inside each bucket.",
    )
    parser.add_argument(
        "--out-prefix",
        default="bigram_probe",
        help="Prefix for output reports.",
    )
    parser.add_argument(
        "--shard-limit",
        type=int,
        default=None,
        help="Optional cap on how many matched shard files to read.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional cap on total tokens consumed from shard input.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.spm:
        raise SystemExit("--spm is required unless TOKENIZER_PATH is set")
    sweep_bucket_counts = parse_optional_int_csv(args.bucket_sweep)
    if args.num_buckets < 2:
        raise SystemExit("--num-buckets must be at least 2")
    if any(num_buckets < 2 for num_buckets in sweep_bucket_counts):
        raise SystemExit("All --bucket-sweep values must be at least 2")
    sp = load_spm(args.spm)

    if args.dump_vocab:
        dump_vocab(sp, args.dump_vocab)
        print(f"Wrote vocab: {args.dump_vocab}")

    if not args.text and not args.token_ids and not args.shard_glob and not args.shard:
        return

    if args.text:
        sequences = iter_tokenized_from_text(
            sp=sp,
            text_path=args.text,
            chunk_chars=args.chunk_chars,
        )
    elif args.shard_glob or args.shard:
        shard_files = resolve_shard_files(
            shard_glob=args.shard_glob,
            shard_list=args.shard,
        )
        sequences = iter_token_ids_from_shards(
            files=shard_files,
            shard_limit=args.shard_limit,
            max_tokens=args.max_tokens,
        )
    else:
        sequences = iter_token_ids_from_file(args.token_ids)

    pair_counts, sequence_count, token_count = count_bigrams(sequences)
    bucket_counts = [args.num_buckets]
    for num_buckets in sweep_bucket_counts:
        if num_buckets not in bucket_counts:
            bucket_counts.append(num_buckets)
    results = [
        build_probe_result(
            pair_counts=pair_counts,
            sequence_count=sequence_count,
            token_count=token_count,
            num_buckets=num_buckets,
            prev_mult=args.prev_mult,
            curr_mult=args.curr_mult,
        )
        for num_buckets in bucket_counts
    ]

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    written_paths: list[str] = []
    if len(results) > 1:
        out_sweep = f"{args.out_prefix}.bucket_sweep.tsv"
        write_sweep_summary_tsv(results=results, out_path=out_sweep)
        written_paths.append(out_sweep)

    for result in results:
        suffix = "" if len(results) == 1 else f".buckets_{result.num_buckets}"
        out_summary = f"{args.out_prefix}{suffix}.summary.txt"
        out_pairs = f"{args.out_prefix}{suffix}.top_pairs.tsv"
        out_bucket_txt = f"{args.out_prefix}{suffix}.bucket_report.txt"
        out_bucket_tsv = f"{args.out_prefix}{suffix}.bucket_summary.tsv"
        write_summary_report(
            stats=result.stats,
            num_buckets=result.num_buckets,
            out_path=out_summary,
        )
        write_top_pairs_report(
            sp=sp,
            pair_counts=pair_counts,
            num_buckets=result.num_buckets,
            prev_mult=args.prev_mult,
            curr_mult=args.curr_mult,
            out_path=out_pairs,
            top_pairs=args.top_pairs,
        )
        write_bucket_report(
            sp=sp,
            buckets=result.buckets,
            out_path=out_bucket_txt,
            top_buckets=args.top_buckets,
            top_pairs_per_bucket=args.top_pairs_per_bucket,
        )
        write_bucket_summary_tsv(
            buckets=result.buckets,
            out_path=out_bucket_tsv,
        )
        written_paths.extend([out_summary, out_pairs, out_bucket_txt, out_bucket_tsv])

    for path in written_paths:
        print(f"Wrote: {path}")
    print(f"Sequences: {sequence_count}")
    print(f"Tokens: {token_count}")
    print(f"Observed unique bigrams: {len(pair_counts)}")
    for result in results:
        print(
            f"Buckets {result.num_buckets}: "
            f"non-empty usable buckets {result.stats.non_empty_buckets} / {result.num_buckets - 1}, "
            f"load {result.stats.mean_pairs_per_non_empty_bucket:.2f} unique pairs per non-empty bucket"
        )
        print(f"Reserved first-token bucket id ({result.num_buckets}): {result.num_buckets - 1}")


if __name__ == "__main__":
    main()
