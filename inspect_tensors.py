from pathlib import Path
import torch
import train_gpt as tg

def tensor_nbytes(t):
    return int(t.numel()) * int(t.element_size())

args = tg.Hyperparameters()

model = tg.GPT(
    vocab_size=args.vocab_size,
    num_layers=args.num_layers,
    model_dim=args.model_dim,
    num_heads=args.num_heads,
    num_kv_heads=args.num_kv_heads,
    mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings,
    tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap,
    rope_base=args.rope_base,
    qk_gain_init=args.qk_gain_init,
)

# Match baseline grouping logic
block_named_params = list(model.blocks.named_parameters())
matrix_params = {
    id(p): name
    for name, p in block_named_params
    if p.ndim == 2 and not any(pattern in name for pattern in tg.CONTROL_TENSOR_NAME_PATTERNS)
}
scalar_params = {
    id(p): name
    for name, p in block_named_params
    if p.ndim < 2 or any(pattern in name for pattern in tg.CONTROL_TENSOR_NAME_PATTERNS)
}
if model.skip_weights.numel() > 0:
    scalar_params[id(model.skip_weights)] = "skip_weights"

rows = []
state_dict = model.state_dict()

for name, p in model.named_parameters():
    if name == "tok_emb.weight":
        opt_group = "token_embedding"
    elif name == "lm_head.weight":
        opt_group = "lm_head"
    elif id(p) in matrix_params:
        opt_group = "block_matrix_muon"
    elif id(p) in scalar_params:
        opt_group = "block_scalar_adam"
    else:
        opt_group = "other"

    keep_float = False
    keep_reason = ""
    if not p.is_floating_point():
        keep_reason = "non_float_passthrough"
    elif p.numel() <= tg.INT8_KEEP_FLOAT_MAX_NUMEL:
        keep_float = True
        keep_reason = f"small_float<= {tg.INT8_KEEP_FLOAT_MAX_NUMEL}"
    elif any(pattern in name for pattern in tg.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        keep_float = True
        keep_reason = "name_pattern_keep_float"

    if keep_reason:
        export_mode = f"keep_float ({keep_reason})"
    else:
        if p.ndim == 2:
            export_mode = "quantize_int8_per_row"
        else:
            export_mode = "quantize_int8_per_tensor"

    rows.append({
        "name": name,
        "shape": tuple(p.shape),
        "ndim": p.ndim,
        "numel": p.numel(),
        "dtype": str(p.dtype),
        "bytes": tensor_nbytes(p),
        "optimizer_group": opt_group,
        "export_mode": export_mode,
    })

rows.sort(key=lambda r: r["bytes"], reverse=True)

total_bytes = sum(r["bytes"] for r in rows)
print(f"Total parameter bytes (current dtype): {total_bytes:,}")
print("=" * 140)
print(f"{'bytes':>12}  {'numel':>12}  {'ndim':>4}  {'optimizer_group':<20}  {'export_mode':<30}  name")
print("=" * 140)
for r in rows:
    print(
        f"{r['bytes']:12,d}  {r['numel']:12,d}  {r['ndim']:4d}  "
        f"{r['optimizer_group']:<20}  {r['export_mode']:<30}  {r['name']} {r['shape']}"
    )

print("\nTop tensors by bytes:")
for r in rows[:20]:
    print(f"{r['bytes']:12,d}  {r['name']} {r['shape']}  -> {r['export_mode']}")

# Group summaries
from collections import defaultdict
by_opt = defaultdict(int)
by_export = defaultdict(int)
for r in rows:
    by_opt[r["optimizer_group"]] += r["bytes"]
    by_export[r["export_mode"]] += r["bytes"]

print("\nBytes by optimizer group:")
for k, v in sorted(by_opt.items(), key=lambda kv: kv[1], reverse=True):
    print(f"{v:12,d}  {k}")

print("\nBytes by export mode:")
for k, v in sorted(by_export.items(), key=lambda kv: kv[1], reverse=True):
    print(f"{v:12,d}  {k}")