## onnx-conv2matmul

`onnx-conv2matmul` is a pre-quantization rewrite tool for ONNX models.

It converts compatible `Conv` 1x1 layers into equivalent `MatMul` subgraphs so your
quantization stack can treat more model weights with `MatMul`-focused kernels
(for example int4 weight-only flows).

Why this matters in practice:

- Many backends optimize/quantize `MatMul` more aggressively than `Conv`
- Pointwise convs often dominate encoder blocks and are good candidates for rewrite
- You can reduce model size and improve throughput potential without changing model semantics

The tool is conservative by default, keeps unsupported layers untouched, and emits a
detailed report of converted vs skipped nodes with explicit reasons.

### Install

From PyPI:

```bash
pip install onnx-conv2matmul
```

### Basic Usage

**Rewrite to a new file:**

```bash
onnx-conv2matmul input.onnx output.onnx
```

**Overwrite in-place:**

```bash
onnx-conv2matmul input.onnx --inplace
```

**Rewrite and verify in one command (CPU):**

```bash
onnx-conv2matmul input.onnx output.onnx \
	--extended-conv1x1 \
	--verify \
	--verify-deterministic-cpu \
	--verify-lengths 64,80,97,128,191
```

### Parakeet Example (Pre-Quantization)

Reference model page:

- https://huggingface.co/efederici/parakeet-tdt-0.6b-v3-onnx-int4

Typical pre-quantization step for the encoder graph:

```bash
onnx-conv2matmul encoder-model.onnx encoder-model.preq.onnx \
	--extended-conv1x1 \
	--allow-non-unit-dilation \
	--max-dilation 4 \
	--skip-checker \
	--report-json encoder-model.preq.report.json
```

This converts compatible pointwise conv layers (`Conv1D k=1` and `Conv2D 1x1`) to `MatMul` and writes a JSON report
with converted/skipped nodes and reasons.

### Hugging Face FP32 -> Pre-Quantization Workflow

If you start from the original FP32 ONNX release:

- https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx

you can download only the files needed for encoder pre-quantization and run the rewrite.

1. Download required files from Hugging Face

```bash
pip install "huggingface_hub[cli]"
hf download istupakov/parakeet-tdt-0.6b-v3-onnx \
	--local-dir ./hf_parakeet_fp32 \
	--include encoder-model.onnx encoder-model.onnx.data config.json vocab.txt
```

2. Run pre-quantization rewrite on encoder

```bash
cd hf_parakeet_fp32
onnx-conv2matmul encoder-model.onnx encoder-model.preq.onnx \
	--extended-conv1x1 \
	--allow-non-unit-dilation \
	--max-dilation 4 \
	--skip-checker \
	--report-json encoder-model.preq.report.json
```

3. Check conversion summary

```bash
cat encoder-model.preq.report.json
```

4. Verify I/O equivalence (robust numerical check)

The CLI includes a built-in strict CPU deterministic verification. It runs inputs through both models to ensure the maximum numerical deviation remains within safe float32 bounds (`max_abs <= 3e-5`, `mean_abs <= 2e-6`).

```bash
onnx-conv2matmul encoder-model.onnx encoder-model.preq.onnx \
	--verify \
	--verify-signal-input-name audio_signal \
	--verify-length-input-name length \
	--verify-output-index 0 \
	--verify-length-output-index 1 \
	--verify-deterministic-cpu \
	--verify-lengths 64,80,97,128,191
```

If it prints `Verification PASSED`, the rewrite is numerically transparent.

Tip: the source model includes `encoder-model.onnx.data` as external weights; keep it in
the same directory as `encoder-model.onnx` while rewriting.

The CLI automatically loads external data and, when needed, writes rewritten artifacts as
`<output>.onnx` plus `<output>.onnx.data`.

### Hybrid Quantization Workflow (Aligned with Parakeet INT4 Release)

If your goal is to reproduce the published Parakeet hybrid pipeline, use this sequence:

1. Rewrite compatible pointwise conv layers (`Conv1D k=1` and `Conv2D 1x1`) to MatMul (this tool).
2. Quantize encoder linear + pointwise layers with int4 `MatMulNBits` (`block_size=64`, asymmetric).
3. Keep depthwise conv layers in FP32 (the ONNX backend manages this automatically if left unconverted).
4. Quantize decoder/joint with int8 dynamic quantization.

**Example Step 2 (INT4 Quantization in Python):**

```python
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

# Must use block_size=64 and is_symmetric=False for audio/speech models
# (like Parakeet) to avoid severe degradation in embedding Cosine Similarity.
quant = MatMulNBitsQuantizer(
    'encoder-model.preq.onnx',
    block_size=64,
    is_symmetric=False,
    accuracy_level=0
)
quant.process()

# Crucial: Save with use_external_data_format=True to de-duplicate Protobuf
# serialization overhead. This saves an extra ~20MB compared to a single file.
quant.model.save_model_to_file('encoder-model.int4.onnx', use_external_data_format=True)
```

Reference release and details:

- https://huggingface.co/efederici/parakeet-tdt-0.6b-v3-onnx-int4

Important: this repository covers step 1 (pre-quantization rewrite). The final 409 MB
hybrid artifact depends on the downstream quantization stack and settings used in steps 2-4.

#### Why this matters

`MatMulBnb4`/NF4 and `MatMulNBits` are not equivalent quantization paths. If you compare
with the wrong quantizer family, you can get size and coverage numbers that do not match
the published hybrid result.

#### Compare package size after full pipeline

After you have produced final artifacts (`encoder-model.int4.onnx`, `decoder_joint-model.int8.onnx`,
optional `nemo128.int8.onnx`, plus `vocab.txt` and `config.json`), you can measure bundle size:

```bash
python - <<'PY'
from pathlib import Path

files = [
    Path("encoder-model.int4.onnx"),
    Path("decoder_joint-model.int8.onnx"),
    Path("nemo128.int8.onnx"),
    Path("vocab.txt"),
    Path("config.json"),
]

existing = [p for p in files if p.exists()]
total = sum(p.stat().st_size for p in existing)

print("files used:")
for p in existing:
    print(f"- {p.name}: {p.stat().st_size / (1024 * 1024):.2f} MB")
print("---")
print(f"bundle total: {total / (1024 * 1024):.2f} MB")
PY
```

### Useful Options

**Enable extended guarded mode:**

Use this when you want to convert more `Conv1x1` layers than strict mode.
It enables extra safe patterns (for example explicit padding or stride > 1),
while keeping guardrails: unsupported or risky layers are skipped, not forced.

```bash
onnx-conv2matmul input.onnx output.onnx --extended-conv1x1
```

**Allow non-unit dilation (explicit opt-in):**

Enable this only if your model uses dilated `Conv1x1` and you want those layers
to be considered for rewrite too. It is opt-in because dilation can increase
conversion risk; keep it off unless you need that extra coverage.

```bash
onnx-conv2matmul input.onnx output.onnx --allow-non-unit-dilation
```

**Set a max dilation guardrail:**

```bash
onnx-conv2matmul input.onnx output.onnx --allow-non-unit-dilation --max-dilation 4
```

**Write a detailed JSON report:**

```bash
onnx-conv2matmul input.onnx output.onnx --report-json report.json
```

**Print JSON report to stdout:**

```bash
onnx-conv2matmul input.onnx output.onnx --report-json-stdout
```

**Skip ONNX checker (useful for very large/external-data models):**

```bash
onnx-conv2matmul input.onnx output.onnx --skip-checker
```

### Compatibility Rules

Strict mode rewrites only conservative cases:

- `Conv` with constant pointwise weights of shape `[M, C, 1]` (Conv1D) or `[M, C, 1, 1]` (Conv2D)
- `group=1`
- unit stride, unit dilation, zero explicit pads, `auto_pad=NOTSET`
- Optional bias is supported only when constant

Extended mode (`--extended-conv1x1`) also supports compatible pointwise conv with explicit
padding and/or stride > 1, with guardrails.

All unsupported `Conv` nodes are left unchanged and reported with a reason.
