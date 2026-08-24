# Ollama GGUF Export

The export pipeline converts a trained LoRA/QLoRA adapter into a local Ollama
model:

1. Merge the PEFT adapter into its Hugging Face base model.
2. Convert the merged model to a high-precision GGUF with llama.cpp.
3. Quantize the GGUF with `llama-quantize`.
4. Write an Ollama `Modelfile` and export metadata.
5. Register and run the model through the Ollama container.

Docker is the default interface. The `main` image contains the project training
dependencies, CUDA-enabled PyTorch, and a built llama.cpp checkout at
`/home/user/workspace/llama.cpp`.

## Export With Docker

Start the validated stack if it is not already running:

```bash
docker/stack.sh up
```

Run the complete export after training:

```bash
docker/stack.sh main python scripts/export_agent_ollama.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3.5_4b.yaml \
  --stages all
```

The stages run in this order:

```text
merged adapter -> merged-hf -> bf16 GGUF -> quantized GGUF -> Modelfile
```

Each model gets an isolated export directory:

```text
outputs/ollama/
`-- <model_name>/
    |-- merged-hf/
    |-- <model_name>.bf16.gguf
    |-- <model_name>.<quantization>.gguf
    |-- Modelfile
    `-- Modelfile.json
```

The Ollama service sees the same directory at `/models/<model_name>`.

### Selected Stages

`--stages` accepts `merge`, `convert`, `quantize`, `modelfile`, or `all`.
Multiple stages can be provided in dependency order:

```bash
docker/stack.sh main python scripts/export_agent_ollama.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3.5_4b.yaml \
  --stages convert quantize modelfile
```

Use selected stages when an earlier artifact is already valid. For example,
changing quantization does not require repeating the model merge.

Generate only the Modelfile and metadata:

```bash
docker/stack.sh main python scripts/export_agent_ollama.py \
  --data-config configs/data.yaml \
  --stages modelfile
```

Use `--dry-run` to print external actions without writing files or running the
heavy merge.

### Multiple Quantizations

Generate more than the configured default quantization:

```bash
docker/stack.sh main python scripts/export_agent_ollama.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3.5_4b.yaml \
  --stages all \
  --quantizations Q4_K_M Q5_K_M
```

Every requested quantization is written into the same model export directory.
The first quantization is selected by default for the Modelfile.

## Register And Run

The export script writes artifacts but does not run `ollama create`. Register
the selected GGUF after export:

```bash
docker/stack.sh ollama create qwen3.5-4b-3dsg-cypher-agent-lora \
  -f qwen3.5-4b-3dsg-cypher-agent-lora/Modelfile
```

Run the registered model:

```bash
docker/stack.sh ollama run qwen3.5-4b-3dsg-cypher-agent-lora \
  "List the objects in room R0."
```

The generated Modelfile uses `FROM ./<model>.gguf`. Ollama resolves that path
relative to the Modelfile, so the nested model directory works from both the
host and the `/models` container mount.

## Configuration

Export settings live under `ollama_export` in `configs/data.yaml`:

- `model_name`: Ollama model name and export directory name.
- `export_root`: parent directory; defaults resolve under
  `<export_root>/<model_name>`.
- `gguf_outtype`: converter output type, usually `bf16` or `f16`.
- `quantizations`: llama.cpp quantizations to generate.
- `selected_quantization`: optional quantization used by the Modelfile.
- `convert_args`: additional arguments for `convert_hf_to_gguf.py`.
- `quantize_args`: additional arguments for `llama-quantize`.
- `target_hardware`: deployment note written to export metadata.
- `system_prompt`: system prompt embedded in the Modelfile.

`merged_model_dir`, `base_gguf_path`, `gguf_path`, and `modelfile_path` are
optional explicit overrides. Omit them for normal use so each model remains
grouped under its own directory.

The student config supplies:

- `model.base_model`: Hugging Face base model used for the merge.
- `model.tokenizer`: optional tokenizer override.
- `model.trust_remote_code`: forwarded to Transformers.
- `training.output_dir`: default LoRA adapter directory.

## Qwen3.5 MTP Metadata

Some merged Qwen3.5 checkpoints retain `mtp_num_hidden_layers` in
`config.json` without containing MTP tensors. A GGUF converted with bundled MTP
metadata can then fail at runtime with a missing final block such as:

```text
missing tensor 'blk.32.attn_norm.weight'
```

Exclude the absent MTP head during conversion:

```yaml
ollama_export:
  convert_args:
    - --no-mtp
```

Then rerun `convert quantize modelfile` and register the model again. The merge
does not need to be repeated.

## Host-Only Export

Host execution is optional. Install this package's export dependencies in the
host project environment:

```bash
pip install -e '.[export]'
```

Install a PyTorch build matching the host CUDA or JetPack environment
separately.

Install llama.cpp and keep its converter dependencies isolated:

```bash
git clone https://github.com/ggml-org/llama.cpp external/llama.cpp
python3 -m venv external/llama.cpp/.venv
external/llama.cpp/.venv/bin/python -m pip install -U pip
external/llama.cpp/.venv/bin/python \
  -m pip install -r external/llama.cpp/requirements.txt
cmake -S external/llama.cpp -B external/llama.cpp/build
cmake --build external/llama.cpp/build --config Release -j
export LLAMA_CPP_DIR=external/llama.cpp
```

Run the host export:

```bash
.venv/bin/python scripts/export_agent_ollama.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3.5_4b.yaml \
  --stages all \
  --llama-cpp-dir external/llama.cpp
```

When `<llama-cpp-dir>/.venv/bin/python` exists, the exporter uses it for the
conversion subprocess. Override it with `--python-bin` only when another
interpreter is required.

Register from a host Ollama installation by running from the exported model
directory:

```bash
cd outputs/ollama/qwen3.5-4b-3dsg-cypher-agent-lora
ollama create qwen3.5-4b-3dsg-cypher-agent-lora -f Modelfile
ollama run qwen3.5-4b-3dsg-cypher-agent-lora \
  "List the objects in room R0."
```
