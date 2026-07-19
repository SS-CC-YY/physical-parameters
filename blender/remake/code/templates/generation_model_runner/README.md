# Frozen-manifest generation runner template

This directory is a source-controlled starting point for people generating the
same frozen benchmark with another local I2V model.  It is deliberately
independent of the benchmark exporter and evaluator.  Copy the directory for a
model, edit only the two `TODO(PROVIDER)` functions in `run_model.py`, and keep
the canonical manifest, prompts, input images, seeds, and job IDs unchanged.

The template uses only the Python standard library.  The selected provider may
of course import its own model dependencies inside `load_model_once()`.

## Provider work

Implement:

1. `load_model_once(config)`: import the official model package and load the
   model/checkpoint exactly once.  Return any in-memory provider state.
2. `generate_video(model, job, input_image, output_video, config)`: generate
   one I2V result and write a non-empty MP4 to the supplied temporary
   `output_video` path.  Return a JSON-compatible mapping of actual generation
   parameters and native media information if available.

The returned mapping should fill these audit fields (use `null`/`false` when a
provider cannot honor one):

```text
provider_prompt, provider_negative_prompt, prompt_transform_id
seed_supported, seed_sent, effective_seed
actual_generation, media_probe, postprocess
```

`media_probe` should contain the actual codec, width, height, FPS, frame count,
duration, and time base obtained from the produced file, not copied from the
nominal manifest request.

Do not change `job["prompt"]`, `job["negative_prompt"]`, `job["seed"]`, or the
conditioning image.  If the provider must transform a prompt or cannot honor a
seed/requested media setting, record that fact explicitly in the returned
metadata.

The common loop already provides:

- streaming JSONL manifest consumption;
- safe resolution of `inputs.image` below the bundle root;
- canonical `videos/<job_id>.mp4` naming;
- model loading once per process;
- atomic video and metadata publication;
- per-job logs and append-only `run_state.jsonl`;
- failure isolation, summary output, `--max-jobs`, and `--resume`.

## Bundle layout

Run against a frozen handoff bundle with this minimum layout:

```text
bundle/
├── manifest.jsonl
└── first_frames_v1_0/
    └── images/...
```

Every relative `inputs.image` in `manifest.jsonl` is resolved from `bundle/`.
Absolute paths and paths escaping this root are rejected.

## Configure and run

Copy and edit the example without adding credentials:

```bash
cp model_config.example.json model_config.json
```

Then run a small smoke test:

```bash
python run_model.py \
  --bundle-root /absolute/path/to/bundle \
  --output-root /absolute/path/to/outputs/helios_smoke \
  --config model_config.json \
  --max-jobs 2
```

Resume the same run after interruption:

```bash
python run_model.py \
  --bundle-root /absolute/path/to/bundle \
  --output-root /absolute/path/to/outputs/helios_full \
  --config model_config.json \
  --resume
```

`--resume` skips a job only when both its non-empty MP4 and successful metadata
exist.  Incomplete jobs are regenerated to temporary files and atomically
replace the incomplete canonical result.

## Output contract

```text
output-root/
├── videos/<job_id>.mp4
├── metadata/<job_id>.json
├── logs/<job_id>.log
├── run_state.jsonl
└── run_summary.json
```

After a full run, validate it against the frozen manifest:

```bash
python code/scripts/validate_generation_outputs.py \
  --manifest /absolute/path/to/bundle/manifest.jsonl \
  --videos /absolute/path/to/output-root/videos \
  --metadata /absolute/path/to/output-root/metadata \
  --expected-jobs 978 \
  --ffprobe required \
  --report /absolute/path/to/output-root/validation_report.json
```

## Credential rule

This template is for locally installed models and never reads, logs, or writes
an API key.  Secret-like config keys (`api_key`, `token`, `password`, and
similar names) are rejected.  Download gated weights separately before running
the benchmark.  Never put credentials in the JSON config, command line,
manifest, metadata, or logs.
