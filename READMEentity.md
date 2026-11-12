# Part 1: Entity identification

The **entity identification** part of our architecture: given a document and a list of
entity labels (e.g. *person, organization, location*), it finds every entity mention and
its type. Coreference and relation extraction build on its output.

| Notebook | Contents |
|---|---|
| `01_entity_model.ipynb` | the whole architecture: settings, helpers, dataset loaders, the model, training |
| `02_tests.ipynb` | unit tests, real-data check, smoke test |
| `03_evaluation.ipynb` | training curve, threshold tuning, test scores, reworded labels, demo |

Notebooks 02 and 03 load the model code from notebook 01, so keep all three in the
same folder.

## How to train the model

Training takes about 20 minutes on an NVIDIA GPU (10 epochs on Re-DocRED). It creates
`entity_identifier.pt` (the trained model, about 430 MB) and `entity_identifier_log.json`
(scores per epoch).

### 1. Install (once)

You need Python, an **NVIDIA GPU**, and VS Code with the **Jupyter** extension.

1. Install PyTorch **with GPU support**: on https://pytorch.org/get-started/locally/ choose
   *Stable → your OS → Pip → Python → CUDA (highest number)* and run the command it shows.
   Use `python -m pip ...` instead of `pip ...`, so it installs into the right Python.
2. Install the rest:
   ```
   python -m pip install transformers matplotlib ipykernel
   ```
3. Check the GPU works:
   ```
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   It must print `True`. If it prints `False` although you followed step 1, uninstall
   PyTorch first (`python -m pip uninstall -y torch torchvision torchaudio`) and repeat step 1:
   a CPU-only version installed earlier is otherwise kept.

### 2. Get the code and data

1. Download this repository (green **Code** button → **Download ZIP**, or `git clone`).
2. Get the team's `data.zip` (not on GitHub) and unzip it so the folders look like this:
   ```
   entity_extraction/
   ├── 01_entity_model.ipynb
   ├── 02_tests.ipynb
   ├── 03_evaluation.ipynb
   └── data/raw/REDocRed/train_revised.json, dev_revised.json, test_revised.json
   ```
   `data` may also sit one or two folders higher; the notebooks find it.

### 3. Run the tests: `02_tests.ipynb`

Open it in VS Code, click **Select Kernel** (top right) and pick the Python you
installed PyTorch into, then **Run All**. The first run downloads BERT (about 430 MB).
Everything should print `passed` or `[ok]`, ending with *All smoke tests passed*.

### 4. Train: `01_entity_model.ipynb`

Open it, select the same kernel, **Run All**. With no `entity_identifier.pt` in the
folder, it trains automatically. Keep the computer awake and VS Code open.

A healthy run looks like this:
- `GPU available: True`
- dev F1 near 0 **before training** (normal: the model starts random)
- the loss falling during epoch 1
- after epoch 1, dev F1 already around 85, and predicted mentions close to gold mentions
- dev F1 levelling off around 90 by epoch 10, then `Done.`

Our reference run: dev F1 **90.26**, test F1 **89.93**. Your numbers may differ slightly,
since GPUs do not compute in exactly the same order.

### 5. Evaluate: `03_evaluation.ipynb`

**Run All**. It loads the trained model and shows the training curve, threshold tuning
on the dev set, the test scores, the reworded-label check and a demo on your own text.

## Settings

All settings are in the **Settings** cell of `01_entity_model.ipynb` (notebooks 02 and 03
use the same ones):

| Setting | Use |
|---|---|
| `RETRAIN = True` | train again although `entity_identifier.pt` exists |
| `QUICK = True` | 1-epoch trial on 40 documents, to test the setup in a few minutes |
| `BATCH_SIZE = 2` | if training stops with `CUDA out of memory` |
| `MODEL_NAME`, `CHECKPOINT` | another backbone, e.g. `"microsoft/deberta-v3-base"` with `"entity_identifier_deberta.pt"` (also `pip install sentencepiece protobuf`) |
| `SEED` with a new `CHECKPOINT` | train repeat runs with different random seeds |
| `THRESHOLD` | confidence threshold; set it to the best value from notebook 03 |
| `DATASETS` | `["redocred"]` by default; `"scierc"` and `"dwie"` are also supported |

## Troubleshooting

| Message | Fix |
|---|---|
| `No module named 'torch'` | the notebook's kernel is a different Python: select the kernel you installed into |
| `GPU available: False` | install the CUDA build of PyTorch (step 1) |
| `CUDA out of memory` | `BATCH_SIZE = 2` |
| `No data/raw folder found` | unzip `data.zip` next to the notebooks (step 2) |
| `No trained model at ...` | run `01_entity_model.ipynb` first |
| yellow `UNEXPECTED` lines or *Token indices sequence length is longer than 512* | harmless: unused BERT parts, and long documents are split into 512-token windows automatically |

## For coreference and relation extraction

- `model.predict(docs)` returns the kept mentions and one 2304-number vector per mention
  (`mention_vectors`).
- During training, use gold mentions: `model.represent_spans(word_vectors, gold_spans)`
  works for spans of any width.
- Relation labels can reuse the shared label encoder: `model.encode_label_text([...])`.
- For joint training, add your losses to `model(docs)["loss"]`.

Section 6 of `03_evaluation.ipynb` shows each of these.

## Not on GitHub

- `entity_identifier.pt`: too large for GitHub (over 100 MB); everyone trains their own,
  or shares it via Google Drive.
- `data/`: shared separately as `data.zip`.

Add a `.gitignore` containing `*.pt`, `data/` and `.ipynb_checkpoints/`.
