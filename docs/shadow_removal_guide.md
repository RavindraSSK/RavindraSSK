# Shadow Removal for Aerial Images: VS Code Insiders Setup Guide

This guide walks through preparing a shadow-removal project focused on aerial/tree shadows using attention-based methods. It assumes you will iterate locally in VS Code Insiders and later push to Git.

## 1) Quick-start checklist (run these exact steps in VS Code Insiders)
1. **Clone + open**: `git clone https://github.com/RavindraSSK/MY_WORK.git` (or your fork) → **File > Open Folder** in VS Code Insiders → click **Trust**.
2. **Integrated terminal**: `Ctrl+`` (backtick) to open a shell at repo root for every command below.
3. **Create env**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```
4. **Install deps** (copy/paste together):
   ```bash
   pip install torch torchvision lightning opencv-python albumentations==1.3.1 \
       einops timm tqdm matplotlib tensorboard
   ```
5. **Pick interpreter**: `Ctrl/Cmd+Shift+P` → `Python: Select Interpreter` → choose `.venv`.
6. **Extensions to install**: Python, Pylance, Jupyter, GitLens, (optional) TensorBoard.
7. **Create folders**:
   ```bash
   mkdir -p data/raw data/masks data/processed models/checkpoints src notebooks outputs
   ```
8. **Drop aerial samples** into `data/raw/` (PNG/JPG). If you have masks, match basenames in `data/masks/`.
9. **Add `.gitignore`** entries for `data/`, `models/`, `.venv/`, `outputs/` before committing.

## 1b) One-pass walkthrough in VS Code Insiders (local test, no Git required)
Follow this if you just want to verify the pipeline on your machine before pushing anything.

1. **Open folder**: In VS Code Insiders, press `Ctrl/Cmd+K` then `Ctrl/Cmd+O` and pick your project folder.
2. **Terminal**: Press ``Ctrl+` `` to open the integrated terminal at the project root.
3. **Fresh env + deps** (copy/paste as one block):
   ```bash
   python -m venv .venv && source .venv/bin/activate && \
   python -m pip install --upgrade pip && \
   pip install torch torchvision lightning opencv-python albumentations==1.3.1 einops timm tqdm matplotlib tensorboard
   ```
4. **Interpreter selection**: `Ctrl/Cmd+Shift+P` → `Python: Select Interpreter` → choose `.venv`.
5. **Drop a sample image** into `data/raw/` (create the folder if missing). Optional: add a matching mask to `data/masks/`.
6. **Quick manifest**: Create `data/quick_eval.csv` with:
   ```csv
   image_path,mask_path
   data/raw/your_image.jpg,
   ```
7. **Dry-run inference** (expects you already have or will download a checkpoint):
   ```bash
   python -m src.infer --checkpoint models/checkpoints/best.ckpt \
     --input_glob "data/raw/*.jpg" --output_dir outputs/ --tile_size 512 --tile_overlap 64
   ```
   - If you do not have a checkpoint yet, skip to step 8 to train a tiny model first.
8. **Tiny training smoke test** (uses the manifest even without masks):
   ```bash
   python -m src.train --train_csv data/quick_eval.csv --val_csv data/quick_eval.csv \
     --epochs 2 --batch-size 1 --lr 2e-4 --num-workers 2 --precision 16
   ```
   - After it finishes, rerun the inference command (step 7) to generate outputs.
9. **Inspect results**: In Explorer, click the PNGs under `outputs/` to visually confirm shadows are reduced.
10. **Keep local only**: Stop here if you do not want to push—your changes and outputs stay on your machine.

## 2) Project structure (recommended scaffold)
```
project-root/
├─ data/
│  ├─ raw/               # provided aerial tiles
│  ├─ masks/             # optional shadow masks (same basename as raw)
│  └─ processed/         # train/val/test splits after preprocessing
├─ models/
│  └─ checkpoints/       # saved model weights
├─ src/
│  ├─ config.py
│  ├─ datasets.py
│  ├─ transforms.py
│  ├─ model.py
│  ├─ train.py
│  └─ infer.py
└─ notebooks/
   └─ exploration.ipynb
```

## 3) Identify good training images (shadow focus)
- In VS Code Insiders, use the built-in **Image Preview** (Explorer) to triage the provided samples in `data/raw/`.
- Prefer tiles with: (a) visible tree/structure shadows, (b) clear non-shadow reference regions of the same material (grass/road/roof), and (c) minimal motion blur.
- If a shadow mask is available, place it in `data/masks/` with the same basename (e.g., `img_001.jpg` ↔ `img_001.png`).
- Create a manifest CSV for train/val (even if masks are missing, leave the column blank):
  ```csv
  image_path,mask_path
  data/raw/img_001.jpg,data/masks/img_001.png
  data/raw/img_002.jpg,
  ```

## 4) Model choices (attention-based)
- **DeShadowFormer / MAST (Mask-Aware Shadow Transformer)**: encoder-decoder with mask conditioning; strong SOTA baseline.
- **Attention U-Net (CBAM / SE blocks on skips)**: leaner, good for limited GPUs.
- Losses: `L1`/`L2`, perceptual (VGG), edge/structure loss; weight shadow pixels higher when masks exist.

Recommended starting recipe:
- Backbone: U-Net encoder-decoder with **Swin-T** or **MobileViT** blocks in the bottleneck.
- Attention: channel + spatial attention (CBAM) on skip connections.

## 5) Minimal training loop (PyTorch Lightning sketch)
```python
# src/model.py
class ShadowRemovalNet(LightningModule):
    def __init__(self, lr=2e-4):
        super().__init__()
        self.net = build_attention_unet()
        self.crit_l1 = nn.L1Loss()
        self.crit_perc = VGGPerceptualLoss()
        self.save_hyperparameters()

    def training_step(self, batch, _):
        img, target, mask = batch
        pred = self.net(torch.cat([img, mask], dim=1) if mask is not None else img)
        loss = self.crit_l1(pred, target) + 0.1 * self.crit_perc(pred, target)
        return loss
```

**How to run training from VS Code terminal** (after creating `data/train.csv` and `data/val.csv`):
```bash
python -m src.train --train_csv data/train.csv --val_csv data/val.csv \
  --epochs 100 --batch-size 4 --lr 2e-4 --num-workers 8 --precision 16
```
Enable callbacks inside `src/train.py`: `ModelCheckpoint(monitor="val/loss")` and `LearningRateMonitor`.

**Step-by-step inside VS Code Insiders**
1. Open `notebooks/exploration.ipynb` → run a quick patch-extraction cell to sanity-check images/masks.
2. Open `src/datasets.py` → implement a `ShadowDataset` with Albumentations transforms (include `mask` in `additional_targets`).
3. Open `src/model.py` → implement `build_attention_unet()` with CBAM or Swin blocks; wire it into `ShadowRemovalNet`.
4. Open `src/train.py` → set up the `LightningDataModule`, callbacks, and Trainer flags matching the command above.
5. Open the **Run and Debug** panel → add a Python launch config pointing to `src/train.py` if you prefer debugging.
6. Run the training command from the terminal. Watch progress in **Terminal** + **TensorBoard** panels.
7. Save the best checkpoint from `models/checkpoints/`.

## 6) Augmentations (Albumentations)
- `RandomBrightnessContrast`, `HueSaturationValue`, `HorizontalFlip`, `RandomRotate90`, `RandomResizedCrop` to 512×512.
- Pass masks with `additional_targets={"mask": "mask"}` so flips/rotations stay aligned.

## 7) Evaluation & visualization
- Metrics: PSNR, SSIM (paired data); NIQE/BRISQUE + visual grids for unpaired.
- Launch TensorBoard from VS Code terminal: `tensorboard --logdir lightning_logs` → open the **TensorBoard** panel.
- In `src/infer.py`, save side-by-side `[input | prediction | target/mask]` for the provided samples.

## 8) Inference command
```bash
python -m src.infer --checkpoint models/checkpoints/best.ckpt \
  --input_glob "data/raw/*.jpg" --output_dir outputs/ --tile_size 512 --tile_overlap 64
```
Tile large aerial images and blend (Hann or average) to avoid seams.

**Fast sanity-check workflow inside VS Code**
1. Copy one aerial tile (with/without mask) into `data/raw/`.
2. Create a tiny CSV `data/quick_eval.csv` with one row pointing to that tile (mask column can be blank).
3. Run `python -m src.infer --checkpoint models/checkpoints/best.ckpt --input_glob "data/raw/*.jpg" --output_dir outputs/`.
4. Open the saved PNGs in the Explorer to visually inspect shadow removal quality.

## 9) Practical tips for the provided samples
- Thin tree-branch shadows: add edge-aware loss or Laplacian sharpening to preserve detail.
- Water/road borders: use **reflection padding** during tiling to prevent boundary halos.
- Start with patch-based training (512×512 crops) on the provided images to validate the pipeline before scaling.

## 10) Git handoff
After you confirm training runs:
```bash
git add src notebooks docs .gitignore
git commit -m "Add attention-based shadow removal scaffold"
git push origin <branch>
```

## 11) Troubleshooting
- If training diverges: reduce `lr` (1e-4 → 5e-5) and raise `weight_decay` (1e-4).
- If bright regions artifact: increase perceptual loss weight and add `SSIMLoss`.
- If GPU memory is tight: use `--accumulate_grad_batches 2` and mixed precision.

Use this as a starting template; refine with DeShadowFormer blocks once the pipeline is verified on your aerial dataset.
