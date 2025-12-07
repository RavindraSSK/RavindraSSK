# Shadow Removal for Aerial Images: VS Code Insiders Setup Guide

This guide walks through preparing a shadow-removal project focused on aerial/tree shadows using attention-based methods. It assumes you will iterate locally in VS Code Insiders and later push to Git.

## 1) Quick-start checklist (do this in order inside VS Code Insiders)
1. **Clone + open folder**: `git clone <your-repo>` → **File > Open Folder** in VS Code Insiders → trust the workspace.
2. **Create env** (Terminal inside VS Code):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```
3. **Install deps** (start minimal, add more later):
   ```bash
   pip install torch torchvision lightning opencv-python albumentations==1.3.1
   pip install einops timm tqdm matplotlib tensorboard
   ```
4. **Select interpreter**: `Ctrl/Cmd+Shift+P` → `Python: Select Interpreter` → pick `.venv`.
5. **Recommended extensions**: Python, Pylance, Jupyter, GitLens, (optional) TensorBoard.
6. **Create folders** (integrated terminal):
   ```bash
   mkdir -p data/raw data/masks data/processed models/checkpoints src notebooks outputs
   ```
7. **Drop sample aerial images** into `data/raw/` (PNG/JPG). If you have masks, match filenames in `data/masks/`.
8. **Add `.gitignore`** for `data/`, `models/`, `.venv/`, `outputs/` before committing.

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
