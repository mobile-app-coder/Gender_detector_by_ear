# Ear Gender Identification (Ear Model)

This repo now focuses on four key assets:
1. **`dataset/`** – small balanced male/female ear crops for classical CNN baselines.
2. **`Images/`** – the full EarVN10 collection (IDs 001–164) used for large‑scale experiments.
3. **`ear.ipynb`** – the only notebook you need for preprocessing, training, evaluation, and Grad‑CAM visualisation.
4. **`ear_model.pth`** – the latest PyTorch checkpoint (MobileNetV2 backbone + attention head) trained on the combined datasets.

Everything else was intentionally removed to keep the project lean and storage‑friendly.

---

## 1. Folder Structure
```
ear model/
├── dataset/
│   ├── male/*.jpg
│   └── female/*.jpg
├── Images/
│   └── 001.ALI_HD/, …, 164.Yen_Nhi_H/  # EarVN10 folders
├── ear.ipynb
├── ear_model.pth
└── README.md
```
- `dataset/` is ideal for quick PyTorch prototyping (128×128 crops, labelled by folder name).
- `Images/` mirrors the EarVN10GenderDataset: IDs **001–098 → male (0)**, **099–164 → female (1)**.
- `ear.ipynb` contains all remaining source code (dataset loader, augmentations, model definition, training loops, metrics, Grad‑CAM).

---

## 2. Getting Started
1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # .\.venv\Scripts\activate on Windows
   pip install torch torchvision opencv-python pillow numpy matplotlib
   ```

2. **Open `ear.ipynb`**
   - Update the data paths at the top of the notebook if your folder layout differs.
   - Run the preprocessing cells to generate train/val splits (uses stratified sampling).
   - Execute the training section (defaults: MobileNetV2 encoder, attention pooling, BCE loss, Adam optimizer).

3. **Checkpoint saving**
   - The notebook saves the best weights to `ear_model.pth`.
   - You can change the filename/location from inside the notebook if needed.

---

## 3. Using `ear_model.pth` for Inference
Below is the minimal snippet mirrored from the notebook. Keep the architecture identical to ensure the checkpoint loads.

```python
import torch
from torchvision import models, transforms
from PIL import Image

class EarNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.mobilenet_v2(weights=None)
        self.encoder.classifier = torch.nn.Identity()
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1280, 2)
        )

    def forward(self, x):
        feats = self.encoder(x)
        return self.head(feats)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = EarNet().to(device)
model.load_state_dict(torch.load("ear_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img = transform(Image.open("sample_ear.jpg")).unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(img)
    prob = torch.softmax(logits, dim=1)[0, 1].item()
    print(f"Female probability: {prob:.2%}")
```

---

## 4. Working with the Datasets
- **`dataset/` workflow**
  - Ideal for quick experiments; simply point the notebook to this folder.
  - Augmentations: random horizontal flip, ±10° rotation, CLAHE for illumination variance.
  - Recommended batch size: 32, image size: 128×128.

- **`Images/` workflow (EarVN10)**
  - Heavier but more diverse. Each subfolder hosts ~150–250 images of one subject.
  - The notebook automatically maps folder IDs (001–164) to labels.
  - Suggested batch size: 16, image size: 224×224, use cosine LR schedule over 30–50 epochs.

---

## 5. Results Summary (latest notebook run)
| Metric             | Value | Notes                                  |
|--------------------|------:|----------------------------------------|
| Val Accuracy       | 93%   | EarVN10 split 80/20                    |
| Val F1 (female)    | 0.92  | Weighted by class frequency            |
| Inference latency  | 6 ms  | Apple M2, batch size = 1               |
| Model size         | 13 MB | `ear_model.pth` (float32, no quant)    |

*Numbers will shift with different random seeds or augmentation mixes.*

---

## 6. Maintenance Notes
- Old helper scripts (`ear.py`, `tensor.py`, `grok.py`, etc.) were deleted.
- If you need reusable modules again, export cells from the notebook or convert it to `.py` via `jupyter nbconvert`.
- Keep large datasets out of version control; this README is the single source of truth for setup.

---

## 7. Ethics & Usage
- Treat ear biometrics as sensitive personal data.
- Obtain explicit consent for every image used.
- Do **not** deploy gender inference systems without bias, privacy, and legal reviews.

---

## 8. Contact
Need the old scripts, want to reproduce experiments, or have dataset contributions?  
→ Drop a note at `contact@earmodel.ai`.

