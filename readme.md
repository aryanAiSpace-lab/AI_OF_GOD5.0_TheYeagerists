# AI of GOD 5.0: Animal Classification from Segmentation Overlays

**Team The Yeagerists** - Aryan Raj (25JE0322) and Arnesh Upadhyaya (25JE0319)
**Result: 3rd place**

---

## 1. Problem statement

Predict which of **72 animal classes** appears in an image. Each image is a **colourised segmentation overlay**: a coloured mask drawn over the original photo. The original photo is not provided separately, though some photo detail (fur, eyes, feathers) stays faintly visible through the tint. The overlay colours are random and are not a reliable clue to the class.

- Task: multi-class image classification (72 classes)
- Metric: **macro-averaged F1** (every class counts equally)
- Platform: Kaggle competition *AI of GOD 5.0*

### Dataset summary

| Item | Value |
|---|---|
| Training images | 3,087 (labelled in `train.csv`) |
| Test images | 1,179 (unlabelled) |
| Classes | 72 (23 to 51 images per class) |
| Format | 8-bit RGB PNG, widths 145 to 4,854 px |
| Submission | CSV with columns `image_id,animal_class` |

---

## 2. Data analysis and preprocessing

| Check | Finding | Decision |
|---|---|---|
| Does image size predict the class? | Random forest on width, height and aspect ratio: 1.65% accuracy vs 1.39% for guessing | No shortcut. Resize everything the same way |
| Exact duplicate files? | 0 inside train, 0 between train and test (MD5) | Nothing to remove |
| Does test differ from train? | A train-vs-test classifier got AUC 0.49 | No domain shift |
| Colours per image | Median about 1,900 unique colours (sample of 200) | Not flat masks. Photo detail remains |

**Preprocessing pipeline**

1. Pad each image to a square with grey borders (no stretching, no cropping off the head)
2. Resize: 224 px for DINOv2, 320 px for EfficientNet
3. Normalise with the pretrained model's mean and std

**Augmentation (EfficientNet training only):** mild random crop (65 to 100% of the area), horizontal flip, hue rotation up to +/-0.5, and random shuffling of the R, G, B channels. Colour is arbitrary, so this stops the model relying on it.

**Test-time augmentation (EfficientNet):** probabilities averaged over 4 views: original, horizontal flip, RGB-to-BGR, and both.

---

## 3. Our two best solutions

### A. DINOv2 ViT-B/14 (frozen) + logistic regression

- `vit_base_patch14_dinov2.lvd142m` (timm), used as a frozen feature extractor at 224 x 224. It outputs 768 numbers per image.
- Classifier: `StandardScaler` + multinomial `LogisticRegression` (C = 1.0, max_iter = 2000), which scores all 72 classes at once with softmax.
- 5-fold stratified CV macro-F1: **0.911** (folds 0.903 to 0.921)
- Public leaderboard: **0.9231**
- Notebook: `Dinov2-b.ipynb`
- Predictions: `submission(Dinov2).csv`

### B. EfficientNetV2-S (fine-tuned)

- `tf_efficientnetv2_s.in21k_ft_in1k` (timm), input 320 x 320
- AdamW (lr 3e-4, weight decay 1e-2), one-cycle schedule, 15 epochs, batch 32, mixed precision (fp16), dropout 0.2, label smoothing 0.1
- Trained on fold 0 only (80% of the training data), best epoch chosen by validation macro-F1
- Validation macro-F1: **0.9207** (0.9206 with TTA)
- Public leaderboard: **0.9235**
- Notebook: `EfficientNetv2fold1.ipynb`
- Predictions: `submission (Efficientnet).csv`

---

## 4. Other experiments (not included in this repo)

| Experiment | Outcome |
|---|---|
| Frozen ensemble: DINOv2-L (336 px) + CLIP-L + SigLIP, with pseudo-labelling | Random-fold CV 0.977, but public score 0.898 |
| ConvNeXt fine-tune | Public score 0.8166 |
| Train-vs-test shift checks | AUC 0.49, no shift found |

**What we learned**

- Plain cross-validation matched the leaderboard for our two simple models, but not for the large ensemble. We suspect near-duplicate photos in the training set inflated its CV score. This is a hypothesis we did not fully prove.
- With about 43 images per class, a well-pretrained model with a simple head, or a careful fine-tune, beat more complex setups.
- Colour is noise here, so colour augmentation made sense.
- The same classes were hard in several models (for example rat, whale, swan, parrot, bear, pig, possum).

---

## 5. Repository structure

```
.
├── README.md
├── Dinov2-b.ipynb                       # DINOv2-B features + logistic regression
├── EfficientNetv2fold1.ipynb            # EfficientNetV2-S fine-tuning (fold 0)
├── submission(Dinov2).csv               # test predictions from model A
├── submission (Efficientnet).csv        # test predictions from model B
├── AI_of_GOD_Round2_Presentation.pptx   # Round 2 presentation
└── data/                                # NOT included, see section 6
```

---

## 6. Dataset (not included)

The dataset belongs to the competition organisers, so it is **not** in this repository. Download it from Kaggle:

1. Create or sign in to a Kaggle account.
2. Open the competition page: https://www.kaggle.com/competitions/ai-of-god-5
3. Click **Join / Late Submission** and accept the rules (you need this to download the data).
4. Download it from the **Data** tab, or with the Kaggle CLI:

```bash
pip install kaggle
# put your API token (kaggle.json) in ~/.kaggle/
kaggle competitions download -c ai-of-god-5
unzip ai-of-god-5.zip -d data/
```

You need these files in `data/`:

```
data/
├── train_images/            # 3,087 PNG images
├── test_images/             # 1,179 PNG images
├── train.csv                # image_id, animal_class
├── sample_submission.csv    # submission template
└── data_description.md
```

If the files extract into a sub-folder (for example `participant_data/`), use that folder as the path below.

---

## 7. How to run

### Clone the repository

```bash
git clone https://github.com/aryanAiSpace-lab/AI_OF_GOD5.0_TheYeagerists.git
cd AI_OF_GOD5.0_TheYeagerists
```

### Install requirements

Python 3.10 or newer, and a GPU is strongly recommended for the EfficientNet notebook.

```bash
pip install torch torchvision timm scikit-learn pandas numpy pillow matplotlib jupyter
```

An internet connection is needed the first time, to download the pretrained weights through `timm`.

### Run

1. Open a notebook: `jupyter notebook Dinov2-b.ipynb` (or `EfficientNetv2fold1.ipynb`).
2. **Set the data path.** Both notebooks were written on Kaggle, where the data was under `/kaggle/input/competitions/ai-of-god-5/...`. In the first cells, change the data directory variable (for example `DATA`) to your local `data/` folder.
3. Run all cells. The notebook writes a `submission.csv` with columns `image_id,animal_class`.

The notebooks were run on a **Kaggle T4 GPU**. Running on a CPU works for the DINOv2 notebook but is slow, and it is not practical for EfficientNet training.

---

## 8. Notes

- Both models were validated on held-out data before any submission.
- Public leaderboard scores use about 54% of the test set, and the final ranking uses the rest.
- This repository contains our code and predictions only. The competition data stays with the organisers and must be downloaded separately.