# Course Lab R33-T006: Bounded CPU Smoke Run (Tying Weights Autoencoder)

**Student:** NAKSHATRA MESHRAM  
**Task:** R33-T006 (6 of 10)  
**Notebook Item:** `17_autoencoders_gans_and_diffusion_models.ipynb`  
**Target Section:** `## Tying weights`  
**Repository:** [S-AI-Projects/handson-ml3](https://github.com/S-AI-Projects/handson-ml3)

---

## 1. Overview & Objective

In Chapter 17 (*Autoencoders, GANs, and Diffusion Models*), the **Tying weights** section demonstrates autoencoders where the decoder's weights are tied to the encoder's weights ($W_{decoder} = W_{encoder}^T$).

This lab provides an isolated, deterministic, CPU-only smoke test that validates:
1. **Weight Tying Contract:** The decoder has no independent weight parameters; it strictly references the transpose of the encoder weight matrix ($W_{dec} \equiv W_{enc}^T$).
2. **Execution Bounds:** Enforces strict execution limits: at most 32 samples, a fixed random seed, and at most 1 epoch / 20 update steps.
3. **Finite Interface Correctness:** Asserts finite outputs and losses, valid prediction dimensions, and non-zero step counts.
4. **Baseline Comparison:** Provides an explicitly labelled naive constant baseline model (`NaiveConstantModel`, `is_baseline=True`).

> **Notice:** The synthetic fixture and results in this lab are intended strictly for **pipeline verification** and interface smoke testing. They do not constitute evidence of real-world or generative accuracy.

---

## 2. Weight Tying Contract & Math

For an input vector $x \in \mathbb{R}^{D}$ (here $D = 16$ for $4 \times 4$ images) and latent dimension $K = 4$:
- **Encoder:**
  $$z = x W_{enc} + b_{enc}, \quad W_{enc} \in \mathbb{R}^{D \times K}, \; b_{enc} \in \mathbb{R}^{K}$$
- **Decoder (Tied Weights):**
  $$\hat{x} = z W_{dec} + b_{dec} = z W_{enc}^T + b_{dec}, \quad W_{dec} \equiv W_{enc}^T \in \mathbb{R}^{K \times D}, \; b_{dec} \in \mathbb{R}^{D}$$
- **Analytical Gradient Update:**
  $$\frac{\partial \mathcal{L}}{\partial W_{enc}} = \frac{\partial \mathcal{L}}{\partial W_{enc}}\Big|_{encoder} + \left(\frac{\partial \mathcal{L}}{\partial W_{dec}}\Big|_{decoder}\right)^T = X^T dZ + d\hat{X}^T Z$$
  Both encoder and decoder gradient signals update $W_{enc}$ simultaneously in each update step.

---

## 3. Synthetic Fixture Specification

The fixture contains six deterministic synthetic $4 \times 4$ grayscale images:
- **`img_001`, `img_002`, `img_003`** (Group `g_zero`): All 16 pixels are 0, labelled `0`.
- **`img_004`, `img_005`, `img_006`** (Group `g_diag`): White main diagonal ($p_{0,0} = p_{1,1} = p_{2,2} = p_{3,3} = 255$, all others 0), labelled `1`.

### Schema:
- `record_id`: Unique identifier (`img_001` .. `img_006`)
- `group_id`: Group identifier (`g_zero`, `g_diag`)
- `split`: Deterministic partition (`train`, `validation`, `test`)
- `pixels`: 16 flat integers in $[0, 255]$
- `pixels_2d`: $4 \times 4$ integer matrix in $[0, 255]$
- `label`: Binary class (`0` = all zero, `1` = white diagonal)

Both `fixture.json` and `fixture.csv` are provided with byte-level parity verification.

---

## 4. Execution Limits & Failure Modes

| Limit | Enforced Threshold | Violation Behavior |
|---|---|---|
| Sample Count | $\le 32$ samples | Raises `ValueError` |
| Step Count | $\le 20$ update steps | Raises `ValueError` |
| Epoch Count | $\le 1$ epoch | Enforced by controller |
| Device | CPU-only | Standard-library implementation; no CUDA/MPS/GPU |
| Network | Offline | 0 network calls, 0 downloads |

---

## 5. Verification Commands

Run unit test discovery:
```bash
python3 -m unittest discover -s course-labs/r33-t006 -p "test_*.py" -v
```

Run comprehensive verification runner:
```bash
python3 work/R33-T006/run_verification.py
```
