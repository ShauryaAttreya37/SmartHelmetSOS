# RoadSoS

RoadSoS is a two-wheeler crash detection and head-injury risk demo built around a Physics-Informed Neural Network (PINN). It uses simulated helmet IMU data to estimate skid probability, friction loss, and injury signals such as HIC15 and BrIC.

Built for IITM Hackathon 2026.

## What It Does

The Streamlit dashboard runs a trained PINN against three simulated riding scenarios:

- Normal Riding
- Oil Patch
- Crash

For each scenario, the app displays:

- Peak acceleration from synthetic accelerometer data
- PINN-estimated effective friction coefficient
- Skid probability derived from predicted friction
- HIC15 head injury criterion
- BrIC rotational brain injury criterion
- A simulated SOS state with rider medical profile information

The model is physics-informed because training combines ordinary data loss with residual losses based on two simplified physical systems:

- Tire-road friction and lean dynamics for a two-wheeler
- Kelvin-Voigt spring-mass-damper dynamics for helmet-to-brain motion

This makes the project more than a plain sensor classifier: the model is encouraged to produce outputs that remain consistent with the governing equations used during training.

## Repository Contents

| Path | Description |
| --- | --- |
| `roadsos_dashboard.py` | Streamlit dashboard for interactive inference and visualization |
| `RoadSoS_PINN.ipynb` | Notebook containing data generation, training, physics losses, and evaluation |
| `roadsos_pinn.pt` | Trained model checkpoint used by the dashboard |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Ignore rules for environments, generated files, and future model artifacts |

## Requirements

- Python 3.10 or newer recommended
- pip
- A CPU is enough for the dashboard
- CUDA-capable GPU is optional for faster training

## Quick Start

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run roadsos_dashboard.py
```

Open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

The dashboard loads `roadsos_pinn.pt` from the project directory, so you do not need to train the model before running the demo.

## Dashboard Usage

Use the sidebar to choose a scenario:

- `Normal Riding`: low-risk riding with stable friction
- `Oil Patch`: friction drops and skid probability rises
- `Crash`: high acceleration and angular velocity pulses trigger crash and injury metrics

You can also adjust the simulated duration and rider medical profile fields. The profile card represents the kind of information that could be released by a real emergency workflow after a crash trigger.

## Training

To retrain the model, open and run:

```text
RoadSoS_PINN.ipynb
```

The notebook generates synthetic IMU samples, trains the PINN, and writes a new `roadsos_pinn.pt` checkpoint. Training can run on CPU, but a GPU is recommended.

If you regenerate the checkpoint, verify that the dashboard still loads it successfully:

```bash
streamlit run roadsos_dashboard.py
```

## Model Summary

| Item | Value |
| --- | --- |
| Input features | `[t, ax, ay, az, gx, gy, gz]` |
| Output values | `[v, theta, mu_eff, x_brain, y_brain, z_brain]` |
| Architecture | Fully connected neural network |
| Hidden layers | 6 |
| Hidden width | 128 |
| Activation | `tanh` |
| Runtime framework | PyTorch |
| App framework | Streamlit |

`mu_eff` is passed through a sigmoid in the dashboard model wrapper, then converted into skid probability by comparing it with the wet-friction threshold.

## Physics Used

Vehicle residual:

```text
M * dv/dt + mu_eff * M * g = 0
dtheta/dt - v * sin(theta) / L = 0
```

Brain model residual:

```text
M_b * x_b'' + C * x_b' + K * x_b = M_b * a_helmet
```

Combined training objective:

```text
L_total = L_data + lambda_vehicle * L_vehicle + lambda_bio * L_bio
```

The current project uses these physics terms as simplified constraints for a hackathon prototype. They are not a substitute for validated crash reconstruction, clinical diagnosis, or certified vehicle safety testing.

## Injury and Alert Thresholds

| Signal | Warning threshold | Critical threshold |
| --- | ---: | ---: |
| HIC15 | 700 | 1000 |
| BrIC | 0.6 | 1.0 |
| Skid probability | 0.65 | 0.65 |
| Peak acceleration | 6 g | 6 g |

The dashboard marks a crash when the crash scenario is selected or peak acceleration exceeds the configured crash threshold. It marks a skid warning when predicted skid probability exceeds the skid threshold and a crash is not already active.

## Deployment Vision

The software demo is designed around a possible helmet-mounted safety system:

- 6-axis IMU such as MPU-6050 or ICM-42688
- Edge compute device such as ESP32-S3
- Haptic warning for skid risk
- BLE, cellular, or paired-phone SOS trigger
- NFC or QR-based emergency medical profile release

Real deployment would require hardware validation, calibrated sensors, field testing, privacy controls, and regulatory review.

## Troubleshooting

If Streamlit cannot find the checkpoint, confirm that `roadsos_pinn.pt` exists beside `roadsos_dashboard.py`.

If PyTorch installation fails, install the PyTorch build that matches your operating system and hardware from the official PyTorch installation selector, then rerun:

```bash
pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the environment again.

## Safety Notice

RoadSoS is a research and hackathon prototype. It uses synthetic data and simplified physics models. Do not use it as a medical device, emergency response system, or production vehicle safety system without proper validation, certification, and real-world testing.
