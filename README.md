# RoadSoS

**Two-wheeler crash detection and head-injury prediction using a Physics-Informed Neural Network.**

Built for IITM Hackathon 2026.

---

## The Problem

India loses over 150,000 lives to road accidents every year. A significant chunk of those are two-wheeler riders who skidded on an oil patch, hit a pothole wrong, or crashed with no one nearby to call for help.

The thing is — the physics of a skid is predictable. Friction drops, the bike tilts faster than the rider can react, and the IMU on your helmet is screaming data that nobody's listening to.

RoadSoS listens.

---

## What It Does

RoadSoS runs a Physics-Informed Neural Network on raw IMU data (accelerometer + gyroscope) and does three things simultaneously:

1. **Predicts skid probability in real time** — by learning the tire-road friction ODE directly inside the network's loss function, not as a post-processing step.

2. **Estimates head injury severity** — using HIC15 (the international helmet safety standard) and BrIC (a brain rotation injury criterion from NHTSA cadaver research), backed by a Kelvin–Voigt spring-mass-damper model of the brain.

3. **Fires an SOS on crash** — and releases the rider's medical profile (blood group, allergies, emergency contact) via NFC/QR from the helmet unit, so first responders have what they need before the ambulance arrives.

The "physics-informed" part matters: a plain neural network can hallucinate safe friction on an oil patch if it hasn't seen that exact sensor signature before. By embedding Newton's laws into the training loss, the model is physically constrained to behave correctly even in edge cases.

---

## Demo

Run the Streamlit dashboard locally — no hardware needed, the model simulates all three scenarios:

```bash
pip install -r requirements.txt
streamlit run roadsos_dashboard.py
```

Open `http://localhost:8501`. Switch between **Normal Riding**, **Oil Patch**, and **Crash** from the sidebar and watch the skid probability, HIC15, and BrIC update live.

> The trained weights (`roadsos_pinn.pt`) are already in the repo — you don't need to retrain to run the demo.

---

## Training Your Own Model

Open [RoadSoS_PINN.ipynb](RoadSoS_PINN.ipynb) and run top to bottom.

The notebook generates 18,800 synthetic IMU samples across the three scenarios, trains for 3000 epochs, and saves `roadsos_pinn.pt`. On a GPU this takes about 5 minutes. On CPU, roughly 30.

```bash
# If you're on Colab or a remote machine, expose the dashboard via ngrok
# Set your token first:
export NGROK_TOKEN=your_token_here
# Then run the last cell in the notebook
```

---

## How the Physics Works

The PINN has two residual losses that act like hard constraints during training:

**Tire-road ODE** — models friction deceleration and tilt kinematics for a two-wheeler:
```
M · dv/dt  +  μ_eff · M · g  =  0
dθ/dt  −  v · sin(θ) / L  =  0
```

**Kelvin–Voigt brain model** — a spring-mass-damper system tuned to the brain's known resonant frequency (~19.5 Hz, right in the concussion-risk band):
```
M_b · x_b''  +  C · x_b'  +  K · x_b  =  M_b · a_helmet
```

These residuals are computed via automatic differentiation and added to the standard MSE data loss:

```
L_total = 1.0 · L_data  +  0.1 · L_vehicle  +  0.05 · L_bio
```

---

## Model at a Glance

| | |
|---|---|
| Input | `[t, ax, ay, az, gx, gy, gz]` — 7 features |
| Output | `[v, θ, μ_eff, x_brain, y_brain, z_brain]` — 6 outputs |
| Architecture | 6 hidden layers × 128 neurons, tanh activation |
| Parameters | ~133,000 |
| Training | Adam + Cosine LR annealing, 3000 epochs |
| Hardware | CUDA GPU (falls back to CPU automatically) |

---

## Injury Thresholds

| Signal | Green | Yellow | Red |
|---|---|---|---|
| HIC15 | < 700 | 700–1000 | > 1000 |
| BrIC | < 0.6 | 0.6–1.0 | > 1.0 |
| P(skid) | < 0.65 | — | ≥ 0.65 |
| Peak accel | < 6 g | — | ≥ 6 g |

HIC15 thresholds come from the ECE 22.06 helmet safety standard. BrIC critical values (66.3 / 56.5 / 42.2 rad/s for pitch/roll/yaw) are from NHTSA cadaver research.

---

## Files

| File | What it is |
|---|---|
| `RoadSoS_PINN.ipynb` | Full training pipeline — data generation, model, physics losses, evaluation plots |
| `roadsos_dashboard.py` | Streamlit live monitor — loads the saved model and runs interactive inference |
| `roadsos_pinn.pt` | Saved weights from the trained model |
| `requirements.txt` | Python dependencies |

---

## Hardware Vision

The demo runs entirely in software, but the system is designed around a real deployment:

- **Sensor**: 6-axis IMU (MPU-6050 or ICM-42688) in the helmet
- **Edge compute**: ESP32-S3 running inference at the edge
- **Alert**: Haptic motor for skid warning, cellular/BLE ping for SOS
- **Medical data**: Encrypted rider profile released via NFC/QR on crash trigger

---

## Physical Constants Used

| Constant | Value | Why |
|---|---|---|
| Brain mass | 1.4 kg | Standard biomechanics value |
| Brain stiffness K | 21,000 N/m | Kelvin–Voigt brain model |
| Brain damping C | 85 N·s/m | Tunes resonance to 19.5 Hz |
| BrIC pitch threshold | 66.3 rad/s | NHTSA cadaver study |
| BrIC roll threshold | 56.5 rad/s | NHTSA cadaver study |
| BrIC yaw threshold | 42.2 rad/s | NHTSA cadaver study |
| Rider + bike mass | 225 kg | 75 kg rider + 150 kg bike |
| Wheelbase | 1.35 m | Typical commuter geometry |
