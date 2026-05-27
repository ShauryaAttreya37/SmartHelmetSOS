import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="RoadSoS", page_icon="🪖", layout="wide")
plt.style.use("dark_background")

G=9.81; M_TOTAL=225; WHEELBASE=1.35; M_BRAIN=1.4; K_BRAIN=2.1e4; C_BRAIN=85
MU_DRY=0.75; MU_WET=0.45; MU_OIL=0.15
BRIC_X=66.3; BRIC_Y=56.5; BRIC_Z=42.2; HIC_LOW=700; HIC_HIGH=1000
FS_N=100; FS_C=1000
CA="#00E5FF"; CW="#FFD600"; CR="#FF1744"; CG="#00E676"; BG="#0d0d0d"


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        d = [7]+[128]*6+[6]
        layers = []
        for i in range(len(d)-1):
            layers.append(nn.Linear(d[i], d[i+1]))
            if i < len(d)-2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def out(self, x):
        o = self.forward(x)
        return {
            "v": o[:,0:1], "theta": o[:,1:2],
            "mu_eff": torch.sigmoid(o[:,2:3]),
            "xb": o[:,3:4], "yb": o[:,4:5], "zb": o[:,5:6]
        }


@st.cache_resource
def load_model():
    ck = torch.load("roadsos_pinn.pt", map_location="cpu", weights_only=False)
    m = PINN()
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m, ck["norm_mean"], ck["norm_std"]


def hic15(ax, ay, az, fs, wms=15):
    ar = np.sqrt(ax**2+ay**2+az**2)/G
    dt = 1/fs; mw = max(2, int(wms*1e-3/dt)); h = 0
    for w in range(2, mw+1):
        cs = np.cumsum(ar)
        ms = (cs[w:]-cs[:-w])/w
        vals = w*dt*(np.maximum(ms,0)**2.5)
        h = max(h, vals.max())
    return h


def bric(gx, gy, gz):
    return np.sqrt(
        (np.max(np.abs(gx))/BRIC_X)**2 +
        (np.max(np.abs(gy))/BRIC_Y)**2 +
        (np.max(np.abs(gz))/BRIC_Z)**2)


def sim(sc, dur, fs, seed=42):
    np.random.seed(seed)
    t = np.linspace(0, dur, int(dur*fs)); n = len(t)
    if sc == 0:
        v = 11.1+0.5*np.sin(0.3*t)+0.1*np.random.randn(n)
        th = 0.05*np.sin(0.2*t)+0.01*np.random.randn(n)
        mu = np.clip(MU_DRY+0.02*np.random.randn(n), 0.6, 0.9)
    elif sc == 1:
        pi = int(0.4*n); v = 13.9+0.8*np.sin(0.3*t)
        mu = np.ones(n)*MU_DRY
        dec = np.exp(-np.arange(n-pi)/(0.2*fs))
        mu[pi:] = MU_OIL+(MU_DRY-MU_OIL)*dec[:n-pi]
        mu = np.clip(mu, 0.05, 0.9)
        th = 0.05*np.sin(0.2*t); th[pi:] += 0.3*(1-dec[:n-pi])
    else:
        ii = int(0.25*n); sw = int(0.1*fs); v = np.ones(n)*16.7
        v[ii:ii+sw] = np.linspace(16.7,0,sw); v[ii+sw:] = 0
        th = np.zeros(n)
        th[ii:] = np.clip(np.linspace(0,np.pi/2*1.2,n-ii), 0, np.pi/2)
        mu = np.ones(n)*MU_DRY; mu[ii:] = MU_OIL
    ax = np.diff(v,prepend=v[0])*fs+0.1*np.random.randn(n)
    ay = v*np.gradient(th,1/fs)+0.1*np.random.randn(n)
    az = G*np.cos(th)+0.05*np.random.randn(n)
    gx = np.gradient(th,1/fs)+0.02*np.random.randn(n)
    gy = 0.05*np.sin(0.1*t)+0.01*np.random.randn(n)
    gz = 0.03*np.cos(0.15*t)+0.01*np.random.randn(n)
    if sc == 2:
        pw = int(0.01*fs); ii2 = int(0.25*n); pulse = np.zeros(n)
        if ii2+pw < n:
            pulse[ii2:ii2+pw] = 10*G*np.sin(np.linspace(0,np.pi,pw))
            gx[ii2:ii2+pw] += 25*np.sin(np.linspace(0,np.pi,pw))
        ax += pulse
    return t, ax, ay, az, gx, gy, gz, v, th, mu


# Sidebar
st.sidebar.title("RoadSoS Controls")
sc_name = st.sidebar.selectbox("Scenario", ["Normal Riding","Oil Patch","Crash"])
sc = {"Normal Riding":0, "Oil Patch":1, "Crash":2}[sc_name]
fs = FS_C if sc==2 else FS_N
dur = st.sidebar.slider("Duration (s)", 1.0, 15.0, 8.0 if sc<2 else 2.0, 0.5)
st.sidebar.markdown("---")
st.sidebar.markdown("**Rider Profile**")
bg  = st.sidebar.selectbox("Blood Group",["O+","O-","A+","A-","B+","B-","AB+","AB-"])
alg = st.sidebar.text_input("Allergies","None")
ec  = st.sidebar.text_input("Emergency Contact","+91-XXXXXXXXXX")

# Header
st.markdown(f"<h1 style='color:{CA};font-family:monospace;'>RoadSoS Live Monitor</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:gray;'>PINN | HIC15 | BrIC | Kelvin-Voigt Brain Model | Tire-Road ODE</p>", unsafe_allow_html=True)

# Inference
model, nm, ns = load_model()
t, ax, ay, az, gx, gy, gz, v, th, mu = sim(sc, dur, fs)
X = np.c_[t,ax,ay,az,gx,gy,gz]
Xn = (X - nm) / (ns + 1e-8)
Xt = torch.tensor(Xn, dtype=torch.float32)
with torch.no_grad():
    o = model.out(Xt)
mu_p = o["mu_eff"].numpy().flatten()
Psk  = np.clip((MU_WET-mu_p)/MU_WET, 0, 1)
ares = np.sqrt(ax**2+ay**2+az**2)/G
hv   = hic15(ax, ay, az, fs)
bv   = bric(gx, gy, gz)
inj  = min(0.6*hv/HIC_HIGH + 0.4*bv, 1.0)

# Alert banner
crash_det = (sc==2 or ares.max()>6)
skid_det  = (Psk.max()>0.65 and not crash_det)
ac = CR if crash_det else (CW if skid_det else CG)
at = "CRASH DETECTED - SOS FIRED" if crash_det else ("SKID WARNING - HAPTIC ALERT" if skid_det else "SAFE")
st.markdown(f"<h2 style='color:{ac};'>{at}</h2>", unsafe_allow_html=True)

# Metrics
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Peak Accel",    f"{ares.max():.1f} g", ">6g" if ares.max()>6  else "OK")
c2.metric("P(skid) max",   f"{Psk.max():.2f}",   "HIGH" if Psk.max()>0.65 else "LOW")
c3.metric("HIC15",         f"{hv:.0f}",           "DANGER" if hv>700 else "OK")
c4.metric("BrIC",          f"{bv:.3f}",           "CRITICAL" if bv>1  else "OK")
c5.metric("Injury Score",  f"{inj:.2f}",          "SEVERE" if inj>0.7 else ("MOD" if inj>0.3 else "LOW"))
st.markdown("---")

# Plots
cl, cr_ = st.columns(2)
with cl:
    fig,(a1,a2) = plt.subplots(2,1,figsize=(8,6),facecolor=BG)
    a1.plot(t,ares,color=CA,lw=1.2); a1.axhline(6,color=CR,ls="--",lw=0.8,label="6g")
    a1.set_title("IMU Acceleration",color="white"); a1.set_ylabel("g",color="white")
    a1.tick_params(colors="white"); a1.grid(alpha=0.15); a1.set_facecolor(BG)
    a1.legend(labelcolor="white",fontsize=8)
    a2.fill_between(t,Psk,alpha=0.35,color=CW); a2.plot(t,Psk,color=CW,lw=1.5)
    a2.axhline(0.65,color=CR,ls="--",lw=0.8,label="Alert 0.65")
    a2.set_ylim(0,1.05); a2.set_title("PINN P(skid)",color="white")
    a2.set_ylabel("Probability",color="white"); a2.set_xlabel("Time (s)",color="white")
    a2.tick_params(colors="white"); a2.grid(alpha=0.15); a2.set_facecolor(BG)
    a2.legend(labelcolor="white",fontsize=8)
    plt.tight_layout(); st.pyplot(fig)

with cr_:
    fig2,(a3,a4) = plt.subplots(2,1,figsize=(8,6),facecolor=BG)
    a3.plot(t,mu_p,color=CG,lw=1.5,label="PINN mu_eff")
    a3.plot(t,mu,color="gray",lw=1,ls="--",alpha=0.6,label="True mu")
    a3.axhline(MU_WET,color=CW,ls=":",lw=0.8,label=f"Wet {MU_WET}")
    a3.axhline(MU_OIL,color=CR,ls=":",lw=0.8,label=f"Oil {MU_OIL}")
    a3.set_title("Friction Coefficient (PINN)",color="white"); a3.set_ylabel("mu_eff",color="white")
    a3.tick_params(colors="white"); a3.grid(alpha=0.15); a3.set_facecolor(BG)
    a3.legend(labelcolor="white",fontsize=7)
    a4.plot(t,np.abs(gx),color=CA,lw=1,label="|wx| roll")
    a4.plot(t,np.abs(gy),color=CW,lw=1,label="|wy| pitch")
    a4.plot(t,np.abs(gz),color=CG,lw=1,label="|wz| yaw")
    a4.set_title("Angular Velocity (Gyro)",color="white"); a4.set_ylabel("rad/s",color="white")
    a4.set_xlabel("Time (s)",color="white"); a4.tick_params(colors="white")
    a4.grid(alpha=0.15); a4.set_facecolor(BG); a4.legend(labelcolor="white",fontsize=7)
    plt.tight_layout(); st.pyplot(fig2)

# Rider card
st.markdown("---")
st.markdown(
    f"<div style='background:#111;padding:16px;border-radius:10px;border:1px solid {CA};'>"
    f"<h4 style='color:{CA};'>Rider Medical Profile - Transmitted on SOS</h4>"
    f"<p style='color:white;'>Blood: <b>{bg}</b> | Allergies: <b>{alg}</b> | Emergency: <b>{ec}</b></p>"
    f"<p style='color:gray;font-size:12px;'>Encrypted on ESP32-S3. Released via NFC/QR on crash detection.</p>"
    f"</div>",
    unsafe_allow_html=True)
