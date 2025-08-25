import numpy as np, os, imageio.v2 as imageio
from datetime import datetime

# ---------- phantom: fixed head+skull, random internals inside skull ----------
def random_head_phantom(n=2048, k_range=(10,18), seed=None):
    rng = np.random.default_rng(seed)
    y,x = np.linspace(-1,1,n), np.linspace(-1,1,n)
    X,Y = np.meshgrid(x,y); img = np.zeros((n,n),float)
    # head + skull
    for A,a,b,x0,y0,phi in [(1.0,0.69,0.92,0,0,0), (-0.8,0.66,0.87,0,-0.018,0)]:
        c,s=np.cos(np.deg2rad(phi)),np.sin(np.deg2rad(phi))
        Xr,Yr=(X-x0)*c+(Y-y0)*s, -(X-x0)*s+(Y-y0)*c
        img[((Xr/a)**2+(Yr/b)**2)<=1]+=A
    # random inner ellipses constrained to skull
    k = rng.integers(*k_range)
    added=0
    while added<k:
        A = rng.uniform(-1,1); a = rng.uniform(0.03,0.22); b = rng.uniform(0.03,0.22)
        x0 = rng.uniform(-0.55,0.55); y0 = rng.uniform(-0.65,0.65); phi = rng.uniform(0,180)
        if ((x0/(0.66-a-0.05))**2 + (y0/(0.87-b-0.05))**2) <= 1:  # inside skull
            c,s=np.cos(np.deg2rad(phi)),np.sin(np.deg2rad(phi))
            Xr,Yr=(X-x0)*c+(Y-y0)*s, -(X-x0)*s+(Y-y0)*c
            img[((Xr/a)**2+(Yr/b)**2)<=1]+=A; added+=1
    return img.astype(np.float32)

# ---------- radon (parallel-beam), compact ----------
def _bilinear(img,x,y):
    H,W=img.shape; x0=np.floor(x).astype(int); y0=np.floor(y).astype(int)
    x1,y1=x0+1,y0+1; wx,wy=x-x0,y-y0
    def get(ix,iy):
        m=(ix>=0)&(ix<W)&(iy>=0)&(iy<H); out=np.zeros_like(ix,float); out[m]=img[iy[m],ix[m]]; return out
    Ia,Ib,Ic,Id=get(x0,y0),get(x1,y0),get(x0,y1),get(x1,y1)
    return Ia*(1-wx)*(1-wy)+Ib*wx*(1-wy)+Ic*(1-wx)*wy+Id*wx*wy

def radon_parallel(img, n_det=256, n_ang=180, oversample=2):
    H,W=img.shape; th=np.linspace(0,180,n_ang,endpoint=False)
    R=np.sqrt(2.0); u=np.linspace(-R,R,n_det); ds=(2.0/max(H,W))/oversample; s=np.arange(-R,R+ds,ds)
    def w2p(wx,wy): return (wx+1)*0.5*(W-1),(wy+1)*0.5*(H-1)
    S=np.zeros((n_det,n_ang),float)
    for ai,t in enumerate(np.deg2rad(th)):
        dx,dy=np.cos(t),np.sin(t); nx,ny=-np.sin(t),np.cos(t)
        x0,y0=u[:,None]*nx,u[:,None]*ny; xs,ys=x0+s*dx,y0+s*dy
        j,i=w2p(xs,ys); S[:,ai]=_bilinear(img,j,i).sum(1)*ds
    return S.astype(np.float32)

# ---------- save helpers (no windowing; min->0, max->65535) ----------
def save_png16(path, arr):
    mn,mx=float(arr.min()),float(arr.max())
    im = np.zeros_like(arr, dtype=np.uint16) if mx==mn else ((arr-mn)/(mx-mn)*65535).astype(np.uint16)
    imageio.imwrite(path, im)

# ---------- main ----------
if __name__=="__main__":
    IMG=2048; DET=256; ANG=180
    ph  = random_head_phantom(n=IMG, k_range=(10,18))
    sino= radon_parallel(ph, n_det=DET, n_ang=ANG)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"shepp_logan_phantom_{ts}"; os.makedirs(out, exist_ok=True)

    np.save(f"{out}/image_{ts}_{IMG}.npy", ph);      save_png16(f"{out}/image_{ts}_{IMG}.png", ph)
    np.save(f"{out}/sinogram_{ts}_{DET}x{ANG}.npy", sino); save_png16(f"{out}/sinogram_{ts}_{DET}x{ANG}.png", sino)
    print("saved to", out)
