import numpy as np, imageio.v2 as imageio, os
from datetime import datetime

# --- modified Shepp–Logan phantom parameters ---
P = np.array([
 [ 1.0, 0.6900, 0.920,  0.00,  0.0000,  0.0],
 [-0.8, 0.6624, 0.874,  0.00, -0.0184,  0.0],
 [-0.2, 0.1100, 0.310,  0.22,  0.0000,-18.0],
 [-0.2, 0.1600, 0.410, -0.22,  0.0000, 18.0],
 [ 0.1, 0.2100, 0.250,  0.00,  0.3500,  0.0],
 [ 0.1, 0.0460, 0.046,  0.00,  0.1000,  0.0],
 [ 0.1, 0.0460, 0.046,  0.00, -0.1000,  0.0],
 [ 0.1, 0.0460, 0.023, -0.08, -0.6050,  0.0],
 [ 0.1, 0.0230, 0.023,  0.00, -0.6060,  0.0],
 [ 0.1, 0.0230, 0.046,  0.06, -0.6050,  0.0],
], float)

# --- phantom generator ---
def phantom(n=1024):
    y,x = np.linspace(-1,1,n), np.linspace(-1,1,n)
    X,Y = np.meshgrid(x,y); img = np.zeros((n,n),float)
    for A,a,b,x0,y0,phi in P:
        c,s = np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi))
        Xr, Yr = (X-x0)*c + (Y-y0)*s, -(X-x0)*s + (Y-y0)*c
        img[(Xr/a)**2 + (Yr/b)**2 <= 1.0] += A
    return img

# --- radon transform ---
def bilinear(img,x,y):
    H,W = img.shape
    x0,y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1,y1 = x0+1,y0+1; wx,wy = x-x0,y-y0
    def get(ix,iy):
        m=(ix>=0)&(ix<W)&(iy>=0)&(iy<H); out=np.zeros_like(ix,float)
        out[m]=img[iy[m],ix[m]]; return out
    Ia,Ib,Ic,Id=get(x0,y0),get(x1,y0),get(x0,y1),get(x1,y1)
    return Ia*(1-wx)*(1-wy)+Ib*wx*(1-wy)+Ic*(1-wx)*wy+Id*wx*wy

def radon_parallel(img,n_det=256,n_ang=180,oversample=2):
    H,W = img.shape; th=np.linspace(0,180,n_ang,endpoint=False)
    R=np.sqrt(2.0); u=np.linspace(-R,R,n_det)
    ds=(2.0/max(H,W))/oversample; s=np.arange(-R,R+ds,ds)
    def w2p(wx,wy): return (wx+1)*0.5*(W-1),(wy+1)*0.5*(H-1)
    S=np.zeros((n_det,n_ang),float)
    for ai,t in enumerate(np.deg2rad(th)):
        dx,dy=np.cos(t),np.sin(t); nx,ny=-np.sin(t),np.cos(t)
        x0,y0=u[:,None]*nx,u[:,None]*ny
        xs,ys=x0+s*dx,y0+s*dy
        j,i2=w2p(xs,ys); S[:,ai]=bilinear(img,j,i2).sum(1)*ds
    return S

# --- save as raw npy + 16bit PNG ---
def save_u16_png(path, arr):
    mn,mx=arr.min(),arr.max()
    img=((arr-mn)/(mx-mn)*65535).astype(np.uint16) if mx>mn else np.zeros_like(arr,np.uint16)
    imageio.imwrite(path,img)

if __name__=="__main__":
    N,DET,ANG=1024,256,180
    ph=phantom(N).astype(np.float32)
    sino=radon_parallel(ph,DET,ANG).astype(np.float32)

    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    folder=f"shepp_logan_modified_{ts}"; os.makedirs(folder,exist_ok=True)

    np.save(f"{folder}/phantom_{N}.npy",ph)
    np.save(f"{folder}/sinogram_{DET}x{ANG}.npy",sino)

    save_u16_png(f"{folder}/phantom_{N}.png",ph)
    save_u16_png(f"{folder}/sinogram_{DET}x{ANG}.png",sino)

    print("saved phantom + sinogram to",folder)
