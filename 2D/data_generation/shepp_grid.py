import numpy as np, imageio.v2 as imageio, os
from datetime import datetime

# --------- core phantom: rectangle with 3–2–3 circular holes ----------
def phantom_rect_holes(n=1024, rect=(0.8,0.55), level=1.0,
                       rows=(3,2,3), hole_r=0.08, gap=(0.18,0.18)):
    """
    n: image size (n x n), world coords in [-1,1]^2
    rect: half-widths (rx, ry) of the rectangle
    level: gray level inside rectangle (background is 0)
    rows: counts per row (top, mid, bottom) → 3–2–3
    hole_r: circle radius
    gap: spacing between hole centers (dx, dy)
    """
    y,x = np.linspace(-1,1,n), np.linspace(-1,1,n)
    X,Y = np.meshgrid(x,y)
    img = (np.abs(X)<=rect[0]) & (np.abs(Y)<=rect[1])
    img = img.astype(np.float32)*level

    # row y-positions: top, mid, bottom
    ry = np.array([ gap[1], 0.0, -gap[1] ], dtype=np.float32)
    # for each row, center holes horizontally around x=0
    for row_idx, cnt in enumerate(rows):
        if cnt<=0: continue
        total_span = (cnt-1)*gap[0]
        xs = np.linspace(-total_span/2, total_span/2, cnt)
        for xc in xs:
            mask = (X-xc)**2 + (Y-ry[row_idx])**2 <= hole_r**2
            img[mask] = 0.0  # punch hole
    return img

# --------- minimal parallel-beam radon (same as before, compact) ---------
def bilinear(img,x,y):
    H,W = img.shape
    x0,y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1,y1 = x0+1,y0+1; wx,wy = x-x0, y-y0
    def get(ix,iy):
        m=(ix>=0)&(ix<W)&(iy>=0)&(iy<H); out=np.zeros_like(ix,float)
        out[m]=img[iy[m],ix[m]]; return out
    Ia,Ib,Ic,Id=get(x0,y0),get(x1,y0),get(x0,y1),get(x1,y1)
    return Ia*(1-wx)*(1-wy)+Ib*wx*(1-wy)+Ic*(1-wx)*wy+Id*wx*wy

def radon_parallel(img,n_det=256,n_ang=180,oversample=2):
    H,W=img.shape; th=np.linspace(0,180,n_ang,endpoint=False)
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

# --------- save helpers: no windowing (exact min→0, max→65535) ----------
def save_u16_png(path, arr):
    mn,mx=arr.min(),arr.max()
    img=((arr-mn)/(mx-mn)*65535).astype(np.uint16) if mx>mn else np.zeros_like(arr,np.uint16)
    imageio.imwrite(path,img)

if __name__=="__main__":
    # configs
    N=1024             # phantom resolution (large & sharp)
    DET, ANG = 256,180 # sinogram resolution (independent)
    RECT=(0.85,0.55)   # rectangle half-widths (x,y)
    HOLE_R=0.08        # hole radius
    GAP=(0.22,0.22)    # hole spacing (x,y)

    ph = phantom_rect_holes(n=N, rect=RECT, level=1.0, rows=(3,2,3),
                            hole_r=HOLE_R, gap=GAP).astype(np.float32)
    sino = radon_parallel(ph, n_det=DET, n_ang=ANG).astype(np.float32)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"rect_3-2-3_holes_{ts}"; os.makedirs(out, exist_ok=True)

    # raw arrays for ML
    np.save(f"{out}/phantom_{N}.npy", ph)
    np.save(f"{out}/sinogram_{DET}x{ANG}.npy", sino)

    # visualization (16-bit PNG, single channel, no windowing)
    save_u16_png(f"{out}/phantom_{N}.png", ph)
    save_u16_png(f"{out}/sinogram_{DET}x{ANG}.png", sino)

    print("saved to", out)
