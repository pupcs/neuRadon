import numpy as np, imageio.v2 as imageio, os
from datetime import datetime

# original Shepp–Logan parameters: A, a, b, x0, y0, phi(deg)
P = np.array([
 [ 2.00,0.69,  0.92,  0.00,  0.0000,  0.0],
 [-0.98,0.6624,0.874, 0.00, -0.0184,  0.0],
 [-0.02,0.11,  0.31,  0.22,  0.0000,-18.0],
 [-0.02,0.16,  0.41, -0.22,  0.0000, 18.0],
 [ 0.01,0.21,  0.25,  0.00,  0.35,   0.0],
 [ 0.01,0.046, 0.046, 0.00,  0.10,   0.0],
 [ 0.01,0.046, 0.046, 0.00, -0.10,   0.0],
 [ 0.01,0.046, 0.023,-0.08, -0.605,  0.0],
 [ 0.01,0.023, 0.023, 0.00, -0.605,  0.0],
 [ 0.01,0.023, 0.046, 0.06, -0.605,  0.0],
], float)

def phantom(n=1024):
    y,x = np.linspace(-1,1,n), np.linspace(-1,1,n)
    X,Y = np.meshgrid(x,y); img = np.zeros((n,n),float)
    for A,a,b,x0,y0,phi in P:
        c,s = np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi))
        Xr, Yr = (X-x0)*c + (Y-y0)*s, -(X-x0)*s + (Y-y0)*c
        img[(Xr/a)**2 + (Yr/b)**2 <= 1] += A
    return img

def save_u16_png_raw(path, arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn: 
        img = np.zeros_like(arr, dtype=np.uint16)
    else:
        img = ((arr - mn) / (mx - mn) * 65535).astype(np.uint16)
    imageio.imwrite(path, img)

if __name__=="__main__":
    IMG_N = 1024
    ph = phantom(IMG_N).astype(np.float32)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"shepp_logan_original_{ts}"; os.makedirs(folder, exist_ok=True)

    np.save(f"{folder}/phantom_{ts}_{IMG_N}.npy", ph)
    save_u16_png_raw(f"{folder}/phantom_{ts}_{IMG_N}.png", ph)

    print("Saved phantom:", folder)
