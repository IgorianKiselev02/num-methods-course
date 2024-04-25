import numpy as np
import argparse
import pickle
import os
from PIL import Image


def simple_svd(A, num_simulations = 1000, num_sing_vals = None):
    m, n = A.shape

    if num_sing_vals is None:
        num_sing_vals = min(m, n)

    AAT = A @ A.T
    eigenvalues = []
    eigenvectors = []

    for _ in range(num_sing_vals):
        eigenvalue, eigenvector = power_iteration(AAT, num_simulations)
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)
        AAT -= eigenvalue * np.outer(eigenvector, eigenvector)

    S = np.sqrt(eigenvalues)
    U = np.vstack(eigenvectors).T
    Sigma_inv = np.diag(1 / S)
    V = (A.T @ U) @ Sigma_inv

    return U, S, V


def power_iteration(A, num_simulations):
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_simulations):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm

    eigenvalue = np.dot(b_k.T, np.dot(A, b_k))
    eigenvector = b_k

    return eigenvalue, eigenvector


def advanced_power_iteration(A, num_simulations, epsilon=1e-10):
    n, _ = A.shape
    ort = np.eye(n)

    for _ in range(num_simulations):
        z = A @ ort
        ort, tri = np.linalg.qr(z)
        diff = np.sum(np.abs(np.diag(tri))) / np.sum(np.abs(tri))

        if diff < epsilon:
            break

    return ort


def advanced_svd(A, num_simulations = 1000, num_sing_vals = None):
    m, n = A.shape

    if num_sing_vals is None or num_sing_vals > min(m, n):
        num_sing_vals = min(m, n)

    Q_right = advanced_power_iteration(A.T @ A, num_simulations)
    V = Q_right[:, :num_sing_vals]
    Sigma_squared = np.maximum(np.diag(V.T @ A.T @ A @ V), 0)
    Sigma = np.sqrt(Sigma_squared)
    U = A @ V @ np.linalg.inv(np.diag(Sigma))

    return U, Sigma, V.T


def compress_image_svd(image, method, compression):
    Rc = np.array(image[:,:,0], dtype=np.float32)
    Gc = np.array(image[:,:,1], dtype=np.float32)
    Bc = np.array(image[:,:,2], dtype=np.float32)

    if method == "numpy":
        Ur, Sr, Vr = np.linalg.svd(Rc, full_matrices=False)
        Ug, Sg, Vg = np.linalg.svd(Gc, full_matrices=False)
        Ub, Sb, Vb = np.linalg.svd(Bc, full_matrices=False)

    elif method == "simple":
        num_sing_vals = min(Rc.shape) // compression
        Ur, Sr, Vr = simple_svd(Rc, num_simulations=1000, num_sing_vals=num_sing_vals)
        Ug, Sg, Vg = simple_svd(Gc, num_simulations=1000, num_sing_vals=num_sing_vals)
        Ub, Sb, Vb = simple_svd(Bc, num_simulations=1000, num_sing_vals=num_sing_vals)
        Vr = Vr.T
        Vg = Vg.T
        Vb = Vb.T

    elif method == "advanced":
        num_sing_vals = min(Rc.shape) // compression
        Ur, Sr, Vr = advanced_svd(Rc, num_simulations=1000, num_sing_vals=num_sing_vals)
        Ug, Sg, Vg = advanced_svd(Gc, num_simulations=1000, num_sing_vals=num_sing_vals)
        Ub, Sb, Vb = advanced_svd(Bc, num_simulations=1000, num_sing_vals=num_sing_vals)

    k = min(len(Sr), len(Sg), len(Sb))
    k = k // compression 

    return ((Ur[:, :k], np.diag(Sr[:k]), Vr[:k, :]),
            (Ug[:, :k], np.diag(Sg[:k]), Vg[:k, :]),
            (Ub[:, :k], np.diag(Sb[:k]), Vb[:k, :]))


def decompress_image_svd(com_channels):
    Ur, Sr, Vr = com_channels[0]
    Ug, Sg, Vg = com_channels[1]
    Ub, Us, Vb = com_channels[2]

    dec_R = (Ur @ np.diag(Sr) @ Vr).clip(0, 255).astype(np.uint8)
    dec_G = (Ug @ np.diag(Sg) @ Vg).clip(0, 255).astype(np.uint8)
    dec_B = (Ub @ np.diag(Us) @ Vb).clip(0, 255).astype(np.uint8)

    if dec_R.ndim == 1:
        dec_R = dec_R[:, np.newaxis]

    if dec_G.ndim == 1:
        dec_G = dec_G[:, np.newaxis]

    if dec_B.ndim == 1:
        dec_B = dec_B[:, np.newaxis]

    dec_R = dec_R.reshape(dec_R.shape[0], dec_R.shape[1], -1)
    dec_G = dec_G.reshape(dec_G.shape[0], dec_G.shape[1], -1)
    dec_B = dec_B.reshape(dec_B.shape[0], dec_B.shape[1], -1)

    dec_image = np.concatenate((dec_R, dec_G, dec_B), axis=2)

    return dec_image


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=["compress", "decompress"])
parser.add_argument("--method", type=str, choices=["numpy", "simple", "advanced"])
parser.add_argument("--compression", type=int)
parser.add_argument("--in_file", type=str)
parser.add_argument("--out_file", type=str)

args = parser.parse_args()


def save_com_image_sv_format(filepath, com_channels, compression, original_size):
    k = min(len(com_channels[0][1]), len(com_channels[1][1]), len(com_channels[2][1]))
    target_size = original_size // compression

    while True:
        estimated_size = sum((ch[0][:, :k].nbytes + ch[1][:k].nbytes + ch[2][:k].nbytes) for ch in com_channels)

        if estimated_size < target_size - 54:
            break

        k -= 1

    com_data = {
        'shape': image.shape,
        'k': k,
    }

    for i, color in enumerate(['Rc', 'Gc', 'Bc']):
        U, S, V = com_channels[i]
        com_data[f'{color}_U'] = U[:, :k].astype(np.float32).ravel()
        com_data[f'{color}_S'] = S[:k].astype(np.float32)
        com_data[f'{color}_V'] = V[:k, :].astype(np.float32).ravel()

    with open(filepath, 'wb') as f:
        pickle.dump(com_data, f)


def load_compressed_image_sv_format(filepath):
    with open(filepath, 'rb') as f:
        com_data = pickle.load(f)

    k = com_data['k']
    shape = com_data['shape']

    channels = []

    for color in ['Rc', 'Gc', 'Bc']:
        U = com_data[f'{color}_U'].reshape((shape[0], k))
        S = np.diag(com_data[f'{color}_S'])
        V = com_data[f'{color}_V'].reshape((k, shape[1]))
        channels.append((U, S, V))

    return channels


if args.mode == "compress":
    image = np.array(Image.open(args.in_file))
    original_size = os.path.getsize(args.in_file)
    com_image = compress_image_svd(image, args.method, args.compression)
    save_com_image_sv_format(args.out_file, com_image, args.compression, original_size)

elif args.mode == "decompress":
    com_image_channels = load_compressed_image_sv_format(args.in_file)
    dec_image = decompress_image_svd(com_image_channels)
    Image.fromarray(dec_image).save(args.out_file)
