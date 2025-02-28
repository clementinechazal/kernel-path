
import numpy as np
import sys

def mixt_gauss(Mu,Sigma,n,p):
    k = len(Mu)  # Nombre de gaussiennes
    d = Mu[0].shape[0]  # Dimension des données
    
    # Choisir la gaussienne pour chaque point en fonction des poids
    indices = np.random.choice(k, size=n, p=p)
    
    # Générer les échantillons
    samples = np.zeros((n, d))
    for i in range(k):
        num_samples = np.sum(indices == i)  # Nombre de points à générer pour cette gaussienne
        if num_samples > 0:
            samples[indices == i] = np.random.multivariate_normal(Mu[i], Sigma[i], size=num_samples)
    
    return samples

def k_gauss(x,y,sigma):
    X_norm = np.sum(x**2, axis=1)
    Y_norm = np.sum(y**2, axis=1)
    xy = np.dot(x, y.T)
    dist = X_norm[:,None] + Y_norm[None,:] - 2 * xy
    return np.exp(-dist/(2*sigma**2))

def dk_gauss(x,y,sigma):
    x_mat = np.tile(x, (len(y), 1, 1))
    y_mat = np.tile(y, (len(x), 1, 1))
    x_mat = np.transpose(x_mat, axes=(1, 0, 2))
    diff = y_mat - x_mat
    return diff/(sigma**2) * k_gauss(x,y,sigma)[:,:,None]

def ddk_gauss(x,y,sigma):
    d = len(x[0])
    X_norm = np.sum(x**2, axis=1)
    Y_norm = np.sum(y**2, axis=1)
    xy = np.dot(x, y.T)
    dist = X_norm[:,None] + Y_norm[None,:] - 2 * xy
    K = np.exp(-dist/(2*sigma**2))
    return 1/sigma**2 * K *(d - 1/sigma**2 * dist)

def sigma_2(t):
    return 1/2 *(1 - np.exp(-2*t)) + 1e-5

def psi(X1,eps,t):
    d = np.shape(X1[0])[0]
    sigma_t = np.sqrt(sigma_2(t)) 
    dot_product = np.einsum('ij,ij->i', X1, eps)  # Produit scalaire ligne par ligne
    #return np.exp(-t)/sigma * (dot_product + np.sum(eps**2, axis=1)) - d * np.exp(-2*t)/sigma
    return - np.exp(-2*t)/sigma_t**2 * np.linalg.norm(eps,axis=1)**2 - np.exp(-t)/sigma_t * dot_product - d * np.exp(-2*t)/sigma_t**2

def Xi(Z,eps,k,dk,ddk,t):
    sigma_t = np.sqrt(sigma_2(t))
    K = k(Z,Z)
    DK = dk(Z,Z)

    K_eps2 = K * (eps @ eps.T)

    DK_T = np.transpose(DK, (1, 0, 2))  # Transposée de DK par rapport aux deux premières variables
    DK_T_eps = np.einsum('ijn,jn->ij', DK_T, eps)  # Produit scalaire ligne par ligne

    #DK_eps = np.einsum('ijk,ik->ij', DK, eps)
    DK_eps_T = np.transpose(DK_T_eps, (1, 0))

    DDK = ddk(Z,Z)

    # print(np.mean(np.abs(K)))
    # print(np.mean(np.abs(DK)))
    # print(np.mean(np.abs(DK_eps)))
    # print(np.min(DDK)))
    # print("hey")

    return 1/sigma_t**2 * K_eps2 + 1/sigma_t * (DK_T_eps + DK_eps_T) + DDK

#sanity check :
# Zi = Zj
#


def fonction():
    print("coucou")
