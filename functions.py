
import numpy as np
import sys

def mixt_gauss(Mu,Sigma,n,p):
    k = len(Mu)  
    d = Mu[0].shape[0]  
    
    indices = np.random.choice(k, size=n, p=p)
    
    samples = np.zeros((n, d))
    for i in range(k):
        num_samples = np.sum(indices == i)  
        if num_samples > 0:
            samples[indices == i] = np.random.multivariate_normal(Mu[i], Sigma[i], size=num_samples)
    
    return samples

def k_gauss(x,y,sigma): #Gram matrix for gaussian kernel
    X_norm = np.sum(x**2, axis=1)
    Y_norm = np.sum(y**2, axis=1)
    xy = np.dot(x, y.T)
    dist = X_norm[:,None] + Y_norm[None,:] - 2 * xy
    return np.exp(-dist/(2*sigma**2))

def dk_gauss(x,y,sigma): #Gram matrix of nabla_1 k(x,y) , dim = (n,m,d)
    x_mat = np.tile(x, (len(y), 1, 1))
    y_mat = np.tile(y, (len(x), 1, 1))
    x_mat = np.transpose(x_mat, axes=(1, 0, 2))
    diff = y_mat - x_mat
    return diff/(sigma**2) * k_gauss(x,y,sigma)[:,:,None]

def ddk_gauss(x,y,sigma): #Gram matrix of nabla_2 . nabla_1 k(x,y), dim = (n,m)
    d = len(x[0])
    X_norm = np.sum(x**2, axis=1)
    Y_norm = np.sum(y**2, axis=1)
    xy = np.dot(x, y.T)
    dist = X_norm[:,None] + Y_norm[None,:] - 2 * xy
    K = np.exp(-dist/(2*sigma**2))
    return 1/sigma**2 * K *(d - 1/sigma**2 * dist)

def sigma_2(t):
    return (1 - np.exp(-2*t)) + 1e-5

def psi(X1,eps,t):
    d = np.shape(X1[0])[0]
    sigma_t = np.sqrt(sigma_2(t)) 
    dot_product = np.einsum('ij,ij->i', X1, eps) 
    return  np.exp(-2*t)/sigma_t**2 * np.linalg.norm(eps,axis=1)**2 + np.exp(-t)/sigma_t * dot_product - d * np.exp(-2*t)/sigma_t**2

def Xi(Z,eps,k,dk,ddk,t):
    sigma_t = np.sqrt(sigma_2(t))
    score = -eps/sigma_t
    K = k(Z,Z)
    DK = dk(Z,Z)

    K_eps2 = K * (score @ score.T)

    # DK_T = np.transpose(DK, (1, 0, 2))  
    # DK_T_eps = np.einsum('ijn,jn->ij', DK_T, score)  
    # DK_eps_T = np.transpose(DK_T_eps, (1, 0))

    gradK_score = np.einsum('ijk, ijk -> ij',DK, (score[None,:,:] - score[:,None,:]))
    DDK = ddk(Z,Z)

    # print((DK_T_eps + DK_eps_T)[10,10:15])
    # print(np.einsum('ijk, ijk -> ij',DK, (score[None,:,:] - score[:,None,:]))[10,10:15])
    # print()

    return  K_eps2 + gradK_score + DDK


def Loss1(v,Z,mu,phi,Xi,t,k,dk,sigma,lambd):
    d = np.shape(Z[0])[0]
    n = len(Z)
    L = 0
    for i in range(n):
        dt_pi_t = -np.dot(Z[i]-np.exp(-t)*mu,np.exp(-t)*mu)
        grad = mu * np.exp(-t) - Z[i]
        dk_Zi = dk(Z, np.array([Z[i]]))  # dk(Z, np.array(Z[i])) de dimension (n,1,d)
        diff = mu[None, :] * np.exp(-t) - Z  # de dimension (n,d)
        product = np.einsum('nid,nd->n', dk_Zi, diff)# produit scalaire dans R^d
        # if i == 0:
        #     print(product[:5])
        div_v = 1/n * np.sum(phi * np.exp(-np.linalg.norm(Z[i] - Z) / sigma**2) * (d - np.linalg.norm(Z[i] - Z)**2 / sigma**2) / sigma**2 + product)
        S = np.dot(v(np.array([Z[i]]))[0],grad) + div_v
        L = L + 1/n * (S - dt_pi_t)**2
    L = L + lambd/n**2 * np.sum(Xi)
    return L 



def Loss2(v,Z,eps,mu,phi,Xi,psi,t,k,dk,sigma,lambd):
    d = np.shape(Z[0])[0]
    n = len(Z)
    L = 0
    sigma_t = np.sqrt(sigma_2(t))
    for i in range(n):

        grad_log = -eps[i]/sigma_t
        dk_Zi = dk(Z, np.array([Z[i]]))  # dk(Z, np.array(Z[i])) de dimension (n,1,d)
        diff = mu[None, :] * np.exp(-t) - Z  # de dimension (n,d)
        product = np.einsum('nid,nd->n', dk_Zi, diff)
        div_v = 1/n * np.sum(phi * np.exp(-np.linalg.norm(Z[i] - Z) / sigma**2) * (d - np.linalg.norm(Z[i] - Z)**2 / sigma**2) / sigma**2 + product)
        
        S = np.dot(v(np.array([Z[i]]))[0],grad_log) + div_v

        L = L + 1/n * (S - psi[i])**2 
    L = L + lambd/n**2 * np.sum(Xi)
    return L




# def Loss11(v, Z, mu, phi, Xi, t, k, dk, sigma, lambd):
#     d = np.shape(Z[0])[0]
#     n = len(Z)
    
#     # Calcul de dt_pi_t pour tous les éléments de Z
#     dt_pi_t = -np.sum((Z - np.exp(-t) * mu) * np.exp(-t) * mu, axis=1)
    
#     # Calcul de grad pour tous les éléments de Z
#     grad = mu * np.exp(-t) - Z
    
#     # Calcul de dk_Zi pour tous les éléments de Z
#     dk_Zi = dk(Z, Z)  # dk(Z, Z) de dimension (n,n,d)
    
#     # Calcul de diff pour tous les éléments de Z
#     diff = mu[None, :] * np.exp(-t) - Z  # de dimension (n,d)
    
#     # Calcul de product pour tous les éléments de Z
#     product = np.einsum('nij,nj->ni', dk_Zi, diff)  # produit scalaire dans R^d, de dimension (n,n)
#     #print(product.T[0,:5])
#     #print()
    
#     # Calcul de div_v pour tous les éléments de Z
#     norm_diff = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)  # de dimension (n,n)
#     exp_term = np.exp(-norm_diff / sigma**2)
#     div_v = 1/n * np.sum(phi[:, None] * exp_term * (d - norm_diff**2 / sigma**2) / sigma**2 + product, axis=0)
    
#     # Calcul de S pour tous les éléments de Z
#     v_Z = v(Z)  # de dimension (n,d)
#     S = np.einsum('nd,nd->n', v_Z, grad) + div_v
    
#     # Calcul de L
#     L = 1/n * np.sum((S - dt_pi_t)**2) + lambd/n**2 * np.sum(Xi)
    
#     return L


    








