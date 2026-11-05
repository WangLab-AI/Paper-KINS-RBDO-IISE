# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import qmc
from sklearn.model_selection import train_test_split
import GPy
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import pickle
from pyDOE import lhs
from functools import partial
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import gaussian_kde

from ccn_trainer import CCNTrainer

# ========== Custom NN Class ==========
class NN:
    def __init__(self, d, d_latent):
        self.d = d
        self.d_latent = d_latent
        self.w = np.random.randn(d * d_latent + d_latent) / 100

    def update(self, w):
        self.w = w

    def predict_latent(self, X):
        w1 = self.w[:self.d * self.d_latent].reshape(self.d, self.d_latent)
        b1 = self.w[self.d * self.d_latent : self.d * self.d_latent + self.d_latent]
        return np.dot(X, w1) + b1

# -------------------- Objective Function -------------------- #
def cost(x):
    J = np.sum(x**2) * 1e-4
    return J

# # -------------------- Limit-State Function --------------------
from G11 import G1
from G22 import G2


def MCS_reliability(w_opt, reg_model, x, x_mcs, x_std, scaler_X, scaler_G):
    X_mean = np.concatenate([x, E_mean, F_mean])          
    X_std = np.concatenate([x_std, E_std, F_std])         
    X_o = Xmcs * X_std + X_mean
    X = scaler_X.transform(X_o)
    nn = NN(d=NX, d_latent=4)
    nn.update(w_opt)
    latent=nn.predict_latent(X)
    y_pred = reg_model.predict(latent)
    y_pred_o=scaler_G.inverse_transform(y_pred.reshape(-1,1)).flatten()
    Pf = np.mean(y_pred_o < 0)
    return Pf

def constraint_g1(w_opt, reg_model,  scaler_X, scaler_G, x):     
    c_pf = Pf_target - MCS_reliability(w_opt, reg_model, x, x_mcs, x_std, scaler_X, scaler_G)
    return c_pf


def plot_prediction_scatter(y_true, y_pred, title='GP Prediction vs True', xlabel='True Values', ylabel='Predicted Values'):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=5, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def datagen(n, d, loc, scale, islhs=False):
    loc = np.asarray(loc)
    scale = np.asarray(scale)
    
    if loc.size == 1:
        loc = np.full(d, loc)
    if scale.size == 1:
        scale = np.full(d, scale)

    if islhs:
        design = lhs(d, samples=n)  
        for i in range(d):
            design[:, i] = norm(loc=loc[i], scale=scale[i]).ppf(design[:, i])
        return design  
    else:
        return np.random.randn(n, d) * scale + loc 


def Standnormal(n,d):
    Xmcs=np.random.randn(n, d)
    return Xmcs


# ========== Parameters ==========
#np.random.seed(42)
Nx = 21
NE = 21
NF = 3
NX = 45

E_mean = np.ones(NE) * 70.0  # E
F_mean = np.ones(NF) * 4.5    # F

E_std = np.ones(NE) * 5
F_std = np.ones(NF) * 0.2


bounds = [(3.2 , 4) for _ in range(Nx)] 

Pf_target = 1 - 0.95
N_MCS = 10000

max_iter = 15      
iner_it=0
iteration_log = []

# ========== Step 2. Generate training, validation, and MCS samples ==========
N_LHS=200
N_train = 100                
N_valid = N_LHS - N_train
valid_size=N_valid/N_LHS
N_MCS=100000

x_center = np.ones(Nx) * 3.6
x_std = np.ones(Nx) * 0.4


X_mean = np.concatenate([x_center, E_mean, F_mean])
X_std = np.concatenate([x_std, E_std, F_std])

X_LHS = datagen(N_LHS, NX, X_mean, X_std, islhs=True)


X_train, X_valid, idx_train, idx_valid = train_test_split(
    X_LHS, np.arange(X_LHS.shape[0]), test_size=valid_size, random_state=42)


x_mcs = Standnormal(N_MCS,Nx)
E_mcs = Standnormal(N_MCS,NE)
F_mcs = Standnormal(N_MCS,NF)

Xmcs=Standnormal(N_MCS,NX) 

X_mcs=Xmcs * X_std + X_mean

G1_train=G1(X_train)
G2_train=G2(X_train)

G1_valid=G1(X_valid)
G2_valid=G2(X_valid)

G1_mcs=G1(X_mcs)
G2_mcs=G2(X_mcs)


# #Train the initial CCNs models for  G1, and G2
# G1 training
print("\n🚀 Start training G1")
trainer_G1 = CCNTrainer(X_train, G1_train, X_valid, G1_valid, X_mcs, G1_mcs, 2000, 4, 0, 0 ,0)
trainer_G1.optimize()
trainer_G1.evaluate()
w_opt_G1 = np.loadtxt("w_opt_best.txt")
reg_model_G1 = joblib.load("reg_model.pkl")
np.savetxt(f"w_opt_best_G1.txt", w_opt_G1)
joblib.dump(reg_model_G1, "reg_model_G1.pkl")
print(f"✅ Training of constraint function G1 completed")
joblib.dump(trainer_G1.Scaler_X, 'Scaler_X_G1.pkl')
joblib.dump(trainer_G1.Scaler_G, 'Scaler_G_G1.pkl')


# # G2 training
print("\n🚀 Start training G2")
trainer_G2 = CCNTrainer(X_train, G2_train, X_valid, G2_valid, X_mcs, G2_mcs, 2000, 4, 0, 0 , 0)
trainer_G2.optimize()
trainer_G2.evaluate()
w_opt_G2 = np.loadtxt("w_opt_best.txt")
reg_model_G2 = joblib.load("reg_model.pkl")
np.savetxt(f"w_opt_best_G2.txt", w_opt_G2)
joblib.dump(reg_model_G2, "reg_model_G2.pkl")
print(f"✅ Training of constraint function G2 completed")
joblib.dump(trainer_G2.Scaler_X, 'Scaler_X_G2.pkl')
joblib.dump(trainer_G2.Scaler_G, 'Scaler_G_G2.pkl')


w_opt_G1=np.loadtxt("w_opt_best_G1.txt")
reg_G1=joblib.load('reg_model_G1.pkl')
Scaler_X_G1 = joblib.load('Scaler_X_G1.pkl')
Scaler_G_G1 = joblib.load('Scaler_G_G1.pkl')


w_opt_G2=np.loadtxt("w_opt_best_G2.txt")
reg_G2=joblib.load('reg_model_G2.pkl')
Scaler_X_G2 = joblib.load('Scaler_X_G2.pkl')
Scaler_G_G2 = joblib.load('Scaler_G_G2.pkl')



pf1 = MCS_reliability(w_opt_G1, reg_G1, x_center, x_mcs, x_std, Scaler_X_G1, Scaler_G_G1)
pf2 = MCS_reliability(w_opt_G2, reg_G2, x_center, x_mcs, x_std, Scaler_X_G2, Scaler_G_G2)
	

x_history = [x_center.copy()]

# ========== Step 3. RBDO Main Loop ==========
for k in range(max_iter):
    print(f"\n---- RBDO outer iteration {k+1} ----")

    constraints = []
    gp_models = [
       (w_opt_G1, reg_G1, Scaler_X_G1, Scaler_G_G1),
       (w_opt_G2, reg_G2, Scaler_X_G2, Scaler_G_G2)] 


    for w_opt, reg_model, scaler_X, scaler_G in gp_models:
        con_fun = partial(constraint_g1, w_opt, reg_model, scaler_X, scaler_G)
        #con_jac = partial(constraint_jac_g1, w_opt, reg_model, scaler_X, scaler_G)
        constraints.append({'type': 'ineq', 'fun': con_fun})#, 'jac': con_jac                          

    result = minimize(
        cost, x_center, method='SLSQP', bounds=bounds,
        constraints=constraints,
        options={'disp': True, 'ftol': 1e-4, 'maxiter': 200, 'eps': 1e-2}
    )

    x_opt = result.x.copy()


    x_center = x_opt.copy()
    x_history.append(x_opt.copy())

    X_mean_new = np.concatenate([x_center, E_mean, F_mean])
    X_std = np.concatenate([x_std, E_std, F_std])

    X_LHS_new = datagen(N_LHS, NX, X_mean_new, X_std, islhs=True)
    X_train_new, X_valid_new, idx_train, idx_valid = train_test_split(
        X_LHS_new, np.arange(X_LHS_new.shape[0]), test_size=valid_size, random_state=42)

    X_mcs_new=Xmcs*X_std+X_mean_new

    G_funcs = [G1, G2]
    names = ["G1", "G2"]
    for i, (Gi, name) in enumerate(zip(G_funcs, names), start=1):
        train_values = Gi(X_train_new)
        valid_values = Gi(X_valid_new)
        MCS_values = Gi(X_mcs_new)

        globals()[f"{name}_train"] = train_values
        globals()[f"{name}_valid"] = valid_values
        globals()[f"{name}_mcs"] = MCS_values

        print(f"\n🚀 Start training constraint function {i}: {name}")
        if name == "G1":
            paras=[2000,4,0,0,0]
        elif name == "G2":
            paras=[2000,4,0,0,0]  


        trainer = CCNTrainer(X_train_new, train_values, X_valid_new, valid_values, X_mcs_new, MCS_values, paras[0], paras[1], paras[2], paras[3], paras[4])
        trainer.optimize()
        trainer.evaluate()
		

        w_opt= np.loadtxt("w_opt_best.txt")
        reg_model = joblib.load("reg_model.pkl")
        np.savetxt(f"w_opt_best_{name}.txt", w_opt)
        joblib.dump(reg_model, f"reg_model_{name}.pkl")
    
        print(f"✅ Training of constraint function {name} completed")


        if name == "G1":
            trainer_G1 = trainer
        elif name == "G2":
            trainer_G2 = trainer

    w_opt_G1=np.loadtxt("w_opt_best_G1.txt")
    reg_G1=joblib.load('reg_model_G1.pkl')

    w_opt_G2=np.loadtxt("w_opt_best_G2.txt")
    reg_G2=joblib.load('reg_model_G2.pkl')

    pf1 = MCS_reliability(w_opt_G1, reg_G1, x_opt, x_mcs, x_std, trainer_G1.Scaler_X, trainer_G1.Scaler_G)
    pf2 = MCS_reliability(w_opt_G2, reg_G2, x_opt, x_mcs, x_std, trainer_G2.Scaler_X, trainer_G2.Scaler_G)
	
    iteration_log.append({
    'x': x_opt.copy(),
    'pf1': pf1,
    'pf2': pf2,
    'Re1': 1 - pf1,
    'Re2': 1 - pf2,
    'cost': cost(x_opt)
      })
	

    Re1=1-pf1
    Re2=1-pf2
    reliable = Re1 >= 0.95 and Re2 >= 0.95
    stable = k > 0 and abs(iteration_log[-1]['cost'] - iteration_log[-2]['cost']) < 5e-4

    if reliable and stable:
        print("The optimization is terminated")
        break


records = []
for i, entry in enumerate(iteration_log):
    row = {'iteration': i + 1}
    for j, val in enumerate(entry['x']):
        row[f'x{j+1}'] = val

    row['cost'] = entry['cost']
    row['Re1'] = 1-entry['pf1']
    row['Re2'] = 1-entry['pf2']
    records.append(row)


# Save as an Excel file
df = pd.DataFrame(records)
print("✅ iteration_log has been successfully saved as an Excel file: RBDO_iteration_log_final.xlsx")
