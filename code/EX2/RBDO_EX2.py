# -*- coding: utf-8 -*-
"""
"""
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
    base=np.sum(x[13:55])-np.sum(x[:13])
    bias_term = np.sum([(x[i] - 3)**2 for i in range(55) ]) 
    return base + bias_term
# # -------------------- Limit-State Function -------------------- 
def G2(X):
    X = np.asarray(X)
    squeeze_out = False
    if X.ndim == 1:         
        X = X[None, :]
        squeeze_out = True
    n, d = X.shape
    c = np.ones(d)
    special = {
        15: 5, 25: 15, 35: 25, 45: 35, 55: 45,
        65: 55, 75: 65, 85: 75, 95: 85, 100: 95
    }
    for idx, val in special.items():
        if idx <= d:
            c[idx - 1] = val

    g = X @ c - 2 * d-300-30

    if squeeze_out:
        return g[0]         
    return g                

def G7(X):
    X = np.asarray(X)
    squeeze_out = False
    if X.ndim == 1:   
        X = X[None, :]
        squeeze_out = True

    d = X.shape[1]               
    last_term = X[:, -1]         
    sum_squares = np.sum(X[:, :-1] ** 2, axis=1)  

    g = 3 - last_term + 0.01 * sum_squares-4

    if squeeze_out:
        return g[0]
    return g

def G12(X):
    X = np.asarray(X)
    squeeze_out = False
    if X.ndim == 1:   
        X = X[None, :]
        squeeze_out = True

    n = X.shape[1]                
    sum_x = np.sum(X, axis=1)
    g = n + 3 * np.sqrt(n) - sum_x + 140-40#-5

    if squeeze_out:
        return g[0]
    return g

def MCS_reliability(w_opt, reg_model, x, Xmcs, std, scaler_X, scaler_G):
    X_o = Xmcs * std + x
    #np.savetxt("X_o.txt", X_o)
	
    X = scaler_X.transform(X_o)

    nn = NN(d=55, d_latent=2)
    nn.update(w_opt)
    latent=nn.predict_latent(X)
    y_pred = reg_model.predict(latent)
    y_pred_o=scaler_G.inverse_transform(y_pred.reshape(-1,1)).flatten()
    Pf = np.mean(y_pred_o < 0)
    return Pf

def constraint_jac(w_opt, reg_model, x, Xmcs, std, scaler_X, scaler_G):
    X_o = Xmcs * std + x        
    X = scaler_X.transform(X_o)
    nn=NN(d=55, d_latent=2)
    nn.update(w_opt)
    latent = nn.predict_latent(X)
    y_pred = reg_model.predict(latent).reshape(-1)
    y_pred_o = scaler_G.inverse_transform(y_pred.reshape(-1,1)).flatten()

    I = (y_pred_o < 0).astype(float)  
    score = Xmcs / (std**2)
    grad = np.mean(I[:, None] * score, axis=0)
	
    return -grad  


def constraint_g1(w_opt, reg_model,  scaler_X, scaler_G, x):
    c_pf = Pf_target - MCS_reliability(w_opt, reg_model, x, Xmcs, std, scaler_X, scaler_G)
    return c_pf


def constraint_jac_g1(w_opt, reg_model, scaler_X, scaler_G, x ):  # 梯度1
    c_jac=constraint_jac(w_opt, reg_model, x, Xmcs, std, scaler_X, scaler_G)
    return c_jac



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
n_x = 55
bounds = np.array([(2, 4) for _ in range(n_x)])
l_bounds = bounds[:, 0]
u_bounds = bounds[:, 1]
Pf_target = 1 - 0.97
N_MCS = 1000000
std = np.full(n_x, 1)
max_iter = 15
iner_it=0
iteration_log = []


#arr = pd.read_excel("x0.xlsx", header=None)
#x_center = arr.to_numpy(dtype=np.float64).reshape(-1)
x_center = np.full(n_x, 3)
# ========== Step 2. Generate training, validation, and MCS samples ==========
N_LHS=150
N_train = 100
N_valid = N_LHS - N_train
valid_size=N_valid/N_LHS
N_MCS=1000000

mean = x_center
std = np.ones(n_x) * 1

X_LHS = datagen(N_LHS, n_x, mean, std, islhs=True)

X_train, X_valid, idx_train, idx_valid = train_test_split(
    X_LHS, np.arange(X_LHS.shape[0]), test_size=valid_size, random_state=42)

Xmcs = Standnormal(N_MCS,n_x)
X_mcs=Xmcs*std+mean

G2_train=G2(X_train)
G7_train=G7(X_train)
G12_train=G12(X_train)


G2_valid=G2(X_valid)
G7_valid=G7(X_valid)
G12_valid=G12(X_valid)


G2_mcs=G2(X_mcs)
G7_mcs=G7(X_mcs)
G12_mcs=G12(X_mcs)



# # #Train the initial CCNs models for  G2, G7, and G12
# # # G2 training
print("\n🚀 Start training the second constraint function: G2")
trainer_G2 = CCNTrainer(X_train, G2_train, X_valid, G2_valid, X_mcs, G2_mcs, 2000, 2, 0, 1 ,1, 0.98)  
trainer_G2.optimize()
trainer_G2.evaluate()
w_opt_G2 = np.loadtxt("w_opt_best.txt")
reg_model_G2 = joblib.load("reg_model.pkl")
np.savetxt(f"w_opt_best_G2.txt", w_opt_G2)
joblib.dump(reg_model_G2, "reg_model_G2.pkl")
print(f"✅ Training of constraint function G2 completed")
joblib.dump(trainer_G2.Scaler_X, 'Scaler_X_G2.pkl')
joblib.dump(trainer_G2.Scaler_G, 'Scaler_G_G2.pkl')

# G7 training
print("Start training the third constraint function: G7")
trainer_G7 = CCNTrainer(X_train, G7_train, X_valid, G7_valid, X_mcs, G7_mcs, 2000, 2, 0, 0 ,0, 0.98)
trainer_G7.optimize()
trainer_G7.evaluate()
w_opt_G7 = np.loadtxt("w_opt_best.txt")
reg_model_G7 = joblib.load("reg_model.pkl")
np.savetxt(f"w_opt_best_G7.txt", w_opt_G7)
joblib.dump(reg_model_G7, "reg_model_G7.pkl")
print(f"✅ Training of constraint function G7 completed")
joblib.dump(trainer_G7.Scaler_X, 'Scaler_X_G7.pkl')
joblib.dump(trainer_G7.Scaler_G, 'Scaler_G_G7.pkl')



# G12 training
print("Start training the third constraint function: G12")
trainer_G12 = CCNTrainer(X_train, G12_train, X_valid, G12_valid, X_mcs, G12_mcs, 2000, 2, 0, 1 ,1, 0.98)  
trainer_G12.optimize()
trainer_G12.evaluate()

w_opt_G12 = np.loadtxt("w_opt_best.txt")
reg_model_G12 = joblib.load("reg_model.pkl")
np.savetxt(f"w_opt_best_G12.txt", w_opt_G12)
joblib.dump(reg_model_G12, "reg_model_G12.pkl")
print(f"✅ Training of constraint function G12 completed")
joblib.dump(trainer_G12.Scaler_X, 'Scaler_X_G12.pkl')
joblib.dump(trainer_G12.Scaler_G, 'Scaler_G_G12.pkl')



w_opt_G2=np.loadtxt("w_opt_best_G2.txt")
reg_G2=joblib.load('reg_model_G2.pkl')
Scaler_X_G2 = joblib.load('Scaler_X_G2.pkl')
Scaler_G_G2 = joblib.load('Scaler_G_G2.pkl')


w_opt_G7=np.loadtxt("w_opt_best_G7.txt")
reg_G7=joblib.load('reg_model_G7.pkl')
Scaler_X_G7 = joblib.load('Scaler_X_G7.pkl')
Scaler_G_G7 = joblib.load('Scaler_G_G7.pkl')

w_opt_G12=np.loadtxt("w_opt_best_G12.txt")
reg_G12=joblib.load('reg_model_G12.pkl')
Scaler_X_G12 = joblib.load('Scaler_X_G12.pkl')
Scaler_G_G12 = joblib.load('Scaler_G_G12.pkl')


pf2 = MCS_reliability(w_opt_G2, reg_G2, x_center, Xmcs, std, Scaler_X_G2, Scaler_G_G2)
pf7 = MCS_reliability(w_opt_G7, reg_G7, x_center, Xmcs, std, Scaler_X_G7, Scaler_G_G7)
pf12 = MCS_reliability(w_opt_G12, reg_G12, x_center, Xmcs, std, Scaler_X_G12, Scaler_G_G12)
	
iteration_log.append({
    'x': x_center,
    'pf2': pf2,
    'pf7': pf7,	
    'pf12': pf12,
    'cost': cost(x_center)})

x_history = [x_center.copy()]


# ========== Step 3. RBDO Main Loop ==========
for k in range(max_iter):
    print(f"\n---- RBDO outer iteration {k+1} ----")

    constraints = []
    gp_models = [
       (w_opt_G2, reg_G2, trainer_G2.Scaler_X, trainer_G2.Scaler_G),
       (w_opt_G7, reg_G7, trainer_G7.Scaler_X, trainer_G7.Scaler_G),	   
       (w_opt_G12, reg_G12, trainer_G12.Scaler_X, trainer_G12.Scaler_G)]


    for w_opt, reg_model, scaler_X, scaler_G in gp_models:
        con_fun = partial(constraint_g1, w_opt, reg_model, scaler_X, scaler_G)
        #con_jac = partial(constraint_jac_g1, w_opt, reg_model, scaler_X, scaler_G)
        constraints.append({'type': 'ineq', 'fun': con_fun})#,# 'jac': con_jac

    #设计变量优化
    result = minimize(
        cost, x_center, method='SLSQP', bounds=bounds,
        constraints=constraints, 
        #callback=iteration_callback,  # 加这一行
        options={'disp': True, 'ftol': 3e-3, 'maxiter': 200, 'eps': 1e-2}
    )

    #优化结果
    x_opt = result.x.copy()

    x_center = x_opt.copy()
    x_history.append(x_opt.copy())
    X_train_new = datagen(N_train, n_x, x_center, std, islhs=True)
    X_valid_new = datagen(N_valid, n_x, x_center, std, islhs=True)
    X_mcs_new=Xmcs*std+x_center

    G_funcs = [G2, G7, G12] #[G1, G2, G3, G4, G6, G7, G9, G10, G12]
    names = ["G2", "G7", "G12"] #["G1", "G2", "G3", "G4", "G6", "G7","G9", "G10", "G12"]
    for i, (Gi, name) in enumerate(zip(G_funcs, names), start=1):
        train_values = Gi(X_train_new)
        valid_values = Gi(X_valid_new)
        MCS_values = Gi(X_mcs_new)

        globals()[f"{name}_train"] = train_values
        globals()[f"{name}_valid"] = valid_values
        globals()[f"{name}_mcs"] = MCS_values

        print(f"\n🚀 Start training constraint function {i}: {name}")
        if name == "G2":
            paras=[2000,2,0,1,1,0.98]
        elif name == "G7":
            paras=[2000,2,0,0,0,0.98]
        elif name == "G12":
            paras=[2000,2,0,1,1,0.98]


        trainer = CCNTrainer(X_train_new, train_values, X_valid_new, valid_values, X_mcs_new, MCS_values, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5])
        #trainer.train_autoencoder()
        trainer.optimize()
        trainer.evaluate()
		
        w_opt= np.loadtxt("w_opt_best.txt")
        reg_model = joblib.load("reg_model.pkl")

        np.savetxt(f"w_opt_best_{name}.txt", w_opt)
        joblib.dump(reg_model, f"reg_model_{name}.pkl")
    
        print(f"✅ Training of constraint function {name} completed")


        if name == "G2":
            trainer_G2 = trainer
        elif name == "G7":
            trainer_G7 = trainer
        elif name == "G12":
            trainer_G12 = trainer			
			
			
    w_opt_G2=np.loadtxt("w_opt_best_G2.txt")
    reg_G2=joblib.load('reg_model_G2.pkl')


    w_opt_G7=np.loadtxt("w_opt_best_G7.txt")
    reg_G7=joblib.load('reg_model_G7.pkl')


    w_opt_G12=np.loadtxt("w_opt_best_G12.txt")
    reg_G12=joblib.load('reg_model_G12.pkl')



    pf2 = MCS_reliability(w_opt_G2, reg_G2, x_opt, Xmcs, std, trainer_G2.Scaler_X, trainer_G2.Scaler_G)
    pf7 = MCS_reliability(w_opt_G7, reg_G7, x_opt, Xmcs, std, trainer_G7.Scaler_X, trainer_G7.Scaler_G)
    pf12 = MCS_reliability(w_opt_G12, reg_G12, x_opt, Xmcs, std, trainer_G12.Scaler_X, trainer_G12.Scaler_G)


    iteration_log.append({
        'x': x_opt.copy(),
        'pf2': pf2,
        'pf7': pf7,	
        'pf12': pf12, 
     'cost': cost(x_opt)})

    Re2=1-pf2
    Re7=1-pf7
    Re12=1-pf12
    reliable = Re2 >= 0.97 and Re7 >= 0.97 and Re12>=0.97
    stable = k > 0 and abs(iteration_log[-1]['cost'] - iteration_log[-2]['cost']) < 6e-2

    if reliable and stable:
        print("The optimization is terminated")
        break 
    
records = []
for i, entry in enumerate(iteration_log):
    row = {'iteration': i + 1}
    for j, val in enumerate(entry['x']):
        row[f'x{j+1}'] = val
    row['cost'] = entry['cost']
    row['pf2'] = entry['pf2']
    row['pf7'] = entry['pf7']
    row['pf12'] = entry['pf12']
    records.append(row)


df = pd.DataFrame(records)
print("✅ iteration_log has been successfully saved as an Excel file: RBDO_iteration_log_final.xlsx")
