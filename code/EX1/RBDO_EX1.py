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
    base = (
        x[0]**2 + x[1]**2 + x[0]*x[1]
        - 14*x[0] - 16*x[1]
        + (x[2] - 10)**2
        + 4*(x[3] - 5)**2
        + (x[4] - 3)**2
        + 2*(x[5] - 1)**2
        + 5*x[6]**2
        + 7*(x[7] - 11)**2
        + 2*(x[8] - 10)**2
        + (x[9] - 7)**2
        + 45-300
    )

    high_dim_term = np.sum((x[10:18] - 3)**2)

    return base + high_dim_term


# # -------------------- Limit-State Function -------------------- 
def G2(X):
    X = np.asarray(X)
    squeeze_out = False
    if X.ndim == 1:
        X = X[None, :]
        squeeze_out = True

    total = np.zeros(X.shape[0])
    for i in range(17):
        total += (
            (1 - X[:, i])
            + 100*(X[:, i+1] - X[:, i]**2)**2
        )
    g = total / 300000 -1.03 #- 1.05

    return g[0] if squeeze_out else g


def G3(X):
    X = np.asarray(X)
    squeeze_out = False
    if X.ndim == 1:
        X = X[None, :]
        squeeze_out = True

    x1, x2, x9, x10 = X[:, 0], X[:, 1], X[:, 8], X[:, 9]

    g = (
        (-8*x1 + 2*x2 + 5*x9 - 2*x10) / 12 - 0.5
        + 0.05*np.sum((X[:, :18] - 3)**2, axis=1)
        #+ 0.2*(x9 - x10)#**2
        + 0.2*np.sin(np.sum(X[:, :15], axis=1))
        + 0.4*np.cos(1.5*X[:, 15] + 1.0*X[:, 16] - 0.5*X[:, 17])
        - 0.5#- 0.182#
    )

    return g[0] if squeeze_out else g


def MCS_reliability(w_opt, reg_model, x, Xmcs, std, scaler_X, scaler_G):
    X_o = Xmcs * std + x
    #np.savetxt("X_o.txt", X_o)
	
    X = scaler_X.transform(X_o)
    nn = NN(d=18, d_latent=2)
    nn.update(w_opt)
    latent=nn.predict_latent(X)
    y_pred = reg_model.predict(latent)
    y_pred_o=scaler_G.inverse_transform(y_pred.reshape(-1,1)).flatten()
	
    Pf = round(np.mean(y_pred_o < 0), 2) #np.mean(y_pred_o < 0)
    return Pf


def constraint_jac(w_opt, reg_model, x, Xmcs, std, scaler_X, scaler_G):
    X_o = Xmcs * std + x      
    
    X = scaler_X.transform(X_o)

    nn=NN(d=18, d_latent=2)
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


def constraint_jac_g1(w_opt, reg_model, scaler_X, scaler_G, x ):
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
n_x = 18
bounds = np.array([(1, 5) for _ in range(n_x)])
l_bounds = bounds[:, 0]
u_bounds = bounds[:, 1]
Pf_target = 1 - 0.95
N_MCS = 100000
std = np.full(n_x, 0.02)
np.savetxt("std.txt",std)
max_iter = 100     
iner_it=0
iteration_log = []

x_center = np.full(n_x, 2.2)

x_center[:10] = np.array([
    4.2, 4.8, 4.8, 4.8, 3.0,
    1.2, 1.2, 4.8, 4.8, 4.8
])

x_center[10:18] = 3.0



# ========== Step 2. Generate training, validation, and MCS samples ==========
N_LHS=150
N_train = 60
N_valid = N_LHS - N_train
valid_size=N_valid/N_LHS
N_MCS=100000

mean = x_center
std = np.ones(n_x) * 0.02

X_train = datagen(N_train, n_x, mean, std, islhs=True)
X_valid = datagen(N_valid, n_x, mean, std, islhs=True)


Xmcs = np.random.normal(0, 1, size=(N_MCS, n_x))
X_mcs=Xmcs*std+mean


G2_train=G2(X_train)
G3_train=G3(X_train)

G2_valid=G2(X_valid)
G3_valid=G3(X_valid)

G2_mcs=G2(X_mcs)
G3_mcs=G3(X_mcs)

# # #Train the initial CCNs models for  G2, and G3
# # G2 training
print("\n🚀 Start training the second constraint function: G2")
trainer_G2 = CCNTrainer(X_train, G2_train, X_valid, G2_valid, X_mcs, G2_mcs, 2000, 2, 0, 1 ,1)
trainer_G2.optimize()
trainer_G2.evaluate()
# Save the optimal weights and linear regression model for G2
w_opt_G2 = np.loadtxt("w_opt_best.txt")
reg_model_G2 = joblib.load("reg_model.pkl")
np.savetxt(f"w_opt_best_G2.txt", w_opt_G2)
joblib.dump(reg_model_G2, "reg_model_G2.pkl")
print(f"✅ Training of constraint function G2 completed")
joblib.dump(trainer_G2.Scaler_X, 'Scaler_X_G2.pkl')
joblib.dump(trainer_G2.Scaler_G, 'Scaler_G_G2.pkl')

# # G3 training
print("\n🚀 Start training the third constraint function: G3")
trainer_G3 = CCNTrainer(X_train, G3_train, X_valid, G3_valid, X_mcs, G3_mcs, 2000, 2, 0, 1 ,1)  
trainer_G3.optimize()
trainer_G3.evaluate()

w_opt_G3 = np.loadtxt("w_opt_best.txt")
reg_model_G3 = joblib.load("reg_model.pkl")
np.savetxt(f"w_opt_best_G3.txt", w_opt_G3)
joblib.dump(reg_model_G3, "reg_model_G3.pkl")
print(f"✅ Training of constraint function G3 completed")
joblib.dump(trainer_G3.Scaler_X, 'Scaler_X_G3.pkl')
joblib.dump(trainer_G3.Scaler_G, 'Scaler_G_G3.pkl')

##Accuracy validation of the CCNs models for G2 and G3
w_opt_G2=np.loadtxt("w_opt_best_G2.txt")
reg_G2=joblib.load('reg_model_G2.pkl')
Scaler_X_G2 = joblib.load('Scaler_X_G2.pkl')
Scaler_G_G2 = joblib.load('Scaler_G_G2.pkl')

w_opt_G3=np.loadtxt("w_opt_best_G3.txt")
reg_G3=joblib.load('reg_model_G3.pkl')
Scaler_X_G3 = joblib.load('Scaler_X_G3.pkl')
Scaler_G_G3 = joblib.load('Scaler_G_G3.pkl')

pf2 = MCS_reliability(w_opt_G2, reg_G2, x_center, Xmcs, std, Scaler_X_G2, Scaler_G_G2)
pf3 = MCS_reliability(w_opt_G3, reg_G3, x_center, Xmcs, std, Scaler_X_G3, Scaler_G_G3)


print(f"Optimization results at iteration {0}: x={np.round(x_center, 5)}")
print(f"Objective function value: cost = {cost(x_center):.4f}")
print(f"Reliability: Re2 = {1-pf2:.6f}, Re3={1-pf3:.6f}")

actual_Re2=1-np.mean(G2(X_mcs)< 0)
actual_Re3=1-np.mean(G3(X_mcs)< 0)

print(f"Reliability：Re2 = {1-pf2:.6f}, Re3={1-pf3:.6f}")

iteration_log.append({
    'x': x_center,
    'pf2': pf2,
    'pf3': pf3,
    'Re2': 1 - pf2,
    'Re3': 1 - pf3,
    'actual_Re2': actual_Re2,
    'actual_Re3': actual_Re3,
    'cost': cost(x_center)
      })

x_history = [x_center.copy()]

# ========== Step 3. RBDO Main Loop ==========
for k in range(max_iter):
    print(f"\n---- RBDO outer iteration {k+1} ----")

    constraints = []
    gp_models = [
       (w_opt_G2, reg_G2, trainer_G2.Scaler_X, trainer_G2.Scaler_G),
       (w_opt_G3, reg_G3, trainer_G3.Scaler_X, trainer_G3.Scaler_G)]
	   

    for w_opt, reg_model, scaler_X, scaler_G in gp_models:
        con_fun = partial(constraint_g1, w_opt, reg_model, scaler_X, scaler_G)
        #con_jac = partial(constraint_jac_g1, w_opt, reg_model, scaler_X, scaler_G)
        constraints.append({'type': 'ineq', 'fun': con_fun})#,# 'jac': con_jac

    result = minimize(
        cost, x_center, method='SLSQP', bounds=bounds,
        constraints=constraints, 
        #callback=iteration_callback,  
        options={'disp': True, 'ftol': 5e-3, 'maxiter': 200, 'eps': 1e-2}
    )


    x_opt = result.x.copy()

    x_center = x_opt.copy()
    x_history.append(x_opt.copy())
    X_train_new = datagen(N_train, n_x, x_center, std, islhs=True)
    X_valid_new = datagen(N_valid, n_x, x_center, std, islhs=True)
    X_mcs_new=Xmcs*std+x_center

    G_funcs = [ G2, G3] #[G1, G2, G3, G4, G6, G7, G9, G10, G12]
    names = [ "G2", "G3"] #["G1", "G2", "G3", "G4", "G6", "G7","G9", "G10", "G12"]
    for i, (Gi, name) in enumerate(zip(G_funcs, names), start=1):
        train_values = Gi(X_train_new)
        valid_values = Gi(X_valid_new)
        MCS_values = Gi(X_mcs_new)


        globals()[f"{name}_train"] = train_values
        globals()[f"{name}_valid"] = valid_values
        globals()[f"{name}_mcs"] = MCS_values

        print(f"\n🚀 Start training constraint function {i}: {name}")

        if name == "G2":
            paras=[2000,2,0,1,1]
        elif name == "G3":
            paras=[2000,2,0,1,1]


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
        elif name == "G3":
            trainer_G3 = trainer 
			
			
    w_opt_G2=np.loadtxt("w_opt_best_G2.txt")
    reg_G2=joblib.load('reg_model_G2.pkl')

    w_opt_G3=np.loadtxt("w_opt_best_G3.txt")
    reg_G3=joblib.load('reg_model_G3.pkl')

    pf2 = MCS_reliability(w_opt_G2, reg_G2, x_opt, Xmcs, std, trainer_G2.Scaler_X, trainer_G2.Scaler_G)
    pf3 = MCS_reliability(w_opt_G3, reg_G3, x_opt, Xmcs, std, trainer_G3.Scaler_X, trainer_G3.Scaler_G)


    print(f"Optimization results at iteration {k+1}: x={np.round(x_opt, 5)}")
    print(f"Objective function value: cost = {cost(x_opt):.4f}")
    print(f"Reliability: Re2 = {1-pf2:.6f}, Re3 = {1-pf3:.6f}")


    iteration_log.append({
    'x': x_opt.copy(),
    'pf2': pf2,
    'pf3': pf3,
    'Re2': 1 - pf2,
    'Re3': 1 - pf3,
    'cost': cost(x_opt)
      })

    Re2=1-pf2
    Re3=1-pf3
    reliable = Re2 >= 0.95 and Re3 >= 0.95
    stable = k > 0 and abs(iteration_log[-1]['cost'] - iteration_log[-2]['cost']) < 1e-4

    if reliable and stable:
        print("The optimization is terminated")
        break


records = []
for i, entry in enumerate(iteration_log):
    row = {'iteration': i + 1}

    for j, val in enumerate(entry['x']):
        row[f'x{j+1}'] = val

    row['cost'] = entry['cost']
    #row['pf1'] = entry['pf1']
    row['pf2'] = entry['pf2']
    row['pf3'] = entry['pf3']
    records.append(row)

records = []
for i, entry in enumerate(iteration_log):
    row = {'iteration': i+1}

    for j, val in enumerate(entry['x']):
        row[f'x{j+1}'] = val

    row['cost'] = entry['cost']
        #row['pf1'] = entry['pf1']
    row['pf2'] = entry['pf2']
    row['pf3'] = entry['pf3']
    #row['Re1'] = 1 - entry['pf1']
    row['Re2'] = 1 - entry['pf2']
    row['Re3'] = 1 - entry['pf3']
    records.append(row)
# Save as an Excel file
df = pd.DataFrame(records)
print("✅ iteration_log has been successfully saved as an Excel file: RBDO_iteration_log_final.xlsx")
