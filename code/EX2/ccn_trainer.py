# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.distributions.empirical_distribution import ECDF
import joblib
from cma import fmin
import mxnet as mx
import matplotlib.pyplot as plt
import warnings
import GPy
from keras.models import Sequential
from keras.layers.core import Dense
from scipy.stats import kurtosis, skew
from scipy.stats import pearsonr
import scipy.stats
from scipy.stats import ks_2samp


warnings.filterwarnings("ignore")

class EarlyStopException(Exception):
    pass

# ========== Autoencoder ==========
class autoencoder(mx.gluon.Block):
    def __init__(self, input_dim, latent_dim):
        super(autoencoder, self).__init__()
        with self.name_scope():
            self.encoder = mx.gluon.nn.Sequential('encoder_')
            with self.encoder.name_scope():
                self.encoder.add(mx.gluon.nn.Dense(latent_dim, activation='sigmoid'))
            self.decoder = mx.gluon.nn.Sequential('decoder_')
            with self.decoder.name_scope():
                self.decoder.add(mx.gluon.nn.Dense(input_dim + 1))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, dataloader, iterations):
        self.collect_params().initialize(mx.init.Xavier(), ctx=mx.cpu())
        criterion = mx.gluon.loss.HuberLoss()
        optimizer = mx.gluon.Trainer(self.collect_params(), 'adam', {'learning_rate': 1e-3, 'wd': 1e-5})
        for epoch in range(iterations):
            running_loss = 0.0
            n_total = 0.0
            for data in dataloader:
                x, _ = data
                with mx.autograd.record():
                    output = self.forward(x)
                    loss = criterion(output, x)
                loss.backward()
                optimizer.step(x.shape[0])
                running_loss += mx.nd.sum(loss).asscalar()
                n_total += x.shape[0]
            # if epoch % 100 == 0:
            #     print(f'epoch [{epoch + 1}/{iterations}], loss:{running_loss / n_total:.4f}')

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

def calculate_metrics(y_true, y_pred):
    y_min = min(np.min(y_true), np.min(y_pred))
    y_max = max(np.max(y_true), np.max(y_pred))
    y_true_normalized = (y_true - y_min) / (y_max - y_min)
    y_pred_normalized = (y_pred - y_min) / (y_max - y_min)
    mse = mean_squared_error(y_true_normalized, y_pred_normalized)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_normalized, y_pred_normalized)
    r2 = r2_score(y_true_normalized, y_pred_normalized)
    smape = np.mean(2 * np.abs(y_true_normalized - y_pred_normalized) / (np.abs(y_true_normalized) + np.abs(y_pred_normalized) + 1e-10)) * 100
    Rou,p_value=pearsonr(y_true_normalized, y_pred_normalized)
	
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2, 'SMAPE': smape, 'pearsonr': Rou}


def check_pf_convergence(SC, idx, delta_thresh=5e-4, avg_thresh=0.980, N=5):
    if idx < N:
        return False
    recent = np.array(SC[idx - N + 1 : idx + 1]).flatten()
    diffs = np.abs(np.diff(recent))
    if np.all(diffs < delta_thresh) and np.mean(recent) > avg_thresh:
        return True
    return False


class CCNTrainer:
    def __init__(self, X_train_o, G_train_o, X_valid_o, G_valid_o, X_mcs_o, G_mcs_o, AE_epoch, CCN_latent, Index_loss3, Index_loss8, Index_lossv, R2_avg):
        self.X_train_o = X_train_o
        self.G_train_o = G_train_o
        self.X_valid_o = X_valid_o
        self.G_valid_o = G_valid_o
        self.X_mcs_o = X_mcs_o
        self.G_mcs_o = G_mcs_o

        self.iter_opt = 0
        self.ii = 0
        self.w_opt_best = None

        #标准化
        self.Scaler_X = StandardScaler()    
        self.Scaler_G = StandardScaler()    

        self.X_train = self.Scaler_X.fit_transform(self.X_train_o)
        self.G_train = self.Scaler_G.fit_transform(self.G_train_o.reshape(-1, 1))
		
        self.X_valid = self.Scaler_X.transform(self.X_valid_o)
        self.G_valid = self.Scaler_G.transform(self.G_valid_o.reshape(-1, 1))

        self.X_mcs = self.Scaler_X.transform(self.X_mcs_o)
        self.G_mcs = self.Scaler_G.transform(self.G_mcs_o.reshape(-1, 1))

        self.n_train = self.X_train.shape[0]
        self.n_valid = self.X_valid.shape[0]
        self.n_mcs = self.X_mcs.shape[0]
        self.d = self.X_train.shape[1]
        self.Rt = (self.G_mcs_o > 0).sum() / self.G_mcs_o.shape[0]


        self.n_test = 1000
        indices = np.random.choice(self.n_mcs, self.n_test, replace=False)
        self.X_test_o = self.X_mcs_o[indices]
        self.G_test_o = self.G_mcs_o[indices]
        self.X_test = self.X_mcs[indices]
        self.G_test = self.G_mcs[indices]

        self.Batch_size = int(self.n_train / 2)

        self.SC = np.zeros((10000,1))
        self.d = self.X_train.shape[1]

        self.d_autoencoder = self.d

        self.G_train_predicted_PR = np.zeros((self.n_train, 100))
        self.G_mcs_predicted_PR = np.zeros((self.n_mcs, 100))
        self.G_valid_predicted_PR = np.zeros((self.n_valid, 100))
        self.G_test_predicted_PR = np.zeros((self.n_test, 100))

        self.autoencoder = None
		


        self.reg_latest = None     # 始终记录最近一次 reg
        self.w_latest = None       # 始终记录最近一次 w


        self.AE_epoch=AE_epoch


        self.d_lrnn = CCN_latent                                              ####高于2维的要改成2
        self.nn = NN(self.d, self.d_lrnn)
        self.Index_loss3=Index_loss3
        self.Index_loss8=Index_loss8
        self.Index_lossv=Index_lossv
        self.avg = R2_avg



    def loss_wrapper(self, w):
        return self.loss(w)

    def loss(self, w):
        self.nn.update(w)
        # Error-based loss (all)
        train_predicted_latent = self.nn.predict_latent(self.X_train)
        reg = LinearRegression().fit(train_predicted_latent, self.G_train.reshape(self.n_train,))
        GG = reg.predict(train_predicted_latent)
        GG_o = self.Scaler_G.inverse_transform(GG.reshape((-1,1)))
        G_tr_o = self.Scaler_G.inverse_transform(self.G_train.reshape((-1,1)))
        if self.Index_loss3==1:
            loss3_all,_ = ks_2samp(GG_o.flatten(), G_tr_o.flatten())   #ks:用于衡量CDF曲线重合程度的
        else:
            loss3_all = mean_squared_error(GG_o, G_tr_o.reshape(self.n_train,))

        # Error-based loss (randomly)
        sampleInd = np.random.choice(self.G_train.shape[0], size=(self.Batch_size,), replace=False)
        Gtr_random = G_tr_o[sampleInd]
        GG_random = GG_o[sampleInd]
        loss3_random = mean_squared_error(GG_random, Gtr_random.reshape(self.Batch_size,))

        #variance-based loss
        mcs_predicted_latent = self.nn.predict_latent(self.X_mcs)
        GG_mcs = reg.predict(mcs_predicted_latent)
        loss4 = np.var(mcs_predicted_latent, axis=0).min()

        #CCNs的精度指标(基于Training samples)
        train_metrics_PR = calculate_metrics(G_tr_o.reshape(self.n_train,), GG_o)

        #CCNs的精度指标（基于valid samples）
        valid_predicted_latent = self.nn.predict_latent(self.X_valid)
        GG_valid = reg.predict(valid_predicted_latent)
        GG_valid_o = self.Scaler_G.inverse_transform(GG_valid.reshape((-1,1)))
        valid_metrics_PR = calculate_metrics(self.G_valid_o.reshape(self.n_valid,), GG_valid_o)

        #CCNs的精度指标(基于MCS samples)
        GG_mcs_o = self.Scaler_G.inverse_transform(GG_mcs.reshape((-1,1)))
		
        #CCNs的精度指标(基于Test samples)
        test_predicted_latent = self.nn.predict_latent(self.X_test)
        GG_test = reg.predict(test_predicted_latent)
        GG_test_o = self.Scaler_G.inverse_transform(GG_test.reshape((-1,1)))
        test_metrics_PR = calculate_metrics(self.G_test_o.reshape(self.n_test,), GG_test_o)


        # total loss
        loss = loss3_all*100+ self.Index_lossv* max(0.1/(loss4+1e-10), 100)


        self.reg_latest = reg
        self.w_latest = w

        if self.iter_opt % 500 == 0:                                                           
            print("**************************************************************************", self.iter_opt // 500)
            print("Error-based loss(all)------>", loss3_all)
            print("Error-based loss (randomly)---------->", loss3_random)
            print("variance-based loss--------------------->", max(0.1/(loss4+1e-10), 100))
            print("tatol loss-->", loss)

            print("validation_metrics_PR---------->", valid_metrics_PR)
            print("Testing_metrics_PR---------->", test_metrics_PR)

            Re_mcs=1-(1 - self.G_mcs_o[self.G_mcs_o > 0].shape[0] / self.n_mcs)
            Re_pred=1-(1 - (GG_mcs_o > 0).sum()/GG_mcs_o.shape[0])
			
            print("True Re=", Re_mcs)
            print("approximate Re by PR====>", Re_pred, np.abs(Re_mcs - Re_pred) / (np.abs(Re_mcs)))


            print("----------------------------------------------------------------")
            print("true statistcs------>", [self.G_mcs_o.mean(), self.G_mcs_o.var(), skew(self.G_mcs_o), kurtosis(self.G_mcs_o)])
            print("por statistcs------>", [GG_mcs_o.mean(), GG_mcs_o.var(), skew(GG_mcs_o), kurtosis(GG_mcs_o)])
            print("----------------------------------------------------------------")


            print("Test_R2",test_metrics_PR["R²"])
            self.SC[self.ii] = test_metrics_PR["R²"]     
            print(f"SC = {self.SC[:self.ii+1].flatten()}")   
                     
            if check_pf_convergence(self.SC, self.ii, delta_thresh=1e-3, avg_thresh=self.avg, N=3):  
                print(f"✅ Converged!")
                self.w_opt_best = w
                if hasattr(self, 'reg_latest'):
                    joblib.dump(self.reg_latest, 'reg_model.pkl')
                else:
                    print("⚠️ Warning: reg_model.pkl was not saved")
                raise EarlyStopException("Custom stopping rule triggered.") 
            self.ii += 1
        self.iter_opt += 1
        return loss


    def optimize(self):
        options = {
            'maxiter': 5000,
            'maxfevals': 10000,
            'tolfun': 1e-8,
            'tolx': 1e-4,
            'tolstagnation': 15,
            'verb_disp': 1,
            'verb_log': 0
        }

        try:
            res = fmin(self.loss_wrapper, self.nn.w, 0.1, options=options)
        except EarlyStopException as e:
            print(f"🎯 CCNs optimization was terminated：{e}")

            #x_opt=res[0]
            self.w_opt_best = self.w_latest
            np.savetxt("w_opt_best.txt", self.w_opt_best)
            # 手动构建回归模型并保存
            if self.reg_latest is not None:
                joblib.dump(self.reg_latest, "reg_model.pkl")
                print("✅ reg_model.pkl has been saved")
            else:
                print("⚠️ No valid reg was found")
            return

        print("⚠️ Optimization terminated normally")
        #x_opt=res[0]
        self.w_opt_best = self.w_latest
        np.savetxt("w_opt_best.txt", self.w_opt_best)

        if self.reg_latest is not None:
            joblib.dump(self.reg_latest, "reg_model.pkl")
            print("✅ reg_model.pkl has been saved")
        else:
            print("⚠️ No valid reg was found")
         
    def evaluate(self):
        self.nn.update(np.loadtxt("w_opt_best.txt"))
        reg_loaded = joblib.load('reg_model.pkl')
        #Test samples
        Test_predicted_latent = self.nn.predict_latent(self.X_test)
        y_pred_test = reg_loaded.predict(Test_predicted_latent)
        y_pred_o_test = self.Scaler_G.inverse_transform(y_pred_test.reshape(-1,1))
        Pf_test = 1 - (y_pred_o_test > 0).sum() / y_pred_o_test.shape[0]
        R2_test = r2_score(self.G_test_o, y_pred_o_test.flatten())
        print("Pf_test and R2_test", 1-Pf_test, R2_test)        


        #MCS samples
        MCS_predicted_latent = self.nn.predict_latent(self.X_mcs)
        y_pred = reg_loaded.predict(MCS_predicted_latent)
        y_pred_o = self.Scaler_G.inverse_transform(y_pred.reshape(-1,1))
        Pf = 1 - (y_pred_o > 0).sum() / y_pred_o.shape[0]
        pf_actual = 1 - (self.G_mcs_o > 0).sum() / self.G_mcs_o.shape[0]
        R2 = r2_score(self.G_mcs_o, y_pred_o.flatten())
        print("Re_actual", 1-pf_actual)
        print("Re_pred and R2_pred", 1-Pf, R2)
		
        self.plot_results(self.G_mcs_o.flatten(), y_pred_o.flatten(), name="Proposed")

    def plot_results(self, y_true, y_pred, name="Result"):
        plt.figure(figsize=(5,5))
        plt.plot(y_true, y_true, color='black', linestyle='--', label="Ideal Line", linewidth=2)
        plt.scatter(y_true, y_pred, color='red', label=name, linewidth=0)
        plt.xlabel("True", fontsize=12)
        plt.ylabel("Predicted", fontsize=12)
        plt.legend(loc="best", fontsize=10)
        plt.grid(True)
        plt.title(f"{name} Scatter Plot")
        plt.savefig(f"{name}_Scatter.png", dpi=300, bbox_inches='tight')
        plt.show()

        plt.figure()
        hist = np.histogram(y_true, bins=100)
        hist_dist = scipy.stats.rv_histogram(hist)
        X = np.linspace(np.min(y_pred), np.max(y_pred), 100)
        plt.plot(X, hist_dist.pdf(X), label='True', color='black')
        hist = np.histogram(y_pred, bins=100)
        hist_dist = scipy.stats.rv_histogram(hist)
        plt.plot(X, hist_dist.pdf(X), label='Predicted', color='red')
        plt.legend(loc='upper right')
        plt.title(f"{name} PDF")
        plt.savefig(f"{name}_PDF.png", dpi=300, bbox_inches='tight')
        plt.show()

        ecdf_true = ECDF(y_true)
        ecdf_pred = ECDF(y_pred)
        plt.figure()
        plt.plot(ecdf_true.x, ecdf_true.y, label='True')
        plt.plot(ecdf_pred.x, ecdf_pred.y, label='Predicted')
        plt.legend(loc='upper right')
        plt.title(f"{name} CDF")
        plt.savefig(f"{name}_CDF.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def assessment(self, w_opt, reg_model, X_valid_o, G_model, model_name="G"):

        G_valid_o = G_model(X_valid_o)
        self.nn.update(w_opt)
        X_valid = self.Scaler_X.transform(X_valid_o)
        latent = self.nn.predict_latent(X_valid)
        y_pred = reg_model.predict(latent).reshape(-1, 1)
        y_pred_o = self.Scaler_G.inverse_transform(y_pred)
        R2 = r2_score(G_valid_o, y_pred_o.flatten())
        Pf = 1 - (y_pred_o > 0).sum() / y_pred_o.shape[0]
        print(f"🎯 {model_name} | Re_valid = {1-Pf:.4f}, R2_valid = {R2:.4f}")

        plt.figure(figsize=(5,5))
        plt.plot(G_valid_o, G_valid_o, color='black', linestyle='--', label="Ideal Line", linewidth=2)
        plt.scatter(G_valid_o, y_pred_o, color='red', label=model_name, s=12)
        plt.xlabel("True", fontsize=12)
        plt.ylabel("Predicted", fontsize=12)
        plt.legend(loc="best", fontsize=10)
        plt.grid(True)
        plt.title(f"{model_name} Scatter Plot")
        plt.savefig(f"{model_name}_scatter.png", dpi=300, bbox_inches='tight')
        plt.show()

        # 2. PDF 
        plt.figure()
        hist_true = np.histogram(G_valid_o, bins=100)
        dist_true = scipy.stats.rv_histogram(hist_true)
        X = np.linspace(min(G_valid_o.min(), y_pred_o.min()), max(G_valid_o.max(), y_pred_o.max()), 100)
        plt.plot(X, dist_true.pdf(X), label='True', color='black')
        hist_pred = np.histogram(y_pred_o, bins=100)
        dist_pred = scipy.stats.rv_histogram(hist_pred)
        plt.plot(X, dist_pred.pdf(X), label='Predicted', color='red')
        plt.legend()
        plt.title(f"{model_name} PDF")
        plt.savefig(f"{model_name}_pdf.png", dpi=300, bbox_inches='tight')
        plt.show()

        # 3. CDF 
        ecdf_true = ECDF(G_valid_o)
        ecdf_pred = ECDF(y_pred_o.flatten())
        plt.figure()
        plt.plot(ecdf_true.x, ecdf_true.y, label='True')
        plt.plot(ecdf_pred.x, ecdf_pred.y, label='Predicted')
        plt.legend()
        plt.title(f"{model_name} CDF")
        plt.savefig(f"{model_name}_cdf.png", dpi=300, bbox_inches='tight')
        plt.show()

    def pred(self, w_opt, reg_model, X_valid_o):
        self.nn.update(w_opt)
        X_valid = self.Scaler_X.transform(X_valid_o)
        latent = self.nn.predict_latent(X_valid)
        y_pred = reg_model.predict(latent).reshape(-1, 1)
        y_pred_o = self.Scaler_G.inverse_transform(y_pred)
        return y_pred_o.flatten()
