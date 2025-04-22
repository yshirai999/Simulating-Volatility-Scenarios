import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import numpy as np
from BaseEnv import BaseClass
import matplotlib.pyplot as plt
from tqdm import tqdm
from Transformer import TransformerModel

# input dim should be 4, output dim is 4
class Transformer4D(BaseClass):
    def __init__(self, tickers, feature_steps, target_steps, scaler,
                input_dim, d_model, num_heads, num_layers, dim_feedforward, output_dim, seq_length, 
                dropout, quantized,quant_all, n_clusters, scale, num_epochs, quantiles=(0.1,0.9), device='cpu'):
        super().__init__(tickers=tickers, feature_steps = feature_steps, target_steps = target_steps, scaler = scaler, quantized=quantized, quant_all=quant_all, nclusters=n_clusters, vals=[0,1,2,3,4])
        # need to change the base class to have al four values be in the time series
        self.num_epochs = num_epochs
        self.valid_pred = {}
        self.train_pred = {}
        self.valid_pred = {}
        self.test_pred = {}
        self.train_errors = {}
        self.valid_errors = {}
        self.test_errors = {}
        self.test_dates = {}
        self.history = {}
        self.models = {}
        self.train_series = {}
        self.quantiles = quantiles
        self.device = device
        for t in self.tickers:
            self.train_series[t] = np.concatenate( (self.y_train[t],self.y_valid[t],self.y_test[t]), axis=0)
            self.models[t] = {}
            self.models[t] = TransformerModel(input_dim, d_model, num_heads, num_layers, dim_feedforward, output_dim, seq_length, dropout, scale, quantiles=quantiles, device = self.device)
            self.train_pred[t] = {}
            self.valid_pred[t] = {}
            self.test_pred[t] = {}
            self.train_errors[t] = {}
            self.valid_errors[t] = {}
            self.test_errors[t] = {}
            self.test_dates[t] = {}
            self.history[t] = {}

    def train(self):
        for t in self.tickers:
            X_train = torch.tensor(self.X_train[t], dtype=torch.float32)
            y_train = torch.tensor(self.y_train[t], dtype=torch.float32)
            X_valid = torch.tensor(self.X_valid[t], dtype=torch.float32)
            y_valid = torch.tensor(self.y_valid[t], dtype=torch.float32)

            y_train = y_train.unsqueeze(1)
            y_valid = y_valid.unsqueeze(1)

            self.train_errors[t], self.valid_errors[t] = self.models[t].train_model(X_train=X_train, y_train=y_train, X_val=X_valid, y_val=y_valid, num_epochs = self.num_epochs)
            print(f"Final Train loss : {self.train_errors[t][-1]} final Val loss: {self.valid_errors[t][-1]}")

    def predict(self):
        for t in self.tickers:
            X_test = torch.tensor(self.X_test[t],dtype=torch.float32)
            X_train = torch.tensor(self.X_train[t], dtype=torch.float32)
            X_valid = torch.tensor(self.X_valid[t], dtype=torch.float32)

            self.test_pred[t] = self.models[t].predict(X_test)

            self.train_pred[t] = self.models[t].predict(X_train)

            self.valid_pred[t] = self.models[t].predict(X_valid)
    
    def plot_predictions(self,t,  title):
        num_samples_train = self.X_train[t].shape[0]
        num_samples_valid = self.X_valid[t].shape[0]
        num_samples_test = self.X_test[t].shape[0]

        bn_train = self.train_pred[t][:,:,0].reshape(num_samples_train).detach().numpy()
        bp_train = self.train_pred[t][:,:,1].reshape(num_samples_train).detach().numpy()

        bn_valid = self.valid_pred[t][:,:,0].reshape(num_samples_valid).detach().numpy()
        bp_valid = self.valid_pred[t][:,:,1].reshape(num_samples_valid).detach().numpy()

        bn_test = self.test_pred[t][:,:,0].reshape(num_samples_test).detach().numpy()
        bp_test = self.test_pred[t][:,:,1].reshape(num_samples_test).detach().numpy()

        fig, axes = plt.subplots(3, 2, figsize=(10, 10))  # 3 rows, 2 columns
        fig.suptitle(f"{t} ({title})", fontsize=16)
            
        axes[0,0].plot(bp_train, label="Predictions")
        axes[0,0].plot(self.y_train[t][:,0][..., np.newaxis, np.newaxis].reshape(num_samples_train), label="True")
        axes[0,0].set_title(f"bp Train Predictions vs Truth")
        axes[0,0].legend()

        axes[0,1].plot(bn_train, label="Predictions")
        axes[0,1].plot(self.y_train[t][:,1][..., np.newaxis, np.newaxis].reshape(num_samples_train), label="True")
        axes[0,1].set_title(f"bn Train Predictions vs Truth")
        axes[0,1].legend()

        axes[1,0].plot(bp_valid, label="Predictions")
        axes[1,0].plot(self.y_valid[t][:,0][..., np.newaxis, np.newaxis].reshape(num_samples_valid), label="True")
        axes[1,0].set_title(f"bp Validation Predictions vs Truth")
        axes[1,0].legend()

        axes[1,1].plot(bn_valid, label="Predictions")
        axes[1,1].plot(self.y_valid[t][:,1][..., np.newaxis, np.newaxis].reshape(num_samples_valid), label="True")
        axes[1,1].set_title(f"bn Validation Predictions vs Truth")
        axes[1,1].legend()

        axes[2,0].plot(bp_test, label="Predictions")
        axes[2,0].plot(self.y_test[t][:,0][..., np.newaxis, np.newaxis].reshape(num_samples_test), label="True")
        axes[2,0].set_title(f"bp Test Predictions vs Truth")
        axes[2,0].legend()

        axes[2,1].plot(bn_test, label="Predictions")
        axes[2,1].plot(self.y_test[t][:,1][..., np.newaxis, np.newaxis].reshape(num_samples_test), label="True")
        axes[2,1].set_title(f"bn Test Predictions vs Truth")
        axes[2,1].legend()

        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()

    def plot_quantile_predicitons(self, t, title):
        num_samples_train = self.X_train[t].shape[0]
        num_samples_valid = self.X_valid[t].shape[0]
        num_samples_test = self.X_test[t].shape[0]
        print(f"Train preds shape : {self.train_pred[t].shape}")
        if self.device == 'cuda':
            self.train_pred[t] = self.train_pred[t].cpu()
            self.valid_pred[t] = self.valid_pred[t].cpu()
            self.test_pred[t] = self.test_pred[t].cpu()

        bn_train_q1 = self.train_pred[t][:,:,0].reshape(num_samples_train).detach().numpy()
        bn_train_q2 = self.train_pred[t][:,:,1].reshape(num_samples_train).detach().numpy()
        bp_train_q1 = self.train_pred[t][:,:,2].reshape(num_samples_train).detach().numpy()
        bp_train_q2 = self.train_pred[t][:,:,3].reshape(num_samples_train).detach().numpy()

        bn_valid_q1 = self.valid_pred[t][:,:,0].reshape(num_samples_valid).detach().numpy()
        bn_valid_q2 = self.valid_pred[t][:,:,1].reshape(num_samples_valid).detach().numpy()
        bp_valid_q1 = self.valid_pred[t][:,:,2].reshape(num_samples_valid).detach().numpy()
        bp_valid_q2 = self.valid_pred[t][:,:,3].reshape(num_samples_valid).detach().numpy()

        bn_test_q1 = self.test_pred[t][:,:,0].reshape(num_samples_test).detach().numpy()
        bn_test_q2 = self.test_pred[t][:,:,1].reshape(num_samples_test).detach().numpy()
        bp_test_q1 = self.test_pred[t][:,:,2].reshape(num_samples_test).detach().numpy()
        bp_test_q2 = self.test_pred[t][:,:,3].reshape(num_samples_test).detach().numpy()

        fig, axes = plt.subplots(3, 2, figsize=(10, 10))  # 3 rows, 2 columns
        fig.suptitle(f"{t} ({title})", fontsize=16)
        
        lq = self.quantiles[0]
        uq = self.quantiles[1]
        axes[0,0].plot(bp_train_q1, linestyle='--', label=f"{lq*100}% q pred")
        axes[0,0].plot(bp_train_q2, linestyle='--', label=f"{uq*100}% q pred")
        axes[0,0].plot(self.y_train[t][:,0][..., np.newaxis, np.newaxis].reshape(num_samples_train), label="True")
        axes[0,0].set_title(f"bp Train Predictions vs Truth")
        axes[0,0].legend()

        axes[0,1].plot(bn_train_q1, linestyle='--', label=f"{lq*100}% q pred")
        axes[0,1].plot(bn_train_q2, linestyle='--', label=f"{uq*100}% q pred")
        axes[0,1].plot(self.y_train[t][:,1][..., np.newaxis, np.newaxis].reshape(num_samples_train), label="True")
        axes[0,1].set_title(f"bn Train Predictions vs Truth")
        axes[0,1].legend()

        axes[1,0].plot(bp_valid_q1, linestyle='--', label=f"{lq*100}% q pred")
        axes[1,0].plot(bp_valid_q2, linestyle='--', label=f"{uq*100}% q pred")
        axes[1,0].plot(self.y_valid[t][:,0][..., np.newaxis, np.newaxis].reshape(num_samples_valid), label="True")
        axes[1,0].set_title(f"bp Validation Predictions vs Truth")
        axes[1,0].legend()

        axes[1,1].plot(bn_valid_q1, linestyle='--', label=f"{lq*100}% q pred")
        axes[1,1].plot(bn_valid_q2, linestyle='--', label=f"{uq*100}% q pred")
        axes[1,1].plot(self.y_valid[t][:,1][..., np.newaxis, np.newaxis].reshape(num_samples_valid), label="True")
        axes[1,1].set_title(f"bn Validation Predictions vs Truth")
        axes[1,1].legend()

        axes[2,0].plot(bp_test_q1, linestyle='--', label=f"{lq*100}% q pred")
        axes[2,0].plot(bp_test_q2, linestyle='--', label=f"{uq*100}% q pred")
        axes[2,0].plot(self.y_test[t][:,0][..., np.newaxis, np.newaxis].reshape(num_samples_test), label="True")
        axes[2,0].set_title(f"bp Test Predictions vs Truth")
        axes[2,0].legend()

        axes[2,1].plot(bn_test_q1, linestyle='--', label=f"{lq*100}% q pred")
        axes[2,1].plot(bn_test_q2, linestyle='--', label=f"{uq*100}% q pred")
        axes[2,1].plot(self.y_test[t][:,1][..., np.newaxis, np.newaxis].reshape(num_samples_test), label="True")
        axes[2,1].set_title(f"bn Test Predictions vs Truth")
        axes[2,1].legend()

        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()


        return