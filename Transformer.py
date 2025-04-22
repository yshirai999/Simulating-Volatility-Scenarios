import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import numpy as np
from BaseEnv import BaseClass
import matplotlib.pyplot as plt
from tqdm import tqdm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]].to(x.device)

class CauchyLoss(nn.Module):
    def __init__(self, scale=1.0):
        super(CauchyLoss, self).__init__()
        self.scale = scale

    def forward(self, predicted, target):
        diff = predicted - target
        
        # Cauchy loss formula: log(1 + (x/scale)^2)
        loss = torch.log(1 + (diff / self.scale) ** 2)
        return loss.mean()


# Input: pred so f shape Bx 1 x4 out put is scalar, assume first two 
# the four dims of output are q1_x, q2_x, q1_y, q2_y
# class QuantileLoss(nn.Module):
#     def __init__(self, quantiles=(0.1, 0.9)):
#         super().__init__()
#         self.quantiles = torch.tensor(quantiles)

#     def forward(self, y_pred, y_true):
#         y_pred = y_pred.view(y_true.shape[0], 2, -1)  # Now shape: (Batch size, 2, 2)
#         y_true = y_true.unsqueeze(-1)  # Shape: (batch_size, 2, 1)

#         errors = y_true - y_pred  # Shape: (batch_size, 2, num_quantiles)
#         loss = torch.max(self.quantiles * errors, (self.quantiles - 1) * errors)

#         return loss.mean() 
class QuantileLoss(nn.Module):
    def __init__(self, quantiles=(0.1, 0.9)):
        super().__init__()
        self.register_buffer('quantiles', torch.tensor(quantiles))  # Register quantiles as a buffer

    def forward(self, y_pred, y_true):
        # Ensure quantiles are on the same device as y_pred
        quantiles = self.quantiles.to(y_pred.device)

        y_pred = y_pred.view(y_true.shape[0], 2, -1)  # Now shape: (Batch size, 2, 2)
        y_true = y_true.unsqueeze(-1)  # Shape: (batch_size, 2, 1)

        errors = y_true - y_pred  # Shape: (batch_size, 2, num_quantiles)
        loss = torch.max(quantiles * errors, (quantiles - 1) * errors)

        return loss.mean()




class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, dim_feedforward, output_dim, seq_length, dropout, scale, quantiles=(0.1, 0.9), device='cpu'):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Linear(input_dim, d_model) # encode the input
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_length) # add positional encoding
        
        self.transformer = nn.Transformer(
            d_model=d_model,  # Input and output dimensionality
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu', 
            norm_first=False,
            batch_first=True,
        )
        self.output_layer = nn.Linear(d_model, output_dim)
        self._init_weights(scale)
        self.quantiles = quantiles    
        self.device = device
    
    def _init_weights(self, scale=2.0):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Initialize Linear layers with Xavier uniform distribution
                init.xavier_uniform_(m.weight)
                m.weight = torch.nn.Parameter(scale * m.weight.data)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                # Initialize MultiheadAttention weights (Q, K, V, and output projection)
                init.xavier_uniform_(m.in_proj_weight)
                m.in_proj_weight = torch.nn.Parameter(scale * m.in_proj_weight.data)
                init.xavier_uniform_(m.out_proj.weight)
                m.out_proj.weight = torch.nn.Parameter(scale * m.out_proj.weight.data)
                if m.in_proj_bias is not None:
                    init.zeros_(m.in_proj_bias)
                if m.out_proj.bias is not None:
                    init.zeros_(m.out_proj.bias)
            elif isinstance(m, nn.TransformerEncoderLayer) or isinstance(m, nn.TransformerDecoderLayer):
                # Initialize feedforward layers within the encoder/decoder layers
                init.xavier_uniform_(m.linear1.weight)
                m.linear1.weight = torch.nn.Parameter(m.linear1.weight.data * scale)
                init.xavier_uniform_(m.linear2.weight)
                m.linear2.weight = torch.nn.Parameter(m.linear2.weight.data * scale) 
                if m.linear1.bias is not None:
                    init.zeros_(m.linear1.bias)
                if m.linear2.bias is not None:
                    init.zeros_(m.linear2.bias)

    def get_mask(self, size):
        return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)

    def forward(self, src):
        """
        src: (batch_size, seq_length, input_dim)
        """
        src = self.embedding(src)
        src = self.positional_encoding(src)

        transformer_output = self.transformer.encoder(src)  # (batch_size, seq_len, d_model)
        last_step_output = transformer_output[:, -1, :]  # Take the last timestep (batch_size, d_model)

        output = self.output_layer(last_step_output)  # (batch_size, output_dim)
        # # Print the output range for debugging
        # print(f'Output range: {output.min().item()} to {output.max().item()}')

        return output.unsqueeze(1)

    def eval_valid(self, X_val, y_val, batch_size):
        self.eval()
        val_loss = 0
        # crit = nn.HuberLoss(delta=5.0)
        crit = QuantileLoss(quantiles=self.quantiles)
        N = X_val.shape[0]
        num_batches =(N + batch_size - 1) // batch_size
        with torch.no_grad():
            for i in range(0, N, batch_size):
                input = X_val[i:i+batch_size].to(self.device)
                labels = y_val[i:i+batch_size].to(self.device)

                output = self.forward(input)
                loss = crit(output, labels)

                val_loss += loss.item()
        
        return val_loss / num_batches 

        

    def predict(self, X, batch_size = 128, max_len=2000):
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No gradients needed for inference
            # Get the output using the forward method
            X = X.to(self.device)
            output = self.forward(X)  # (batch_size, 1, output_dim)

        return output

    def train_model(self, X_train, y_train, X_val, y_val, num_epochs=100, lr=5e-3 , batch_size=128):
        self.to(self.device)
        # crit = nn.HuberLoss(delta=5.0) #nn.MSELoss()
        crit = QuantileLoss(self.quantiles)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        N = X_train.shape[0]
        num_batches =(N + batch_size - 1) // batch_size

        train_errors = []
        val_errors = []

        for e in range(num_epochs):
            self.train()
            epoch_loss = 0

            indices = torch.randperm(N)
            X_train, y_train = X_train[indices], y_train[indices]
            for i in range(0, N, batch_size):
                input = X_train[i:i+batch_size].to(self.device)
                labels = y_train[i:i+batch_size].to(self.device)

                output = self.forward(input)
                loss = crit(output, labels)
                # print(f"loss = {loss}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            avg_loss = epoch_loss / num_batches
            avg_val_loss = self.eval_valid(X_val, y_val, batch_size) 
            train_errors.append(avg_loss)
            val_errors.append(avg_val_loss)
            # print(f"Epoch {e+1}/{num_epochs}, Loss: {avg_loss:.4f}, Validation loss: {avg_val_loss}")
        # for name, param in self.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: {param.grad.abs().mean().item()}")
        plt.figure()
        plt.plot(train_errors[2:], label="Train")
        plt.plot(val_errors[2:], label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Huber Loss")
        plt.title("Train/Val Loss")
        plt.legend()
        plt.show()
        return train_errors, val_errors



class Transformer(BaseClass):
    def __init__(self, tickers, feature_steps, target_steps, scaler,
                input_dim, d_model, num_heads, num_layers, dim_feedforward, output_dim, seq_length, 
                dropout, quantized, quant_all, n_clusters, scale, num_epochs, device='cpu'):
        super().__init__(tickers=tickers, feature_steps = feature_steps, target_steps = target_steps, scaler = scaler, quantized=quantized,quant_all=quant_all, nclusters=n_clusters)
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
        for t in self.tickers:
            self.train_series[t] = np.concatenate( (self.y_train[t],self.y_valid[t],self.y_test[t]), axis=0)
            self.models[t] = {}
            self.models[t]['bp'] = TransformerModel(input_dim, d_model, num_heads, num_layers, dim_feedforward, output_dim, seq_length, dropout, scale, device = device)
            self.models[t]['bn'] = TransformerModel(input_dim, d_model, num_heads, num_layers, dim_feedforward, output_dim, seq_length, dropout, scale, device = device)
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
            # train the transformer

            # change this to better 
            X_train0 = torch.tensor(self.X_train[t][:,:,0][..., np.newaxis],dtype=torch.float32)
            y_train0 = torch.tensor(self.y_train[t][:,0][..., np.newaxis, np.newaxis],dtype=torch.float32)
            X_valid0 = torch.tensor(self.X_valid[t][:,:,0][..., np.newaxis],dtype=torch.float32)
            y_valid0 = torch.tensor(self.y_valid[t][:,0][..., np.newaxis, np.newaxis],dtype=torch.float32)

            X_train1 = torch.tensor(self.X_train[t][:,:,1][..., np.newaxis],dtype=torch.float32)
            y_train1 = torch.tensor(self.y_train[t][:,1][..., np.newaxis, np.newaxis],dtype=torch.float32)
            X_valid1 = torch.tensor(self.X_valid[t][:,:,1][..., np.newaxis],dtype=torch.float32)
            y_valid1 = torch.tensor(self.y_valid[t][:,1][..., np.newaxis, np.newaxis], dtype=torch.float32)

            print(f"Xtrain shape: {X_train0.shape}")

            self.train_errors[t]['bp'], self.valid_errors[t]['bp'] = self.models[t]['bp'].train_model(X_train=X_train0, y_train=y_train0, X_val=X_valid0, y_val=y_valid0, num_epochs = self.num_epochs)

            self.train_errors[t]['bn'], self.valid_errors[t]['bn'] = self.models[t]['bn'].train_model(X_train=X_train1, y_train=y_train1, X_val=X_valid1, y_val=y_valid1,num_epochs = self.num_epochs)

            # predict train and valid?

    def predict(self):
        for t in self.tickers:
            X_test0 = torch.tensor(self.X_test[t][:,:,0][..., np.newaxis],dtype=torch.float32)
            X_train0 = torch.tensor(self.X_train[t][:,:,0][..., np.newaxis],dtype=torch.float32)
            X_valid0 = torch.tensor(self.X_valid[t][:,:,0][..., np.newaxis],dtype=torch.float32)

            X_train1 = torch.tensor(self.X_train[t][:,:,1][..., np.newaxis],dtype=torch.float32)
            X_valid1 = torch.tensor(self.X_valid[t][:,:,1][..., np.newaxis],dtype=torch.float32)
            X_test1 = torch.tensor(self.X_test[t][:,:,1][..., np.newaxis],dtype=torch.float32)
            y_test1 = torch.tensor(self.y_test[t][:,1][..., np.newaxis, np.newaxis],dtype=torch.float32)
            
            self.test_pred[t] = {}
            self.test_pred[t]['bp'] = self.models[t]['bp'].predict(X_test0)
            self.test_pred[t]['bn'] = self.models[t]['bn'].predict(X_test1)

            self.train_pred[t] = {}
            self.train_pred[t]['bp'] = self.models[t]['bp'].predict(X_train0)
            self.train_pred[t]['bn'] = self.models[t]['bn'].predict(X_train1)
            
            self.valid_pred[t] = {}
            self.valid_pred[t]['bp'] = self.models[t]['bp'].predict(X_valid0)
            self.valid_pred[t]['bn'] = self.models[t]['bn'].predict(X_valid1)
    
    def plot_predictions(self,t,  title):
        num_samples_train = self.X_train[t].shape[0]
        num_samples_valid = self.X_valid[t].shape[0]
        num_samples_test = self.X_test[t].shape[0]

        fig, axes = plt.subplots(3, 2, figsize=(10, 10))  # 3 rows, 2 columns
        fig.suptitle(f"{t} ({title})", fontsize=16)
            
        axes[0,0].plot(self.train_pred[t]['bp'].reshape(num_samples_train).detach().numpy(), label="Predictions")
        axes[0,0].plot(self.y_train[t][:,0][..., np.newaxis, np.newaxis].reshape(num_samples_train), label="True")
        axes[0,0].set_title(f"bp Train Predictions vs Truth")
        axes[0,0].legend()

        axes[0,1].plot(self.train_pred[t]['bn'].reshape(num_samples_train).detach().numpy(), label="Predictions")
        axes[0,1].plot(self.y_train[t][:,1][..., np.newaxis, np.newaxis].reshape(num_samples_train), label="True")
        axes[0,1].set_title(f"bn Train Predictions vs Truth")
        axes[0,1].legend()

        axes[1,0].plot(self.valid_pred[t]['bp'].reshape(num_samples_valid).detach().numpy(), label="Predictions")
        axes[1,0].plot(self.y_valid[t][:,0][..., np.newaxis, np.newaxis].reshape(num_samples_valid), label="True")
        axes[1,0].set_title(f"bp Validation Predictions vs Truth")
        axes[1,0].legend()

        axes[1,1].plot(self.valid_pred[t]['bn'].reshape(num_samples_valid).detach().numpy(), label="Predictions")
        axes[1,1].plot(self.y_valid[t][:,1][..., np.newaxis, np.newaxis].reshape(num_samples_valid), label="True")
        axes[1,1].set_title(f"bn Validation Predictions vs Truth")
        axes[1,1].legend()

        axes[2,0].plot(self.test_pred[t]['bp'].reshape(num_samples_test).detach().numpy(), label="Predictions")
        axes[2,0].plot(self.y_test[t][:,0][..., np.newaxis, np.newaxis].reshape(num_samples_test), label="True")
        axes[2,0].set_title(f"bp Test Predictions vs Truth")
        axes[2,0].legend()

        axes[2,1].plot(self.test_pred[t]['bn'].reshape(num_samples_test).detach().numpy(), label="Predictions")
        axes[2,1].plot(self.y_test[t][:,1][..., np.newaxis, np.newaxis].reshape(num_samples_test), label="True")
        axes[2,1].set_title(f"bn Test Predictions vs Truth")
        axes[2,1].legend()

        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()




class Transformer2D(BaseClass):
    def __init__(self, tickers, feature_steps, target_steps, scaler,
                input_dim, d_model, num_heads, num_layers, dim_feedforward, output_dim, seq_length, 
                dropout, quantized,quant_all, n_clusters, scale, num_epochs, quantiles=(0.1,0.9), device='cpu'):
        super().__init__(tickers=tickers, feature_steps = feature_steps, target_steps = target_steps, scaler = scaler, quantized=quantized, quant_all=quant_all, nclusters=n_clusters)
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
            print(X_train.shape)
            print(self.X[t].shape)

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


class ProbTransformer(TransformerModel):
    def __init__(self, input_dim, d_model, num_heads, num_layers, dim_feedforward, output_dim, seq_length, dropout, scale, quantiles=(0.1, 0.9), device='cpu'):
        super().__init__(self, input_dim, d_model, num_heads, num_layers, dim_feedforward, output_dim, seq_length, dropout, scale, quantiles, device)
        self.prob_layer = nn.Softmax()

    def forward(self,src):
        out = super().forward(self, src) # output batch size, output dim
        return self.prob_layer(out, dim=-1) # perform softmax on it     




