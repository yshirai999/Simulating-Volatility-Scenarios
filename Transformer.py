import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from BaseEnv import BaseClass

"""
Encodes position of the input, other wise transfomer views all inputs the same
"""
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

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, dim_feedforward, output_dim, seq_length, dropout, decoder=False):
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
        # Print the output range for debugging
        print(f'Output range: {output.min().item()} to {output.max().item()}')

        return output.unsqueeze(1)

    def eval_valid(self, X_val, y_val, batch_size, device='cpu'):
        self.eval()
        val_loss = 0
        crit = nn.MSELoss()
        N = X_val.shape[0]
        num_batches =(N + batch_size - 1) // batch_size
        with torch.no_grad():
            for i in range(0, N, batch_size):
                input = X_val[i:i+batch_size].to(device)
                labels = y_val[i:i+batch_size].to(device)

                output = self.forward(input)
                loss = crit(output, labels)

                val_loss += loss.item()
        
        return val_loss / num_batches 

        

    def predict(self, X, batch_size = 128, max_len=2000, device='cpu'):
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No gradients needed for inference
            # Get the output using the forward method
            X = X.to(device)
            output = self.forward(X)  # (batch_size, 1, output_dim)

        return output

    def train_model(self, X_train, y_train, X_val, y_val, num_epochs=10, lr=5e-3 , batch_size=128, device='cpu'):
        self.to(device)
        crit = nn.HuberLoss(delta=5.0) #nn.MSELoss()
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
                input = X_train[i:i+batch_size].to(device)
                labels = y_train[i:i+batch_size].to(device)

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
            print(f"Epoch {e+1}/{num_epochs}, Loss: {avg_loss:.4f}, Validation loss: {avg_val_loss}")
            # for name, param in self.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: {param.grad.abs().mean().item()}")
        return train_errors, val_errors



class Transformer(BaseClass):
    def __init__(self, tickers, feature_steps, target_steps, scaler,
                input_dim, d_model, num_heads, num_layers, dim_feedforward, output_dim, seq_length, dropout):
        super().__init__(tickers=tickers, feature_steps = feature_steps, target_steps = target_steps, scaler = scaler)
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
            self.models[t]['mp'] = TransformerModel(input_dim, d_model, num_heads, num_layers, dim_feedforward, output_dim, seq_length, dropout)
            self.models[t]['mn'] = TransformerModel(input_dim, d_model, num_heads, num_layers, dim_feedforward, output_dim, seq_length, dropout)
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

            self.train_errors[t]['mp'], self.valid_errors[t]['mp'] = self.models[t]['mp'].train_model(X_train0, y_train0, X_valid0, y_valid0)

            self.train_errors[t]['mn'], self.valid_errors[t]['mn'] = self.models[t]['mn'].train_model(X_train1, y_train1,X_valid1, y_valid1)

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

            self.test_pred['mp'] = self.models[t]['mp'].predict(X_test0)
            self.test_pred['mn'] = self.models[t]['mn'].predict(X_test1)

            self.train_pred['mp'] = self.models[t]['mp'].predict(X_train0)
            self.train_pred['mn'] = self.models[t]['mn'].predict(X_train1)

            self.valid_pred['mp'] = self.models[t]['mp'].predict(X_valid0)
            self.valid_pred['mn'] = self.models[t]['mn'].predict(X_valid1)
            




