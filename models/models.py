import torch


def phi(D_in, dropout_prob=0.25):

    D_h0  = 256
    D_h1  = 32 

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, D_h0),
        torch.nn.BatchNorm1d(D_h0),
        torch.nn.ELU(),
        torch.nn.Dropout(dropout_prob),
        torch.nn.Linear(D_h0, D_h1),
        torch.nn.BatchNorm1d(D_h1),
        torch.nn.ELU(),
        torch.nn.Dropout(dropout_prob),
        torch.nn.Linear(D_h1, D_h0),
        torch.nn.BatchNorm1d(D_h0),
        torch.nn.ELU(),
        torch.nn.Dropout(dropout_prob),
        torch.nn.Linear(D_h0, D_in),
        torch.nn.ReLU()
    )
    return model


def rho(D_in):
    D_h0  = 64  
    D_out = 1

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, D_h0),
        torch.nn.BatchNorm1d(D_h0),
        torch.nn.ReLU(),
        torch.nn.Linear(D_h0, D_out),
        torch.nn.ReLU()
    )
    return model