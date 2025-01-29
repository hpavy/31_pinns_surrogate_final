from model import PINNs
import json
import torch
from transfert_learning.run import run

folder_name = "5_huge"
epoch = "epoch420"
param_adim = {"V": 1.0, "L": 0.025, "rho": 1.2}
folder_name_transfert = 'first_try'
folder_name_transfert = folder_name + '/' + folder_name_transfert

hyper_param_transfert = {
    "H": [
        230.67,
    ],
    "ya0": [
        0.00125,
    ],
    "m": 1.57,
    "file": [
        "data_john_4_case_2.csv",
    ],
    "nb_epoch": 1000,
    "save_rate": 20,
    "batch_size": 10000,
    "nb_points_pde": 1000000,
    "Re": 100,
    "lr": 0.0001,
    "n_data_test": 5000,
    "x_min": -0.03,
    "x_max": 0.03,
    "y_min": -0.03,
    "y_max": 0.03,
    "nb_points_border": 25,
}

# On charge le modèle
with open("results/" + folder_name + "/hyper_param.json", "r") as file:
    hyper_param = json.load(file)

with open('results/'+folder_name+'/mean_std.json') as file: 
    mean_std = json.load(file)

# On crée notre modèle
model = PINNs(hyper_param)
checkpoint = torch.load(
    "results/" + folder_name + "/" + epoch + "/" + "model_weights.pth",
    map_location=torch.device("cpu"),
)
model.load_state_dict(checkpoint["model_state_dict"])

hyper_param_transfert['t_min'] = hyper_param['t_min']
hyper_param_transfert['nb_period'] = hyper_param['nb_period']
hyper_param_transfert['nb_period_plot'] = hyper_param['nb_period_plot']
hyper_param_transfert['force_inertie_bool'] = hyper_param['force_inertie_bool']


