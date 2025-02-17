import os
import argparse
import logging
import json
import torch
import datetime
import pathlib
from copy import deepcopy

import numpy as np
import tqdm

from source import *
from network.vanila import *

logger = logging.getLogger(__name__)

def build_model(model_type:str, **hparams):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        logger.warning("CUDA is not available. Running on CPU")
        
    if model_type == "CNN":
        model = CNN(device, **hparams)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    return model

def train_model(
    model_type:str,
    dataset,
    n_epochs:int,
    batch_size:int,
    lr:float,
    test_idx:int,
    save_path: str
    ):
    
    hparams = {
        "num_DV" : 12
    }
    
    ml_model = build_model("CNN", **hparams)

    logger.info(f"LOOCV Iteration: Bush {test_idx} started")

    ml_model.train(dataset, n_epochs, batch_size, lr, test_idx, save_path=result_path)
    
def model_test(
    model_type:str,
    dataset,
    test_idx:int,
    model_path: str
    ):
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        logger.warning("CUDA is not available. Running on CPU")
        
    if model_type == "CNN":
        model = CNN.load(model_path, device)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    _, _, test_loader, _, _, _, _ = dataset.get_loader()
    
    for idx, (inputs, outputs) in enumerate(test_loader):
        prediction = model.predict(inputs) # input은 스케일이 이미 된 상태로 들어옴
        prediction = prediction 
        gt_output = outputs.numpy() # output은 굳이 스케일링해서 넣을 필요 없음
        
        input_unscaled = model.input_scaler.inverse_transform(inputs.numpy())
        
        sub_axes_inverse = input_unscaled[:,-2]
        main_axes_inverse = input_unscaled[:,-1]
        
        print("hi")
        
        
        
        
if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model_type", type=str, default="CNN")
    args = parser.parse_args()
    
    data_path = rf'.\resource\\combined_data_10.npy'
    gt_data_path = rf'.\resource\combined_data_10.npy'
    # test_set = [0,5,13,18,29,31,70,71] # 몇번 인덱스로 테스트 하실래여?
    test_set = [0] # 몇번 인덱스로 테스트 하실래여?
    result_path = pathlib.Path("results") / f"{args.model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # for test_idx in test_set:
    #     dataset = VEPDataset(batch=args.batch_size, output_path=data_path, gt_path=gt_data_path, field_range=256, num_stiffness=6)
    #     dataset.update_test_idx(test_idx)
    #     train_model(args.model_type, dataset, args.n_epochs, args.batch_size, args.lr, test_idx=test_idx, save_path=result_path)
        
    
    for test_idx in test_set:
        dataset = VEPDataset(batch=args.batch_size, output_path=data_path, gt_path=gt_data_path, field_range=256, num_stiffness=6)
        dataset.update_test_idx(test_idx)
        model_test(args.model_type, dataset=dataset, test_idx=test_idx, model_path=r'E:\Dongwoo\TeamWork\Hyundai_bush_2\github\bush_prediction_CNN\results\CNN_20250217_154338\bush_idx[0]')