import os
import numpy as np
import pandas as pd
import torch
from torch import nn
# from network.CNN import build
from network import build
import matplotlib.pyplot as plt
import csv
from polynomial_reg import polynomial_regression, loocv_optimization, predict_on_grid
import warnings
warnings.filterwarnings(action='ignore')

from source import *


def GetDevice():
    seed = 2025
    np.random.seed(seed)
    torch.manual_seed(seed)

    print('\n>> GPU setting ...')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    print('cuda index :', torch.cuda.current_device())
    print('number of gpu :', torch.cuda.device_count())
    print('graphic name:', torch.cuda.get_device_name())

    return device
    
# In[4. LOOCV WMAPE Calculation] ############################################################################################


def train_model(device, dataset, epochs, learning_rate, num_DV, pred_percentage, test_set, save_path):
    num_samples = len(dataset.np_train_input)

    
    result_file = save_path + f'/loocv_results_{pred_percentage*10}[%].csv'
    result_file2 = save_path + '/loocv_results_postprocessed.csv'
    
    results_df = pd.DataFrame(columns=["Test Index", "WMAPE", "Mean"])
    results_df2 = pd.DataFrame(columns=["Test Index", "WMAPE", "Mean"])
    results_df3 = pd.DataFrame(columns=["Test Index", "WMAPE", "Mean"])

    # Initialize result file
    with open(result_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Test Index', 'WMAPE'])

    with open(result_file2, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Test Index', 'WMAPE'])

    best_val_error = float('inf')
    best_model_weights = None

    models_dir = os.path.join(result_path, "saved_models")
    
    for test_idx in test_set:
    # for test_idx in range(int(num_samples/num_stiff)):
        print(f'LOOCV Iteration {test_idx}/{int(num_samples/num_stiff)}')
        best_model_weights = None
        best_val_error = float('inf')
        # Update test index
        dataset.update_test_idx(test_idx)

        # Get data loaders
        train_loader, val_loader, test_loader, np_output_data_gt, output_scalers, input_scaler, field_range = dataset.get_loader()

        # Initialize model
        model = build(num_DV).to(device)
        initialize_weights(model)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=94, T_mult=1, eta_min=0, verbose=False)

        # Training
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                # mono_loss = monotonicity_loss(outputs)
                # smooth_loss = smoothness_loss(outputs)

                total_loss = loss #+ mono_loss + smooth_loss
                total_loss.backward()
                optimizer.step()

            model.eval()
            val_error = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    val_error += criterion(outputs, targets).item()
                    # val_error += monotonicity_loss(outputs)
                    # val_error += smoothness_loss(outputs)

            val_error /= len(val_loader)
            print(f'Epoch {epoch + 1}, Validation Error: {val_error:.10f}')

            # Save the model weights if validation error is the lowest
            if val_error < best_val_error:
                best_val_error = val_error
                best_model_weights = model.state_dict()

            scheduler.step()
        
        model_save_path = os.path.join(models_dir, f"model_test_idx_{test_idx}.pth")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        save_dict = {
            "model_state_dict": best_model_weights,
            "input_scaler": input_scaler,
            "output_scalers": output_scalers
        }
        
        torch.save(save_dict, model_save_path)
        print(f"Saved LOOCV model and scalers for test index {test_idx} at {model_save_path}")
        
        
        # Load the best model weights
        model.load_state_dict(best_model_weights)

        model.eval()

        wmape = 0
        wmape2 = 0
        wmape3 = 0

        test_results1=[]
        test_results2=[]
        test_results_post_pro = []
        folder_path = f'{save_path}\\visualization\\test_idx_{test_idx}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with torch.no_grad():
            for case_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                inverse_input = input_scaler.inverse_transform(inputs.cpu().numpy())
                sub_axes_inverse = inverse_input[:,-2]
                main_axes_inverse = inverse_input[:,-1]

                predictions = model(inputs).cpu().numpy()

                inverse_predictions = inverse_scale_data(predictions, output_scalers[int(inverse_input[:,9])-1], field_range)
                linear_stiff = linear_stiffness_extaraction_each(inverse_predictions[0, :, 0], main_axes_inverse)
                
                linear_inverse = np.expm1(inverse_input[:,-4])
                scaling_factor = linear_inverse/linear_stiff
                # scaling_factor = 1
                scaled_predict = scaling_factor * inverse_predictions
                targets = targets.cpu().numpy()
                temp_wmape = calculate_wmape(scaled_predict, targets)
                # temp_wmape2 = calculate_wmape(inverse_predictions, targets)



                ground_truth = targets[0]
                vmin = ground_truth.min()
                vmax = ground_truth.max()

                # Calculate absolute error
                error = np.abs(inverse_predictions[0] - ground_truth)
                error_vmin = error.min()
                error_vmax = error.max()

                pred_percentage_ = pred_percentage / 10
                grid_x1, grid_y1 = np.meshgrid(np.linspace(0, sub_axes_inverse*pred_percentage_, 16),
                                            np.linspace(0, main_axes_inverse*pred_percentage_, 16))

                grid_x, grid_y = np.meshgrid(np.linspace(0, sub_axes_inverse, 16),
                                            np.linspace(0, main_axes_inverse, 16))

                train_X1 = np.column_stack([grid_x1.ravel(), grid_y1.ravel()])
                train_X = np.column_stack([grid_x.ravel(), grid_y.ravel()])

                optimal_degree = loocv_optimization(train_X1, inverse_predictions[0,:,:].flatten())
                poly_model, poly = polynomial_regression(train_X1, inverse_predictions[0,:,:].flatten(), optimal_degree)
                
                Z_pred_1 = predict_on_grid(poly_model, poly, train_X1)
                Z_pred_1 = Z_pred_1.reshape(16, 16)
                temp_wmape2 = calculate_wmape(Z_pred_1, ground_truth)
                
                
                Z_pred = predict_on_grid(poly_model, poly, train_X)
                Z_pred = Z_pred.reshape(16, 16)
                ground_truth_full = np_output_data_gt[case_idx]
                temp_wmape3 = calculate_wmape(Z_pred, ground_truth_full)
                
                wmape3+=float(temp_wmape3)
                test_results_post_pro.append(float(temp_wmape3))
                
                test_results1.append(float(temp_wmape))
                test_results2.append(float(temp_wmape2))

                wmape += float(temp_wmape)
                wmape2 += float(temp_wmape2)
                print("scaled_after: ",temp_wmape)
                print("scaled_before: ",temp_wmape2)
                print("post_processing: ",temp_wmape3)

                
                # Plot actual and predicted polynomials
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(grid_x1, grid_y1, ground_truth, color='blue', alpha=0.5, label=f'Actual_{pred_percentage_*100}[%]')
                ax.plot_surface(grid_x1, grid_y1, Z_pred_1, color='red', alpha=0.5, label=f'Predicted_{pred_percentage_*100}[%]')
                ax.set_xlabel('SubAxes')
                ax.set_ylabel('MainAxes')
                ax.set_zlabel('Value')
                plt.legend()
                file_name = os.path.join(folder_path, f"test_result_plot_{case_idx}_{pred_percentage_*100}[%].png")
                plt.savefig(file_name, dpi=300)
                # plt.show()

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(grid_x, grid_y, ground_truth_full, color='blue', alpha=0.5, label='Actual_100[%]')
                ax.plot_surface(grid_x, grid_y, Z_pred, color='yellow', alpha=0.5, label='predicted-extrapolated_100[%]')
                ax.set_xlabel('SubAxes')
                ax.set_ylabel('MainAxes')
                ax.set_zlabel('Value')
                plt.legend()
                file_name = os.path.join(folder_path, f"test_result_plot_{case_idx}_full.png")
                plt.savefig(file_name, dpi=300)



        wmape = wmape/num_stiff
        wmape2 = wmape2/num_stiff
        wmape3 = wmape3/num_stiff

        new_row1 = pd.DataFrame({"Test Index": test_idx, "WMAPE": [test_results1], "Mean": wmape} )
        new_row2 = pd.DataFrame({"Test Index": test_idx, "WMAPE": [test_results2], "Mean": wmape2})
        new_row3 = pd.DataFrame({"Test Index": test_idx, "WMAPE": [test_results_post_pro], "Mean": wmape3})

        results_df = pd.concat([results_df, new_row1], ignore_index=True)
        results_df2 = pd.concat([results_df2, new_row2], ignore_index=True)
        results_df3 = pd.concat([results_df3, new_row3], ignore_index=True)

        # results_df.to_excel(os.path.join(save_path, "test_results_scale_O.xlsx"), index=False)
        results_df2.to_excel(os.path.join(save_path, f"test_results_{pred_percentage*10}[%].xlsx"), index=False)
        results_df3.to_excel(os.path.join(save_path, "test_results_post.xlsx"), index=False)

        # Save results
        with open(result_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([test_idx, wmape2])

        with open(result_file2, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([test_idx, wmape3])

    
        # print(f'Test Index {test_idx}: WMAPE = {wmape:.2f}%')
        
def inverse_scale_data(data, scaler, field_range):
    reshaped = data.reshape(-1, field_range)
    inversed = scaler.inverse_transform(reshaped)
    return inversed.reshape(data.shape)
# In[Main Execution] ########################################################################################################

if __name__ == "__main__":
    # In[1. Parameter setting] ########################################################################################

    # mode ------------------------------------------------------------------------
    mode = 'training'  # 'training' or 'test'
    network = 'CNN'  # 'CNN' or 'VAE'

    # training parameter ----------------------------------------------------------
    epochs = 3000
    batch = 64
    learning_rate = 0.0005
    num_DV = 12
    field_range = 16*16
    num_stiff = 6
    
    ## User_defined parameters
    pred_percentages = [4, 5] # 몇 퍼센트 데이터로 학습하실래여? [4,5], [6,7], [8,9]로 나눠서 학습
    test_set = [0,5,13,18] # 몇번 인덱스로 테스트 하실래여?
    
    # result directory ------------------------------------------------------------
    
    for pred_percentage in pred_percentages:
        if mode == 'training':
            result_path = f'[{len(os.listdir("./results")) + 1}]_{pred_percentage}0[%]_{network}_ep{epochs}_bat{batch}_lr{learning_rate}_dropout'
            if not os.path.exists(f'.\\results\\{result_path}'):
                os.makedirs(f'.\\results\\{result_path}')

        elif mode == 'test':
            result_path = '[5]CNN-Iter1_ep1000_bat4_lr0.00030693161628128087'

        result_path = '.\\results\\' + result_path
        
        data_path = rf'.\resource\combined_data_9_squared_103\combined_data_{pred_percentage}.npy'
        # data_path = rf'E:\Dongwoo\TeamWork\Hyundai_bush_2\0204_CNN\resource\combined_data_9_squared\combined_data_{pred_percentage}.npy'
        gt_data_path = rf'.\resource\combined_data_10.npy'
        # Device setting
        device = GetDevice()

        # Dataset preparation
        dataset = VEPDataset(batch, output_path=data_path, gt_path=gt_data_path, field_range=field_range, num_stiffness=num_stiff , mode=mode)

        # LOOCV WMAPE calculation
        train_model(device, dataset, epochs=epochs, learning_rate=learning_rate, num_DV=num_DV, pred_percentage=pred_percentage, test_set=test_set, save_path=result_path)
