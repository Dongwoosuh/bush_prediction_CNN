import os 
import torch
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from network import build
from source import *
from polynomial_reg import polynomial_regression, loocv_optimization, predict_on_grid

def load_model_and_scalers(model_file: str, num_DV: int, device):
    """
    저장된 모델과 스케일러를 로드한다.
    
    인자:
      - model_file: 모델 파일 경로 (예, '.../model_test_idx_0.pth')
      - num_DV: 입력 변수 개수
      
    반환:
      - model: 재구성된 모델 (평가 모드)
      - input_scaler: 입력 데이터에 사용된 QuantileTransformer 객체
      - output_scalers: 출력 데이터에 사용된 QuantileTransformer 객체 리스트
    """
    saved_dict = torch.load(model_file)
    model = build(num_DV).to(device)
    model.load_state_dict(saved_dict['model_state_dict'])
    model.eval()
    
    input_scaler = saved_dict['input_scaler']
    output_scalers = saved_dict['output_scalers']
    
    return model, input_scaler, output_scalers

def test_model(model, device, input_scaler, output_scalers, test_dataset, gt_dataset, field_range: int = 256, save_path: str = None):
    predictions = [] 
    predictions_post = []
    
    gt_dataset = gt_dataset.original_output_gt
    
    wmape1_acc_total = 0
    wmape2_acc_total = 0
    for idx, (input_data, _) in enumerate(test_dataset):
        
        input_data_unscaled = input_scaler.inverse_transform(input_data)
        input_data = input_data.to(device)
        # 입력 데이터에 스케일러 적용
        prediction = model(input_data).detach().cpu().numpy()
        # 예측 결과를 numpy 배열로 변환
        
        # prediction = prediction
        
        prediction_flat = prediction.reshape(-1, field_range)
        output_scaler = output_scalers[int(input_data_unscaled[:, -3])-1]
        prediction_flat = output_scaler.inverse_transform(prediction_flat)
        
        prediction = prediction_flat.reshape(prediction.shape)
        
        sub_axes_inverse = input_data_unscaled[:,-2]
        main_axes_inverse = input_data_unscaled[:,-1]
        
        grid_x, grid_y = np.meshgrid(np.linspace(0, sub_axes_inverse, 16),
                                    np.linspace(0, main_axes_inverse, 16))
        
        grid_x1, grid_y1 = np.meshgrid(np.linspace(0, sub_axes_inverse/2, 8),
                                    np.linspace(0, main_axes_inverse/2, 8))


        train_X = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        
        optimal_degree = loocv_optimization(train_X, prediction[0,:,:].flatten())
        poly_model, poly = polynomial_regression(train_X, prediction[0,:,:].flatten(), optimal_degree)

        Z_pred = predict_on_grid(poly_model, poly, train_X)
        Z_pred = Z_pred.reshape( 16, 16)
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(grid_x1, grid_y1, gt_dataset[idx][:8,:8], color='blue', alpha=0.5, label='Actual_50[%]')
        ax.plot_surface(grid_x1, grid_y1, Z_pred[:8,:8], color='yellow', alpha=0.5, label='predicted-extrapolated_50[%]')
        ax.set_xlabel('SubAxes')
        ax.set_ylabel('MainAxes')
        ax.set_zlabel('Value')
        plt.legend()
        folder_path = os.path.join(save_path, 'test_result_plot2')
        file_name = os.path.join(folder_path, f"test_result_plot_{idx}_post_50%.png")
        plt.savefig(file_name, dpi=300)
        print(f"Saved: {file_name}")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(grid_x1, grid_y1, gt_dataset[idx][:8,:8], color='blue', alpha=0.5, label='Actual_50[%]')
        ax.plot_surface(grid_x1, grid_y1, prediction.reshape(16,16)[:8,:8], color='yellow', alpha=0.5, label='predicted-extrapolated_50[%]')
        ax.set_xlabel('SubAxes')
        ax.set_ylabel('MainAxes')
        ax.set_zlabel('Value')
        plt.legend()
        folder_path = os.path.join(save_path, 'test_result_plot2')
        file_name = os.path.join(folder_path, f"test_result_plot_{idx}_50%.png")
        plt.savefig(file_name, dpi=300)
        print(f"Saved: {file_name}")
        
        wmape1_acc = calculate_wmape(prediction.reshape(16,16)[:8,:8], gt_dataset[idx][:8,:8])
        # wmape2_acc = calculate_wmape(Z_pred, gt_dataset[idx])   
        wmape2_acc = calculate_wmape(Z_pred[:8,:8], gt_dataset[idx][:8,:8])   
        
        wmape1_acc_total += wmape1_acc
        wmape2_acc_total += wmape2_acc
        
        predictions_post.append(Z_pred)
        predictions.append(prediction)
        
    predictions = np.vstack(predictions)  # (N, field_range)
    predictions_post = np.vstack(predictions_post)
    
    acc1 = wmape1_acc_total / 6
    acc2 = wmape2_acc_total / 6
    
    print(f"WMAPE1: {wmape1_acc:.4f}, WMAPE2: {wmape2_acc:.4f}")
    return predictions, wmape1_acc, wmape2_acc


    

if __name__ == "__main__":
    # 저장된 경로
    save_path = r"E:\Dongwoo\TeamWork\Hyundai_bush_2\github\bush_prediction_CNN\results\[1]CNN-Iter0__ep5000_bat64_lr0.0005"
    model_path = os.path.join(save_path, 'saved_models\\model_test_idx_0_Q.pth' )
    input_path = './resource/inference/input_data_.xlsx'
    
    gt_source = r'E:\Dongwoo\TeamWork\Hyundai_bush_2\github\bush_prediction_CNN\resource\combined_data_10.npy'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, input_scaler, output_scalers = load_model_and_scalers(model_path, 12, device)
    # 디바이스 설정

    # 데이터셋 로드 (이 부분은 데이터셋 구조에 맞게 수정 필요)
    dataset = VEPDataset_inference(input_path = input_path, field_range=256, num_stiffness=6)
    test_dataloder = dataset.get_loader(input_scaler=input_scaler)
    dataset_gt = VEPDataset(batch =1 , output_path = gt_source, gt_path= gt_source, field_range=256, num_stiffness=6)

    prediction, acc, acc_post = test_model(model, device, input_scaler, output_scalers, test_dataloder, gt_dataset=dataset_gt, field_range=256, save_path=save_path)
