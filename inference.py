import os 
import torch
import json
import pickle
import numpy as np
import pandas as pd
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

def test_model(model, device, input_scaler, output_scalers, test_dataset, gt_dataset, target_bush, field_range: int = 256, save_path: str = None):
    predictions = [] 
    predictions_post = []
    
    gt_dataset = gt_dataset.original_output_gt
    
    _ , bush_idx = target_bush

    result_df = pd.DataFrame(columns=["stiffness_num", "40%", "50%", "60%", "70%", "80%","100%"])
    
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
        
        train_X = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        
        optimal_degree = loocv_optimization(train_X, prediction[0,:,:].flatten())
        poly_model, poly = polynomial_regression(train_X, prediction[0,:,:].flatten(), optimal_degree)

        Z_pred = predict_on_grid(poly_model, poly, train_X)
        Z_pred = Z_pred.reshape( 16, 16)
        
        pred_percentages = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        wmape1_acc_total = 0
        wmape2_acc_total = 0
        wmape_per_percent_list = []
        wmape_full_range_list = []
        for pred_percengtage in pred_percentages:
          
          grid_x1, grid_y1 = np.meshgrid(np.linspace(0, sub_axes_inverse*pred_percengtage, int(16*pred_percengtage)),
                                      np.linspace(0, main_axes_inverse*pred_percengtage, int(16*pred_percengtage)))


          
          
          folder_path_ = os.path.join(save_path, f'test_result_plot_bush_{bush_idx}')
          
          if not os.path.exists(folder_path_):
              os.makedirs(folder_path_)
              
          folder_path = os.path.join(folder_path_, f'{pred_percengtage*100}%')
          if not os.path.exists(folder_path):
              os.makedirs(folder_path)
              
          fig = plt.figure()
          ax = fig.add_subplot(111, projection='3d')
          ax.plot_surface(grid_x1, grid_y1, gt_dataset[bush_idx*6 + idx][:int(16*pred_percengtage),:int(16*pred_percengtage)], color='blue', alpha=0.5, label='Actual_50[%]')
          ax.plot_surface(grid_x1, grid_y1, Z_pred[:int(16*pred_percengtage),:int(16*pred_percengtage)], color='yellow', alpha=0.5, label='predicted-extrapolated_50[%]')
          ax.set_xlabel('SubAxes')
          ax.set_ylabel('MainAxes')
          ax.set_zlabel('Value')
          plt.legend()
          file_name = os.path.join(folder_path, f"test_result_plot_{idx}_post_50%.png")
          plt.savefig(file_name, dpi=300)
          print(f"Saved: {file_name}")

          fig = plt.figure()
          ax = fig.add_subplot(111, projection='3d')
          ax.plot_surface(grid_x, grid_y, gt_dataset[bush_idx*6 + idx], color='blue', alpha=0.5, label='Actual_100[%]')
          ax.plot_surface(grid_x, grid_y, Z_pred, color='yellow', alpha=0.5, label='predicted-extrapolated_100[%]')
          ax.set_xlabel('SubAxes')
          ax.set_ylabel('MainAxes')
          ax.set_zlabel('Value')
          plt.legend()
          file_name = os.path.join(folder_path, f"test_result_plot_{idx}_100%.png")
          plt.savefig(file_name, dpi=300)
          print(f"Saved: {file_name}")
          
          wmape1_acc = calculate_wmape(Z_pred.reshape(16,16), gt_dataset[bush_idx*6 + idx])
          # wmape2_acc = calculate_wmape(Z_pred, gt_dataset[idx])   
          wmape2_acc = calculate_wmape(Z_pred[:int(16*pred_percengtage),:int(16*pred_percengtage)], gt_dataset[bush_idx*6 + idx][:int(16*pred_percengtage),:int(16*pred_percengtage)])   
          
          wmape_full_range_list.append(float(wmape1_acc))
          wmape_per_percent_list.append(float(wmape2_acc))

        new_row = pd.DataFrame({"stiffness_num": idx+1, "40%": [wmape_per_percent_list[0]], "50%": [wmape_per_percent_list[1]], "60%": [wmape_per_percent_list[2]], "70%": [wmape_per_percent_list[3]], "80%": [wmape_per_percent_list[4]], "100%": [wmape_full_range_list[0]]})
          
        result_df = pd.concat([result_df, new_row], ignore_index=True)
        
    
    mean_row =  pd.DataFrame({"stiffness_num": 'mean', "40%": [result_df['40%'].mean()], "50%": [result_df['50%'].mean()], "60%": [result_df['60%'].mean()], "70%": [result_df['70%'].mean()], "80%": [result_df['80%'].mean()], '100%': [result_df['100%'].mean()]})
    result_df = pd.concat([result_df, mean_row], ignore_index=True)
    result_df.to_csv(os.path.join(save_path, f"result_df_bush[{bush_idx}].csv"), index=False)
    
    #     predictions_post.append(Z_pred)
    #     predictions.append(prediction)
        
    # predictions = np.vstack(predictions)  # (N, field_range)
    # predictions_post = np.vstack(predictions_post)
    
    acc1 = wmape1_acc_total / 6
    acc2 = wmape2_acc_total / 6
    
    print(f"WMAPE1: {acc1:.4f}, WMAPE2: {acc2:.4f}")
    return predictions, wmape1_acc, wmape2_acc


    

if __name__ == "__main__":
    # 저장된 경로
    save_path = r"E:\Dongwoo\TeamWork\Hyundai_bush_2\github\bush_prediction_CNN\results\inference_참설계변수"
    # save_path = r"E:\Dongwoo\TeamWork\Hyundai_bush_2\github\bush_prediction_CNN\results\inference_예측설계변수"
    # target_bush = ('06_04_NX4', 0)
    # target_bush = ('06_11_MQ4', 5)
    # target_bush = ('G_06_04_IK', 13)
    target_bushes = [('06_04_NX4', 0), ('06_11_MQ4', 5), ('G_06_04_IK', 13) ,('G_11_01_IK', 18)]
    target_bushes = [('06_04_NX4', 0), ('06_11_MQ4', 5), ('G_06_04_IK', 13) ]
    for target_bush in target_bushes:
      # input_path = f'./resource/inference/{target_bush[0]}_.xlsx'
      input_path = f'./resource/inference/{target_bush[0]}.xlsx'
      model_path = os.path.join(save_path, f'saved_models\\model_test_idx_{target_bush[1]}.pth' ) # 모델 경로
      
      gt_source = r'E:\Dongwoo\TeamWork\Hyundai_bush_2\github\bush_prediction_CNN\resource\combined_data_10.npy'
      
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      model, input_scaler, output_scalers = load_model_and_scalers(model_path, 12, device)
      # 디바이스 설정

      # 데이터셋 로드 (이 부분은 데이터셋 구조에 맞게 수정 필요)
      dataset = VEPDataset_inference(input_path = input_path, field_range=256, num_stiffness=6)
      test_dataloder = dataset.get_loader(input_scaler=input_scaler)
      dataset_gt = VEPDataset(batch =1 , output_path = gt_source, gt_path= gt_source, field_range=256, num_stiffness=6)

      prediction, acc, acc_post = test_model(model, device, input_scaler, output_scalers, test_dataloder, gt_dataset=dataset_gt, target_bush= target_bush, field_range=256, save_path=save_path)
