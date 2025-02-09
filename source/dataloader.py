import numpy as np
import torch
import ast
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import  QuantileTransformer
# from network.CNN import build
import warnings
warnings.filterwarnings(action='ignore')

__all__ = ["LogScaler", "Dataset", "VEPDataset", "VEPDataset_inference"]
# In[3. Data setting] #############################################################################################

class LogScaler:
    def fit(self, X, y=None):
        self.min_ = np.min(X, axis=0)
        return self

    def transform(self, X):
        return np.log1p(X - self.min_)

    def inverse_transform(self, X):
        return np.expm1(X) + self.min_
    
class Dataset(Dataset):
    def __init__(self, input_dataset, output_dataset):
        super(Dataset, self).__init__()

        self.inputs = input_dataset
        self.outputs = output_dataset

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx]).float()
        y = torch.from_numpy(self.outputs[idx]).float()

        return x, y

class VEPDataset():
    def __init__(self, batch, output_path, gt_path, field_range, num_stiffness, mode='test'):
        self.mode = mode
        self.batch = batch
        self.test_idx = 0  # Default value
        self.field_range = field_range
        self.num_stiff = num_stiffness
        
        # output_path = r"D:\hyundai_bush_2nd\2D_predict_CNN\2D_prediction\2D_prediction\MLP_extrapolation\output\combined_data.npy"
        output_path = output_path
    
        total_output_data = np.load(output_path, allow_pickle=True).item()
        total_output_data_gt = np.load(gt_path, allow_pickle=True).item() 
        
        self.original_input = total_output_data["inputs"]  # 원본 데이터를 유지
        self.original_output = total_output_data["outputs"]
        self.bush_names = total_output_data.get("bush_names", None)
        
        self.original_input_gt = total_output_data_gt["inputs"]  # 원본 데이터를 유지
        self.original_output_gt = total_output_data_gt["outputs"]
        self.bush_names_gt = total_output_data_gt.get("bush_names", None)
        

        if self.bush_names is not None:
            self.bush_indices = {name: np.where(self.bush_names == name)[0] for name in np.unique(self.bush_names)}
        else:
            self.bush_indices = None
            
        if self.bush_names_gt is not None:
            self.bush_indices_gt = {name: np.where(self.bush_names_gt == name)[0] for name in np.unique(self.bush_names_gt)}
        else:
            self.bush_indices_gt = None
            
        # bush_numbers = np.array([int(name.split('_')[-2]) for name in self.bush_names])
        bush_numbers_gt = np.array([int(name.split('_')[-1][-1]) for name in self.bush_names_gt])
        bush_numbers = np.array([int(name.split('_')[-1][-1]) for name in self.bush_names_gt])
        # valid_indices = bush_numbers == 1
        # valid_indices = (bush_numbers == 1)| (bush_numbers == 2) | (bush_numbers == 4) |  (bush_numbers == 5)
        # valid_indices = (bush_numbers == 4) | (bush_numbers == 5)
        valid_indices = (bush_numbers == 1) | (bush_numbers == 2) | (bush_numbers == 3) | (bush_numbers == 4) | (bush_numbers == 5) | (bush_numbers == 6)
        
        self.original_input = self.original_input[valid_indices]
        self.original_output = self.original_output[valid_indices]
        bush_numbers = bush_numbers[valid_indices]
        
        self.original_output_gt = self.original_output_gt[valid_indices]
        

        # Append bush numbers to inputs
        self.original_input = np.hstack((self.original_input, bush_numbers.reshape(-1, 1)))

        subAxes, mainAxes = get_extrapolation_range(self.original_input[:,-1], self.original_input[:,:8])

        self.original_input = np.hstack((self.original_input, subAxes.reshape(-1, 1), mainAxes.reshape(-1, 1)))
        self.original_input[:,8] = np.log1p(self.original_input[:,8])  ## log scale로 변환 선형강성값

        # Placeholders for dynamic updates
        self.np_train_input = self.original_input.copy()
        self.np_train_output = self.original_output.copy()
        self.np_test_input = None
        self.np_test_output = None
        self.inp_minmax = None

    def update_test_idx(self, idx):
        self.test_idx = idx
        self.input_scalers = QuantileTransformer()
        self.output_scalers = [QuantileTransformer() for _ in range(6)]

        def scale_output_data(data, scaler):
            reshaped = data.reshape(-1, self.field_range)
            scaled = scaler.transform(reshaped)
            return scaled.reshape(data.shape)
        
        # Split train and test data dynamically from the original data
        self.np_test_input = self.original_input[self.test_idx*self.num_stiff:(self.test_idx + 1)*self.num_stiff]
        self.np_test_output = self.original_output[self.test_idx*self.num_stiff:(self.test_idx + 1)*self.num_stiff]
        self.np_test_output_gt = self.original_output_gt[self.test_idx*self.num_stiff:(self.test_idx + 1)*self.num_stiff]

        mask = np.ones(self.original_input.shape[0], dtype=bool)
        mask[self.test_idx*self.num_stiff:(self.test_idx + 1)*self.num_stiff] = False  # Exclude test data
        self.np_train_input = self.original_input[mask]
        self.np_train_output = self.original_output[mask]

        for i in range(6):
            mask = self.np_train_input[:, 9] == (i + 1)
            reshaped_output = self.np_train_output[mask].reshape(-1, self.field_range)
            if reshaped_output.shape[0] == 0:
                continue
            self.output_scalers[i].fit(reshaped_output)
            self.np_train_output[mask] = scale_output_data(self.np_train_output[mask], self.output_scalers[i])

    def get_loader(self):
        
        # # Split train data into train and validation sets
        # train_input, val_input, train_output, val_output = train_test_split(
        #     self.np_train_input, self.np_train_output, test_size=0.1, random_state=42
        # )

        # Scale train, validation, and test inputs
        scaled_train_input = self.input_scalers.fit_transform(self.np_train_input)
        # scaled_val_input = self.input_scalers.transform(val_input)
        scaled_test_input = self.input_scalers.transform(self.np_test_input)

        # Create datasets and data loaders
        train_dataset = Dataset(scaled_train_input, self.np_train_output)
        # val_dataset = Dataset(scaled_val_input, val_output)
        test_dataset = Dataset(scaled_test_input, self.np_test_output)

        train_loader = DataLoader(train_dataset, batch_size=self.batch, shuffle=True, drop_last=False)
        # val_loader = DataLoader(val_dataset, batch_size=self.batch, shuffle=True, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Single sample for LOOCV

        return train_loader, test_loader, self.np_test_output_gt, self.output_scalers, self.input_scalers, self.field_range
    
    
class VEPDataset_inference():
    def __init__(self, input_path="./resource/inference/input_data.xlsx", field_range=256, num_stiffness=6):
        self.input_path = input_path
        # 엑셀 파일 불러오기 (경로 수정 필요)
        df = pd.read_excel(input_path, dtype=str)  # 모든 데이터를 문자열로 로드
        df["predictions"] = df["predictions"].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))
        df.iloc[:, 3] = df.iloc[:, 3].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))
        df.iloc[:, 4] = df.iloc[:, 4].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.int32))
        
        input_data_all = []        
        for idx in range(len(df)):
            
            linear_stiffness = df.iloc[idx, 3]  # (6,)
            bush_number = df.iloc[idx, 4]  # (6,)
            self.input_data = df.loc[idx, "predictions"]  # (8,)
            
            expanded_data = np.zeros((6, 10))         
            
            for i in range(6):
                expanded_data[i, :8] = self.input_data  # 8개 데이터 동일하게 복사
                expanded_data[i, 8] = linear_stiffness[i]  # 9번째 열에 값 추가
                expanded_data[i, 9] = bush_number[i]  # 10번째 열에 값 추가

            subAxes, mainAxes = get_extrapolation_range(expanded_data[:,-1], expanded_data[:,:8])
            expanded_data = np.hstack((expanded_data, subAxes.reshape(-1, 1), mainAxes.reshape(-1, 1)))        
            input_data_all.append(expanded_data)
            
        self.input_data = np.vstack(input_data_all)
        print(self.input_data.shape)

    def get_loader(self, input_scaler):
        
        scaled_input = input_scaler.transform(self.input_data)
        # scaled_tensor = torch.tensor(scaled_input, dtype=torch.float32)
        
        dataset = Dataset(scaled_input, np.zeros((scaled_input.shape[0], 256)))
        # for i in range(1, num_stiffness+1):
        return DataLoader(dataset, batch_size=1, shuffle=False)
        
    
# In[4. LOOCV WMAPE Calculation] ############################################################################################

# def inverse_scale_data(data, scaler):
#     reshaped = data.reshape(-1, field_range)
#     inversed = scaler.inverse_transform(reshaped)
#     return inversed.reshape(data.shape)

def get_extrapolation_range(stiffness_value_to_train, df):

    scale_factor = 1.0487
    # Calculate rubber parameters
    D_O_RUBBER = 2 * (df[:, 0] + df[:, 1])
    D_I_RUBBER = 2 * df[:, 0]
    L_O_RUBBER = 2 * df[:, 2]
    L_I_RUBBER = 2 * (df[:, 2] + df[:, 3])

    # Calculate displacements and angles
    x_disp = (D_O_RUBBER - D_I_RUBBER) / 2 - df[:, 6]
    z_disp = (L_I_RUBBER * scale_factor - L_O_RUBBER) / 2
    theta_x = (np.arctan(D_O_RUBBER / L_O_RUBBER) - np.arcsin(D_I_RUBBER / np.sqrt(D_O_RUBBER**2 + L_O_RUBBER**2)))

    subAxes = np.zeros_like(stiffness_value_to_train, dtype=float)
    mainAxes = np.zeros_like(stiffness_value_to_train, dtype=float)

    mask_1_2 = np.isin(stiffness_value_to_train, [1, 2])
    mask_3 = np.isin(stiffness_value_to_train, [3])

    mask_4_5 = np.isin(stiffness_value_to_train, [4, 5])
    mask_6 = stiffness_value_to_train == 6

    subAxes[mask_1_2] = theta_x[mask_1_2]
    mainAxes[mask_1_2] = x_disp[mask_1_2]

    subAxes[mask_3] = theta_x[mask_3]
    mainAxes[mask_3] = z_disp[mask_3]

    subAxes[mask_4_5] = z_disp[mask_4_5]
    mainAxes[mask_4_5] = theta_x[mask_4_5]

    subAxes[mask_6] = z_disp[mask_6]
    mainAxes[mask_6] = 0.2617993877991494  # 15 degree

    return subAxes, mainAxes