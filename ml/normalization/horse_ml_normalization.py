import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import VarianceThreshold

title_list = [
'surgery','Age','hospital_number','rectal_temperature','pulse','respiratory_rate'\
,'temperature_of_extremities','peripheral_pulse','mucous_membranes','capillary_refill_time','pain'\
,'peristalsis','abdominal_distension','nasogastric_tube','nasogastric_reflux','nasogastric_reflux_PH'\
,'rectal_examination_feces','abdomen','packed_cell_volume','total_protein','abdominocentesis_appearance'\
,'abdomcentesis_total_protein','outcome','surgical_lesion','symptom1','symptom2','symptom3','cp_data']

train = pd.read_csv('horse-colic.csv', header = None, delim_whitespace=True)

# 插入題目訂的headers
train.columns = title_list

# 將空值('?')轉換為null準備給simpleimputer處理
train = train.replace('?',np.nan)
# 取出最適合當作y的feature
train_output = train.loc[:, ['outcome','symptom1','symptom2','symptom3']]

data_use_mean = ['rectal_temperature','total_protein','abdominocentesis_appearance','abdomcentesis_total_protein']
# 將某些適合用mean來表示的feature使用mean來補上空值
train_mean_data = train.loc[:, data_use_mean]
train_missing_value = SimpleImputer(missing_values = np.nan, strategy='mean')
train_missing_value = train_missing_value.fit(train_mean_data)
train_mean_data.iloc[:,:] = train_missing_value.transform(train_mean_data)
# 丟掉mean的那幾個column
train = train.drop(data_use_mean, axis = 1)
# 將丟掉的column與原先train組合
train = pd.concat([train,train_mean_data], axis = 1)
# 其餘空值使用median來表示
train_missing_value = SimpleImputer(missing_values = np.nan, strategy='median')
train_missing_value = train_missing_value.fit(train)
#iloc要重新定位train
train.iloc[:,:] = train_missing_value.transform(train)

# hospital_number 無意義所以drop掉
train = train.drop(['outcome','hospital_number','symptom1','symptom2','symptom3'], axis = 1)
print(train)

# L2 normalization
L2no = Normalizer()
L2_train = L2no.fit_transform(train)

# 刪掉variance太小的column
vt = VarianceThreshold(threshold=1e-04)
L2_train_vt = vt.fit_transform(L2_train)
print(L2_train_vt.var(axis=1))