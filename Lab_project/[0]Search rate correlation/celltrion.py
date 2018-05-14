import pandas as pd
import math
from antQuant.utils import DataManager
import matplotlib.pyplot as plt


celltrion = DataManager()
celltrion_stock = celltrion.get_daily_ohlcv('068270')
celltrion_stock.index.get_loc('2016-01-04')
celltrion_stock = celltrion_stock[celltrion_stock.index.get_loc('2016-01-04'):]


samsung = pd.read_excel('datalab_samsung.xlsx', skiprows=6)
samsungbio = pd.read_excel('datalab_samsungbio.xlsx',skiprows=6)
hynix = pd.read_excel('datalab_skhynix.xlsx',skiprows=6)
celltrion = pd.read_excel('datalab_celltrion.xlsx',skiprows=6)
hyundai = pd.read_excel('datalab_hyundai.xlsx',skiprows=6)

for df in [samsung, samsungbio, hynix, celltrion, hyundai]:
    df.columns = ['date', 'search']
    df.date = pd.to_datetime(df.date)

celltrion = celltrion.set_index('date')
celltrion = celltrion.shift(1)
all_set = pd.concat([celltrion, celltrion_stock], axis=1)
all_set = all_set.dropna(axis=0)

log_close = []
for i in range(len(all_set.index)):
    a = math.log(all_set['close'][i])
    log_close.append(a)

all_set['log_close']=log_close
all_set['log_close_diff'] = all_set['log_close'].diff()
all_set = all_set[1:]
all_set = pd.DataFrame(all_set)

from sklearn.preprocessing import StandardScaler
stdscaler = StandardScaler()
x_search = all_set[['search']].values.astype(float)
std_search = stdscaler.fit_transform(x_search)

norm_log_close_diff = all_set[['log_close_diff']].values.astype(float)
norm_log_close_diff = stdscaler.fit_transform(norm_log_close_diff)

all_set['std_search'] = std_search
all_set['norm_log_close_diff'] = norm_log_close_diff
all_set['std_search'].plot()
all_set['norm_log_close_diff'].plot()
plt.legend(['std_search', 'norm_log_close_diff'])

from scipy.stats import pearsonr
pearsonr(all_set['std_search'], all_set['norm_log_close_diff'])

print(all_set.corr())
plt.show()
