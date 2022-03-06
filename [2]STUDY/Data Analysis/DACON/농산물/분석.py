import pandas as pd

data = pd.read_csv('D:/[0] Study/[9] 데이터분석/농산물_가격예측/public_data/train.csv')
data.head()

week_day_map = {}
for i, d in enumerate(data['요일'].unique()):
    week_day_map[d] = i
data['요일'] = data['요일'].map(week_day_map)