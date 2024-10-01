import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

train_path = './playground-series-s4e8/train.csv'
test_path = './playground-series-s4e8/test.csv'
sub_path = './playground-series-s4e8/sample_submission.csv'

train_df = pd.read_csv(train_path).drop('id', axis=1)
test_df = pd.read_csv(test_path).drop('id', axis=1)
sub = pd.read_csv(sub_path)

# def handle_missed(df, n_freq=50):
#     numeric = ['stem-width', 'stem-height', 'cap-diameter']
#     category = [x for x in df.columns if x not in numeric]
#     if 'class' in category:
#         category.remove('class')
#     # for numerical cols
#     for col in numeric:
#         if df[col].isnull().sum() > 0:
#             mean_col = df[col].mean()
#             df[col].fillna(mean_col, inplace=True)
    
#     # for categorical cols
#     for col in category:
#         if df[col].isnull().sum() > 0:
#             # Replace missing values with 'NO'
#             df[col].fillna('NO', inplace=True)
#             unique_vals = df[col].unique()
#             freq_list = []
#             for x in unique_vals:
#                 if np.sum(df[col] == x) > n_freq:
#                     freq_list.append(x)

#             # Replace less frequent values with 'LF'
#             df[col] = df[col].apply(lambda x: x if x in freq_list else 'LF')
    
#     return df

# # train data
# train_df = handle_missed(train_df, n_freq=100)

# # test data
# test_df = handle_missed(test_df, n_freq=100)

# numeric = ['stem-width', 'stem-height', 'cap-diameter']
# category = [x for x in test_df.columns if x not in numeric]
# train_df[category] = train_df[category].astype('category')
# test_df[category] = test_df[category].astype('category')

# # log tranformation for nearly normal distribution of numerical columns
# for col in numeric:
#     train_df[col] = train_df[col].apply(lambda x: np.log(x+1.00001))
#     test_df[col] = test_df[col].apply(lambda x: np.log(x+1.00001))

# from sklearn.preprocessing import LabelEncoder

# le = LabelEncoder() # converts categorical data into numerical ones: e=0, 1=p
# train_df['class'] = le.fit_transform(train_df['class'])

predictor = TabularPredictor(label='class',
                            eval_metric='mcc',
                            problem_type='binary').fit(train_df,
                                                       presets='best_quality',
                                                        time_limit=3600*20,
#                                                        hyperparameters=hyperparameters,
                                                       num_bag_folds=7,
                                                       num_stack_levels=2,
                                                       excluded_model_types=['KNN'],
#                                                        ag_args_fit={'num_gpus': 1}
                                                      )
results = predictor.fit_summary()

predictor.leaderboard()

y_pred = predictor.predict(test_df)

# sub = pd.read_csv('/kaggle/input/playground-series-s4e8/sample_submission.csv')
sub['class'] = y_pred
# sub['class'] = sub['class'].map({0:'e', 1:'p'})
sub.to_csv('submission_4.csv', index=False)