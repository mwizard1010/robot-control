"""
@author: Hung Son
"""

import pandas as pd

def save(rewards, feedbacks, file):
    pairs = {'Reward': rewards, 'Feedback': feedbacks}
    data_df = pd.DataFrame.from_dict(pairs)
    data_df.to_csv(file)