"""
@author: Hung Son
"""

import pandas as pd

def save(rewards, file):
    rewards_df = pd.DataFrame(rewards)
    rewards_df.to_csv(file)
