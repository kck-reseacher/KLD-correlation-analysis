import operator
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


def KL_divergence(x, y):
    kl = -1 * np.sum(x * np.log(y / x))
    return kl


# Jensen-Shannon Divergence: KL Divergence를 symmetric 하게 개량한 척도
def JS_divergence(x, y):
    M = (x + y) / 2
    a = KL_divergence(x, M)
    b = KL_divergence(y, M)
    c = (a + b) / 2
    return c


# Pattern Matching Coefficient
def PMC(x_val, y_val):
    # y: true values, 원본 확률 분포
    # x: predicted values, 근사된 분포

    x = x_val.copy()
    y = y_val.copy()
    if len(y) < len(x):
        print(f"y length is {len(y)}, y variable is not sufficient")
        return None

    if np.sum(y) == 0:
        print(f"y variable all values is zero")
        return None

    if np.sum(x) == 0:
        print(f"x variable all values is zero")
        return None

    if np.unique(y, return_counts=True)[1][0] == 25:
        print(f"y variable all values is same")
        return None

    # divide by zero 예외처리
    if 0 in x:
        x = np.where(x == 0, np.min(x[np.where(x > 0)]), x)

    if 0 in y:
        y = np.where(y == 0, np.min(y[np.where(y > 0)]), y)

    y_dst = y / np.sum(y)
    x_dst = x / np.sum(x)

    # create mirrored distribution
    y_temp = -1 * y_dst
    mr_y_dst = y_temp + np.abs(np.min(y_temp)) + np.min(y_dst)

    if not JS_divergence(y_dst, mr_y_dst) == 0:
        a = 1 - JS_divergence(y_dst, x_dst) / JS_divergence(y_dst, mr_y_dst)
    else:
        a = 1

    # repeat with opposite baseline
    # create mirrored distribution
    x_temp = -1 * x_dst
    mr_x_dst = x_temp + np.abs(np.min(x_temp)) + np.min(x_dst)
    if not JS_divergence(x_dst, mr_x_dst) == 0:
        b = 1 - JS_divergence(x_dst, y_dst) / JS_divergence(x_dst, mr_x_dst)
    else:
        b = 1

    pmc = (a + b) / 2

    return pmc


if __name__ == "__main__":
    print("Kullback Leibler Divergence start !!")

    result_dir = Path("./KLD_result")
    result_dir.mkdir(exist_ok=True, parents=True)

    train_df = pd.read_csv("./train_data/happyscore_income.csv", index_col=0)
    x_valiables = ['adjusted_satisfaction', 'avg_satisfaction', 'std_satisfaction', 'avg_income', 'median_income',
                   'income_inequality', 'GDP']

    pmc_result_dict = dict()
    y_feat = 'happyScore'
    for x_feat in x_valiables:
        y = train_df[y_feat].values
        x = train_df[x_feat].values
        pmc_result_dict[x_feat] = PMC(x, y)
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        scaled_x = x_scaler.fit_transform(train_df[x_feat].values.reshape(-1, 1))
        scaled_y = y_scaler.fit_transform(train_df[y_feat].values.reshape(-1, 1))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=train_df.index, y=scaled_x.reshape(1, -1)[0], mode="lines+markers", name="scaled x value"))
        fig.add_trace(
            go.Scatter(x=train_df.index, y=scaled_y.reshape(1, -1)[0], mode="lines+markers", name="scaled y value",
                       marker=dict(color='violet')))
        fig.update_layout(
            title=f'<b></b> KLD result x: {x_feat}, y: {y_feat} <br> pattern matching coefficient : {pmc_result_dict[x_feat]}')
        fig.show()
        fig.write_image(f"./KLD_result/KLD_result_y_{y_feat}_x_{x_feat}.png")

    print("Kullback Leibler Divergence finish !!")