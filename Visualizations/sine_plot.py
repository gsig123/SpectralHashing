import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_two_subfigs_besides():
    x = np.linspace(0, 2 * np.pi, 400)
    df = pd.DataFrame({'x': x, 'y': np.sin(x ** 2)})
    df.index.names = ['obs']
    df.columns.names = ['vars']

    idx = np.array(df.index.tolist(), dtype='float')  # make an array of x-values

    # call regplot on each axes
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
    sns.regplot(x=idx, y=df['x'], ax=ax1)
    sns.regplot(x=idx, y=df['y'], ax=ax2)

    plt.show()

def plot_sine():
    x = np.linspace(0, np.pi, 1300)
    x_intersect = np.arange(np.pi/20.0, np.pi, np.pi/10.0)
    df = pd.DataFrame({'x': x, 'y': np.sin(np.pi/2.0 + 2.0 * np.pi / (np.pi/5.0) * x)})
    df_intersect = pd.DataFrame({'x': x_intersect, 'y': np.sin(np.pi/2.0 + 2.0 * np.pi / (np.pi/5.0) * x_intersect)})
    # sin(pi/2 + 2* pi/(pi/5)*x)
    df.index.names = ['obs']
    df.columns.names = ['vars']

    # idx = np.array(df.index.tolist(), dtype='float')  # make an array of x-values

    # call regplot on each axes
    # fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
    # sns.regplot(x=idx, y=df['x'], ax=ax1)
    sns.regplot(x=df['x'], y=df['y'], ci=None, color="#3d3d3d", marker=".", scatter_kws={"color": "#55A868", "s": 50}, line_kws={"lw": 1, "alpha": .9})
    sns.regplot(x=df_intersect['x'], y=df_intersect['y'], ci=None, color="#3d3d3d", marker=".", scatter_kws={"color": "#C44F53", "marker": "h", "s": 200, "alpha": 1}, line_kws={"lw": 1, "alpha": 0.8})

    greek_letterz = [chr(code) for code in range(945, 970)]
    plt.xticks([3.1415], [greek_letterz[15]])

    print(np.arange(np.pi/20.0, np.pi, np.pi/10.0))

    plt.show()

def main():# create df
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)

    # plot_two_subfigs_besides()
    plot_sine()

if __name__ == '__main__':
    main()