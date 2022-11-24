"""
Group Members: Nathan Dennis, Kevin Wang, Cheeson Lau
This file answers the second question, which zone affects the game more, in the
paint, mid range, or behind the three point line? A variety of visualizations
and calculations will be done in this file in order to answer the question.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
pd.options.mode.chained_assignment = None


def initialize_dataset(df):
    """
    Initializes the dataset and returns it. Adds the columns that I need and
    removes the columns that are not useful in question 2. All data is rounded
    off to 1 decimal place, except WIN%.
    """
    df = df.iloc[:, [0, 1, 5, 11, 12, 13, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                     64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                     78, 79, 80, 81, 82, 83, 84]]
    df['CRFGM'] = df['RAFGM'] + df['NRAFGM']
    df['CRFGA'] = df['RAFGA'] + df['NRAFGA']
    df['CRFG%'] = round(df['CRFGM'] / df['CRFGA'] * 100, 1)
    df['OCRFGM'] = df['ORAFGM'] + df['ONRAFGM']
    df['OCRFGA'] = df['ORAFGA'] + df['ONRAFGA']
    df['OCRFG%'] = round(df['OCRFGM'] / df['OCRFGA'] * 100, 1)
    df['O3PM'] = df['OC3M'] + df['OATB3M']
    df['O3PA'] = df['OC3A'] + df['OATB3A']
    df['O3P%'] = round(df['O3PM'] / df['O3PA'] * 100, 1)
    df['TEAMY'] = df['YEAR'].astype('str') + ' ' + df['TEAM']
    df = df.loc[:, ['YEAR', 'TEAMY', 'WIN%', 'CRFGM', 'CRFGA', 'CRFG%',
                    'MRFGM', 'MRFGA', 'MRFG%', '3PM', '3PA', '3P%', 'OCRFGM',
                    'OCRFGA', 'OCRFG%', 'OMRFGM', 'OMRFGA', 'OMRFG%', 'O3PM',
                    'O3PA', 'O3P%']]
    return df


def modelling(df):
    """
    Creates a model that predicts the WIN% with 4 variables: 3P%, CRFG%, O3P%
    and OCRFG%. Prints out mean absolute error and r squared value of both the
    test set and train set.
    """
    x_var = df.loc[:, ['3P%', 'CRFG%', 'O3P%', 'OCRFG%']]
    y_var = df['WIN%']
    x_train, x_test, y_train, y_test = train_test_split(x_var, y_var,
                                                        test_size=0.2)
    # After trying max_depth from 1 to 50 for many times, it seems that
    # max_depth = 5 optimizes the accuracy of the model.
    # The MAE to predict win% is around 0.06 to 0.10.
    mae_arr = []
    for i in range(1, 50):
        model = DecisionTreeRegressor(max_depth=i)
        model.fit(x_train, y_train)
        train_predictions = model.predict(x_train)
        train_mae = mean_absolute_error(y_train, train_predictions)
        test_predictions = model.predict(x_test)
        test_mae = mean_absolute_error(y_test, test_predictions)
        train_r2 = r2_score(y_train, train_predictions)
        test_r2 = r2_score(y_test, test_predictions)
        mae_arr.append({'Max depth': i, 'Train MAE': train_mae,
                        'Test MAE': test_mae, 'Train r2': train_r2,
                        'Test r2': test_r2})
    mae_arr = pd.DataFrame(mae_arr)
    print(mae_arr)


def plot_indep_var(df):
    """
    Generates and saves the regression plots of WIN% against any independent
    variables in the plots directory. Each image has 6 regression plots, each
    one represents one year.
    """
    dep_var = 'WIN%'
    for indep_var in df:
        if indep_var != 'YEAR' and indep_var != 'TEAMY' \
           and indep_var != 'WIN%':
            sns.lmplot(x=indep_var, y=dep_var, col='YEAR', col_wrap=3,
                       hue='YEAR', data=df)
            plt.xlabel = (indep_var)
            plt.ylabel = (dep_var)
            plt.savefig('plots/' + indep_var + '.png', bbox_inches='tight')


def print_r2(rowsno, df):
    """
    Saves the r squared value in a dictionary and returns the dictionary.
    Prints out the r squared values with the variables and years.
    """
    r2dict = {
        'CRFGA': [], 'CRFGM': [], 'CRFG%': [], 'MRFGA': [], 'MRFGM': [],
        'MRFG%': [], '3PA': [], '3PM': [], '3P%': [], 'OCRFGA': [],
        'OCRFGM': [], 'OCRFG%': [], 'OMRFGA': [], 'OMRFGM': [], 'OMRFG%': [],
        'O3PA': [], 'O3PM': [], 'O3P%': [], 'YEAR': []
    }
    years = df['YEAR'].unique()
    rows_per_year = rowsno / len(years)
    for year in reversed(years):
        filtered_df = df[df['YEAR'] == year]
        r2dict['YEAR'].append(year)
        for x in filtered_df:
            if x != 'YEAR' and x != 'TEAMY' and x != 'WIN%':
                r_square = round(cal_r2(rows_per_year, x, 'WIN%', filtered_df),
                                 3)
                print('R2 for WIN%' + ' against ' + x + ' in ' + str(year) +
                      ': ' + str(r_square))
                r2dict[x].append(r_square)
    return r2dict


def cal_r2(rowsno, x, y, df):
    """
    Calculates and returns the r squared value with the given number of rows,
    independent variable, dependent variable and dataset.
    """
    slope = (rowsno * ((df[x] * df[y]).sum()) - (df[x].sum()) *
             (df[y].sum())) / (rowsno * ((df[x] ** 2).sum()) -
                               (df[x].sum()) ** 2)
    y_intercept = (df[y].sum() - slope * (df[x].sum())) / rowsno
    y_pred = y + '_pred'
    df[y_pred] = slope * df[x] + y_intercept
    y_mean = df[y].mean()
    r_square = 1 - (((df[y] - df[y_pred]) ** 2).sum()) / (((df[y] - y_mean)
                                                           ** 2).sum())
    return r_square


def graph_r2(r2dict, num):
    """
    Generates and saves two graphs that show how r squared varies from 2017 to
    2022. One graph involves offensive variables and another involves
    defensive variables (opponent offensive variables). Prints out the
    dictionary inputted.
    """
    r2df = pd.DataFrame(r2dict)
    print(r2df)
    fig1, ax1 = plt.subplots(1)
    fig2, ax2 = plt.subplots(1)
    for indep_var in r2df:
        if indep_var != 'YEAR' and indep_var[0] != 'O':
            r2df.plot(ax=ax1, x='YEAR', y=indep_var)
        if indep_var != 'YEAR' and indep_var[0] == 'O':
            r2df.plot(ax=ax2, x='YEAR', y=indep_var)
    ax1.set_xlabel('Year')
    ax2.set_xlabel('Year')
    ax1.set_ylabel('R Squared Value')
    ax2.set_ylabel('R Squared Value')
    ax1.set_title('Offensive Shooting Stats R Squared Value From 2017 to 2022')
    ax2.set_title('Defensive Shooting Stats R Squared Value From 2017 to 2022')
    fig1.savefig('plots/r2comparison1 ' + str(num) + '.png')
    fig2.savefig('plots/r2comparison2 ' + str(num) + '.png')


def main():
    """
    Reads the csv files and runs all the functions above. Testing is done
    here.
    """
    sns.set()
    df = pd.read_csv('NBA_stats.csv')
    df = initialize_dataset(df)
    modelling(df)
    plot_indep_var(df)
    r2dict = print_r2(df['TEAMY'].count(), df)
    graph_r2(r2dict, 1)
    # testing my result using playoff stats
    df2 = pd.read_csv('NBA_stats.csv')
    teams6 = (df2['TEAM'] == 'Golden State Warriors') |\
             (df2['TEAM'] == 'Houston Rockets') |\
             (df2['TEAM'] == 'Detroit Pistons') |\
             (df2['TEAM'] == 'Brooklyn Nets') |\
             (df2['TEAM'] == 'New York Knicks') |\
             (df2['TEAM'] == 'New Orleans Pelicans') |\
             (df2['TEAM'] == 'Milwaukee Bucks') |\
             (df2['TEAM'] == 'Sacramento Kings') |\
             (df2['TEAM'] == 'Phoenix Suns') |\
             (df2['TEAM'] == 'Boston Celtics')
    df2 = df2[teams6]
    df2 = initialize_dataset(df2)
    r2dict2 = print_r2(df2['TEAMY'].count(), df2)
    graph_r2(r2dict2, 2)


if __name__ == '__main__':
    main()
