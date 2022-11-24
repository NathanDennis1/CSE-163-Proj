"""
Group Members: Nathan Dennis, Kevin Wang, Cheeson Lau
Creates tests for question 1
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


def years_test(data):
    """
    Tests the data by comparing the results to a visualization
    representing different years within our data set.
    """
    sns.relplot(data=data, x='OffRtg', y='WIN%', size='WIN%',
                col='YEAR', col_wrap=3)
    plt.savefig('Offensive_Years_Test.png', bbox_inches='tight')
    sns.relplot(data=data, x='DefRtg', y='WIN%', size='WIN%',
                col='YEAR', col_wrap=3)
    plt.savefig('Defensive_Years_Test.png', bbox_inches='tight')


def result_test(high_off, low_off, high_def, low_def):
    """
    Tests some results from the calculated statistics using the high and low
    offensive rating, and high and low defensive rating, calculating mean
    win percent for these variables. Uses data over the year 2021, so 2021
    and 2022.
    """
    high_off = high_off[high_off['YEAR'] >= 2021]
    low_off = low_off[low_off['YEAR'] >= 2021]
    high_off_percent = high_off['WIN%'].mean()
    low_off_percent = low_off['WIN%'].mean()
    print('Test results for high offensive rating win percent:',
          high_off_percent)
    print('Test results for low offensive rating win percent:',
          low_off_percent)
    high_def = high_def[high_def['YEAR'] >= 2021]
    low_def = low_def[low_def['YEAR'] >= 2021]
    high_def_percent = high_def['WIN%'].mean()
    low_def_percent = low_def['WIN%'].mean()
    print('Test results for high defensive rating win percent:',
          high_def_percent)
    print('Test results for low defensive rating win percent:',
          low_def_percent)


def main():
    data = pd.read_csv('NBA_stats.csv')
    avg_off = data['OffRtg'].mean()
    avg_def = data['DefRtg'].mean()
    high_off = data[data['OffRtg'] >= avg_off]
    low_off = data[data['OffRtg'] < avg_off]
    high_def = data[data['DefRtg'] >= avg_def]
    low_def = data[data['DefRtg'] < avg_def]
    years_test(data)
    result_test(high_off, low_off, high_def, low_def)


if __name__ == '__main__':
    main()
