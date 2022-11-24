"""
Group Members: Nathan Dennis, Kevin Wang, Cheeson Lau
This file answers the first question, how does offensive and defensive rating
affect win percentage. A variety of visualizations and calculations will be
done in this file in order to answer the question.
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy import stats
import statsmodels.stats.api as sms
sns.set()


def off_plot(data):
    """
    Takes the data set, data, and creates a plot comparing offensive rating to
    win percentage, called OffRtg.png.
    """
    sns.relplot(x='OffRtg', y='WIN%', kind='line', data=data)
    plt.title('Offensive Rating Compared to Win Percent')
    plt.xlabel('Offensive Rating')
    plt.ylabel('Win Percent (%)')
    plt.savefig('OffRtg.png', bbox_inches='tight')


def def_plot(data):
    """
    Takes the data set, data, and creates a plot comparing defensive rating to
    win percentage, called DefRtg.png.
    """
    sns.relplot(x='DefRtg', y='WIN%', kind='line', data=data)
    plt.title('Defensive Rating Compared to Win Percent')
    plt.xlabel('Defensive Rating')
    plt.ylabel('Win Percent (%)')
    plt.savefig('DefRtg.png', bbox_inches='tight')


def high_off_plt(high_off):
    """
    Takes the data set, high_off, which represents the data for teams with an
    above average/high offensive rating and creates a line plot comparing
    these offensive ratings to win percent, labelled high_off.png. Also
    calculates the mean win percent for high offensive rating teams.
    """
    sns.relplot(x='OffRtg', y='WIN%', kind='line', data=high_off)
    plt.ylim(0, 1)
    plt.title('High Offensive Rating Compared to Win Percent')
    plt.xlabel('Offensive Rating')
    plt.ylabel('Win Percent (%)')
    plt.savefig('high_off.png', bbox_inches='tight')
    high_off_percent = high_off['WIN%'].mean()
    print('High offensive rating Average Win%:', high_off_percent)


def low_off_plt(low_off):
    """
    Takes the data set, low_off, which represents the data for teams with a
    below average offensive rating and creates a line plot comparing these
    offensive ratings to win percent, labelled low_off.png. Also
    calculates the mean win percent for low offensive rating teams.
    """
    sns.relplot(x='OffRtg', y='WIN%', kind='line', data=low_off)
    plt.ylim(0, 1)
    plt.title('Low Offensive Rating Compared to Win Percent')
    plt.xlabel('Offensive Rating')
    plt.ylabel('Win Percent (%)')
    plt.savefig('low_off.png', bbox_inches='tight')
    low_off_percent = low_off['WIN%'].mean()
    print('Low offensive rating Average Win%:', low_off_percent)


def off_ci(high_off, low_off):
    """
    Creates the offensive confidence interval using the data sets high_off,
    representing the data set for teams which had an above average offensive
    rating and low_off representing the teams which had a below average
    offensive rating. Creates both a confidence interval and p-test for these
    results.
    """
    high_off_percent = []
    for i in high_off['WIN%']:
        high_off_percent.append(i)
    low_off_percent = []
    for i in low_off['WIN%']:
        low_off_percent.append(i)
    cm = sms.CompareMeans(sms.DescrStatsW(high_off_percent),
                          sms.DescrStatsW(low_off_percent))
    print('Confidence Interval for offensive ratings',
          cm.tconfint_diff(usevar='unequal'))
    p_val = stats.ttest_ind(high_off_percent, low_off_percent, equal_var=False)
    print('Offensive Difference p value:', p_val)


def high_def_plt(high_def):
    """
    Takes the data set, high_def, which represents the data for teams with an
    above average/high defensive rating and creates a line plot comparing
    these defensive ratings to win percent, labelled high_def.png. Also
    calculates the mean win percent for high defensive rating teams.
    """
    sns.relplot(x='DefRtg', y='WIN%', kind='line', data=high_def)
    plt.ylim(0, 1)
    plt.title('High Defensive Rating Compared to Win Percent')
    plt.xlabel('Defensive Rating')
    plt.ylabel('Win Percent (%)')
    plt.savefig('high_def.png', bbox_inches='tight')
    high_def_percent = high_def['WIN%'].mean()
    print('High Defensive rating Average Win%:', high_def_percent)


def low_def_plt(low_def):
    """
    Takes the data set, low_def, which represents the data for teams with a
    below average/low defensive rating and creates a line plot comparing
    these defensive ratings to win percent, labelled low_def.png. Also
    calculates the mean win percent for low defensive rating teams.
    """
    sns.relplot(x='DefRtg', y='WIN%', kind='line', data=low_def)
    plt.ylim(0, 1)
    plt.title('Low Defensive Rating Compared to Win Percent')
    plt.xlabel('Defensive Rating')
    plt.ylabel('Win Percent (%)')
    plt.savefig('low_def.png', bbox_inches='tight')
    low_def_percent = low_def['WIN%'].mean()
    print('Low Defensive rating Average Win%:', low_def_percent)


def def_ci(high_def, low_def):
    """
    Creates the defensive confidence interval using the data sets high_def,
    representing the data set for teams which had an above average/high
    defensive rating and low_def representing the teams which had a below
    average/low defensive rating. Creates both a confidence interval and p-test
    for these results.
    """
    high_def_percent = []
    for i in high_def['WIN%']:
        high_def_percent.append(i)
    low_def_percent = []
    for i in low_def['WIN%']:
        low_def_percent.append(i)
    cm = sms.CompareMeans(sms.DescrStatsW(high_def_percent),
                          sms.DescrStatsW(low_def_percent))
    print('Confidence Interval for defensive ratings',
          cm.tconfint_diff(usevar='unequal'))
    p_val = stats.ttest_ind(high_def_percent, low_def_percent, equal_var=False)
    print('Defensive Difference p value:', p_val)


def offmodel(data):
    """
    Creates the model using the data set, data, and offensive rating as
    variable to predict win percent. Creates the model, then calculates
    the adjusted r squared and rmse score for this model to test its validity.
    """
    data = data.loc[:, ['OffRtg', 'WIN%']]
    model = DecisionTreeRegressor()
    features = data.loc[:, data.columns != 'WIN%']
    labels = data['WIN%']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2, random_state=1000)
    model.fit(features_train, labels_train)
    test_predictions = model.predict(features_test)
    print('Mean Squared Error for Offensive rating Model:',
          mean_squared_error(labels_test, test_predictions))
    rsquared = r2_score(labels_test, test_predictions)
    adj_rsquared = 1-(1-rsquared)*(180-1)/(180-2)
    print('Offensive model adjusted r-squared value:', adj_rsquared)


def defmodel(data):
    """
    Creates the model using the data set, data, and defensive rating as
    variable to predict win percent. Creates the model, then calculates
    the adjusted r squared and rmse score for this model.
    """
    data = data.loc[:, ['DefRtg', 'WIN%']]
    model = DecisionTreeRegressor()
    features = data.loc[:, data.columns != 'WIN%']
    labels = data['WIN%']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2, random_state=1000)
    model.fit(features_train, labels_train)
    test_predictions = model.predict(features_test)
    print('Mean Squared Error for Defensive rating Model:',
          mean_squared_error(labels_test, test_predictions))
    rsquared = r2_score(labels_test, test_predictions)
    adj_rsquared = 1-(1-rsquared)*(180-1)/(180-2)
    print('Defensive model adjusted r-squared value:', adj_rsquared)


def combinemodel(data):
    """
    Creates a new model using the data set, data, using offensive and defensive
    rating as the predictor variables for win percent. Calculates the adjusted
    r squared and rmse score again to assess its validity.
    """
    data = data.loc[:, ['OffRtg', 'DefRtg', 'WIN%']]
    model = DecisionTreeRegressor()
    features = data.loc[:, data.columns != 'WIN%']
    labels = data['WIN%']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2, random_state=1000)
    model.fit(features_train, labels_train)
    test_predictions = model.predict(features_test)
    print('Mean Squared Error for Combined Rating Model:',
          mean_squared_error(labels_test, test_predictions))
    rsquared = r2_score(labels_test, test_predictions)
    adj_rsquared = 1-(1-rsquared)*(180-1)/(180-2)
    print('Combined model adjusted r-squared value:', adj_rsquared)


def main():
    data = pd.read_csv('NBA_stats.csv')
    avg_off = data['OffRtg'].mean()
    avg_def = data['DefRtg'].mean()
    high_off = data[data['OffRtg'] >= avg_off]
    low_off = data[data['OffRtg'] < avg_off]
    high_def = data[data['DefRtg'] >= avg_def]
    low_def = data[data['DefRtg'] < avg_def]
    off_plot(data)
    def_plot(data)
    high_off_plt(high_off)
    low_off_plt(low_off)
    off_ci(high_off, low_off)
    high_def_plt(high_def)
    low_def_plt(low_def)
    def_ci(high_def, low_def)
    offmodel(data)
    defmodel(data)
    combinemodel(data)


if __name__ == '__main__':
    main()
