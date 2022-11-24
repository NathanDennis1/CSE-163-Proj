# Kevin Wang
# group member: Nathan Dennis, Cheeson Lau
# This function provides the necessary function for the part 3 of our
# final project, mainly addressing machine learning challenge goal
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import shap


def machine_learning(merged_df, playoff_df):
    # this function includes all of the neccessary processes for testing three
    # different regression models including decision tree regression, linear
    # regression and polynomial degree regression models, given the regular
    # season data for training and playoff data for testing. The specific
    # processes includes calculating mean absoulate error to compare accuracy
    # and making plots to facilitate the justification of what a certain plot
    # outperfrom another
    features_train = merged_df.loc[:, ['REB', 'AST', 'TOV', 'STL', 'BLK',
                                       'OPP_REB', 'OPP_AST', 'OPP_TOV',
                                       'OPP_STL', 'OPP_BLK']]
    # define training and testing features and labels
    labels_train = merged_df['PLUS/MINUS']
    playoff_df = playoff_df.loc[:, 'REB':'OPP_BLK']
    features_test = playoff_df.loc[:, playoff_df.columns != 'plus/minus']
    labels_test = playoff_df['plus/minus']
    # first model: decision tree regression model
    model1 = DecisionTreeRegressor()
    model1.fit(features_train, labels_train)
    test_prediction1 = model1.predict(features_test)
    mean_abs_error1 = mean_absolute_error(test_prediction1, labels_test)
    mean_sq_error1 = mean_squared_error(test_prediction1, labels_test)
    r2_score1 = r2_score(test_prediction1, labels_test)
    # print errors
    print(mean_abs_error1)
    print(mean_sq_error1)
    print(r2_score1)
    print()
    # second model: linear regression model
    model2 = LinearRegression()
    model2.fit(features_train, labels_train)
    test_prediction2 = model2.predict(features_test)
    mean_abs_error2 = mean_absolute_error(test_prediction1, labels_test)
    mean_sq_error2 = mean_squared_error(test_prediction2, labels_test)
    r2_score2 = r2_score(test_prediction1, labels_test)
    # print errors
    print(mean_abs_error2)
    print(mean_sq_error2)
    print(r2_score2)
    print()
    # plot each independent variables against dependent variables
    fig1, [[ax1, ax2, ax3, ax4, ax5], [ax6, ax7, ax8, ax9, ax10]] = \
        plt.subplots(2, 5, figsize=(20, 10))
    fig2, ax11 = plt.subplots(1, figsize=(20, 10))
    sns.regplot(x="REB", y="PLUS/MINUS", data=merged_df, ax=ax1)
    ax1.set_title('effect of rebound on +/-')
    sns.regplot(x="AST", y="PLUS/MINUS", data=merged_df, ax=ax2)
    ax2.set_title('effect of assist on +/-')
    sns.regplot(x="TOV", y="PLUS/MINUS", data=merged_df, ax=ax3)
    ax3.set_title('effect of turnover on +/-')
    sns.regplot(x="STL", y="PLUS/MINUS", data=merged_df, ax=ax4)
    ax4.set_title('effect of steal on +/-')
    sns.regplot(x="BLK", y="PLUS/MINUS", data=merged_df, ax=ax5)
    ax5.set_title('effect of block on +/-')
    sns.regplot(x="OPP_REB", y="PLUS/MINUS", data=merged_df, ax=ax6)
    ax6.set_title('effect of opponent rebound on +/-')
    sns.regplot(x="OPP_AST", y="PLUS/MINUS", data=merged_df, ax=ax7)
    ax7.set_title('effect of opponent assist on +/-')
    sns.regplot(x="OPP_TOV", y="PLUS/MINUS", data=merged_df, ax=ax8)
    ax8.set_title('effect of opponent turnover on +/-')
    sns.regplot(x="OPP_STL", y="PLUS/MINUS", data=merged_df, ax=ax9)
    ax9.set_title('effect of opponent steal on +/-')
    sns.regplot(x="OPP_BLK", y="PLUS/MINUS", data=merged_df, ax=ax10)
    ax10.set_title('effect of opponent block on +/-')
    # make polynomial degree regression model for 2, 3, 4, 5 degree
    # as well as print out errors for each degree model
    poly_regressor(2, features_train, labels_train, features_test, labels_test)
    poly_regressor(3, features_train, labels_train, features_test, labels_test)
    poly_regressor(4, features_train, labels_train, features_test, labels_test)
    poly_regressor(5, features_train, labels_train, features_test, labels_test)
    # set up explainer to interpret the weighting of each independent variable
    # on the dependent variable
    explainer = shap.Explainer(model2, features_train)
    shap_values = explainer(features_train)
    # show the weighting plot as a bar graph
    shap.summary_plot(shap_values, features_train, plot_type='bar')


def poly_regressor(degree_num,  features_train, labels_train, features_test,
                   labels_test):
    # Given the features and lables for both the training and testing dataset,
    # as well the number of intended degree, construct a polynomical regression
    # of the given degree, train and predict then print out the mean abosolute
    # error, mean squared error and R2 error
    poly = PolynomialFeatures(degree=degree_num, include_bias=False)
    features_train_poly = poly.fit_transform(features_train)
    model = LinearRegression()
    model.fit(features_train_poly, labels_train)
    features_test_poly = poly.fit_transform(features_test)
    test_prediction = model.predict(features_test_poly)
    mean_abs_error = mean_absolute_error(test_prediction, labels_test)
    mean_sq_error = mean_squared_error(test_prediction, labels_test)
    r2_scored = r2_score(test_prediction, labels_test)
    print(mean_abs_error)
    print(mean_sq_error)
    print(r2_scored)
    print()


def main():
    # read in regular season and playoff season data, merge and filter down to
    # 2021 data as the 2022 playoff data is still constantly being updated
    regular_df = pd.read_csv("NBA_stats.csv")
    regular_opp_df = pd.read_csv("opponent_stats.csv")
    playoff_df = pd.read_csv("playoff_stats.csv")
    regular_df = regular_df[regular_df["YEAR"] != 2022]
    merged_df = pd.merge(regular_df, regular_opp_df, on=['YEAR', 'TEAM'],
                         how='left')
    machine_learning(merged_df, playoff_df)


if __name__ == '__main__':
    main()
