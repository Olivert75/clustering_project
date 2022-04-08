import pandas as pd
from pydataset import data

#Sklearn
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor 
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


def select_kbest  (X_df, y_df, n_features):
    '''
    Takes in the predictors, the target, and the number of features to select (k) ,
    and returns the names of the top k selected features based on the SelectKBest class
    
    X_df : the predictors
    y_df : the target
    n_features : the number of features to select (k)
    Example
    select_kbest(X_train_scaled, y_train, 2)
    '''
    
    f_selector = SelectKBest(score_func=f_regression, k= n_features)
    f_selector.fit(X_df, y_df)
    mask = f_selector.get_support()
    X_df.columns[mask]
    top = list(X_df.columns[mask])
    print(f'The top {n_features} selected feautures based on the SelectKBest class are: {top}' )
    return top


def select_rfe (X_df, y_df, n = 1, model = LinearRegression(normalize=True), rank = False):
    '''
    Takes in the predictors, the target, and the number of features to select (k) ,
    and returns the names of the top k selected features based on the Recursive Feature Elimination (RFE)
    
    X_df : the predictors
    y_df : the target
    n_features : the number of features to select (k)
    method : LinearRegression, LassoLars, TweedieRegressor
    Example
    select_rfe(X_train_scaled, y_train, 2, LinearRegression())
    '''
    
    rfe = RFE(estimator=model, n_features_to_select= n)
    rfe.fit_transform(X_df, y_df)
    mask = rfe.get_support()
    rfe_feature = X_df.iloc[:,mask].columns.tolist()
    # check if rank=True
    if rank == True:
        # get the ranks
        var_ranks = rfe.ranking_
        # get the variable names
        var_names = X_df.columns.tolist()
        # combine ranks and names into a df for clean viewing
        rfe_ranks_df = pd.DataFrame({'Var': var_names, 'Rank': var_ranks})
        # sort the df by rank
        rfe_ranks_df = rfe_ranks_df.sort_values('Rank')
        # print DataFrame of rankings
    return rfe_feature, rfe_ranks_df


def create_baseline(y_train, y_validate, target):
    '''
    Take in a y_train and y_validate dataframe and target variable(logerror). 
    Calculate the mean and median of the target variable and print the result side by side comparsion
    Select the one has lowest RMSE
    And then append into a dataframe called metric_df
    '''
    #tax_value mean
    logerror_pred_mean = y_train[target].mean()
    y_train['logerror_pred_mean'] = logerror_pred_mean
    y_validate['logerror_pred_mean'] = logerror_pred_mean

    #tax_value_median
    logerror_pred_median = y_train[target].median()
    y_train['logerror_pred_median'] = logerror_pred_median
    y_validate['logerror_pred_median'] = logerror_pred_median


    #RMSE of tax_value_pred_mean
    rmse_mean_train = mean_squared_error(y_train[target], y_train.logerror_pred_mean)**(1/2)
    rmse_mean_validate = mean_squared_error(y_validate[target], y_validate.logerror_pred_mean)**(1/2)


    #RMSE of tax_value_pred_median
    rmse_median_train = mean_squared_error(y_train[target], y_train.logerror_pred_median)**(1/2)
    rmse_median_validate = mean_squared_error(y_validate[target], y_validate.logerror_pred_median)**(1/2)

    #R2 score for the baseline
    r2_baseline = r2_score(y_validate[target], y_validate.logerror_pred_mean)

    #Append rmse validate and r2 score into a dataframe
    metric_df = pd.DataFrame(data=[{
    'model': 'Mean Baseline',
    'rmse_train': rmse_mean_train,
    'rmse_validate': rmse_mean_validate,
    'r^2_value': r2_baseline}])

    return  metric_df, rmse_mean_train, rmse_mean_validate, rmse_median_train, rmse_median_validate, r2_baseline

def create_model(model, X_train_scaled, X_validate_scaled, y_train, y_validate, target):
    '''
    take in features scaled df  and target df (tax_value), and
    type of model (LinearRegression, LassoLars, TweedieRegressor, PolynomialFeatures) and hyper parameter and
    calculate the mean square error and r2 score
    and return mean square error and r2 score
    '''
    #fit the model to our training data, specify column since it is a dataframe
    model.fit(X_train_scaled,y_train[target])

    #predict train
    y_train['logerror_pred_lm'] = model.predict(X_train_scaled)
    y_train['logerror_pred_lars'] = model.predict(X_train_scaled)
    y_train['logerror_pred_glm'] = model.predict(X_train_scaled)
    y_train['logerror_pred_lm3'] = model.predict(X_train_scaled)

    #evaluate the RMSE for train
    rmse_train = mean_squared_error(y_train[target], y_train.logerror_pred_lm)**(1/2)
    rmse_train = mean_squared_error(y_train[target], y_train.logerror_pred_lars)**(1/2)
    rmse_train = mean_squared_error(y_train[target], y_train.logerror_pred_glm)**(1/2)
    rmse_train = mean_squared_error(y_train[target], y_train.logerror_pred_lm3)**(1/2)

    #predict validate
    y_validate['logerror_pred_lm'] = model.predict(X_validate_scaled)
    y_validate['logerror_pred_lars'] = model.predict(X_validate_scaled)
    y_validate['logerror_pred_glm'] = model.predict(X_validate_scaled)
    y_validate['logerror_pred_lm3'] = model.predict(X_validate_scaled)
   
    #evaluate the RMSE for validate
    rmse_validate = mean_squared_error(y_validate[target], y_validate.logerror_pred_lm)**(1/2)
    rmse_validate = mean_squared_error(y_validate[target], y_validate.logerror_pred_lars)**(1/2)
    rmse_validate = mean_squared_error(y_validate[target], y_validate.logerror_pred_glm)**(1/2)
    rmse_validate = mean_squared_error(y_validate[target], y_validate.logerror_pred_lm3)**(1/2)

    #r2 score for model
    r2_model_score = r2_score(y_validate[target], y_validate.logerror_pred_lm)
    r2_model_score = r2_score(y_validate[target], y_validate.logerror_pred_lars)
    r2_model_score = r2_score(y_validate[target], y_validate.logerror_pred_glm)
    r2_model_score = r2_score(y_validate[target], y_validate.logerror_pred_lm3)

    return rmse_train, rmse_validate, r2_model_score

def best_model(X_test_scaled, y_test,target, model):
    '''
    This function is similar to create_model function but only using the test dataset
    '''
    #let's do linear regression again with our new degree
    lm3 = LinearRegression(normalize=True)
    #fit the model using scaled X_train, once again specify y_train column
    lm3.fit(X_test_scaled, y_test[target])
    # predicting on our test model
    y_test[model] = lm3.predict(X_test_scaled)
    # evaluate: rmse
    rmse_test = mean_squared_error(y_test[target], y_test[model])**(1/2)
    #R2 score
    r2_model_score = r2_score(y_test[target], y_test[model])
    return rmse_test, r2_model_score

def report(metric_df):
    '''
    This funtion will take in a dataframe and convert to a html and passed to the display function, 
    it will result in dataframe being displayed in the frontend (only works in the notebook).
    '''
    
    from IPython.display import display, HTML
    rmse_base = metric_df.iloc[0,2]
    print(f'These are the models that perform better than our baseline rmse: {rmse_base}')
    dfs =metric_df[['model', 'rmse_validate']][metric_df['rmse_validate'] < rmse_base]
    display(HTML(dfs.to_html()))
    
    
    min_val = metric_df['rmse_validate'].idxmin()
    metric_df.iloc[min_val][0]
    rsme_bet = round(metric_df['rmse_validate'].iloc[min_val], 6)
    print('-----------------------------------------------------------------------------------------------')
    print(f'   ********** The model with the less  rmse_validate  is {metric_df.iloc[min_val][0] }  rmse:{rsme_bet} **********             ')
    print('-----------------------------------------------------------------------------------------------')
    print(' ')
    min_val = metric_df['r^2_value'].idxmax()
    metric_df.iloc[min_val][0]
    print(f'The model with r^2 validate closer to 1 is ', metric_df.iloc[min_val][0])
    
    display(HTML(metric_df.to_html()))
    return