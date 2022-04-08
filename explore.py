import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def explore_univariate(df, variable):
    '''
    functions to create clusters and scatter-plot:
    explore_univariate will take in a dataframe, and one feature or variable. It graphs a box plot and a distribution 
    of the single variable.
    '''
    #set figure size, font for axis ticks, and turns off gridlines.
    plt.figure(figsize=(30,10))
    sns.set(font_scale = 2)
    sns.set_style("whitegrid", {'axes.grid' : False})
    
    # boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(x=variable, data=df)
    plt.xlabel('')
    plt.title('Box Plot', fontsize=30)
    
    # distribution
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x=variable, element='step', kde=True, color='blue')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Distribution', fontsize=30)
    
    #title
    plt.suptitle(f'{variable}', fontsize = 45)
    plt.tight_layout()
    plt.show()

def explore_bivariate(df, feature, target):
    '''
    function to create boxplot and barplot
    explore bivariate will take in a dataframe, one feature and one target.
    '''
    #set up figure size, font size, and turn off grid.
    plt.figure(figsize=(30,10))
    sns.set(font_scale = 2)

    #boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(df[target], df[target])
    plt.ylim(-.2, .2)

    #barplot
    plt.subplot(1, 2, 2)
    sns.barplot(data=df, x=feature,y=target)

    #title
    plt.suptitle('Log Error Across Counties', fontsize = 45)
    plt.tight_layout()
    plt.show()

def create_join_plot(df, feature, target):
    '''
    this function will take in a dataframe, one feature and a target.
    create a join plot (bar + scatter)
    '''
    #Property age and log error
    print('Age and LogError')
    plt.figure(figsize=(7,9))
    sns.jointplot(x=feature, y=target, data=df)
    plt.xlabel('Age')
    plt.ylabel('Log Error')
    plt.show()

def make_scatter_plot(df, feature, target):
    #Dollar/Sqft and Log Error
    plt.figure(figsize=(7,9))
    sns.scatterplot(x=feature, y=target, data=df)
    plt.xlabel('Dollar Per Sqft')
    plt.ylabel('Log Error')
    plt.title('Dollar per Sq Ft and Log Error')
    plt.show()

def create_cluster(train,validate, test, X, k,name):
    
    """ Takes in df, X (dataframe with variables you want to cluster on) and k
    # It scales the X, calcuates the clusters and return train (with clusters), the Scaled dataframe,
    #the scaler and kmeans object and unscaled centroids as a dataframe"""
    
    scaler = StandardScaler(copy=True).fit(train[X])
    X_scaled = pd.DataFrame(scaler.transform(train[X]), columns=train[X].columns.values).set_index([train[X].index.values])
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(X_scaled)
    kmeans.predict(X_scaled)

    train[name] = kmeans.predict(X_scaled)
    train[name] = 'cluster_' + train[name].astype(str)
    
    v_scaled = pd.DataFrame(scaler.transform(validate[X]), columns=validate[X].columns.values).set_index([validate[X].index.values])
    validate[name] = kmeans.predict(v_scaled)
    validate[name] = 'cluster_' + validate[name].astype(str)
    
    t_scaled = pd.DataFrame(scaler.transform(test[X]), columns=test[X].columns.values).set_index([test[X].index.values])
    test[name] = kmeans.predict(t_scaled)
    test[name] = 'cluster_' + test[name].astype(str)

    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    return train, X_scaled, scaler, kmeans, centroids

def create_scatter_plot(x,y,df,kmeans, X_scaled, scaler,name):
    
    """ Takes in x and y (variable names as strings, along with returned objects from previous
    function create_cluster and creates a plot"""
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x = x, y = y, data = df, hue = name)
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    centroids.plot.scatter(y=y, x= x, ax=plt.gca(), s=1000, c='black', marker='x')

def scatter_plot_ks (X_df, X_scaled_df, x, y, start, finish):
    fig, axs = plt.subplots(2, 2, figsize=(13, 13), sharex=True, sharey=True)

    for ax, k in zip(axs.ravel(), range(start , finish)):
        clusters = KMeans(k).fit(X_scaled_df).predict(X_scaled_df)
        ax.scatter(X_df[x], X_df[y], c=clusters)
        ax.set(title='k = {}'.format(k), xlabel=x, ylabel=y)

def elbow_chart ( X, end_range):
    '''
    this function will take in features as X and end point to graph to 
    '''
    scaler = StandardScaler().fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns= X.columns).set_index([X.index.values])

    # let is explore what values of k might be appropriate
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(9, 6))
        pd.Series({k: KMeans(k).fit(X_scaled).inertia_ for k in range(2, end_range)}).plot(marker='x')
        plt.xticks(range(2, end_range))
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.title('Change in inertia as k increases')

def pairplot_zillow(df, target, x, y):
    #let's add log error bins
    df['logerror_bins'] = pd.cut(df.target, [-5, -.2, -.05, .05, .2, 4])#the pairplot
    sns.pairplot(data = df, hue = 'logerror_bins', x =[], y=[])
 
def get_zillow_heatmap(train,target_variable):
    '''
    returns a heatmap and correlations of how each feature relates to tax_value
    '''
    sns.set()
    plt.figure(figsize=(9, 17))
    heatmap = sns.heatmap(train.corr()[[target_variable]].sort_values(by=target_variable, ascending=False), vmin=-2, vmax=2, annot=True, cmap='coolwarm')
    heatmap.set_title('Feautures Correlating with Value')
    
    return heatmap

def plot_variable_pairs(train, cols, hue=None):
    '''
    This function takes in a df, a list of cols to plot, and default hue=None 
    and displays a pairplot with a red regression line.
    '''
    plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.7}}
    sns.pairplot(train[cols], hue=hue, kind="reg",plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})
    plt.show()

def plot_pairplot(train, cols, hue=None):
    '''
    Take in train df, list of columns to plot, and hue=None
    and display scatter plots and hists.
    '''
    sns.pairplot(train[cols], corner=True, hue=hue)
    plt.show()
    
def correlation_exploration(train, feature_x, feature_y, t, p):
    '''
    This function takes in a df, a string for an x-axis variable in the df, 
    and a string for a y-axis variable in the df and displays a scatter plot, the r-
    squared value, and the p-value. It explores the correlation between input the x 
    and y variables.
    '''
    train.plot.scatter(feature_x, feature_y)
    plt.title(f"{feature_x}'s Relationship with {feature_y}")
    print(f'The p-value is: {p}. There is {round(p,3)}% chance that we see these results by chance.')
    print(f't = {round(t, 2)}')
    plt.show()

def distribution_plot(df,feature_lst):
    '''
    This function will take in a dataframe(train) and features to create a barplot for us to check distributions
    of our selected features/univeriate exploration
    '''
    plt.figure(figsize=(13,25))
    plt.subplot(5,1,1, xlabel = 'Property Square Footage', ylabel= 'No. Properties', title='Distribution of Sq Ft')
    plt.hist(data=df, x=feature_lst[0], bins = 30,ec='black')

    plt.subplot(5,1,2, xlabel = 'No. of Bathrooms on Property', ylabel= 'No. Properties', title='Distribution of No. of Bathrooms')
    plt.hist(data=df, x=feature_lst[1], ec='black')

    plt.subplot(5,1,3, xlabel = 'No. of Bedrooms on Property', ylabel= 'No. Properties', title='Distribution of No. of Bedrooms')
    plt.hist(data=df, x=feature_lst[2],ec='black')

    plt.subplot(5,1,4, xlabel = 'County', ylabel= 'No. Properties', title='Distribution of FIPS')
    plt.hist(data=df, x=feature_lst[3],ec='black')

    plt.subplot(5,1,5, xlabel = 'Age of Property', ylabel= 'No. Properties', title='Distribution of House Age')
    plt.hist(data=df, x=feature_lst[4],ec='black')

    plt.subplots_adjust(hspace=1)
    plt.show()

def plot_residuals(y_validate):
    '''
    take in a df, display a scatter plot. 
    The closer a dot is to the line means that the closer the prediction was to the actual value
    '''
    # plot the residuals for the best performing model
    plt.figure(figsize=(16,8))
    plt.axhline(label="No Error")
    plt.scatter(y_validate.tax_value, y_validate.tax_value_pred_lm - y_validate.tax_value, 
            alpha=.5, color="blue", s=100, label="Model: Linear Regression")
    plt.scatter(y_validate.tax_value, y_validate.tax_value_pred_lars - y_validate.tax_value, 
            alpha=.5, color="yellow", s=100, label="Model: LassoLars Regression")
    plt.scatter(y_validate.tax_value, y_validate.tax_value_pred_glm - y_validate.tax_value, 
            alpha=.5, color="green", s=100, label="Model: TweedieRegressor")
    plt.scatter(y_validate.tax_value, y_validate.tax_value_pred_lm3 - y_validate.tax_value, 
            alpha=.5, color="red", s=100, label="Model 3rd degree Polynomial")
    plt.legend()
    plt.xlabel("Actual tax Value")
    plt.ylabel("Residual/Error: Predicted Tax value - Actual")
    plt.title("Residuals")

    plt.show()

def plot_polynomial(y_test, y_validate):
    plt.figure(figsize=(20,8))

    sns.regplot(data=y_test, x=y_validate.tax_value, y=y_validate.tax_value_pred_lm3, 
            scatter_kws={'color':'blue'}, line_kws={'color':'red'})
    plt.xlabel("Actual Tax Value of Property", fontdict={'fontsize':15})
    plt.ylabel("Predicted Tax Value of Property W/ Polynomial Regression Model", fontdict={'fontsize':15})
    plt.title("Polynomial Regression Model", fontdict={'fontsize': 20})

    plt.show()