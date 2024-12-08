import pandas as pd
import numpy as np

def outliers(data):
    """ 
    Input(data): Takes in a data frame 

    Make sure the dataset contains only `numeric` variable otherwise 
    it will be removed.

    The method being applied in this function uses IQR formular
    to detect outliers    

    Output(data frame): Returns a dataframe containing the no
                        outliers in each column
    Example:

      ```
      x = pd.DataFrame(...)
      x = x.select_dtype(includes = 'number')
      outliers(x)
      ```
    """
    if str(type(data)) == "<class 'pandas.core.frame.DataFrame'>":
        data = data.select_dtypes(include ='number')
    else:
        return 'DataError:🧐Oops! data is not a dataframe check type(data)'
    to_list = list()
    final_df = pd.DataFrame(columns=data.columns,index=['Number of outliers:'])
    for n,col in enumerate(data):
        Q1=np.quantile(a=data[col],q=0.25)
        Q3=np.quantile(a=data[col],q=0.75)
        IQR =Q3-Q1
        for obs in data[col]:
            if (obs <=(Q3+IQR*1.5)) & (obs >=(Q1-IQR*1.5)):
                to_list.append(obs)
            else:
                obs= np.nan
                to_list.append(obs) 
        to_df =pd.Series(to_list,name=col)
        val = to_df.isna().sum()
        final_df[col] = val

            
    return final_df

import matplotlib.pyplot as plt
import seaborn as sns

def plot_category(data,figsize=(10,8),n_row=None,n_col=None):
    """ 
    Input(data,figsize,n):
                           `data`= should be pd.DataFrame with only objects variables atleast 2
                           `figsize` = a tupple default is (10,8)
                           `n_row(int)` = The number of rows for the subplots
                           `n_col(int)` = The number of col for the subplots

    Data should only contain categorical columns else it will be removed
    return: Plots of the categorical variables

    `Example:`👇🏿

    ```
    data = data.select_dtypes(include = 'object')
    figsize = (15,8)
    # Assuming the dataframe has 6 columns
    nrow= 2 # no of rows
    ncol= 3 # no of columns
    
    # Plot the categorical variables using Barplot
    plot_category(data,figsize=figsize,n_row=nrow,n_col=ncol)
    ```
    """
    if str(type(data))=="<class 'pandas.core.frame.DataFrame'>":
        data = data.select_dtypes(include ='object')
    else:
        return "🙏🏿 My Bad it is not a dataframe"
    
    fig,axes = plt.subplots(nrows=n_row,ncols=n_col,figsize=figsize)
    axes = axes.ravel() #Flatten the 2D array into 1D array for easier indexing
    
    for n,col in enumerate(data):
        val = data[col].value_counts().values
        ind = data[col].value_counts().index
        ax =axes[n]
        sns.barplot(y=val,x=ind,ax=ax)
        plt.title('{}'.format(col));

    # Remove any unused subplots
    for i in range(len(data.columns), len(axes)):
        fig.delaxes(axes[i])

    # Adjust the layout
    plt.tight_layout()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def OneShotEncoding(DF_data,drop =True):
    """ 
    `OneShotEncoding(DF_data)` -> DataFrame
    This an update version of the One-Hot Encoder from
    sklearn but modified to take in more than one categorical 
    variable.
    NOTE:By default the drop first column is initiated which
    deletes the first level in a categorical variable

    """
    # Takes in Data
    categorical_data=DF_data.select_dtypes(include ="object")
    df_dummy_list = []
    # list_data = []
    for col in categorical_data:
        if drop:
            Enc_ohe, Enc_label = OneHotEncoder(drop='first'), LabelEncoder() # Create an instance of the class OneHot encoding
        else:
            Enc_ohe, Enc_label = OneHotEncoder(), LabelEncoder()
        Enc_label.fit_transform(DF_data[col])
        DF_dummies2 = pd.DataFrame(Enc_ohe.fit_transform(DF_data[[col]]).todense(), columns = Enc_label.classes_[1:])
        df_dummy_list.append(DF_dummies2)
    df_dummy_list.append(DF_data)
    result =pd.concat(df_dummy_list, axis=1)

    return(result)

    return(result)

import itertools
import tensorflow as tf
from sklearn.metrics import confusion_matrix

## evaluate using a matrix function

def plot_confusion_matrix(y_preds,y_test,classes=None,text_size=18,figsize=(15,15)):
  '''
  Make sure to have imported the itertools library
  '''
  #figsize=(10,10)
  #Definine the confusion_matrix
  y_preds=tf.round(y_preds) #Convert them from probabilities to labels
  cm=confusion_matrix(y_true=y_test,y_pred=y_preds)
  cm_norm=confusion_matrix(y_pred=y_preds,y_true=y_test,normalize='true') #Normalized data
  n_classes=cm.shape[0] #Essential fo mulit-classification

  #Prettify the confusion matrix
  fig,ax=plt.subplots(figsize=figsize)

  #plot the confusion matrix
  cax=ax.matshow(cm,cmap=plt.cm.Blues) #,cmap=plt.cm.Reds, interpolation='nearest'
  fig.colorbar(cax)



  if classes:
    labels=classes
  else:
    labels=np.arange(n_classes)
  # label axis
  ax.set(title='Confusion matrix',
         xlabel='Predicted values',
         ylabel='Actual values',
         yticks=np.arange(n_classes),
         xticks=np.arange(n_classes),
         xticklabels=labels,
         yticklabels=labels
         )

  #Set axis labels to bottom
  ax.xaxis.set_label_position('bottom')
  ax.xaxis.tick_bottom()

  #Adjust label size
  ax.xaxis.label.set_size(text_size)
  ax.yaxis.label.set_size(text_size)
  ax.title.set_size(text_size)

  #Set threshold for different colors
  threshold=(cm.max()+cm.min())/2.

  #Plot the text on each cell
  for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
    plt.text(j,i,f'{cm[i,j]} ({cm_norm[i,j]*100:.1f}%)',
             horizontalalignment='center',
             color='white' if cm[i,j]>threshold else "black",size=9)
  plt.show()

