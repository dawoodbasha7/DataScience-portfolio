import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt


#with warnings.catch_warnings():
#   warnings.simplefilter(action='ignore', category=FutureWarning)

def count_values_sort(df, df_columnName, df_columnCount):
    '''
    Resultant series will be converted to dataframe with the count by values in provided column.
    
    '''
    col_val=df.value_counts().to_frame().reset_index()
    col_val.columns=[df_columnName,df_columnCount]
    return col_val



def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),size=12,
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def encoding_columns(data, column_names):
    ''' This function generates the encoded output of categorical data in the dataframe.
    
    example:
        encoding_columns(data, ['column1','column 3'])
        
    The function will return new dataframe with encoded columns  by removing the original categorical columns
    '''
    df=data
    for i in column_names:
        encoded_col=pd.get_dummies(data[i])
        new_df=pd.concat([df, encoded_col], axis=1)
        df=new_df.drop([i], axis=1)
    
    return df
    
    
    
class PreCheck:
    '''
    Using methods from this class preliminary check of the dataset canbe performed.
    '''
    
    def __init__(self, data):
        self.data=data
        
    
    def duplicates(self):
        '''
        The function checks duplicated entries.
        '''
        duplicates=self.data.duplicated().sum()
        shape=self.data.shape[0]

        if duplicates==0:
            print(f"\033[1mThere are '{shape} non duplicated entries' are recorded in dataset.\
Further analysis can be performed.\033[0m ")
        else:
            if duplicates==1:
                print(f"\033[1mThere are {shape} entries totally but it has {duplicates} duplicated entry. \
Advised to remove the duplicated entry for further analysis.\033[0m ")
            else:
                print(f"\033[1mThere are {shape} entries totally but it has {duplicates} duplicated entries. \
Advised to remove the duplicated entries for further analysis.\033[0m ")
                
    def missing_values(self):
        if np.any(self.data.isnull())==True:
            for col in self.data:
                if sum(pd.isnull(self.data[col]))>0:
                    print(f"\033[1mFollowing columns have missing values\033[0m\n{col}:", sum(pd.isnull(self.data[col])))
                else:
                    pass
        else:
            print("There are no null values in any of the columns in the dataset")
            
            
            
def find_outliers(df):
    
    ''' this function checks how many outliers each columns of dataframe has.'''
    
    print("\033[1m \033[4mColumn Name\033[0m \033[0m ",
          "\033[1m \033[4mOutliers\033[0m \033[0m ")
    warnings.filterwarnings("ignore") # this line is to prevent FutureWarning that occur
    q1=df.quantile(0.25)
    q3=df.quantile(0.75)
    IQR=q3-q1
    outliers=((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))
    return outliers.sum()


def corr_heatmap(df, colour_range:float):
    '''
    This function gives nice plots with customised colour range of correlation
    For example:
        corr_heatmap(dataframe,0.5) #values of greater/lessthan than 0.5 have intense colours
    '''
    corr = df.corr()

    # Generate a mask for the upper triangle; True = do NOT show
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio

    sns.heatmap(corr,          # The data to plot
                mask=mask,     # Mask some cells
                cmap=cmap,     # What colors to plot the heatmap as
                annot=True,    # Should the values be plotted in the cells?
                vmax=colour_range,       # The maximum value of the legend. All higher vals will be same color
                vmin=-colour_range,      # The minimum value of the legend. All lower vals will be same color
                center=0,      # The center value of the legend. With divergent cmap, where white is
                square=True,   # Force cells to be square
                linewidths=.5, # Width of lines that divide cells
                cbar_kws={"shrink": .5}  # Extra kwargs for the legend; in this case, shrink by 50%
               )


