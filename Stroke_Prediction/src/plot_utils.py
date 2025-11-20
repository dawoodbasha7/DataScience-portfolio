import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_categorical_heatmaps(df, categorical_cols, target_col,
                              cols=3, cmap='Blues', vmax=None,
                              figsize_per_subplot=(6,5), normalize_axis='index'):
    """
    Plot heatmaps of proportions for categorical columns vs a target column.

    Parameters:
    - df: pandas DataFrame
    - categorical_cols: list of categorical column names (excluding target)
    - target_col: target column name (string)
    - cols: number of subplots per row (default 3)
    - cmap: matplotlib colormap (default 'Blues')
    - vmax: max value for heatmap color scale (default None)
    - figsize_per_subplot: tuple (width, height) per subplot (default (6,5))
    - normalize_axis: 'index' (default) or 'columns' for pd.crosstab normalization

    Returns:
    - fig, axes: matplotlib figure and axes array
    """

    total = len(categorical_cols)
    rows = (total + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_subplot[0] * cols,
                                                 figsize_per_subplot[1] * rows))
    axes = axes.flatten()

    for i, col in enumerate(categorical_cols):
        ct = pd.crosstab(df[col], df[target_col], normalize=normalize_axis).T
        sns.heatmap(ct, annot=True, cmap=cmap, vmax=vmax, fmt='.2f',
                    ax=axes[i], cbar=i == total - 1)
        axes[i].set_title(f'{col} by {target_col}')
        axes[i].set_ylabel(col)
        axes[i].set_xlabel(target_col)

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

    return fig, axes
    
    
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_categorical_vs_numerical(df, categorical_cols, target_col,
                                  cols=3, figsize_per_subplot=(6,5)):
    """
    Plot boxplots of numerical target vs categorical features.

    Parameters:
    - df: pandas DataFrame
    - categorical_cols: list of categorical column names
    - target_col: numerical target column name (string)
    - cols: number of subplots per row (default 3)
    - figsize_per_subplot: tuple (width, height) per subplot (default (6,5))

    Returns:
    - fig, axes: matplotlib figure and axes array
    """

    total = len(categorical_cols)
    rows = (total + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_subplot[0]*cols,
                                                 figsize_per_subplot[1]*rows))
    axes = axes.flatten()

    for i, col in enumerate(categorical_cols):
        sns.boxplot(x=col, y=target_col, data=df, ax=axes[i])
        axes[i].set_title(f'{target_col} by {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(target_col)
        axes[i].tick_params(axis='x', rotation=45)

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

    return fig, axes
    
    
    
import matplotlib.pyplot as plt
import seaborn as sns

def plot_numerical_distributions(df, numerical_cols, cols=3, figsize_per_subplot=(6,4)):
    total = len(numerical_cols)
    rows = (total + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_subplot[0]*cols, figsize_per_subplot[1]*rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
    
    # Hide unused axes
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return fig, axes


import matplotlib.pyplot as plt
import seaborn as sns

def plot_num_target_vs_cat_features(df, cat_cols, target_col, cols=3):
    total = len(cat_cols)
    rows = (total + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        sns.boxplot(x=col, y=target_col, data=df, ax=axes[i])
        axes[i].set_title(f'{target_col} distribution by {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(target_col)
        axes[i].tick_params(axis='x', rotation=45)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


