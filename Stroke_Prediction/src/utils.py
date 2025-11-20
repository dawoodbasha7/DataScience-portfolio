import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report
import seaborn as sns



def plot_decorator(func):
    """
    A decorator to apply a plotting function for each column of data and manage axis visibility.
    """
    def wrapper(data, axes):
        for i, column in enumerate(data.columns):
            ax = axes[i]
            func(data, column, ax)

        # Hide any unused axes if the number of columns is odd
        for j in range(len(data.columns), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

    return wrapper


def plot_continuos_data(data):
    # Set up a grid of subplots
    num_vars = len(data.columns)
    cols = 2  # Number of columns in the grid
    rows = (num_vars + cols - 1) // cols  # Number of rows in the grid

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = axes.flatten()  # Flatten the 2D array of axes to easily iterate over them

    # Loop through each variable and create a histogram with KDE
    for ax, variable in zip(axes, data.columns):
        # axs = sns.histplot(data[variable], kde=True, ax=ax, linewidth=2)
        axs = sns.histplot(data[variable], bins="doane", kde=True, ax=ax)
        axs.lines[0].set_color("crimson")
        axs.set_title(f"Distribution of {variable} ", fontsize=14)
        axs.set_xlabel(variable, fontsize=12)
        axs.set_ylabel("Frequency", fontsize=12)

    # Hide unused subplots if there are any
    for ax in axes[len(data.columns) :]:
        ax.set_visible(False)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

@plot_decorator
def categorical_columns_proportions(data, column, ax):
    df = data.groupby(column)[column].agg(value_count="value_counts").reset_index()
    df["proportions"] = (
        df["value_count"].apply(lambda a: round(a / df["value_count"].sum(), 2)) * 100
    )

    # Plotting on the given axis
    sns.barplot(x=column, y="proportions", data=df, ax=ax)
    ax.set_title(f"Proportions of {column}")
    ax.set_xlabel("Category")
    ax.set_ylabel("Proportions")


def build_and_tune_model(
    model,
    param_grid,
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    scoring="roc_auc",
    cv=5,
):
    """
    Generalized function for model building, hyperparameter tuning, and evaluation.

    Parameters:
        model: The machine learning model to tune and evaluate.
        param_grid: A dictionary of hyperparameters to test.
        model_name: Name of the model (for display purposes).
        X_train: Training features.
        y_train: Training labels.
        X_test: Testing features.
        y_test: Testing labels.
        scoring: The scoring metric for hyperparameter optimization (default: 'roc_auc').
        cv: Number of cross-validation folds (default: 5).

    Returns:
        A dictionary containing evaluation metrics and best parameters.
    """
    print(f"Building and Tuning {model_name}:\n")

    # Perform hyperparameter tuning using GridSearchCV
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Get the best model, parameters, and score
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best Parameters for {model_name}: {best_params}")
    print(f"Best Cross-Validated {scoring}: {best_score:.2f}")

    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]  # For AUC

    # Generate performance metrics
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC for {model_name}: {auc:.2f}")

    # Return metrics and the tuned model
    metrics = {
        "Model": model_name,
        "Best Parameters": best_params,
        "Cross-Validated AUC": best_score,
        "Test AUC-ROC": auc,
        "Accuracy": report["accuracy"],
        "Precision (Class 1)": report["1"]["precision"],
        "Recall (Class 1)": report["1"]["recall"],
        "F1-Score (Class 1)": report["1"]["f1-score"],
    }

    return metrics, best_model