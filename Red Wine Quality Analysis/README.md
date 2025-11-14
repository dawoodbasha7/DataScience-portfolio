# Red Wine Quality Analysis

## Overview

This project analyzes which varibales influence the quality of wine using linear regression statitsical learning approach by fitting OLS. It is to estimate the function to estimate the coefficients of the variables from the best fitted model.

## Dataset

The analysis utilizes a Kaggle dataset containing information about Red wine quality.

Dataset Source: [Kaggle - Podcast Reviews](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)

## Tools and Technologies

**Python**: Core programming language for data manipulation and visualization.
**Pandas**: Efficient data manipulation and analysis.
**NumPy**: Mathematical computations for data analysis.
**opendatasets**: To downloading datasets from online sources like Kaggle using API.
**Matplotlib** & **Seaborn**: High-quality visualizations.
**Scikit-learn**: Clustering and advanced data processing.
**Scipy**: For scientific computing.
**Seaborn**: statistical data visualization.
**Jupyter Notebook**: Interactive environment for analysis and documentation.


## Key Findings:

* There were few independent variables that are positively skewed but are not skewed very highly
* Box plot and correlation annalysis of the predictors with quality have shown initial sign of relationship with target variable. However there were evidances of existance of multicolliniarity among independent variables.
* In the investigation of relationship of predictors with response it is estimated that not all the varibles are influencing the quality of the wine. The key predictors which alters the varainace of quality are alcohol, volatile acidity, sulphates, chlorides & total sulfur dioxide.
* Where as alcohol and sulphates have positive impact on quality. On the other hand volatile acidity, chlorides and total sulfur dioxide have negative impact on the quality of the red wine.
* However the R^2 attained after estimating the coefficients of the variables is not that influencing. Only 36.4% of the variance in wine quality is explained by the predictors on training data.

## Future Scope
* The nature of response variable is ordinal, Oridinal Logistic Regression will defenite will provide better estimations. There is always a scope to work on the error which is reducible by applyin gbetter statistical learning technique.
* One shall also work on irreducible error by finding more appropriate varaibles which infleunce the vraince of quality of wine.
* Advanced variable selection approaches shall be used in variable selection.
* The analysis furtherly shall be continued for prediction purposes as well.

## File Structure

- `Red Wine Quality analysis.ipynb`: Jupyter Notebook containing the Python code for EDA and Inferential statistics.
- `Requirements.txt`: The list of dependencies used for the analysis.
- `README.md`: Overview of the repository.

## Usage

1. Clone the repository to your local machine:

```bash
git clone https://github.com/TuringCollegeSubmissions/ddudek-DA.3.4.git
```

2. Navigate to the repository directory:

```bash
cd ./src/Red Wine Quality analysis.ipynb
```

3. Run each cell in the notebook to execute the code and generate results/visualizations.

## Install the dependencies

```bash
pip install -r requirements.txt
```

## License

```Groovy
MIT License

Copyright (c) 2024 Dawood Basha Dudekula

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
