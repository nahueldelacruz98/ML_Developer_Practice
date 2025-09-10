Useful links to browse:
- Statsquest Linear Regression: https://www.youtube.com/watch?v=7ArmBVF2dCs
- Statsquest Logistic Regression: https://www.youtube.com/watch?v=yIYKR4sgzI8

ML Cheatsheet:
1. https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-machine-learning-tips-and-tricks
2. https://www.datacamp.com/cheat-sheet/machine-learning-cheat-sheet
3. https://elitedatascience.com/machine-learning-algorithms
4. https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

# LINEAR REGRESSION

1. Quantifies the relationship between the data --> R^2
    - The larger the R^2 is, the better
2. How reliable is the relationship of the data
    - You get the F. 

Topics
    - Fitting a line with the data (to find a linear function that gets the best R^2)
        - Calculate them by
            I. Sum of squares around the mean (data - line of mean)
                - The mean of the values of column Y
            II. Sum of squares around the fit (data - line of fit)
                - The fit line that cross close to all the points of the data.
            III. R^2 = ( I - II) / I
                - If it is 100%. Data totally related.
                - If it is 0%. Data is not compatible.

    - Adding variables will never reduce R^2

# Regression problem: Bike Sharing
Link of dataset: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

Problem: predict the amount of bikes rented based on the datetime and weather conditions
Amount of instances: ~17K

R2 Score: 0.683
Models: XGBoost ; LinearRegression ; SGDRegressor

# Classification problem: CDC Diabetes Health Indicators
Link of dataset: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

Problem: predict whether a patient has diabetes, is pre-diabetic, or healthy
Amount of instances: ~253K

ROC AUC SCORE: 0.82

# TUNADROMD: detect malware 
Link of dataset: https://archive.ics.uci.edu/dataset/813/tunadromd
Target: Label