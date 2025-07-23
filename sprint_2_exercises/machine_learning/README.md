Useful links to browse:
- Statsquest Linear Regression: https://www.youtube.com/watch?v=7ArmBVF2dCs
- Statsquest Logistic Regression: https://www.youtube.com/watch?v=yIYKR4sgzI8

ML Cheatsheet:
1. https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-machine-learning-tips-and-tricks
2. https://www.datacamp.com/cheat-sheet/machine-learning-cheat-sheet
3. https://elitedatascience.com/machine-learning-algorithms
4. https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

# TO DO
1. Read and recap ML basics
    - Start with Linear Regression and Logistic Regression      # Linear Regression DONE
    - Study Classification metrics (Confusion Matrix, F1 score, ROC, AUC)
    - Study Regression metrics (basic metrics, Coefficient of determination, AIC, BIC)
2. Keep reading ML Cheatsheet to continue learning and put everything in Practice
REMEMBER: DO NOT START WITH PRACTICE YET (until finishing step 1 at least)
REMEMBER: Not everything is about theory, but helps a lot. Try not to avoid practice. PRACTICE ALL THE TIME



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
# TO DO
    - Look for your first Linear Regression Model and do it from scratch
        - Look for a webpage that has lots of datasets
        - Study the data and find
            - The hipotesis (Y) and the corresponding features (X) to train your model
            - Train a basic Logistic or Linear Regression using Scikit Learn.
                - TODO: preprocessing (datetime, binary, continuous and classes to the same scale)
                - HINT: you can use StandardScaler from sklearn
                - GOAL: Reach out to R2Score > 0.85
        - Do it for at least 2 datasets.
    - Next to study: Classification problem
    - See link: https://mlu-explain.github.io/
