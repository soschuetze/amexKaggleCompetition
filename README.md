# American Express Kaggle Competition

*From the Kaggle Data Description*

Objective: "The objective of this competition is to predict the probability that a customer does not pay back their credit card balance amount in the future based on their monthly customer profile. The target binary variable is calculated by observing 18 months performance window after the latest credit card statement, and if the customer does not pay due amount in 120 days after their latest statement date it is considered a default event.

The dataset contains aggregated profile features for each customer at each statement date. Features are anonymized and normalized, and fall into the following general categories:

D_* = Delinquency variables

S_* = Spend variables

P_* = Payment variables

B_* = Balance variables

R_* = Risk variables

with the following features being categorical:

['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']

Your task is to predict, for each customer_ID, the probability of a future payment default (target = 1).

Note that the negative class has been subsampled for this dataset at 5%, and thus receives a 20x weighting in the scoring metric."

**The training data and testing data files are too large to upload here, so they will need to be accessed through the Kaggle competition site:** https://www.kaggle.com/competitions/amex-default-prediction/data

## My Approach
1. Clean data to remove variables with potential to cause multicollinearity - used VIF to determine these variables
2. Impute missing data for numerical variables using median imputation and mode imputation for categorical variables
3. Partition data into train and test sections - used 10,000 observations for each since dataset is so large
4. Fit Classficiation Tree, Random Forest Model, and Bagging Model
5. Used SVM for final model - achieved 85% accuracy
