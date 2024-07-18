# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 06:11:44 2024

@author: Nikola Anđelić
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, 
                              GradientBoostingRegressor)
from sklearn.linear_model import (LinearRegression, 
                                  Ridge,
                                  Lasso,
                                  ElasticNet, 
                                  SGDRegressor) 
from sklearn.model_selection import train_test_split

data = pd.read_csv("All_Best_GPSCHYPE_AverageLength.csv")
print(data)
y = data.pop('Mean length')


X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

# Feature importance using tree-based models 
# Decision Tree Regressor  
# step 1 - train the model 
model1 = DecisionTreeRegressor()
model1.fit(X_train,y_train)

# step 2 - Compute feature importances 
importances1 = model1.feature_importances_ 

# Create a DataFrame for better visualization 
FI_DF_1 = pd.DataFrame({
    'feature':data.columns,
    'importance': importances1
    })


# FI_DF_1 = FI_DF_1.sort_values(by='importance', ascending=False)

# # Step 6: Visualize Feature Importances
# plt.figure(figsize=(10, 6))
# plt.barh(FI_DF_1['feature'], FI_DF_1['importance'])
# plt.xlabel('Feature Importance')
# plt.ylabel('Feature')
# plt.title('Feature Importance using Decision Tree Regressor')
# plt.gca().invert_yaxis()
# plt.show()





# Step 4: Train the Model
model = RandomForestRegressor()#n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Compute Feature Importances
importances = model.feature_importances_

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({
    'feature': data.columns,
    'importance': importances
})

# # Sort the DataFrame by importance
# feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# # Step 6: Visualize Feature Importances
# plt.figure(figsize=(10, 6))
# plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
# plt.xlabel('Feature Importance')
# plt.ylabel('Feature')
# plt.title('Feature Importance using RandomForestRegressor')
# plt.gca().invert_yaxis()
# plt.show()


# ExtraTreesRegressor
# step 1 - train the model 
model3 = ExtraTreesRegressor()
model3.fit(X_train,y_train)

# step 2 - Compute feature importances 
importances3 = model3.feature_importances_ 

# Create a DataFrame for better visualization 
FI_DF_3 = pd.DataFrame({
    'feature':data.columns,
    'importance': importances3
    })


# FI_DF_3 = FI_DF_3.sort_values(by='importance', ascending=False)

# # Step 6: Visualize Feature Importances
# plt.figure(figsize=(10, 6))
# plt.barh(FI_DF_3['feature'], FI_DF_3['importance'])
# plt.xlabel('Feature Importance')
# plt.ylabel('Feature')
# plt.title('Feature Importance using ExtraTreesRegressor')
# plt.gca().invert_yaxis()
# plt.show()


# GradientBoostingRegressor
# step 1 - train the model 
model4 = GradientBoostingRegressor()
model4.fit(X_train,y_train)

# step 2 - Compute feature importances 
importances4 = model3.feature_importances_ 

# Create a DataFrame for better visualization 
FI_DF_4 = pd.DataFrame({
    'feature':data.columns,
    'importance': importances4
    })


# FI_DF_4 = FI_DF_4.sort_values(by='importance', ascending=False)

# # Step 6: Visualize Feature Importances
# plt.figure(figsize=(10, 6))
# plt.barh(FI_DF_4['feature'], FI_DF_4['importance'])
# plt.xlabel('Feature Importance')
# plt.ylabel('Feature')
# plt.title('Feature Importance using ExtraTreesRegressor')
# plt.gca().invert_yaxis()
# plt.show()

x = np.arange(0,len(FI_DF_1['importance'])*3,3)
plt.figure(figsize=(12,8))
plt.bar(x-0.75, FI_DF_1['importance'], width=0.5,zorder=3, label="Decision\nTree\nRegressor")
plt.bar(x-0.25, feature_importance_df['importance'],width=0.5,zorder=3, label="Random\nForest\nRegressor")
plt.bar(x+0.25, FI_DF_3['importance'], width=0.5,zorder=3, label="Extra\nTrees\nRegressor")
plt.bar(x+0.75, FI_DF_4['importance'], width=0.5,zorder=3, label="Gradient\nBoosting\nRegressor")
plt.xticks(x, list(data.columns),rotation=90)
plt.title("Feature Importance Analysis on Mean Length")
plt.grid(True, zorder=0)
plt.ylim(0,0.25)
plt.ylabel("Feature Importance Value")
plt.xlabel("GPSC hyperparameters")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
          fancybox=True, shadow=True, ncol=5)
# plt.show()
plt.savefig("FI_Mean_Length.png",
            dpi = 300,
            bbox_inches = "tight")