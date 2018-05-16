##Project topic
Inspired by the Kaggle Competition:

https://www.kaggle.com/c/home-depot-product-search-relevance

Tasks: Given product title and product description, predict the relative relevance score for the search query and the product
Example: 2,100001,"Simpson Strong-Tie 12-Gauge Angle","angle bracket",3
3,100001,"Simpson Strong-Tie 12-Gauge Angle","l bracket",2.5
Relevance Score is slightly right skewed distributed


Either regression problem or classification problem.
Reason: NLP problem, less computational burden, challenging
Time: 3 weeks, 2 weeks for feature engineering
Results: 25% of total 2125 teams, RMSE 0.47214, Top RMSE 0.43192

Preprocessing: Dropping Symbols, word replacement, stemming.
Feature Extraction: Counting features, distance features, TF-IDF features, query encoding.
Ensemble: XGBoost linear booster, XGBoost tree booster, Gradientboosting regressor, ExtraTressRegressor, RandomForest Regressor. SVR, Ridge, and Lasso.
