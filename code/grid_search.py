from sklearn import svm,linear_model,grid_search

DF_TRAIN_LOW = pd.read_csv('features/DF_TRAIN_LOW_LIN.csv')
DF_TEST_LOW = pd.read_csv('features/DF_TEST_LOW_LIN.csv')
y_train = pd.read_csv('Y_lable.csv')

svm_lin = svm.LinearSVR()
svm_rbf = svm.SVR()

ridge = linear_model.Ridge()
lasso = linear_model.Lasso()


svm_lin_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

svm_rbf_grid = {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 'kernel':['rbf','sigmoid','poly'], 'gamma':[0.001, 0.01, 0.1]}

ridge_grid = {'alpha':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 'normalize':[True,False], 
              'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']}

lasso_grid = {'alpha':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 'normalize':[True,False],
             'selection':['random','cyclic']}

model = [svm_lin, svm_rbf, ridge, lasso]
grid = [svm_lin_grid,svm_rbf_grid,ridge_grid,lasso_grid]

best_para = []
best_score = []
for i in range(len(model)):
    gs = grid_search.GridSearchCV(estimator = model[i], param_grid = grid[i], n_jobs = -1, cv = 3, verbose = 20, scoring=RMSE)
    gs.fit(DF_TRAIN_LOW_LIN, y_train)

    best_para.append(gs.best_params_)
    best_score.append(gs.best_score_)
    print i