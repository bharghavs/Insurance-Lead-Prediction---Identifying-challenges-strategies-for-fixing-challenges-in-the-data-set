# Name: Bharghav Srikhakollu
# Date: 03-20-2023 
#######################################################################################################
# Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from tabulate import tabulate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

# Reference Citation:
# RandomForest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier    
# Encoding: https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features
# SMOTE: https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html
# GridSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
# Classification Report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report
# Dummy Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
# Confusion Matrix Display: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay
# Matplotlib: https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html

# Matplotlib - settings and plotting
plt.rcParams['font.family'], plt.rcParams['figure.dpi'], plt.rcParams['savefig.dpi']  = "serif", 300, 300

# Tabular form format
challenge_1_acc, challenge_2_acc, challenge_3_acc  = ["Accuracy"], ["Accuracy"], ["Accuracy"]
#######################################################################################################
# Read CSV file using pd:(df) and Describe - To show minimum, maximum, mean and median values for all the numeric features
#######################################################################################################
df = pd.read_csv('Health Insurance Lead Prediction Raw Data.csv')

print('**********************************************************')
data_shape = df.shape
print('The shape of the dataset is: ', data_shape)
print('**********************************************************')
print(df.dtypes)
print('**********************************************************')

df_new = df.drop(['ID', 'Response', 'Region_Code', 'Holding_Policy_Type', 'Reco_Policy_Cat'], axis = 1)
output = df_new.describe().T
output = output.drop(['count', 'std', '25%', '75%'], axis = 1)
output.rename(columns = {'50%':'median'}, inplace = True)
print(output)
print('**********************************************************')
print('Number of missing values in each feature is as below')
print(df.isnull().sum())
print('**********************************************************')

df_plot = df.drop(['ID'], axis = 1)
df_plot = df_plot.select_dtypes(exclude = ['object'])
#######################################################################################################
# Histogram Plot
#######################################################################################################
df_plot.hist(column=['Region_Code', 'Upper_Age', 'Lower_Age', 'Holding_Policy_Type', 'Reco_Policy_Cat', 'Reco_Policy_Premium'],figsize = (10,10), color='green')
plt.savefig('Histogram.jpg')
#######################################################################################################
# Heatmap with correlation (correlation values between the features)
#######################################################################################################
plt.figure(figsize = (15,10))
sns.heatmap(df_plot.corr(),annot = True, cmap = 'YlOrRd',linewidths=0.5)
plt.savefig('Heatmap.jpg')
#######################################################################################################
# Methods - To reduce repetition of code
#######################################################################################################
# Initial Setup
def initial_setup(df_initial):
    # For city code, health indicator - removing the prefix of "C" and "X" respectively, replace 14+ with 15
    df_initial['City_Code'] = (df_initial['City_Code'].str.split('C').str[1]).astype(float)
    df_initial['Health Indicator'] = df_initial['Health Indicator'].str.split('X').str[1]
    df_initial['Holding_Policy_Duration'].replace({'14+': 15}, inplace=True)
    return df_initial

# Random Forest Classifier method useful for all the strategies
def random_forest(X_train_rf, Y_train_rf, X_test_rf, Y_test_rf, method):
    clf = RandomForestClassifier(criterion = 'gini', random_state = 10, n_estimators = 250)
    clf.fit(X_train_rf, Y_train_rf)
    Y_pred = clf.predict(X_test_rf)
    acc = round(accuracy_score(Y_test_rf, Y_pred) * 100, 2)
    print('**********************************************************')
    print("Classification Report for " + method + " is as below" )
    print('**********************************************************')
    print(classification_report(Y_test_rf, Y_pred))
    print('**********************************************************')
    return Y_pred, acc

# Count/Frequency Encoding method useful for challenge 2 + challenge 3 : all strategies   
def count_freq_encoding(X_train_encode, X_test_encode):
    df_freq_map_acc = X_train_encode.Accomodation_Type.value_counts().to_dict()
    X_train_encode.Accomodation_Type = X_train_encode.Accomodation_Type.map(df_freq_map_acc)
    df_freq_map_ins = X_train_encode.Reco_Insurance_Type.value_counts().to_dict()
    X_train_encode.Reco_Insurance_Type = X_train_encode.Reco_Insurance_Type.map(df_freq_map_ins)
    df_freq_map_sp = X_train_encode.Is_Spouse.value_counts().to_dict()
    X_train_encode.Is_Spouse = X_train_encode.Is_Spouse.map(df_freq_map_sp)
    X_test_encode.Accomodation_Type = X_test_encode.Accomodation_Type.map(df_freq_map_acc)
    X_test_encode.Reco_Insurance_Type = X_test_encode.Reco_Insurance_Type.map(df_freq_map_ins)
    X_test_encode.Is_Spouse = X_test_encode.Is_Spouse.map(df_freq_map_sp)
    return X_train_encode, X_test_encode

# Deletion of rows method useful for challenge 3: all strategies
def x_y_del_rows():
    df_main = pd.read_csv('Health Insurance Lead Prediction Raw Data.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    df_main = initial_setup(df_main)
    df_main['Health Indicator'] = df_main['Health Indicator'].astype(float)
    df_main['Holding_Policy_Duration'] = df_main['Holding_Policy_Duration'].astype(float)
    df_main['Holding_Policy_Type'] = df_main['Holding_Policy_Type'].astype(float)
    XY = df_main.to_numpy()
    # split features and target label
    X, Y = XY[:, :12], XY[:, 12]
    Y = Y.astype('int')
    # missing values
    missing = np.sum(pd.isna(X), axis = 1) > 0
    # Items with "no missing values"
    X_use, Y_use = X[~missing], Y[~missing]
    return X_use, Y_use

# Challenge 1: Best Strategy (Count/Frequency encoding) + Challenge 2: Best Strategy (Deletion of Rows)
def challenge1_challenge2_best():
    # Applying Challenge 2 - Best Strategy: Deletion of rows with missing values
    X_use, Y_use = x_y_del_rows()
    # Train on a random 80% subset, test on the remaining 20% (train_test_split)
    X_train_c1c2, X_test_c1c2, Y_train_c1c2, Y_test_c1c2 = train_test_split(X_use, Y_use, test_size=0.2, random_state=0)
    X_train_c1c2_mod, X_test_c1c2_mod = pd.DataFrame(X_train_c1c2, columns=col), pd.DataFrame(X_test_c1c2, columns=col)
    # Applying Challenge 1 - Best Strategy: Count/Frequency encoding
    X_train_c1c2_mod, X_test_c1c2_mod = count_freq_encoding(X_train_c1c2_mod, X_test_c1c2_mod)
    X_train_c1c2, X_test_c1c2 = X_train_c1c2_mod.to_numpy(), X_test_c1c2_mod.to_numpy()
    return X_train_c1c2, X_test_c1c2, Y_train_c1c2, Y_test_c1c2
#######################################################################################################
# Read CSV file using Pandas: dataframe (df)
#######################################################################################################
df_main = pd.read_csv('Health Insurance Lead Prediction Raw Data.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
df_main = initial_setup(df_main)
df_main['Health Indicator'].fillna(0, inplace = True)
df_main['Health Indicator'] = df_main['Health Indicator'].astype(float)
df_main['Holding_Policy_Duration'].fillna(0, inplace = True)
df_main['Holding_Policy_Duration'] = df_main['Holding_Policy_Duration'].astype(float)
df_main['Holding_Policy_Type'].fillna(0, inplace = True)
df_main['Holding_Policy_Type'] = df_main['Holding_Policy_Type'].astype(float)

XY = df_main.to_numpy()
# split features and target label
X, Y = XY[:, :12], XY[:, 12]
Y = Y.astype('int')

# Train on a random 80% subset, test on the remaining 20% (train_test_split)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
val_classes = np.unique(Y_train)
# column names and categorical col names to be used everywhere in the code
col = ['City_Code', 'Region_Code', 'Accomodation_Type', 'Reco_Insurance_Type', 'Upper_Age',	'Lower_Age', 'Is_Spouse', 'Health Indicator', 'Holding_Policy_Duration', 'Holding_Policy_Type', 'Reco_Policy_Cat', 'Reco_Policy_Premium']
cat_col = ['Accomodation_Type', 'Reco_Insurance_Type', 'Is_Spouse']

# train and test useful for initial start of each challenge + strategy
X_train_c1s1, X_train_c1s2, X_train_c1s3 = X_train, X_train, X_train
X_test_c1s1, X_test_c1s2, X_test_c1s3 = X_test, X_test, X_test
Y_pred_label, challenge_acc = [], []
Y_pred_c1, Y_pred_c2, Y_pred_c3 = [], [], []
X_train_c1s1_mod, X_train_c1s2_mod, X_train_c1s3_mod = pd.DataFrame(X_train, columns = col), pd.DataFrame(X_train, columns = col), pd.DataFrame(X_train, columns = col)
X_test_c1s1_mod, X_test_c1s2_mod, X_test_c1s3_mod = pd.DataFrame(X_test, columns = col), pd.DataFrame(X_test, columns = col), pd.DataFrame(X_test, columns = col)

# Baseline Classifier
clf_baseline = DummyClassifier(strategy = 'stratified', random_state = 190)
clf_baseline.fit(X_train, Y_train)
Y_pred_baseline = clf_baseline.predict(X_test)
acc_bl = round(accuracy_score(Y_test, Y_pred_baseline)*100, 2)
print('Test Accuracy with baseline classifier is:', acc_bl)
challenge_1_acc.append(acc_bl)
challenge_2_acc.append(acc_bl)
challenge_3_acc.append(acc_bl)
#######################################################################################################
# Challenge 1: Handling Categorical data(representation challenge)
# Strategy 1 - Label encoding - Fit & Transform Label encoding on Train, Transform the same on Test
#######################################################################################################
for x in cat_col:
    le = LabelEncoder()
    X_train_c1s1_mod[x] = le.fit_transform(X_train_c1s1_mod[x])
    X_test_c1s1_mod[x] = le.transform(X_test_c1s1_mod[x])

X_train_c1s1, X_test_c1s1 = X_train_c1s1_mod.to_numpy(), X_test_c1s1_mod.to_numpy()
Y_pred_label, challenge_acc = random_forest(X_train_c1s1, Y_train, X_test_c1s1, Y_test, method = 'Challenge-1: Strategy-1(Label Encoding)')
Y_pred_c1.append(Y_pred_label)
challenge_1_acc.append(challenge_acc)
#######################################################################################################
# Challenge 1: Handling Categorical data(representation challenge)
# Strategy 2 - One hot encoding - Fit & Transform One hot encoding on Train, Transform the same on Test
#######################################################################################################
X_train_c1s2_new, X_test_c1s2_new = [], []
ohe = OneHotEncoder()

X_train_c1s2_new = pd.DataFrame(ohe.fit_transform(X_train_c1s2_mod[cat_col]).toarray())
X_train_c1s2_new.columns = ohe.get_feature_names_out(cat_col)
X_train_c1s2_mod = X_train_c1s2_mod.join(X_train_c1s2_new)
X_train_c1s2_mod.drop(cat_col , axis = 1, inplace = True)

X_test_c1s2_new = pd.DataFrame(ohe.transform(X_test_c1s2_mod[cat_col]).toarray())
X_test_c1s2_new.columns = ohe.get_feature_names_out(cat_col)
X_test_c1s2_mod = X_test_c1s2_mod.join(X_test_c1s2_new)
X_test_c1s2_mod.drop(cat_col , axis = 1, inplace = True)

X_train_c1s2, X_test_c1s2 = X_train_c1s2_mod.to_numpy(), X_test_c1s2_mod.to_numpy()
Y_pred_label, challenge_acc = random_forest(X_train_c1s2, Y_train, X_test_c1s2, Y_test, method = 'Challenge-1: Strategy-2(One Hot Encoding)')
Y_pred_c1.append(Y_pred_label)
challenge_1_acc.append(challenge_acc)
#######################################################################################################
# Challenge 1: Handling Categorical data(representation challenge)
# Strategy 3 - Count/Frequency encoding - Fit & Transform Count/Frequency encoding on Train, Transform the same on Test
#######################################################################################################
X_train_c1s3_mod, X_test_c1s3_mod = count_freq_encoding(X_train_c1s3_mod, X_test_c1s3_mod)

X_train_c1s3, X_test_c1s3 = X_train_c1s3_mod.to_numpy(), X_test_c1s3_mod.to_numpy()
Y_pred_label, challenge_acc = random_forest(X_train_c1s3, Y_train, X_test_c1s3, Y_test, method = 'Challenge-1: Strategy-3(Count/Frequency Encoding)')
Y_pred_c1.append(Y_pred_label)
challenge_1_acc.append(challenge_acc)
#######################################################################################################
# Challenge 2 - Missing Values:
# Strategy 1 - Delete the observations(rows) which have missing values
#######################################################################################################
# Combining Challenge 1 best strategy and Challenge 2: Strategy 1
X_train_c2s1, X_test_c2s1, Y_train_c2s1, Y_test_c2s1 = challenge1_challenge2_best()

Y_pred_label, challenge_acc = random_forest(X_train_c2s1, Y_train_c2s1, X_test_c2s1, Y_test_c2s1, method = 'Challenge-1(Count/Frequency Encoding) + Challenge-2: Strategy-1(Delete the rows)')
Y_pred_c2.append(Y_pred_label)
challenge_2_acc.append(challenge_acc)
#######################################################################################################
# Challenge 2 - Missing Values:
# Strategy 2 - Delete the features(columns) which have missing values
#######################################################################################################
df_main = pd.read_csv('Health Insurance Lead Prediction Raw Data.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 11, 12, 13])
df_main['City_Code'] = (df_main['City_Code'].str.split('C').str[1]).astype(float)
XY = df_main.to_numpy()
# split features and target label
X, Y = XY[:, :9], XY[:, 9]
Y = Y.astype('int')

# Train on a random 80% subset, test on the remaining 20% (train_test_split)
X_train_c2s2, X_test_c2s2, Y_train_c2s2, Y_test_c2s2 = train_test_split(X, Y, test_size = 0.2, random_state = 0)
X_train_c2s2_mod = pd.DataFrame(X_train_c2s2, columns = ['City_Code', 'Region_Code', 'Accomodation_Type', 'Reco_Insurance_Type', 'Upper_Age',	'Lower_Age', 'Is_Spouse', 'Reco_Policy_Cat', 'Reco_Policy_Premium'])
X_test_c2s2_mod = pd.DataFrame(X_test_c2s2, columns = ['City_Code', 'Region_Code', 'Accomodation_Type', 'Reco_Insurance_Type', 'Upper_Age',	'Lower_Age', 'Is_Spouse', 'Reco_Policy_Cat', 'Reco_Policy_Premium'])

# Applying Challenge 1 - Best Strategy: Count/Frequency encoding
X_train_c2s2_mod, X_test_c2s2_mod = count_freq_encoding(X_train_c2s2_mod, X_test_c2s2_mod)
X_train_c2s2, X_test_c2s2 = X_train_c2s2_mod.to_numpy(), X_test_c2s2_mod.to_numpy()

Y_pred_label, challenge_acc = random_forest(X_train_c2s2, Y_train_c2s2, X_test_c2s2, Y_test_c2s2, method = 'Challenge-1(Count/Frequency Encoding) + Challenge-2: Strategy-2(Delete the columns)')
Y_pred_c2.append(Y_pred_label)
challenge_2_acc.append(challenge_acc)
#######################################################################################################
# Challenge 2 - Missing Values:
# Strategy 3 - Replace the missing values with the most frequent value
#######################################################################################################
df_main = pd.read_csv('Health Insurance Lead Prediction Raw Data.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
df_main = initial_setup(df_main)
df_main['Health Indicator'] = df_main['Health Indicator'].astype(float)
df_main['Holding_Policy_Duration'] = df_main['Holding_Policy_Duration'].astype(float)
df_main['Holding_Policy_Type'] = df_main['Holding_Policy_Type'].astype(float)
XY = df_main.to_numpy()
# split features and target label
X, Y = XY[:, :12], XY[:, 12]
Y = Y.astype('int')

# Train on a random 80% subset, test on the remaining 20% (train_test_split)
X_train_c2s3, X_test_c2s3, Y_train_c2s3, Y_test_c2s3 = train_test_split(X, Y, test_size = 0.2, random_state = 0)
X_train_c2s3_mod, X_test_c2s3_mod = pd.DataFrame(X_train_c2s3, columns = col), pd.DataFrame(X_test_c2s3, columns = col)

# Applying Challenge 1 - Best Strategy: Count/Frequency encoding
X_train_c2s3_mod, X_test_c2s3_mod = count_freq_encoding(X_train_c2s3_mod, X_test_c2s3_mod)
X_train_c2s3, X_test_c2s3 = X_train_c2s3_mod.to_numpy(), X_test_c2s3_mod.to_numpy()

imp = SimpleImputer(strategy = 'most_frequent')
X_train_c2s3 = imp.fit_transform(X_train_c2s3)
X_test_c2s3 = imp.transform(X_test_c2s3)

Y_pred_label, challenge_acc = random_forest(X_train_c2s3, Y_train_c2s3, X_test_c2s3, Y_test_c2s3, method = 'Challenge-1(Count/Frequency Encoding) + Challenge-2: Strategy-3(Replace with most frequent)')
Y_pred_c2.append(Y_pred_label)
challenge_2_acc.append(challenge_acc)
#######################################################################################################
# Challenge 3 - Imbalance in the dataset:
# Strategy 1 - Under Sampling – Down Sampling the majority class
# Strategy 2 - Random Over Sampling – Over Sampling the minority class
# Strategy 3 - SMOTE(Over Sampling)
#######################################################################################################
# Output of Challenge 1 + Challenge 2: Best Strategies
X_train_c3, X_test_c3, Y_train_c3, Y_test_c3 = challenge1_challenge2_best()
X_train_c31, X_train_c32, X_train_c33 = X_train_c3, X_train_c3, X_train_c3
Y_train_c31, Y_train_c32, Y_train_c33 = Y_train_c3, Y_train_c3, Y_train_c3
X_train_arr, Y_train_arr = [X_train_c31, X_train_c32, X_train_c33], [Y_train_c31, Y_train_c32, Y_train_c33]

methods = ['Challenge-1(Count/Frequency Encoding) + Challenge-2(Delete the rows) + Challenge-3: Strategy-1(Under Sampling)', 
            'Challenge-1(Count/Frequency Encoding) + Challenge-2(Delete the rows) + Challenge-3: Strategy-2(Over Sampling)',
            'Challenge-1(Count/Frequency Encoding) + Challenge-2(Delete the rows) + Challenge-3: Strategy-3(SMOTE - Over Sampling)']
i = 0
# For loop for all the three strategies to address imbalance in the dataset challenge
for clf_imb in [RandomUnderSampler(random_state = 70),
                RandomOverSampler(random_state = 90),
                SMOTE(random_state = 110)]:
    X_imb, Y_imb = clf_imb.fit_resample(X_train_arr[i], Y_train_arr[i])
    X_imb_pd = pd.DataFrame(X_imb, columns = ['City_Code', 'Region_Code', 'Accomodation_Type', 'Reco_Insurance_Type', 'Upper_Age',	'Lower_Age', 'Is_Spouse', 'Health Indicator', 'Holding_Policy_Duration', 'Holding_Policy_Type', 'Reco_Policy_Cat', 'Reco_Policy_Premium'])
    Y_imb_pd = pd.DataFrame(Y_imb, columns= ['Response'])
    df_imb = X_imb_pd.join(Y_imb_pd)
    XY_imb_new = df_imb.to_numpy()
    X_new_c3, Y_new_c3 = XY_imb_new[:, :12], XY_imb_new[:, 12]
    Y_new_c3 = Y_new_c3.astype('int')
    
    # Start - Hyper Parameter Tuning Logic
    # 1. Uncomment the below code only if we want to test the hyper parameter tuning logic
    # 2. Also comment out the last part of this code (confusion matrix) for the time being when checking hyperparameter tuning
    # 3. Also comment out the lines 340 to 342
    """
    if isinstance(clf_imb, SMOTE):
        n_estimators = [100, 250]
        max_depth = [None, 10, 20]
        grid = dict(n_estimators = n_estimators, max_depth = max_depth)
        cv = RepeatedStratifiedKFold(n_splits = 10, random_state = 250)
        grid_search = GridSearchCV(estimator = RandomForestClassifier(), param_grid = grid, n_jobs = -1, cv = cv, scoring = 'accuracy', error_score = 0)
        grid_result = grid_search.fit(X_new_c3, Y_new_c3)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
    else:
        Y_pred_label, challenge_acc = random_forest(X_new_c3, Y_new_c3, X_test_c3, Y_test_c3, method = methods[i])
        Y_pred_c3.append(Y_pred_label)
        challenge_3_acc.append(challenge_acc)
    """
    # End - End of Hyper Parameter Tuning Logic
    
    Y_pred_label, challenge_acc = random_forest(X_new_c3, Y_new_c3, X_test_c3, Y_test_c3, method = methods[i])
    Y_pred_c3.append(Y_pred_label)
    challenge_3_acc.append(challenge_acc)
    i = i + 1
#######################################################################################################
# Tabular Form
#######################################################################################################
table_header = [['Challenge-1(Categorical Data(Representation Challenge))','Baseline','Strategy-1:Label Encoding','Strategy-2:One Hot Encoding','Strategy-3:Count/Frequency Encoding'],
                ['Count/Frequency Encoding + Challenge-2(Missing Values)','Baseline','Strategy-1:Delete Rows(Observations)','Strategy-2:Delete Columns(Features)','Strategy-3:Fill With Most Frequent Value'],
                ['Count/Frequency Encoding + Delete Rows + Challenge-3(Imbalance in the dataset)','Baseline','Strategy-1:Random Under Sampling','Strategy-2:Random Over Sampling','Strategy-3:SMOTE(Over Sampling)']]
challenge_acc_table = [challenge_1_acc, challenge_2_acc, challenge_3_acc]

for i in range(len(table_header)):
    table_data = [table_header[i], challenge_acc_table[i]]
    print(tabulate(table_data, headers = 'firstrow', tablefmt = 'grid'))
#######################################################################################################
# Confusion Matrix
#######################################################################################################
inp_true, inp_pred = [Y_test, Y_test_c2s1, Y_test_c3], [Y_pred_c1[2], Y_pred_c2[0], Y_pred_c3[2]]
titles = ['Conf Mat: Challenge-1(Strategy-3:Count/Freq Encoding)', 'Conf Mat: Count/Freq Encoding+Delete Rows', 'Conf Mat: Count/Freq Encoding+Delete Rows+SMOTE']
fig_names = ['cm_c1_s3.jpg', 'cm_c1s3_c2s1.jpg', 'cm_c1s3_c2s1_c3s3.jpg']

for i in range(len(inp_true)):
    conf_mat = confusion_matrix(inp_true[i], inp_pred[i])
    disp = ConfusionMatrixDisplay(confusion_matrix = conf_mat, display_labels = val_classes)
    disp.plot()
    plt.title(titles[i])
    plt.savefig(fig_names[i])
    plt.show()
