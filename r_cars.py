### import libraries ###
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import optuna
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from xgboost import plot_importance
import plotly.graph_objects as go


### import dataset ###
df = pd.read_csv("D:/PYTHON/vscode_car/dataset.csv")

### visualize the data ###

### top 20 carmodels by amount ###
sns.barplot(y="index",x="carmodel" ,data=df['carmodel'].str.strip().value_counts().reset_index().head(20))
plt.subplots_adjust(left=0.221 , right=0.938)
plt.show()
### top carmodels by brand name
sns.barplot(y='index', x='carmodel', data=df['carmodel'].str.strip().str.split().str[0].value_counts().reset_index().head(20));
plt.show()

#get name of brand
df['carmodel'] = df['carmodel'].str.strip().str.split().str[0]

### Histogram of Price
df['price'] = df["price"].str.strip().str.extract(r'([0-9 ]+)')[0].str.replace(' ','').astype('int')

df['price'].hist(bins=50)
plt.xlabel("price in euro")
plt.ylabel("frequnecy")
plt.show()

### Year of registration

df = df[~df['année'].isna()]
df.loc[:,("année")] = df["année"].astype("int")
df = df.rename({'année':"year"} , axis = 1)

# bar plot of registration
sns.barplot(x= 'index', y="year", data=df['year'].value_counts().reset_index().head(20))
plt.xlabel("year")
plt.ylabel("number of cars")
plt.show()

### does car require a technical check 
df["technical_check"] = df["contrôletechnique"].str.strip()=="requis"
sns.countplot(df["technical_check"])
plt.show()

### kilometers
df['kilometrage'] = df['kilométragecompteur'].str.extract(r'([0-9\s]+)')[0].str.replace(' ', '').astype('int')
df['kilometrage'].hist(bins=50)
plt.xlabel("frequency")
plt.ylabel("total kilometer")
plt.show()

### most fual used by cars
sns.barplot(x="énergie" , y="index" , data= df['énergie'].value_counts().reset_index().head(20))
plt.xlabel("year")
plt.ylabel("numbar of cars")
plt.subplots_adjust(left=0.293)
plt.show()

#after dropping unbalanceed data
df = df[df['énergie'].str.strip().isin(['Essence', 'Diesel', 'Hybride essence électrique', 'Electrique'])]
df.loc[:,("car_type")] = df['énergie']

sns.barplot(x="énergie" , y="index" , data= df['énergie'].value_counts().reset_index().head(20))
plt.xlabel("year")
plt.ylabel("numbar of cars")
plt.show()

### manual/automatic transmission
df.loc[:,('transmission')] = df['boîtedevitesse'].str.strip()
df["transmission"].unique()
df['transmission'].replace('', np.nan, inplace=True)
df['transmission'].unique()
df.dropna(subset=['transmission'] , inplace=True)
df['transmission'].unique()

sns.countplot(df['transmission'])
plt.show()

### car color
df['color'] = df['couleurextérieure'].str.strip().str.lower()

sns.barplot(x="color" , y="index" , data=df["color"].value_counts().reset_index().head(20))
plt.show()

### number of doors & seats
df=df[~df["nombredeportes"].isna()]
df['door_number'] = df["nombredeportes"].astype("int")
df['seat_number'] = df["nombredeplaces"].fillna(4).astype("int")

sns.countplot(df["door_number"])
plt.show()
sns.countplot(df["seat_number"])
plt.show()

### Guaranty of cars
df["garantie"] = df["garantie"].fillna("0")
df['guaranty'] = df['garantie'].str.extract(r"([0-9]+)")[0].astype("int")
sns.countplot(df["guaranty"])
plt.show()

df = df[df["guaranty"].isin([0,3,6,8,12,24,36])]
df["guaranty"].unique()
sns.countplot(df["guaranty"])
plt.show()

### is car first hand
df["is_first_hand"] = df["premièremain(déclaratif)"].str.strip() == "oui"
sns.countplot(df["is_first_hand"])
plt.show()

# car metrics
df = df[~df["puissancefiscale"].isna()]
df = df[~df["puissancedin"].isna()]
df["cv"] = df["puissancefiscale"].str.strip().str.extract(r'([0-9]+)')[0].astype("int")
df["car_power"] = df["puissancedin"].str.strip().str.extract(r'([0-9]+)')[0].astype("int")
df["pollution"] = df["crit'air"].fillna(2).astype("str")

sns.countplot(df["cv"])
plt.show()

sns.countplot(df["car_power"])
plt.show()

sns.countplot(df["pollution"])
plt.show()

#create new data set 
columns = ['carmodel', 'price', 'year', 'technical_check','kilometrage', 'car_type', 'transmission', 'color','door_number', 'seat_number', 'guaranty', 'is_first_hand', 'cv', 'car_power']
df = df[columns]

#correlation matrix
corr = df.corr()
sns.heatmap(corr,cmap="YlGnBu" , annot=True)
plt.show()

### model building 
y = df["price"]
x = df.drop(["price"], axis = 1)
numeric_columns = list(x.select_dtypes(include=np.number).columns)
dummie_columns = ["car_type", 'carmodel', 'transmission', 'color']
x = pd.get_dummies(x , columns = dummie_columns , prefix = dummie_columns)

x_train , x_test , y_train , y_test = train_test_split(x , y , random_state=0)
scaler = StandardScaler()

x_train.loc[:, numeric_columns] = pd.DataFrame(scaler.fit_transform(x_train[numeric_columns]),columns=numeric_columns, index=x_train.index)
x_test.loc[:,numeric_columns] = pd.DataFrame(scaler.fit_transform(x_test[numeric_columns]),columns=numeric_columns,index=x_test.index)

def objective(trial,x,y):
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = 101)
    param = {
                'eta':trial.suggest_uniform('eta', 0.005, 0.01),
                "n_estimators" : trial.suggest_int('n_estimators', 400, 2000),
                'max_depth':trial.suggest_int('max_depth', 7, 9),
                'reg_alpha':trial.suggest_uniform('reg_alpha', 0.01, 1),
                'reg_lambda':trial.suggest_uniform('reg_lambda', 0.01, 1),
                'min_child_weight':trial.suggest_uniform('min_child_weight', 1, 10),
                'gamma':trial.suggest_uniform('gamma', 0.01, 1),
                'learning_rate':trial.suggest_uniform('learning_rate', 0.001, 0.01),
                'colsample_bytree':trial.suggest_discrete_uniform('colsample_bytree',0.1,1,0.01),
                'nthread' : -1
            }
    model = XGBRegressor()
    model.set_params(**param)

    model.fit(train_X,train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=0)
    y_preds = model.predict(test_X)
    return np.sqrt(mean_squared_error(test_y, y_preds))

study = optuna.create_study(direction='minimize')
study.optimize(lambda trial : objective(trial, x_train, y_train), n_trials = 15)

##visualize model 
fig =optuna.visualization.plot_optimization_history(study)
fig.show()

fig1 = optuna.visualization.plot_parallel_coordinate(study)
fig1.show()

fig2 = optuna.visualization.plot_contour(study)
fig2.show()

fig3 = optuna.visualization.plot_param_importances(study)
fig3.show()

fig4 = optuna.visualization.plot_edf(study)
fig4.show()


### Model Performace

params = study.best_params
xgb_model = XGBRegressor(n_jobs = -1)
xgb_model.set_params(**params)
xgb_model.fit(x_train , y_train)
y_pred = xgb_model.predict(x_test)

print(f"RSME = {np.sqrt(mean_squared_error(y_pred , y_test))}")



f ,ax =  plt.subplots()
plot_importance(xgb_model , max_num_features=20 ,ax = ax)
plt.show()
