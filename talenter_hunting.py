#Bussines Problem
#Predicting which class (average, highlighted) player is according to the points given to the characteristics of the players. (Scenario)

# Dataset Story
#The data set consists of information from Scoutium, which includes the features and scores of the football players evaluated by the scouts according to the characteristics of the footballers observed in the matches.

import pandas as pd
import numpy as np
from termcolor import colored
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,roc_auc_score
pd.set_option('display.width', 600)

attributes = pd.read_csv("/kaggle/input/attribute/scoutium_attributes.csv",sep=";")
potential_labels = pd.read_csv("/kaggle/input/potential/scoutium_potential_labels.csv",sep=";")

result = pd.merge(attributes,potential_labels, how="right", on=["task_response_id", "match_id","evaluator_id","player_id"])

df = result.copy()
df

#Examine the data
def missing_values_analysis(df):
    na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (df[na_columns].isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["Total Missing Values", "Ratio"])
    missing_df = pd.DataFrame(missing_df).sort_values(by="Ratio", ascending=False)
    return missing_df


def check_df(df, head=5, tail=5):
    print("*** Shape ***")
    print('Observations:', df.shape[0])
    print('Features:', df.shape[1])
    print("*** Head ***")
    print(df.head())
    print("*** Tail ***")
    print(df.tail())
    print("*** Missing Values ***")
    print(missing_values_analysis(df))
    print("*** Duplicate Values ***")
    print(df.duplicated().sum())
    print("*** Quantiles ***")
    print(df.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

#Drop Position 1 (Goolkeeper)
df.drop(df[df["potential_label"] == "below_average"].index, inplace=True)

df["position_id"].value_counts()

# Drop Below Average

df.drop(df[df["potential_label"] == "below_average"].index, inplace=True)
df["potential_label"].value_counts()

# Regulate datafra for machine learning
df.head()

last_df = pd.pivot_table(data=df,
                       index=["player_id","position_id","potential_label"],
                       columns="attribute_id",
                       values="attribute_value")

last_df.head()

last_df.info()

# index is converting into variables
last_df.reset_index(inplace=True)
#all variables convert into string
last_df = last_df.astype(str)
last_df


#Finding numeric and categorical variables.

def grab_col_names(df, cat_th=10, car_th=20):

    cat_cols = [col for col in df.columns if
                str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and df[col].dtypes in ["int",
                                                                                              "float"]]
    cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and str(df[col].dtypes) in ["category",
                                                                                                   "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observation: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car, num_but_cat

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(last_df)

print(f"cat_cols: {cat_cols}")
print(f"num_cols: {num_cols}")
print(f"cat_but_car: {cat_but_car}")
print(f"num_but_cat: {num_but_cat}")

# Encoding

def label_encoder(df,column):
    labelencoder = LabelEncoder()
    df[column] = labelencoder.fit_transform(df[column])
    return df

binary_cols = [col for col in df.columns if df[col].dtype not in [int,float] and df[col].nunique()==2]
print("binary column: ",binary_cols)

for col in binary_cols:
    last_df=label_encoder(last_df,col)

last_df.head()

#Variable name converts into str

last_df.columns = last_df.columns.astype(str)
last_df.columns

#Standartization

num_cols = last_df.columns[3:]
num_cols

standard_scaler=StandardScaler()
last_df[num_cols] = standard_scaler.fit_transform(last_df[num_cols])

#Machine Learning Model

y= last_df["potential_label"]
X=last_df.drop(["potential_label","player_id"],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state = 42,
                                                    stratify = y,
                                                    test_size = 0.2,
                                                    shuffle = True)

print("X_train:",X_train.shape)
print("X_test:",X_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)


def classification_models(model):
    y_pred = model.fit(X_train, y_train).predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    roc_score = roc_auc_score(y_pred, model.predict_proba(X_test)[:, 1])
    f1 = f1_score(y_pred, y_test)
    precision = precision_score(y_pred, y_test)
    recall = recall_score(y_pred, y_test)

    results = pd.DataFrame({"Values": [accuracy, roc_score, f1, precision, recall],
                            "Metrics": ["Accuracy", "ROC-AUC", "F1", "Precision", "Recall"]})

    # Visualize Results:
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Bar(x=[round(i, 5) for i in results["Values"]],
                         y=results["Metrics"],
                         text=[round(i, 5) for i in results["Values"]], orientation="h", textposition="inside",
                         name="Values",
                         marker=dict(color=["indianred", "firebrick", "palegreen", "skyblue", "plum"],
                                     line_color="beige", line_width=1.5)), row=1, col=1)
    fig.update_layout(title={'text': model.__class__.__name__,
                             'y': 0.9,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'},
                      template='plotly_white')
    fig.update_xaxes(range=[0, 1], row=1, col=1)

    iplot(fig)


my_models = [
    LogisticRegression(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    GaussianNB()
]

for model in my_models:
    classification_models(model)

