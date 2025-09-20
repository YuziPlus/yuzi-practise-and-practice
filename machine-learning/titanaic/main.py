import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
#打开文件
train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')
#超级数据清洗
y = train['Survived']
drop_cols = ['Survived','PassengerId','Cabin','Name','Ticket']
drop_cols2 = ['PassengerId','Cabin','Name','Ticket']
X = train.drop(columns=drop_cols)
X_submit = test.drop(columns=drop_cols2)
test_ids = test['PassengerId']
#超级识别数据类型
nums_col = X.select_dtypes(exclude='object').columns
cat_col = X.select_dtypes(include='object').columns
#超级填充数据空缺
preprocess = ColumnTransformer(
    transformers=[
        ('nums',SimpleImputer(strategy='median'),nums_col),
        ('cat',Pipeline([
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('onehot',OneHotEncoder(handle_unknown="ignore"))
        ]),cat_col),
    ]
)
#建模流水线
model = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", RandomForestClassifier(n_estimators=100, max_depth=7, random_state=55)) # 使用随机森林
])
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=3,stratify=y)
model.fit(X_train,y_train)
val_pred = model.predict(X_val)
print(f"VAL ACCURACY:{accuracy_score(y_val,val_pred):.4f}")
#彻底的训练部分
model.fit(X,y)
test_pred = model.predict(X_submit)
submisson = pd.DataFrame({"PassengerId":test_ids,"Survived":test_pred})
submisson.to_csv('submission.csv',index=False)
print("Saved suceesful")









