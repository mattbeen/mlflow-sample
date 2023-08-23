import mlflow
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

with mlflow.start_run() as run:
    mlflow.autolog()
    db = load_diabetes()

    x_train, x_test, y_train, y_test = train_test_split(db.data,db.target,test_size=0.2,random_state=13)

    rf = RandomForestRegressor(n_estimators=100,max_depth=6,max_features=3)
    rf.fit(x_train,y_train)

    prediction = rf.predict(x_test)
    signature = infer_signature(x_test,prediction)
    mlflow.sklearn.log_model(rf,"model",signature=signature,
                             registered_model_name='diabete-random-forest-reg-model')