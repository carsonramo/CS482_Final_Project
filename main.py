import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder
import pickle
import matplotlib.pyplot as plt

# Linear Regression Model
class LinearModel:
    def __init__(self):
        self.split_data()

    def split_data(self):
        df = pd.read_csv("prices.csv", low_memory=False)
        self.int_columns = ['quarter', 'nsmiles', 'passengers']
        self.string_columns = ['city1', 'city2', 'airport_1', 'airport_2', 'carrier_lg']

        self.encoder = OneHotEncoder(sparse_output=False)

        string_encoded = self.encoder.fit_transform(df[self.string_columns].values)
        int_data = df[self.int_columns].to_numpy()
        self.X = np.concatenate([string_encoded, int_data], axis=1)

        self.y = df[['fare']].to_numpy()
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    def train_model(self):
        self.model = LinearRegression(fit_intercept=True)
        self.model.fit(self.X_train, self.y_train)
        with open("LinearModel", "wb") as files:
            pickle.dump(self.model, files)

    def test_accuracy(self):
        self.train_model()
        predictions = self.model.predict(self.X_test)
        return mean_squared_error(self.y_test, predictions)

    def prediction(self, X):
        string_X = np.array(X[:len(self.string_columns)]).reshape(1, -1)
        int_X = np.array(X[len(self.string_columns):len(self.string_columns) + len(self.int_columns)]).reshape(1, -1)

        string_X_encoded = self.encoder.transform(string_X)

        X_encoded = np.concatenate([string_X_encoded, int_X], axis=1)
        return self.model.predict(X_encoded)

# Random Forest Model    
class RFModel:
    def __init__(self):
        self.split_data()

    def split_data(self):
        df = pd.read_csv("prices.csv", low_memory=False)
        self.int_columns = ['quarter', 'nsmiles', 'passengers']
        self.string_columns = ['city1', 'city2', 'airport_1', 'airport_2', 'carrier_lg']
        self.encoder = OneHotEncoder(sparse_output=False)
        string_encoded = self.encoder.fit_transform(df[self.string_columns].values)
        with open("OneHotEncoder", "wb") as files:
            pickle.dump(self.encoder, files)
        int_data = df[self.int_columns].to_numpy()
        self.X = np.concatenate([string_encoded, int_data], axis=1)
        df['fare_category'] = pd.cut(df['fare'], bins=[0, 100, 300, np.inf], labels=[0, 1, 2], right=False)
        self.y = df[['fare_category']].values.ravel()
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    def train_model(self):
        self.model = RandomForestClassifier()
        self.model.fit(self.X_train, self.y_train)
        with open("RFModel", "wb") as files:
            pickle.dump(self.model, files)

    def test_accuracy(self):
        self.train_model()
        training_predictions = self.model.predict(self.X_train)
        test_predictions = self.model.predict(self.X_test)
        val_predictions = self.model.predict(self.X_val)

        train_accuracy = accuracy_score(self.y_train, training_predictions)
        test_accuracy = accuracy_score(self.y_test, test_predictions)
        val_accuracy = accuracy_score(self.y_val, val_predictions)
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(self.y_train, training_predictions, average='weighted')
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(self.y_test, test_predictions, average='weighted')
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(self.y_val, val_predictions, average='weighted')

        print("Training: ")
        print("\t Accuracy: ", train_accuracy)
        print("\t Precision: ", train_precision)
        print("\t Recall: ", train_recall)
        print("\t F1: ", train_f1)

        print("Validation: ")
        print("\t Accuracy: ", val_accuracy)
        print("\t Precision: ", val_precision)
        print("\t Recall: ", val_recall)
        print("\t F1: ", val_f1)

        print("Test: ")
        print("\t Accuracy: ", test_accuracy)
        print("\t Precision: ", test_precision)
        print("\t Recall: ", test_recall)
        print("\t F1: ", test_f1)
        return train_accuracy, val_accuracy, test_accuracy

    def prediction(self, X):

        string_X = np.array(X[:len(self.string_columns)]).reshape(1, -1)
        int_X = np.array(X[len(self.string_columns):len(self.string_columns) + len(self.int_columns)]).reshape(1, -1)

        string_X_encoded = self.encoder.transform(string_X)
        X_encoded = np.concatenate([string_X_encoded, int_X], axis=1)
        return self.model.predict(X_encoded)

# XGBoost Model
class XGBoost:
    def __init__(self):
        self.split_data()

    def split_data(self):
        df = pd.read_csv("prices.csv", low_memory=False)
        self.int_columns = ['quarter', 'nsmiles', 'passengers']
        self.string_columns = ['city1', 'city2', 'airport_1', 'airport_2', 'carrier_lg']

        self.encoder = OneHotEncoder(sparse_output=False)

        string_encoded = self.encoder.fit_transform(df[self.string_columns].values)
        int_data = df[self.int_columns].to_numpy()
        self.X = np.concatenate([string_encoded, int_data], axis=1)
        df['fare_category'] = pd.cut(df['fare'], bins=[0, 100, 300, np.inf], labels=[0, 1, 2], right=False)
        self.y = df[['fare_category']].values.ravel()
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    def train_model(self):
        self.model = XGBClassifier(
            objective='multi:softmax',  
            num_class=3,               
            eval_metric='mlogloss',    
            max_depth=10,               
            learning_rate=0.1,         
            n_estimators=100,        
            random_state=42
        )

        self.model.fit(self.X_train, self.y_train)

        with open("XGBoost", "wb") as files:
            pickle.dump(self.model, files)

    def test_accuracy(self):
        self.train_model()

        training_predictions = self.model.predict(self.X_train)
        test_predictions = self.model.predict(self.X_test)
        val_predictions = self.model.predict(self.X_val)

        train_accuracy = accuracy_score(self.y_train, training_predictions)
        test_accuracy = accuracy_score(self.y_test, test_predictions)
        val_accuracy = accuracy_score(self.y_val, val_predictions)
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(self.y_train, training_predictions, average='weighted')
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(self.y_test, test_predictions, average='weighted')
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(self.y_val, val_predictions, average='weighted')

        print("Training: ")
        print("\t Accuracy: ", train_accuracy)
        print("\t Precision: ", train_precision)
        print("\t Recall: ", train_recall)
        print("\t F1: ", train_f1)

        print("Validation: ")
        print("\t Accuracy: ", val_accuracy)
        print("\t Precision: ", val_precision)
        print("\t Recall: ", val_recall)
        print("\t F1: ", val_f1)

        print("Test: ")
        print("\t Accuracy: ", test_accuracy)
        print("\t Precision: ", test_precision)
        print("\t Recall: ", test_recall)
        print("\t F1: ", test_f1)

        return train_accuracy, val_accuracy, test_accuracy

    def prediction(self, X):
        string_X = np.array(X[:len(self.string_columns)]).reshape(1, -1)
        int_X = np.array(X[len(self.string_columns):len(self.string_columns) + len(self.int_columns)]).reshape(1, -1)

        string_X_encoded = self.encoder.transform(string_X)

        X = np.concatenate([string_X_encoded, int_X], axis=1)
        return self.model.predict(X)

# Logistic Regression Model
class LogisticModel:
    def __init__(self):
        self.split_data()

    def split_data(self):
        df = pd.read_csv("prices.csv", low_memory=False)
        self.int_columns = ['quarter', 'nsmiles', 'passengers']
        self.string_columns = ['city1', 'city2', 'airport_1', 'airport_2', 'carrier_lg']
        self.encoder = OneHotEncoder(sparse_output=False)

        string_encoded = self.encoder.fit_transform(df[self.string_columns].values)
        int_data = df[self.int_columns].to_numpy()
        self.X = np.concatenate([string_encoded, int_data], axis=1)
        df['fare_category'] = pd.cut(df['fare'], bins=[0, 100, 300, np.inf], labels=[0, 1, 2], right=False)
        self.y = df[['fare_category']].values.ravel()
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    def train_model(self):
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)
        with open("LogisticModel", "wb") as files:
            pickle.dump(self.model, files)

    def test_accuracy(self):
        self.train_model()
        training_predictions = self.model.predict(self.X_train)
        test_predictions = self.model.predict(self.X_test)
        val_predictions = self.model.predict(self.X_val)

        train_accuracy = accuracy_score(self.y_train, training_predictions)
        test_accuracy = accuracy_score(self.y_test, test_predictions)
        val_accuracy = accuracy_score(self.y_val, val_predictions)
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(self.y_train, training_predictions, average='weighted')
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(self.y_test, test_predictions, average='weighted')
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(self.y_val, val_predictions, average='weighted')

        print("Training: ")
        print("\t Accuracy: ", train_accuracy)
        print("\t Precision: ", train_precision)
        print("\t Recall: ", train_recall)
        print("\t F1: ", train_f1)

        print("Validation: ")
        print("\t Accuracy: ", val_accuracy)
        print("\t Precision: ", val_precision)
        print("\t Recall: ", val_recall)
        print("\t F1: ", val_f1)

        print("Test: ")
        print("\t Accuracy: ", test_accuracy)
        print("\t Precision: ", test_precision)
        print("\t Recall: ", test_recall)
        print("\t F1: ", test_f1)
        return train_accuracy, val_accuracy, test_accuracy

    def prediction(self, X):
        string_X = np.array(X[:len(self.string_columns)]).reshape(1, -1)
        int_X = np.array(X[len(self.string_columns):len(self.string_columns) + len(self.int_columns)]).reshape(1, -1)

        string_X_encoded = self.encoder.transform(string_X)

        X = np.concatenate([string_X_encoded, int_X], axis=1)
        return self.model.predict(X)

# gather data for the report
rf = RFModel()
rf_train_acc, rf_val_acc, rf_test_acc = rf.test_accuracy()

xg = XGBoost()
xg_train_acc, xg_val_acc, xg_test_acc = xg.test_accuracy()

logistic = LogisticModel()
lg_train_acc, lg_val_acc, lg_test_acc = logistic.test_accuracy()

train_acc = [rf_train_acc, xg_train_acc, lg_train_acc]
val_acc = [rf_val_acc, xg_val_acc, lg_val_acc]
test_acc = [rf_test_acc, xg_test_acc, lg_test_acc]
x_labels = ['Random Forest', 'XGBoost', 'Logistic Regression']

plt.plot(x_labels, train_acc, label='training')
plt.plot(x_labels, val_acc, label='validation')
plt.plot(x_labels, test_acc, label='test')
plt.title('Classifier Model Accuracies')
plt.xlabel('Model')
plt.ylabel('Accuracies')
plt.legend()
plt.savefig('accuracy_plots.jpg')
plt.show()