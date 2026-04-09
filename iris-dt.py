import mlflow
from sklearn.ensemble import RandomForestClassifier
mlflow.set_tracking_uri("http://127.0.0.1:5000")
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Load the Iris dataset
iris = load_iris()
X = iris.data   
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
max_depth = 30

mlflow.set_experiment('iris-dt')


# Start an MLflow run
with mlflow.start_run():
    # Train a Decision Tree Classifier
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, name="decision_tree_model")


    #create a confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('actual')
    plt.title('Confusion Matrix') 

    plt.savefig('confusion_matrix.png')

    #mlflow code
    mlflow.log_artifact('confusion_matrix.png')   

    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(model,'decision-tree')

    mlflow.set_tag('auther','shivam')
    mlflow.set_tag('model','decisiontree')
    print(f"Accuracy: {accuracy}")
