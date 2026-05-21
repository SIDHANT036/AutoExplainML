from autoexplainml import explain
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# fake dataset
X = pd.DataFrame({
    "age": [20, 30, 40, 50],
    "income": [2000, 3000, 4000, 5000]
})

y = [0, 1, 1, 0]

model = RandomForestClassifier()
model.fit(X, y)

result = explain(model, X, task="tabular")

print(result)