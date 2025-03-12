import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

np.random.seed(42)

customer_data = pd.DataFrame({
    'customer_id': np.arange(1, 501),
    'customer_age': np.random.randint(18, 65, size=500),
    'customer_gender': np.random.choice(['Male', 'Female'], size=500),
    'customer_location': np.random.choice(['Urban', 'Suburban', 'Rural'], \
                                          size=500)
})

product_data = pd.DataFrame({
    'product_id': np.arange(1, 11),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', \
                                          'Sports'], size=10),
    'product_brand': np.random.choice(['Brand_A', 'Brand_B', 'Brand_C'], size=10),
    'product_price': np.random.uniform(50, 500, size=10),
    'min_price': np.random.uniform(40,4900, size=10),
    'cm': np.random.uniform(50,500, size = 10)/1000
})

sales_data = pd.DataFrame({
    'customer_id': np.random.choice(np.arange(1, 501), size=1000),
    'product_id': np.random.choice(np.arange(1, 11), size=1000),
    'sales_quantity': np.random.randint(1, 10, size=1000)
})

all_data = pd.merge(customer_data, sales_data, on="customer_id")
all_data = pd.merge(all_data, product_data, on="product_id")

print("######### BEFORE")
print(all_data[['product_price','min_price']].head(10))

features = ['customer_age', 'product_price']

target_variable = 'sales_quantity'

X_train, X_test, y_train, y_test = train_test_split(all_data[features], \
                    all_data[target_variable], test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

model = LinearRegression()

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

y_pred = model.predict(X_test_scaled)
test_rmse = mean_squared_error(y_test, y_pred)
print(f"Test RMSE: {test_rmse:.2f}")

all_data["predicted_sales"] = model.predict(scaler.transform(all_data[features]))

mean_sales = all_data["predicted_sales"].mean()
std_sales = all_data["predicted_sales"].std()
scaling_factor = 1 + (0.1 * (all_data["predicted_sales"] - mean_sales) / std_sales)
all_data['scf'] = scaling_factor.astype(float)

all_data['adj_psc'] = all_data['product_price'] * all_data['scf']

all_data["apbm"] = all_data["product_price"] * (1 + all_data["cm"])

all_data['adj_price'] = np.maximum(\
                                   np.maximum(all_data['min_price'], \
                                              all_data['adj_psc']) , \
                                       all_data['apbm'])

print("######### AFTER")
print(all_data[['product_price','scf','min_price',\
               'adj_psc','apbm','adj_price']].head(10))
