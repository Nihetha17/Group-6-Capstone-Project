import matplotlib.pyplot as plt
import pandas as pd

customer_data = pd.read_csv('/Users/santoshr/Documents/TNS_project_1/unsupervised/customer-segmentation/customers.csv')
print(customer_data)
print(min(customer_data['annual_income_k']),max(customer_data['annual_income_k']),min(customer_data['spending_score']),max(customer_data['spending_score']))
print(customer_data.describe())
customer_data.plot(
    kind="scatter",
    x="annual_income_k",
    y="spending_score",
    color="blue"
)
plt.title('Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()