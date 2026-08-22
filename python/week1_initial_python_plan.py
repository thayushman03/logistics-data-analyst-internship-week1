import pandas as pd

orders = pd.read_csv("orders.csv")

orders["purchase_date"] = pd.to_datetime(
    orders["purchase_date"]
)

orders["delivery_date"] = pd.to_datetime(
    orders["delivery_date"]
)

orders["delivery_days"] = (
    orders["delivery_date"]
    - orders["purchase_date"]
).dt.days

orders["is_late"] = (
    orders["delivery_date"]
    > orders["estimated_delivery_date"]
)

on_time_rate = (~orders["is_late"]).mean() * 100
avg_delivery_days = orders["delivery_days"].mean()

print(f"On-time delivery rate: {on_time_rate:.2f}%")
print(f"Average delivery time: {avg_delivery_days:.2f} days")