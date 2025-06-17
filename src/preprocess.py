# src/preprocess.py

def clean_and_preprocess(data):
    data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)
    data['Item_Visibility'].replace(0, data['Item_Visibility'].median(), inplace=True)
    data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0], inplace=True)
    # Add more steps...
    return data
