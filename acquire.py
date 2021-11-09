import pandas as pd
import env
import os


    
def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_titanic_data():
    return pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))

def get_titanic_data():
    filename = "titanic.csv"
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = new_titanic_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df  

def new_iris_data():
    return pd.read_sql('SELECT * FROM measurements JOIN species USING(species_id)', get_connection('iris_db'))

def get_iris_data():
    filename = "iris.csv"
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = new_iris_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df  

def new_telco_data():
    return pd.read_sql('SELECT * FROM customers JOIN contract_types USING(contract_type_id) JOIN internet_service_types USING(internet_service_type_id) JOIN payment_types USING(payment_type_id)', get_connection('telco_churn'))

def get_telco_data():
    filename = "telco.csv"
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = new_telco_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df  