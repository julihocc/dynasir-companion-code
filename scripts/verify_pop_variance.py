
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
try:
    from dynasir import process_data_from_owid, DataContainer
except ImportError:
    print("dynasir not installed")
    exit(1)

print("Starting verification (simulating report logic)...")

# 1. Load initial data
print("Loading standard OWID data...")
try:
    global_dataframe = process_data_from_owid(include_vaccination=True)
except Exception as e:
    print(f"Failed to load OWID data: {e}")
    exit(1)

# 2. Simulate the interpolation logic added to report.py
try:
    print("Fetching historical population data...")
    pop_df = pd.read_csv('https://ourworldindata.org/grapher/population.csv')
    
    # Filter for World (OWID_WRL) because that's what process_data_from_owid defaults to
    world_pop = pop_df[(pop_df['Entity'] == 'World') & (pop_df['Year'] > 2000)].copy()
    
    # Create a date index from the Year column 
    world_pop['date'] = pd.to_datetime(world_pop['Year'].astype(str) + '-01-01')
    world_pop = world_pop.set_index('date').sort_index()
    
    # Rename column
    world_pop = world_pop.rename(columns={'Population (historical)': 'N'})
    
    # Reindex and interpolate
    daily_pop = world_pop['N'].resample('D').interpolate(method='time')
    
    # Align
    common_dates = global_dataframe.index.intersection(daily_pop.index)
    
    if len(common_dates) > 0:
        print("Successfully interpolated population data.")
        # Update
        global_dataframe.loc[common_dates, 'N'] = daily_pop.loc[common_dates]
        
        # Verify
        n_unique = global_dataframe['N'].nunique()
        print(f"Population column now has {n_unique} unique values.")
        print("First 5 values:")
        print(global_dataframe['N'].head())
        print("Last 5 values:")
        print(global_dataframe['N'].tail())

        # Check DataContainer
        dc = DataContainer(global_dataframe)
        dc_unique = dc.data['N'].nunique()
        print(f"DataContainer N column unique values: {dc_unique}")
        if dc_unique > 1:
            print("SUCCESS: Population is time-varying.")
        else:
            print("FAILURE: Population is still constant in DataContainer.")

    else:
        print("Warning: No overlapping dates found.")

except Exception as e:
    print(f"Error during verification: {e}")
