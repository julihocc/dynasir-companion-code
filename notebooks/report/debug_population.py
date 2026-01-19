
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

try:
    from dynasir import process_data_from_owid, DataContainer
except ImportError:
    print("dynasir not installed")
    sys.exit(1)

print("Fetching OWID data...")
df = process_data_from_owid(include_vaccination=True)
print(f"Data shape: {df.shape}")
print("Columns:", df.columns.tolist())

if 'population' in df.columns:
    pop = df['population']
    print("\nPopulation column stats:")
    print(pop.describe())
    print("\nUnique population values:")
    print(pop.unique())
    
    # Check if it varies over time
    is_constant = pop.nunique() <= 1
    print(f"\nIs population constant over time? {is_constant}")
    
    if not is_constant:
        print("Population changes over time.")
        print(pop.head())
        print(pop.tail())
else:
    print("No 'population' column found.")

# Check DataContainer mapping
dc = DataContainer(df)
if 'N' in dc.data.columns:
    print("\nDataContainer 'N' column stats:")
    print(dc.data['N'].describe())
    print("Unique 'N':", dc.data['N'].unique())
else:
    print("\nDataContainer has no 'N' column.")
