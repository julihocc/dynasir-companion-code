
import warnings
warnings.filterwarnings("ignore")
try:
    from dynasir import process_data_from_owid, DataContainer
except ImportError:
    print("dynasir not installed")
    exit(1)

print("Loading data...")
# Use local data if download fails, or try download
try:
    df = process_data_from_owid(include_vaccination=True)
except Exception as e:
    print(f"Download failed: {e}")
    # Fallback to local if possible, but the function handles it. 
    # If function raised exception, it means both failed.
    exit(1)

dc = DataContainer(df)
n_series = dc.data['N']

print(f"Population stats:")
print(f"Min: {n_series.min()}")
print(f"Max: {n_series.max()}")
print(f"Unique values count: {n_series.nunique()}")
print(f"Is constant? {n_series.nunique() <= 1}")

if n_series.nunique() > 1:
    print("Population varies. First 5 changes:")
    print(n_series.drop_duplicates().head())
else:
    print("Population is exactly constant.")
