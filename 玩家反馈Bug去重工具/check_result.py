import pandas as pd
import glob
import os

# Find the latest test result
list_of_files = glob.glob(r'D:\douzhongjun\work\260112-260116\玩家反馈Bug去重工具\cleandata\test_cleaning_result_*.xlsx') 
latest_file = max(list_of_files, key=os.path.getctime)
print(f"Checking file: {latest_file}")

df = pd.read_excel(latest_file)
print("Columns:", df.columns.tolist())

if 'scene' in df.columns and 'pos' in df.columns:
    print("\nFirst 10 rows of scene and pos:")
    print(df[['scene', 'pos']].head(10))
    
    print("\nNon-empty count:")
    print("Scene:", df['scene'].astype(str).replace('nan', '').replace('', 'empty').value_counts().head())
    print("Pos:", df['pos'].astype(str).replace('nan', '').replace('', 'empty').value_counts().head())
else:
    print("Columns 'scene' or 'pos' missing!")
