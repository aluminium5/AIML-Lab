"""
Pandas Module - Data Analysis and Manipulation
This module contains examples for working with Pandas DataFrames
"""

import pandas as pd


def read_data_examples():
    """Demonstrate different ways to read data"""
    # Read CSV
    df = pd.read_csv('pokemon_data.csv')
    print("Reading CSV file:")
    print(df.head(5))
    
    # Read Excel
    # df_excel = pd.read_excel('pokemon_data.xlsx')
    # print("\nReading Excel file:")
    # print(df_excel.head(5))
    
    # Read specific columns
    df_specific = pd.read_csv('pokemon_data.csv', usecols=['Name', 'Type 1', 'HP'])
    print("\nReading specific columns:")
    print(df_specific.head(5))
    
    return df


def basic_data_exploration(df):
    """Demonstrate basic data exploration techniques"""
    print("\n" + "=" * 50)
    print("BASIC DATA EXPLORATION")
    print("=" * 50)
    
    # Read headers
    print("\nColumn names:")
    print(df.columns)
    
    # Read first few rows
    print("\nFirst 3 rows:")
    print(df.head(3))
    
    # Read last few rows
    print("\nLast 3 rows:")
    print(df.tail(3))
    
    # Get info about the dataframe
    print("\nDataFrame info:")
    print(df.info())
    
    # Get statistical summary
    print("\nStatistical summary:")
    print(df.describe())


def reading_data_methods(df):
    """Demonstrate different methods to read data from DataFrame"""
    print("\n" + "=" * 50)
    print("READING DATA METHODS")
    print("=" * 50)
    
    # Read a specific column
    print("\nReading 'Name' column:")
    print(df['Name'].head())
    
    # Read multiple columns
    print("\nReading multiple columns:")
    print(df[['Name', 'Type 1', 'HP']].head())
    
    # Read a specific row
    print("\nReading row at index 2:")
    print(df.iloc[2])
    
    # Read specific rows
    print("\nReading rows 1-3:")
    print(df.iloc[1:4])
    
    # Read specific cell
    print("\nReading cell at row 2, column 1:")
    print(df.iloc[2, 1])
    
    # Using loc (label-based)
    print("\nUsing loc to read row 2:")
    print(df.loc[2])


def filtering_data(df):
    """Demonstrate data filtering techniques"""
    print("\n" + "=" * 50)
    print("FILTERING DATA")
    print("=" * 50)
    
    # Filter by condition
    print("\nPokemon with HP > 100:")
    high_hp = df[df['HP'] > 100]
    print(high_hp.head())
    
    # Filter by specific value
    print("\nFire type Pokemon:")
    fire_type = df[df['Type 1'] == 'Fire']
    print(fire_type.head())
    
    # Multiple conditions (AND)
    print("\nFire type Pokemon with HP > 80:")
    fire_high_hp = df[(df['Type 1'] == 'Fire') & (df['HP'] > 80)]
    print(fire_high_hp.head())
    
    # Multiple conditions (OR)
    print("\nFire or Water type Pokemon:")
    fire_or_water = df[(df['Type 1'] == 'Fire') | (df['Type 1'] == 'Water')]
    print(fire_or_water.head())
    
    # Filter by string contains
    print("\nPokemon with 'Mega' in name:")
    mega_pokemon = df[df['Name'].str.contains('Mega')]
    print(mega_pokemon.head())
    
    # Filter with regex
    print("\nPokemon starting with 'Pi':")
    pi_pokemon = df[df['Name'].str.contains('^Pi', regex=True)]
    print(pi_pokemon)


def sorting_data(df):
    """Demonstrate data sorting"""
    print("\n" + "=" * 50)
    print("SORTING DATA")
    print("=" * 50)
    
    # Sort by single column
    print("\nSorted by Name:")
    sorted_name = df.sort_values('Name')
    print(sorted_name.head())
    
    # Sort by multiple columns
    print("\nSorted by Type 1, then HP (descending):")
    sorted_multi = df.sort_values(['Type 1', 'HP'], ascending=[True, False])
    print(sorted_multi.head(10))


def modifying_data(df):
    """Demonstrate data modification"""
    print("\n" + "=" * 50)
    print("MODIFYING DATA")
    print("=" * 50)
    
    # Create a copy to avoid modifying original
    df_copy = df.copy()
    
    # Add a new column
    df_copy['Total'] = df_copy['HP'] + df_copy['Attack'] + df_copy['Defense'] + \
                       df_copy['Sp. Atk'] + df_copy['Sp. Def'] + df_copy['Speed']
    print("\nAdded 'Total' column:")
    print(df_copy[['Name', 'HP', 'Attack', 'Defense', 'Total']].head())
    
    # Reorder columns
    cols = list(df_copy.columns)
    df_copy = df_copy[cols[:4] + [cols[-1]] + cols[4:12]]
    print("\nReordered columns:")
    print(df_copy.head())
    
    return df_copy


def saving_data(df):
    """Demonstrate saving data to different formats"""
    print("\n" + "=" * 50)
    print("SAVING DATA")
    print("=" * 50)
    
    # Save to CSV
    df.to_csv('modified_pokemon.csv', index=False)
    print("Saved to modified_pokemon.csv")
    
    # Save to Excel
    # df.to_excel('modified_pokemon.xlsx', index=False)
    # print("Saved to modified_pokemon.xlsx")
    
    # Save to CSV with tab separator
    df.to_csv('modified_pokemon.txt', index=False, sep='\t')
    print("Saved to modified_pokemon.txt")


def aggregate_statistics(df):
    """Demonstrate aggregate statistics using groupby"""
    print("\n" + "=" * 50)
    print("AGGREGATE STATISTICS")
    print("=" * 50)
    
    # Group by Type 1 and get mean
    print("\nMean stats by Type 1:")
    type_stats = df.groupby(['Type 1']).mean(numeric_only=True)
    print(type_stats.head())
    
    # Group by Type 1 and get sum
    print("\nSum of stats by Type 1:")
    type_sum = df.groupby(['Type 1']).sum(numeric_only=True)
    print(type_sum.head())
    
    # Group by Type 1 and count
    print("\nCount by Type 1:")
    type_count = df.groupby(['Type 1']).count()
    print(type_count['Name'].sort_values(ascending=False))


def working_with_large_files():
    """Demonstrate working with large files using chunks"""
    print("\n" + "=" * 50)
    print("WORKING WITH LARGE FILES")
    print("=" * 50)
    
    # Read in chunks
    print("\nReading file in chunks:")
    chunk_size = 100
    for chunk in pd.read_csv('pokemon_data.csv', chunksize=chunk_size):
        print(f"Processing chunk with {len(chunk)} rows")
        # Process each chunk
        print(chunk.head(2))
        break  # Just show first chunk for demo


def main():
    """Main function to run all demonstrations"""
    print("=" * 50)
    print("PANDAS MODULE DEMONSTRATIONS")
    print("=" * 50)
    
    try:
        # Read data
        df = read_data_examples()
        
        # Basic exploration
        basic_data_exploration(df)
        
        # Reading methods
        reading_data_methods(df)
        
        # Filtering
        filtering_data(df)
        
        # Sorting
        sorting_data(df)
        
        # Modifying
        df_modified = modifying_data(df)
        
        # Saving
        # saving_data(df_modified)
        
        # Aggregate statistics
        aggregate_statistics(df)
        
        # Large files
        # working_with_large_files()
        
    except FileNotFoundError:
        print("\nError: pokemon_data.csv not found!")
        print("Please ensure the data file is in the same directory.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()
