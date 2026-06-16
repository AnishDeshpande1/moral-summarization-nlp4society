import pickle
import pandas as pd
import sys

def main():
    try:
        with open('dict_of_dfs.pickle', 'rb') as f:
            metrics = pickle.load(f)
    except FileNotFoundError:
        print("Error: dict_of_dfs.pickle not found. Please run run_evaluation.py first.")
        sys.exit(1)

    # The metrics dict has the structure: metrics[metric_name] = multi_index_dataframe
    for metric_name, df in metrics.items():
        print("\n" + "="*50)
        print(f" METRIC: {metric_name.upper()}")
        print("="*50)
        
        # Display the summary rows (mean and std) for each model/prompt type
        if isinstance(df, pd.DataFrame):
            # Print the dataframe
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print(df)
        else:
            print(df)

if __name__ == '__main__':
    main()
