import pandas as pd
import os

def analyze_basic_stats(files, output_dir="analysis_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for name, path in files.items():
        df = pd.read_csv(path)
        if "WHE" not in df.columns:
            raise ValueError(f"{name} 缺少 WHE 列")

        s = df["WHE"].dropna()

        results.append({
            "Dataset": name,
            "Count": len(s),
            "Mean": s.mean(),
            "Std": s.std(),
            "Min": s.min(),
            "Max": s.max(),
            "Median": s.median()
        })

    result_df = pd.DataFrame(results)
    print(result_df)

    result_df.to_csv(os.path.join(output_dir, "basic_stats.csv"), index=False, encoding="utf-8-sig")
    return result_df


if __name__ == "__main__":
    files = {
        "London_Block0": "./dataset/London/Block0_Final_1h.csv",
        "London_Block1": "./dataset/London/Block1_Final_1h.csv",
        "AMPds": "./dataset/AMPds/AMPds_Final.csv",
        "Mexico": "./dataset/Mexico/Mexico_Final.csv",
    }
    analyze_basic_stats(files)