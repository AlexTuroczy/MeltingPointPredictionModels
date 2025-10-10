import pandas as pd
import os
import subprocess

def main():

    submission_path = "out/submission.csv"

    train_df = pd.read_csv("data/melting-point/train.csv")
    pred = train_df["Tm"].mean()
    print(pred)

    test_df = pd.read_csv("data/melting-point/test.csv")
    submission = test_df[["id"]]
    submission["Tm"] = pred
    print(submission)

    os.makedirs("out", exist_ok=True)
    submission.to_csv(submission_path, index=False)

    # submit to kaggle
    competition_slug = "melting-point"
    message = "Trivial mean prediction."

    subprocess.run([
    "kaggle", "competitions", "submit",
    "-c", competition_slug,
    "-f", submission_path,
    "-m", message
])
    
if __name__ == "__main__":
    main()