import clean_email_data
import os
import argparse
import pandas as pd
from glob import glob


def main():
    # To make this script work the user needs to provide two things:
    # 1. The path to the data you want to parse
    # 2. The location where you want to save the resuling .csv file
    parser = argparse.ArgumentParser()
    parser.add_argument("--email_path", help="Location of email files",
                        type=str)
    parser.add_argument("--save_path", help="Location to save final data",
                        type=str)
    args = parser.parse_args()

    # Get all of the files in email_path
    email_files = glob(os.path.join(args.email_path, "*"))

    # Parse each of the email files
    n = len(email_files)
    df_list = [pd.DataFrame()] * n
    for (i, email) in enumerate(email_files):
        df_list[i] = clean_email_data.clean_emails(email)

    # Join the emails
    df = pd.concat(df_list, ignore_index=True)

    # Save the result to disk
    df.to_csv(args.save_path, index=False)
    return None


if __name__ == "__main__":
    main()
