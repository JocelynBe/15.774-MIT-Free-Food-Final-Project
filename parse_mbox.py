import mailbox
import pandas as pd
import re
from datetime import datetime


def parse_email(x: mailbox.mboxMessage) -> pd.DataFrame:
    """
    Parses the email to get the date, subject, sender, and message
    """

    # Get each of the elements we're interested in
    date = x["date"]
    subject = x["subject"]
    author = x["from"]
    message = x.get_payload(decode=True)

    # We need to add a "0" in front of days like 6 May 2013 so that
    # we can parse the date
    regex = r"\s[0-9]\s"
    search_res = re.search(regex, date)
    if search_res:
        tmp_str = " 0" + search_res.group().strip() + " "
        date = re.sub(regex, tmp_str, date)

    # If the time string for example specifies (EDT) then remove this because
    # we can infer this from the +0000 string
    date = re.sub(r"\([A-Z]{3}\)", "", date).rstrip()

    # Parse the date to get the weekday, day, month, year, hour, and minute
    date_res = datetime.strptime(date, "%a, %d %b %Y %H:%M:%S %z")

    # Clean up the author string so we just have their email
    author = author.lower()
    regex = r"<.*>$"
    author_res = re.search(regex, author)
    if author_res:
        author = re.sub(r"[<>]", "", author_res.group())

    # Combine the subject and body since we search those together
    email = str(subject) + " " + str(message)

    # Remove \n from the email
    email = re.sub(r"\\n", "", email)

    # Get the final DataFrame for the given email
    df = pd.DataFrame(
        {"weekday": [date_res.isoweekday()], "day": [date_res.day],
         "month": [date_res.month], "year": [date_res.year],
         "hour": [date_res.hour], "minute": [date_res.minute],
         "email": email, "author": [author]}
    )
    return df


def parse_mbox(mbox_file: str) -> pd.DataFrame:

    # Get the .mbox file into a mailbox.mbox object
    emails = mailbox.mbox(mbox_file)

    # Parse all of the emails
    n = len(emails)
    df_list = [pd.DataFrame()] * n
    for (i, email) in enumerate(emails):
        df_list[i] = parse_email(email)

    # Combine all of the emails
    return pd.concat(df_list, ignore_index=True)
