import email
import re
from datetime import datetime
import pandas as pd


def read_file(file: str) -> list:
    """
    Reads the .txt file and grabs all of the email messages contained in it
    """
    with open(file) as f:
        data = f.readlines()

    positions = []
    tag = "Subject: [Vultures]"
    for i in range(len(data)):
        if data[i].startswith(tag):
            positions.append(i)

    starts = list(map(lambda x: x - positions[0], positions)) + [len(data)]
    messages = [data[starts[i]:starts[i+1]] for i in range(len(starts) - 1)]
    return messages


def parse_message(message: list) -> dict:
    """
    Parses the message using the email package
    """
    email_dict = {}
    msg = email.message_from_string("".join(message))

    # Add the elements of the message
    email_dict["subject"] = msg["subject"]
    email_dict["date"] = msg["date"]
    email_dict["body"] = msg.get_payload(decode=True)
    email_dict["author"] = msg["from"]

    # Check to make sure that we have a valid email
    for key in email_dict.keys():
        if email_dict[key] is None:
            return {}

    return email_dict


def clean_message(email_dict: dict) -> pd.DataFrame:
    """
    Cleans an email message and puts it into a DataFrame format
    """

    # To clean the date we first need to add a trailing zero in the case
    # of a date like Oct 1 2013 so that we can parse it correctly
    regex = r"\s[0-9]\s"
    search_res = re.search(regex, email_dict["date"])
    if search_res:
        tmp_str = "0" + search_res.group().strip() + " "
        email_dict["date"] = re.sub(regex, tmp_str, email_dict["date"])

    # If the time string for example specifies (EDT) then remove this because
    # we can infer this from the +0000 string
    date = re.sub(r"\([A-Z]{3}\)", "", email_dict["date"]).rstrip()

    # The dates have the following expected formats
    # 1. Fri Oct 01 15:09:58 2004
    # 2. Fri, 04 Jan 2013 13:51:36 -0500
    # So we need to account for both cases
    fmts = ["%a %b %d %H:%M:%S %Y", "%a, %d %b %Y %H:%M:%S %z"]
    for fmt in fmts:
        try:
            date_res = datetime.strptime(date, fmt)
            break
        except ValueError:
            pass

    # Grab just the email from the author of the message
    # Either this will be in the format <...@...> or it will be
    # ... at .... (name); we will account for both cases
    at_regex = r"^.*(at).*\("
    bracket_regex = r"<.*>"
    at_search = re.search(at_regex, email_dict["author"])
    bracket_search = re.search(bracket_regex, email_dict["author"])

    if at_search:
        email_dict["author"] = re.sub(r"\s\(", "", at_search.group())
    else:
        email_dict["author"] = re.sub(r"[<>]", "", bracket_search.group())

    # Combine the subject and body since we search those together
    body = str(email_dict["subject"]) + " " + str(email_dict["body"])

    # Remove \n from the email
    body = re.sub(r"\\n", "", body)

    # Build the message DataFrame
    df = pd.DataFrame(
        {"weekday": [date_res.isoweekday()], "day": [date_res.day],
         "month": [date_res.month], "year": [date_res.year],
         "hour": [date_res.hour], "minute": [date_res.minute],
         "email": body, "author": [email_dict["author"]]}
    )
    return df


def parse_txt(txt_file: str) -> pd.DataFrame:
    """
    Parses an email .txt file and returns a DataFrame containing the
    date, subject, body, and author
    """
    # Read the emails
    emails = read_file(txt_file)

    # Get all of the emails into a DataFrame form
    n = len(emails)
    df_list = [pd.DataFrame()] * n
    for (i, msg) in enumerate(emails):
        email_dict = parse_message(msg)
        df_list[i] = clean_message(email_dict)

    return pd.concat(df_list, ignore_index=True)
