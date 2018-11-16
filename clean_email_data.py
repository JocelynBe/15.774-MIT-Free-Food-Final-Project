import re
import numpy as np
import pandas as pd
import hashlib
import parse_txt
import parse_mbox
from tqdm import tqdm


def change_ordinal_phrasing(data: np.ndarray) -> np.ndarray:
    """
    Converts things like first --> 1st, second --> 2nd, ...
    """
    ordinal_dict = {"first": "1st", "second": "2nd", "third": "3rd",
                    "fourth": "4th", "fifth": "5th", "sixth": "6th",
                    "seventh": "7th", "eighth": "8th"}

    ordinal_regex = r"((first)|(second)|(third)|(fourth)|(fifth)|(sixth)|" \
                    r"(seventh)|(eighth))"

    # Convert the values
    n = data.shape[0]
    for i in range(n):
        str_match = re.search(ordinal_regex, data[i])
        if str_match:
            change_val = ordinal_dict[str_match.group()]
            data[i] = re.sub(ordinal_regex, change_val, data[i])

    return data


def letter_number_loc(loc: str, email_id: str) -> pd.DataFrame:
    """
    Handles ex: e62-107
    """
    building = loc[0:3]
    floor = loc[4:5]
    room = loc[4:]
    return pd.DataFrame({"id": [email_id], "building": [building],
                         "floor": [floor], "room": [room]})


def all_number_loc(loc: str, email_id: str) -> pd.DataFrame:
    """
    Handles ex: 1-207
    """
    building = re.search(r"^[0-9]{1,2}", loc).group()
    loc = re.sub(r"^[0-9]{1,2}-", "", loc)
    floor = loc[0]
    room = loc
    return pd.DataFrame({"id": [email_id], "building": [building],
                         "floor": [floor], "room": [room]})


def stata_rooms(loc: str, email_id: str) -> pd.DataFrame:
    """
    Handles ex: 32-G882
    """
    loc = re.sub(r"32-", "", loc)
    floor = re.search(r"[dg][0-9]", loc).group()
    return pd.DataFrame({"id": [email_id], "building": [32],
                         "floor": [floor], "room": [loc]})


def stata_kitchens(loc: str, email_id: str) -> pd.DataFrame:
    """
    Handles ex: D4 or D408
    """
    floor = re.search(r"[dg][0-9]", loc).group()

    # Check if we were given a room location
    room_search = re.search(r"[0-9]{3}", loc)
    if room_search:
        room = loc
    else:
        room = np.nan

    return pd.DataFrame({"id": [email_id], "building": [32],
                         "floor": [floor], "room": [room]})


def building_floor_loc(loc: str, email_id: str) -> pd.DataFrame:
    """
    ex: building 9 4th floor
    """
    building = re.search(r"[0-9]{1,2}", loc).group()
    loc = re.sub(r"((building)|(bldg))\s[0-9]{1,2}", "", loc)
    floor = re.search(r"[0-9]", loc).group()
    return pd.DataFrame({"id": [email_id], "building": [building],
                         "floor": [floor], "room": [np.nan]})


def floor_building_loc(loc: str, email_id: str) -> pd.DataFrame:
    """
    ex: 4th floor building 10
    """
    floor = re.search(r"^[0-9]", loc).group()
    loc = re.sub(r"[0-9]((st)|(nd)|(rd)|(th))\sfloor", "", loc)
    building = re.search(r"[0-9]{1,2}", loc).group()
    return pd.DataFrame({"id": [email_id], "building": [building],
                         "floor": [floor], "room": [np.nan]})


def just_building(loc: str, email_id: str) -> pd.DataFrame:
    """
    Handles the case where we just detect the building
    """
    building = re.search(r"[0-9]{1,2}", loc).group()
    return pd.DataFrame({"id": [email_id], "building": [building],
                         "floor": [np.nan], "room": [np.nan]})


def just_floor(loc: str, email_id: str) -> pd.DataFrame:
    """
    Handles the case where we just detect the floor (ex: 1st floor)
    """
    floor = loc[0]
    return pd.DataFrame({"id": [email_id], "building": [np.nan],
                         "floor": [floor], "room": [np.nan]})


def no_room_loc(loc: str, email_id: str) -> pd.DataFrame:
    """
    ex: w20
    """
    return pd.DataFrame({"id": [email_id], "building": [loc],
                         "floor": [np.nan], "room": [np.nan]})


def letter_after_loc(loc: str, email_id: str) -> pd.DataFrame:
    """
    ex: 14e-101
    """
    building = re.search(r"[0-9]{1,2}[a-z]", loc).group()
    loc = re.sub(r"[0-9]{1,2}[a-z]-", "", loc)
    return pd.DataFrame({"id": [email_id], "building": [building],
                         "floor": [loc[0]], "room": [loc]})


def special_loc(loc: str, email_id: str) -> pd.DataFrame:
    """
    Handles special locations where only the name is listed
    (ex: student center)
    """
    # Check if we can extract floor and room information from the location
    floor = re.search(r"[0-9]((st)|(nd)|(rd)|(th))\sfloor", loc)
    room = re.search(r"[0-9]{3}", loc)

    loc_dict = {
        "stud": {"building": ["w20"]},
        "student center": {"building": ["w20"]},
        "student centre": {"building": ["w20"]},
        "walker": {"building": ["50"]},
        "media lab": {"building": ["e17"]},
        "stata": {"building": ["32"]},
        "next house": {"building": ["w71"]},
        "kresge": {"building": ["w16"]},
        "forbes": {"building": ["32"], "floor": ["1"]},
        "spxce": {"building": ["w31"], "floor": ["1"], "room": ["110"]},
        "lobby 10": {"building": ["10"], "floor": ["1"], "room": ["lobby"]},
        "morss": {"building": ["50"]}
    }

    # Get the initial location information
    res_dict = loc_dict[loc]

    # See if we can add any of the floor or room information otherwise
    # just add a NaN
    if floor is not None:
        floor = re.search(r"[0-9]", floor.group()).group()
        res_dict["floor"] = [int(floor)]
    else:
        res_dict["floor"] = [np.nan]

    if room is not None:
        res_dict["room"] = [int(room.group())]
    else:
        res_dict["room"] = [np.nan]

    # Add the email ID
    res_dict["id"] = [email_id]
    return pd.DataFrame(data=res_dict)


def get_proposed_locations(email: str, email_id: str,
                           regex_handlers: list) -> pd.DataFrame:
    """
    Goes through all provided regular expressions and gives the best
    possible location data available
    """

    # Get all possible proposed locations in the data
    loc_df = pd.DataFrame({"id": [email_id], "building": [np.nan],
                           "floor": [np.nan], "room": [np.nan]})
    for regex, fun in regex_handlers:
        str_match = re.search(regex, email)
        if str_match:
            loc_df = fun(str_match.group(), email_id)
            break

    return loc_df


def get_location(emails: np.ndarray, email_ids: np.ndarray,
                 email_file: str) -> pd.DataFrame:
    """
    Gets the location from the emails
    """

    # Define the proposed regular expressions to detect the location
    letter_number = r"[a-z]{1}[0-9]{1,2}-[0-9]{1,3}"

    # ex: 1-207
    all_number = r"[0-9]{1,2}-[0-9]{1,3}"

    # ex: 32-G882
    stata_room = r"32-[dg][0-9]{3}"

    # ex: D4 or D507
    stata_kitchen = r"[dg][0-9]{1,3}"

    # ex: building 4
    building_regex = r"((building)|(bldg))\s[0-9]{1,2}"

    # ex: 1st floor
    floor_regex = r"[0-9]{1}((st)|(nd)|(rd)|(th))\sfloor"

    # ex: building 4 1st floor
    building_floor = re.compile(building_regex + "\s" + floor_regex)

    # ex: 4th floor building 10
    floor_builidng = re.compile(floor_regex + "\s" + building_regex)

    # ex: w11
    no_room = r"[a-z]{1}[0-9]{1,2}"

    # ex: 14e-101
    letter_after = r"[0-9]{1,2}[a-z]{1}-[0-9]{1,3}"

    # # ex: student center
    # student_center = r"((stud)|(student center)|(student centre))"

    # special locations
    special = r"((stud)|(student center)|(student centre))|(walker)|" \
              r"(media lab)|(stata)|(next house)|(kresge)|(forbes)|" \
              r"(spxce)|(morss)|(lobby 10)"

    # Define our set of regex handlers
    regex_handlers = [
        (letter_number, letter_number_loc),
        (all_number, all_number_loc),
        (stata_room, stata_rooms),
        (stata_kitchen, stata_kitchens),
        (building_floor, building_floor_loc),
        (floor_builidng, floor_building_loc),
        (building_regex, just_building),
        (floor_regex, just_floor),
        (no_room, no_room_loc),
        (letter_after, letter_after_loc),
        (special, special_loc)
    ]

    # Get the proposed locations for each of the emails
    n = emails.shape[0]
    loc_dfs = [pd.DataFrame()] * n
    desc = "Processing file: " + email_file
    for i in tqdm(range(n), desc=desc):
        loc_dfs[i] = get_proposed_locations(emails[i], email_ids[i],
                                            regex_handlers)

    # Combine the results
    return pd.concat(loc_dfs, ignore_index=True, sort=True)


def make_email_id(email: np.ndarray) -> str:
    """
    Creates a hashed ID for each email so that we can track them for
    consistency
    """

    # Merge the email into a single string
    email = "".join(email)
    email = email.encode("utf8")
    return hashlib.sha1(email).hexdigest()


def remove_spam(emails: np.ndarray, email_ids: np.ndarray) -> tuple:
    """
    Removes spam emails from the data
    """

    # I noticed that mail is typically spam if it contains the phrase
    # "unsubscribe" or "special offer"; therefore if we can find this message
    #  in the email, we will remove it from the data
    n = emails.shape[0]
    regex = r"(unsubscribe)|(special offer)"
    idx = [i for i in range(n) if re.search(regex, "".join(emails[i]))]
    good_idx = np.setdiff1d(np.arange(n), idx)
    return emails[good_idx], email_ids[good_idx]


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate emails
    """

    # Duplicate emails fall into a couple of cases. First if they have
    # the same id, then it is very likely that it is a duplicate email
    # because the hash is created by using email text and the collision
    # probability is quite low. Second, if emails are sent by the same
    # author on the same day then it is likely that they are duplicates

    # Remove rows with duplicate IDs
    good_ids = ~df.duplicated(subset="id", keep="first")
    df = df.loc[good_ids, :]

    # Detect same author and day
    bad_idx = []
    df = df.sort_values(by="datetime")
    df_idx = df.index.values
    n = len(df_idx)
    for i in range(n - 1):
        if (df.loc[df_idx[i+1], "day"] == df.loc[df_idx[i], "day"]) and \
                (df.loc[df_idx[i+1], "author"] == df.loc[df_idx[i], "author"]):
            bad_idx.append(df_idx[i+1])

    good_idx = np.setdiff1d(df_idx, bad_idx)
    return df.loc[good_idx, :]


def csail_heuristic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implement the CSAIL heuristic where if we see the author has a csail
    domain, then the building is probably Stata
    """
    # Get all of the rows where the building is missing and the author
    # has a CSAIL domain
    idx = df.index[(df["building"].isna()) &
                   (df["author"].str.contains("csail.mit.edu"))]
    df.loc[idx, "building"] = 32
    return df


def clean_emails(email_file: str):

    # Parse the email file into a DataFrame
    if ".txt" in email_file:
        df = parse_txt.parse_txt(email_file)
    else:
        df = parse_mbox.parse_mbox(email_file)

    # To make life easier we are going to convert the weekday from
    # 1 --> mon, 2 --> tue, and so forth, and add a datetime column
    # so we can work with the email more easily
    # Update the weekday so that we map 1 --> mon, 2 --> tue, and so forth
    weekday_dict = {1: "mon", 2: "tue", 3: "wed", 4: "thu", 5: "fri",
                    6: "sat", 7: "sun"}
    df["weekday"] = df["weekday"].map(weekday_dict)

    # Create a datetime column -- this is useful for sorting chronologically
    # but is not as good for querying specific days or times
    df.loc[:, "datetime"] = pd.to_datetime(df.loc[:, ["day", "month", "year",
                                                      "hour", "minute"]])

    # Make the relevant columns all lowercase
    str_cols = ["email", "author"]
    df.loc[:, str_cols] = df.loc[:, str_cols].apply(lambda x: x.str.lower())

    # Change any instances of first, second, ... to 1st, 2nd, ...
    emails = change_ordinal_phrasing(df["email"].values.astype(str))

    # Remove the punctuation
    emails = np.char.replace(emails, ",", "")
    emails = np.char.replace(emails, ".", "")

    # Create unique IDs for each email
    email_ids = np.array([make_email_id(email) for email in emails])

    # Add the email_id column to the overall DataFrame so we can merge later
    df.loc[:, "id"] = email_ids

    # Remove spam from the data
    emails, email_ids = remove_spam(emails, email_ids)

    # Get the location data
    loc_df = get_location(emails, email_ids, email_file)

    # Merge the location data with the rest of the DataFrame
    df = df.merge(right=loc_df, how="inner", on="id")

    # To finish, remove duplicate records
    df = remove_duplicates(df)

    # Implement the CSAIL heuristic to catch any potential missing buildings
    df = csail_heuristic(df)
    return df
