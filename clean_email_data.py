import re
import numpy as np
import pandas as pd
import hashlib
import parse_txt
import parse_mbox
from tqdm import tqdm


# def remove_bad_from(data: np.ndarray) -> np.ndarray:
#     """
#     Removes the "bad From" from the email tag line so that there is only
#     one "FROM" tag line per email
#     """
#
#     # In almost every email there are two "FROM" tags listed in the email
#     # one contains "From email@email.com" and the other is
#     # "From: <email@email.com>"; we only need one of these values
#     # so we will get rid of the first one since every email contains the
#     # second tag, but not every email contains the first
#     bad_from = r"(from)\s.*@"
#     n = data.shape[0]
#     idx = [i for i in range(n) if re.search(bad_from, data[i])]
#     good_idx = np.setdiff1d(np.arange(n), idx)
#     return data[good_idx]
#
#
# def split_emails(data: np.ndarray) -> np.ndarray:
#     """
#     Splits the emails by going from the date: to date: tags which define
#     the start of a new email
#     """
#
#     # First identify the indices in the array where we can find the
#     # DATE tag
#     n = data.shape[0]
#     idx = [i for i in range(n) if re.search(r"(date:)", data[i])]
#
#     # Using those start point indices for the date, we need to get
#     # the indices which contain each unique email
#     n_idx = len(idx)
#     email_idx = [range(0, 0)] * n_idx
#     for i in range(n_idx):
#         if i != (n_idx - 1):
#             email_idx[i] = range(idx[i], idx[i+1])
#         else:
#             email_idx[i] = range(idx[i], n)
#
#     # Use the indices to separate the emails
#     return np.array([data[email_idx[i]] for i in range(n_idx)])
#
#
# def correct_bad_date_format(date_tags: np.ndarray, fmt_dict: dict,
#                             regex: str) -> np.ndarray:
#     """
#     Generic function that takes into the date, a correction dictionary,
#     and a regex, and puts things in the expected format
#     """
#     n = date_tags.shape[0]
#     idx = []
#     match = []
#     for i in range(n):
#         str_match = re.search(regex, date_tags[i])
#         if str_match:
#             idx.append(i)
#             match.append(str_match.group())
#
#     n_bad_instances = len(idx)
#     corrected_fmt = [fmt_dict[match[i]] for i in range(n_bad_instances)]
#     for i in range(n_bad_instances):
#         date_tags[idx[i]] = re.sub(regex, corrected_fmt[i], date_tags[i])
#
#     return date_tags
#
#
# def correct_bad_day(date_tags: np.ndarray) -> np.ndarray:
#     """
#     Corrects the issue where days are listed as their full name instead
#     of the abbreviation
#     """
#     full_days = r"(monday)|(tuesday)|(wednesday)|(thursday)|(friday)|" \
#                 r"(saturday)|(sunday)"
#
#     day_dict = {"monday": "mon", "tuesday": "tue", "wednesday": "wed",
#                 "thursday": "thu", "friday": "fri", "saturday": "sat",
#                 "sunday": "sun"}
#
#     # Get the corrected day format
#     return correct_bad_date_format(date_tags, day_dict, full_days)
#
#
# def correct_bad_month(date_tags: np.ndarray) -> np.ndarray:
#     """
#     Corrects the month to be in the expected (short) format
#     """
#
#     # First we're going to get all of the months into their shortened format
#     full_months = r"(january)|(february)|(march)|(april)|(june)|(july)|" \
#                   r"(august)|(september)|(october)|(november)|(december)"
#
#     month_dict = {"january": "jan", "february": "feb", "march": "mar",
#                   "april": "apr", "june": "jun", "july": "jul",
#                   "august": "aug", "september": "sep", "october": "oct",
#                   "november": "nov", "december": "dec"}
#
#     date_tags = correct_bad_date_format(date_tags, month_dict, full_months)
#     return date_tags
#
#
# def extract_day(date_tags: np.ndarray) -> np.ndarray:
#     """
#     Extracts the day from the date string
#     """
#
#     # Most often the day listed in either a full or short form
#     days = np.empty_like(date_tags)
#     short_days = r"((mon)|(tue)|(wed)|(thu)|(fri)|(sat)|(sun))"
#     full_days = r"((monday)|(tuesday)|(wednesday)|(thursday)|(friday)|" \
#                 r"(saturday)|(sunday))"
#     regex = re.compile(short_days + "|" + full_days)
#     n = date_tags.shape[0]
#     for i in range(n):
#         str_match = re.search(regex, date_tags[i])
#         if str_match:
#             days[i] = str_match.group()
#
#     return days
#
#
# def extract_date(date_tags: np.ndarray) -> dict:
#     """
#     Extracts the date from the DATE tag
#     """
#
#     # Grab the day, month, and year
#     day_regex = r"[0-9]{1,2}"
#     month_regex = r"((jan)|(feb)|(mar)|(apr)|(may)|(jun)|(jul)|(aug)|(sep)|" \
#                   r"(oct)|(nov)|(dec))"
#
#     year_regex = r"[0-9]{4}"
#
#     days = np.empty_like(date_tags)
#     months = np.empty_like(date_tags)
#     years = np.empty_like(date_tags)
#
#     n = date_tags.shape[0]
#     for i in range(n):
#         day_match = re.search(day_regex, date_tags[i])
#         month_match = re.search(month_regex, date_tags[i])
#         year_match = re.search(year_regex, date_tags[i])
#         if day_match:
#             days[i] = day_match.group()
#
#         if month_match:
#             months[i] = month_match.group()
#
#         if year_match:
#             years[i] = year_match.group()
#
#     # Convert the entries into integers
#     month_dict = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
#                   "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11,
#                   "dec": 12}
#
#     new_days = np.empty(shape=(n,), dtype=int)
#     new_months = np.empty(shape=(n,), dtype=int)
#     new_years = np.empty(shape=(n,), dtype=int)
#
#     for i in range(n):
#         # Try to convert the month into an integer
#         try:
#             new_months[i] = month_dict[months[i]]
#         except KeyError:
#             # This error would occur if we were not able to find a month in
#             # in the string and thus we won't do anything
#             new_months[i] = -999999
#
#         # Try to convert the day and year into an integer
#         try:
#             new_days[i] = int(days[i])
#         except ValueError:
#             # This error would occur if it was not able to find a number
#             # representation for the day (or year)
#             new_days[i] = -999999
#
#         try:
#             new_years[i] = int(years[i])
#         except ValueError:
#             new_years[i] = -999999
#
#     return {"day": new_days, "month": new_months, "year": new_years}
#
#
# def extract_time(date_tags: np.ndarray) -> dict:
#     """
#     Extracts the time from the date tag
#     """
#
#     # There are two standard formats for the time:
#     # HH:MM:SS and HH:MM AM/PM; I don't care about the second that the
#     # message was sent, so we will just grab the HH:MM portion of the time
#     # string
#     standard_fmt = r"[0-9]{2}:[0-9]{2}"
#     alt_fmt = r"[0-9]{1,2}:[0-9]{2}\s((am)|(pm))"
#
#     # Get all of the time strings
#     n = date_tags.shape[0]
#     hours = np.empty(shape=(n,), dtype=int)
#     minutes = np.empty(shape=(n,), dtype=int)
#     for i in range(n):
#         standard_match = re.search(standard_fmt, date_tags[i])
#         alt_match = re.search(alt_fmt, date_tags[i])
#         if standard_match:
#             tmp_str = standard_match.group()
#             hours[i] = int(tmp_str[0:2])
#             minutes[i] = int(tmp_str[3:5])
#         elif alt_match:
#             time = datetime.strptime(alt_match.group(), "%I:%M %p")
#             hours[i] = time.hour
#             minutes[i] = time.minute
#         else:
#             hours[i] = -999999
#             minutes[i] = -999999
#
#     return {"hour": hours, "minute": minutes}
#
#
# def get_date_time(emails: np.ndarray, email_ids: np.ndarray) -> pd.DataFrame:
#     """
#     Extracts the day, date, and time from the date tag in the email
#     """
#
#     # Grab the date tags from the emails
#     n = emails.shape[0]
#     date_tags = np.array([emails[i][0] for i in range(n)])
#
#     # Correct the full day format to be the expected format
#     date_tags = correct_bad_day(date_tags)
#
#     # Correct so we have the abbreviated month
#     date_tags = correct_bad_month(date_tags)
#
#     # Extract the vector of days
#     days = extract_day(date_tags)
#
#     # Extract the date information
#     dates = extract_date(date_tags)
#
#     # Get the times
#     times = extract_time(date_tags)
#
#     # Combine all of the data
#     df = pd.DataFrame({"weekday": days, "day": dates["day"],
#                        "month": dates["month"], "year": dates["year"],
#                        "hour": times["hour"], "minute": times["minute"],
#                        "id": email_ids})
#     return df


# def combine_email_body(email: np.ndarray) -> str:
#     """
#     Finds the subject line in an email and combines everything after
#     it into a single string
#     """
#     # Detect the start point for the subject line
#     n = email.shape[0]
#     idx = [i for i in range(n) if re.search(r"(to:)", email[i])]
#     if idx == list():
#         return ""
#
#     # Combine everything after it
#     email_body = email[(idx[0] + 1):]
#     email_body = " ".join(email_body)
#
#     # Remove all of the "\n" from the body
#     return re.sub(r"\\n", "", email_body)


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


# def student_center_loc(loc: str, email_id: str) -> pd.DataFrame:
#     """
#     Handles student center identification
#     """
#     return pd.DataFrame({"id": [email_id], "building": ["w20"],
#                          "floor": [np.nan], "room": [np.nan]})


def letter_after_loc(loc: str, email_id: str) -> pd.DataFrame:
    """
    ex: 14e-101
    """
    building = re.search(r"[0-9]{1,2}[a-z]", loc).group()
    loc = re.sub(r"[0-9]{1,2}[a-z]-", "", loc)
    return pd.DataFrame({"id": [email_id], "building": [building],
                         "floor": [loc[0]], "room": [loc]})


# def walker_loc(loc: str, email_id: str) -> pd.DataFrame:
#     """
#     Walker memorial
#     """
#     return pd.DataFrame({"id": [email_id], "building": ["50"],
#                          "floor": [np.nan], "room": [np.nan]})
#
#
# def media_lab_loc(loc: str, email_id: str) -> pd.DataFrame:
#     """
#     Media lab
#     """
#     return pd.DataFrame({"id": [email_id], "building": ["e14"],
#                          "floor": [np.nan], "room": [np.nan]})
#
#
# def stata_loc(loc: str, email_id: str) -> pd.DataFrame:
#     """
#     Stata Center
#     """
#     return pd.DataFrame({"id": [email_id], "building": ["32"],
#                          "floor": [np.nan], "room": [np.nan]})


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

    # # Concatenate the results
    # df = pd.concat(loc_dfs, ignore_index=True)
    #
    # # The proposed building is the least likely to be troublesome so we
    # # will just take the first entry in the data
    # building = df.loc[0, "building"]
    #
    # # For the floor and room, since they are integers, we can take the
    # # "mean" and get a result
    # floor = df.groupby("id").mean()["floor"].values
    # room = df.groupby("id").mean()["room"].values
    # return pd.DataFrame({"id": [email_id], "building": [building],
    #                      "floor": floor, "room": room})
    return loc_df


# def detect_location(emails: np.ndarray) -> dict:
#     """
#     Detects the location from the email
#     """
#
#     # # There are significantly many more ways to express the location
#     # # some of the most common ways include: e62-105, 1-207, building 4,
#     # # 2nd floor, etc.; I will encode the most common ways and hard-code
#     # # some rules such as the "stud" --> "Student Center"
#     # letter_fmt = r"[a-z]{1}[0-9]{1,2}-[0-9]{1,3}"
#     # std_fmt = r"[0-9]{1,2}-[0-9]{1,3}"
#     # building_fmt = r"((building)|(bldg))\s[0-9]{1,2}"
#     # floor_fmt = r"((1st)|(2nd)|(3rd)|(4th)|(5th)|(6th)|(7th)|(8th)|(9th))\s" \
#     #             r"floor"
#     # no_room_fmt = r"[a-z][0-9]{1,2}"
#     # letter_after = r"[0-9]{1,2}[a-z]-[0-9]{1,3}"
#     # student_center = r"((stud)|(student center)|(student centre))"
#     # spxce = r"spxce"
#     # stata = r"stata"
#     # walker = r"walker"
#     # next_house = r"next house"
#     # kresge = r"kresge"
#     # forbes_cafe = r"forbes"
#     # morss = r"morss"
#     # media_lab = r"media lab"
#     # lobby_10 = r"(lobby 10)"
#     # killian_hall = r"(killian)"
#
#     # ex: e62-107
#     letter_number = r"[a-z]{1}[0-9]{1,2}-[0-9]{1,3}"
#
#     # ex: 1-207
#     all_number = r"[0-9]{1,2}-[0-9]{1,3}"
#
#     # ex: building 4
#     building_regex = r"((building)|(bldg))\s[0-9]{1,2}"
#
#     # ex: 1st floor
#     floor_regex = r"[0-9]{1}((st)|(nd)|(rd)|(th))\sfloor"
#
#     # ex: w11
#     no_room = r"[a-z]{1}[0-9]{1,2}"
#
#     # ex: 14e-101
#     letter_after = r"[0-9]{1,2}[a-z]{1}-[0-9]{1,3}"
#
#     # ex: student center
#     student_center = r"((stud)|(student center)|(student centre))"
#
#     # Define our set of regex handlers
#     regex_handlers = [
#         (letter_number, letter_number_loc),
#         (all_number, all_number_loc),
#         (building_regex, just_building),
#         (floor_regex, just_floor),
#         (no_room, no_room_loc),
#         (student_center, student_center_loc),
#         (letter_after, letter_after_loc)
#     ]


def get_location(emails: np.ndarray, email_ids: np.ndarray,
                 email_file: str) -> pd.DataFrame:
    """
    Gets the location from the emails
    """

    # Get the subject line and bodies for each emails
    n = emails.shape[0]
    # email_bodies = np.array([combine_email_body(emails[i]) for i in range(n)])

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

    # # walker memorial
    # walker = r"(walker)"
    #
    # # media lab
    # media_lab = r"(media lab)"
    #
    # # Stata Center
    # stata = r"(stata)"

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
        # (student_center, student_center_loc),
        (letter_after, letter_after_loc),
        # (walker, walker_loc),
        # (media_lab, media_lab_loc),
        # (stata, stata_loc)
        (special, special_loc)
    ]

    # Get the proposed locations for each of the emails
    loc_dfs = [pd.DataFrame()] * n
    desc = "Processing file: " + email_file
    for i in tqdm(range(n), desc=desc):
        # loc_dfs[i] = get_proposed_locations(email_bodies[i], email_ids[i],
        #                                     regex_handlers)
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


# def get_from_tag(email: np.ndarray) -> str:
#     """
#     Gets the from tag from a given email
#     """
#     from_regex = r"from:.*<.*>$"
#     n = len(email)
#     from_tag = ""
#     for i in range(n):
#         if re.search(from_regex, email[i]):
#             from_tag = email[i]
#             break
#     return from_tag


# def get_author(emails: np.ndarray, email_ids: np.ndarray) -> pd.DataFrame:
#     """
#     Gets the authors from the emails
#     """
#
#     # Get the lines that correspond to the FROM tag
#     n = emails.shape[0]
#     from_tags = list(map(get_from_tag, emails))
#
#     # Grab the author
#     author_regex = r"<.*>"
#     authors = [""] * n
#     for i in range(n):
#         author_search = re.search(author_regex, from_tags[i])
#         if author_search:
#             authors[i] = re.sub(r"[<>]", "", author_search.group())
#         else:
#             authors[i] = np.nan
#
#     # Create the final DataFrame
#     return pd.DataFrame({"id": email_ids, "author": authors})


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
    bad_ids = ~df.duplicated(subset="id", keep="first")
    df = df.loc[bad_ids, :]

    # Detect same author and day
    idx = []
    n = df.shape[0]
    df = df.sort_values(by="datetime")
    for i in range(n - 1):
        if (df.loc[i+1, "day"] == df.loc[i, "day"]) and \
                (df.loc[i+1, "author"] == df.loc[i, "author"]):

            idx.append(i+1)

    idx = np.setdiff1d(np.arange(n), idx)
    return df.loc[idx, :]


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

    # # Read in the .txt file
    # with open(args["email_file"], encoding="utf8") as f:
    #     data = f.readlines()
    # data = np.array(data)
    #
    # # Make everything lowercase so we do not have to worry about
    # # punctuation
    # data = np.char.lower(data)

    # # Remove the bad FROM tag line from the emails
    # data = remove_bad_from(data)

    # Change any instances of first, second, ... to 1st, 2nd, ...
    emails = change_ordinal_phrasing(df["email"].values.astype(str))

    # Remove the punctuation
    emails = np.char.replace(emails, ",", "")
    emails = np.char.replace(emails, ".", "")

    # # Split the data into the discrete emails
    # emails = split_emails(data)

    # Create unique IDs for each email
    email_ids = np.array([make_email_id(email) for email in emails])

    # Add the email_id column to the overall DataFrame so we can merge later
    df.loc[:, "id"] = email_ids

    # Remove spam from the data
    emails, email_ids = remove_spam(emails, email_ids)

    # # Get the date data
    # date_df = get_date_time(emails, email_ids)

    # Get the location data
    loc_df = get_location(emails, email_ids, email_file)

    # # Get the email authors
    # author_df = get_author(emails, email_ids)

    # # Join the DataFrames
    # df = date_df.merge(loc_df, on="id").merge(author_df, on="id")

    # Merge the location data with the rest of the DataFrame
    df = df.merge(right=loc_df, how="inner", on="id")

    # To finish, remove duplicate records
    df = remove_duplicates(df)

    # Implement the CSAIL heuristic to catch any potential missing buildings
    df = csail_heuristic(df)
    return df


if __name__ == "__main__":
    clean_emails("c:/users/zqb0731/downloads/2004.txt")
