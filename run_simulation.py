import agent_simulation
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import os


def get_edge_data(time_path: str) -> tuple:
    """
    Gets the edge data used for the simulation
    """

    # Read in the data
    time_mat = pd.read_csv(time_path, index_col=0)

    # Get the list of buildings we will use for the simulation
    buildings = time_mat.index.values
    return time_mat.values, buildings


def parse_freefood(food_path: str, node_path: str, buildings: np.ndarray,
                   year: int, months: tuple) -> tuple:
    """
    Parses the free food data to be in the expected format for the simulation
    """
    # Get the free food data
    df = pd.read_csv(food_path)

    # Build a map from the building number to its ID
    node_df = pd.read_csv(node_path)
    node_df["node_num"] = np.arange(node_df.shape[0])
    nodes = dict(zip(node_df.node_num, zip(node_df.lat, node_df.lon)))
    node_dict = dict(zip(node_df.building, node_df.node_num))


    # Subset the DataFrame according to the year we're interested in
    df = df.loc[((df["year"] == year) & (df["month"].isin(months))), :]

    # Grab the relevant values and sort the data chronologically so that we
    # can properly access the food elements
    df = df.loc[(df["new_building"].isin(buildings)),
                ["weekday", "datetime", "new_building"]]

    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values(by="datetime", inplace=True)

    # Get the first datetime index
    df.reset_index(drop=True, inplace=True)
    first_datetime = df.loc[0, "datetime"]

    # Grab all of the relevant timestamps for the free food format
    building_ids = [node_dict[val] for val in df["new_building"]]
    return list(zip(df["datetime"], building_ids)), first_datetime, nodes, \
           node_dict


def run_simulation(time_path: str, food_path: str, node_path: str,
                   prob_path: str, agent_type: str, year=2017,
                   months=(9, 10, 11, 12)) -> pd.DataFrame:
    """
    Runs the simulation
    """

    # Get the edge data
    time_mat, buildings = get_edge_data(time_path)

    # Get expected format for the free food for the simulation
    freefood, first_datetime, nodes, node_dict = parse_freefood(
        food_path, node_path, buildings, year, months
    )

    # Get the probability data
    prob_df = pd.read_csv(prob_path)
    prob_df.columns = map(str.lower, prob_df.columns)

    # Convert the date from MON => 1, ..., SUN => 7
    day_dict = {"MON": 1, "TUE": 2, "WED": 3, "THU": 4, "FRI": 5, "SAT": 6,
                "SUN": 7}
    prob_df["weekday"] = prob_df["weekday"].apply(lambda x: day_dict[x])

    # Subset the buildings to only contain the ones we care about
    prob_df = prob_df.loc[prob_df["building_number"].isin(buildings)]

    # Change all hours that are listed as 24 to 0 since that is what
    # pandas expected for the hour stamp
    prob_df.loc[prob_df["hour"] == 24, "hour"] = 0

    # Infer the total number of minutes for the provided number of months
    max_step = 60 * 24 * 30 * len(months)

    # Run the simulation
    sim = agent_simulation.Simulation(
        nodes=nodes, edges=time_mat, node_dict=node_dict, freefood=freefood,
        prob_df=prob_df, first_timestamp=first_datetime, delta=1,
        max_step=max_step
    )

    # Add the agent to the simulation
    if agent_type == "random":
        agent = agent_simulation.BaseAgent("random", agent_type, len(nodes))
    elif agent_type == "stata":
        agent = agent_simulation.BaseAgent("stata", agent_type, len(nodes))
    else:
        agent = agent_simulation.BaseAgent("prob", agent_type, len(nodes))

    sim.add_agent(agent)
    sim.predefine_log()
    sim.parse_freefood()

    # For 100 steps, run the simulation
    for _ in tqdm(range(max_step)):
        sim.next()

    # Return the log from the simulation
    return sim.log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wd", help="Working directory of the script",
                        type=str)
    parser.add_argument("--agent_type", help="Agent type for the simulation",
                       type=str)
    args = parser.parse_args()

    # The data sources have an expected name, so we will get them by
    # combining the working directory with their file names
    time_path = os.path.join(args.wd, "time_mat.csv")
    food_path = os.path.join(args.wd, "clean_emails_liz_1.csv")
    node_path = os.path.join(args.wd, "buildings_loc.csv")
    prob_path = os.path.join(args.wd, "Prob_top20buildings_dayhr.csv")

    # Run the simulation
    df = run_simulation(time_path, food_path, node_path, prob_path,
                        args.agent_type)

    # Save the log to disk
    save_path = os.path.join(args.wd, "sim_res_" + args.agent_type + ".csv")
    df.to_csv(save_path, index=False)


if __name__ == '__main__':
    main()
