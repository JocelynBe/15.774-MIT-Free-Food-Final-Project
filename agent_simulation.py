import random
import numpy as np
import pandas as pd
from typing import Union


class BaseAgent:
    def __init__(self, name: str, agent_type: str, nnodes: int, seed=17):
        """
        name: Name of the agent
        agent_type: Type of agent (random or probability)
        nnodes: Number of nodes
        seed: Random seed
        """

        self.rng = random.Random(seed)
        self.name = name
        self.agent_type = agent_type
        self.nnodes = nnodes
        self.position = self.rng.randint(0, self.nnodes - 1)

        # Either they are browsing, moving, or eating their food
        self.state = "browsing"
        self.destination = None  # if state = moving, then destination is the nodeId the agent is going to
        self.departure_time = None  # if state = moving, departure_time is the time at which the agent left position

        # Amount of time the agent has been browsing for food
        self.browse_time = 0

        # Amount of time agent has been eating food
        self.eat_time = 0

    def make_decision(self, time: pd.datetime, browse_lim: int, eat_lim: int,
                      delta: int, node_dict: dict, edges: np.ndarray,
                      prob_df: pd.DataFrame):
        """
        Defines the decision function for various types of agents
        """

        # If the agent is not eating then they need to make a decision
        # based on their type
        if self.state != "eating":
            self.get_decision(time, browse_lim, delta, node_dict, edges,
                              prob_df)

        # If an agent is eating we have to check if they've eaten for long
        # enough; if they have then the agent needs to make a decision
        # according to their type
        else:
            if self.eat_time < eat_lim:
                self.eat_time += delta
            else:
                # We've reached the eating time limit and thus need to look
                # for more food
                self.eat_time = 0
                self.browse_time = browse_lim
                self.get_decision(time, browse_lim, delta, node_dict, edges,
                                  prob_df)

    def get_decision(self, time: pd.datetime, browse_lim: int, delta: int,
                     node_dict: dict, edges: np.ndarray, prob_df: pd.DataFrame):
        """
        Gets the decision based on the agent type
        """
        if self.agent_type == "random":
            self.rand_decision(time, browse_lim, delta)
        elif self.agent_type == "stata":
            stata_id = node_dict["32"]
            self.stata_decision(time, stata_id)
        elif self.agent_type == "prob":
            self.prob_decision(time, browse_lim, delta, edges, prob_df)

    def rand_decision(self, time: pd.datetime, browse_lim: int, delta: int):
        """
        Defines the decision criteria for an agent that acts randomly
        """

        # A random agent will continue searching for free food in the
        # current location if he is below the browsing limit; otherwise,
        # he will randomly select his next location
        if self.browse_time < browse_lim:
            # Continue searching and update the number of steps he's been
            # searching
            self.browse_time += delta
        else:
            # We have reached the browse limit, we need to select a new
            # destination to look for food
            while True:
                dest = self.rng.randint(0, self.nnodes - 1)
                if dest != self.position:
                    break

            self.move(time, dest)

    def stata_decision(self, time: pd.datetime, stata_id: int):
        """
        Defines the decision function for the Stata bot -- an agent who
        will only look for food in Stata
        """

        # If an agent is browsing and in Stata then there is nothing to do
        if self.state == "browsing" and self.position == stata_id:
            pass

        # If the agent is eating and the code has reached this point,
        # this implies that the agent has eaten for one hour and thus
        # needs to start looking for more food in Stata
        elif self.state == "eating":
            self.position = stata_id
            self.state = "browsing"
            self.destination = None
            self.departure_time = None

        # The agent could also not be in Stata in which case they need
        # to head towards there
        elif self.position != stata_id:
            self.move(time, stata_id)

    def prob_decision(self, time: pd.datetime, browse_lim: int, delta: int,
                      edges: np.ndarray, prob_df: pd.DataFrame):
        """
        Implements the decision function for the probability maximizing agent
        """
        # If the agent has not reached the browse limit, then it can
        # continue to search for food
        if self.browse_time < browse_lim:
            self.browse_time += delta
        else:
            # We have the reached the browsing limit and thus need to make
            # to select the next destination to look for food; we will
            # do this by solving pi* = argmax_{j \in N \ i} p_{jt} / t_{ij}
            # where i is the current node the agent is at, N is the set of
            # nodes in the graph, p_{jt} is the probability of finding food
            # at node j at time t and t_{ij} is the time to travel from node
            # i to j

            # Get the day and hour from the provided timestamp
            weekday = time.isoweekday()
            hour = time.hour

            # Subset the DataFrame to only correspond to the particular
            # weekday and hour
            sub_df = prob_df.loc[
                (prob_df["hour"] == hour) & (prob_df["weekday"] == weekday),
                "probability"
            ].to_frame()

            sub_df["building_id"] = np.arange(sub_df.shape[0])
            sub_df.reset_index(drop=True, inplace=True)

            # Get all of the t_{ij} values for i != j
            pos = self.position
            other_nodes = np.setdiff1d(np.arange(edges.shape[0]), pos)
            time_vect = np.empty(shape=(self.nnodes,))
            time_vect[pos] = 1e6
            time_vect[other_nodes] = np.array([edges[pos, node]
                                               for node in other_nodes])

            # Get the p_{jt} values for j != self.position (we fix the
            # current location to have zero mass because this makes finding
            # the argmax index easier)
            prob_vect = np.empty(shape=(self.nnodes,))
            prob_vect[pos] = 0.
            prob_vect[other_nodes] = sub_df.loc[
                sub_df["building_id"].isin(other_nodes), "probability"
            ]

            # Compute the element-wise division of p_{jt} / t_{ij} and
            # get the argmax
            edge_weights = prob_vect / time_vect
            dest = edge_weights.argmax()
            self.move(time, dest)

    def move(self, time: pd.datetime, destination: int):
        """
        Tells the agent to move to a provided destination in the graph
        """
        self.state = "moving"
        self.destination = destination
        self.departure_time = time
        self.browse_time = 0

    def check_position(self, time: pd.datetime, edges: np.ndarray):
        """
        Check the agent's position at a given time
        """
        pos = self.position
        dest = self.destination

        # Check if the agent has arrived at their desination
        if (time - self.departure_time).seconds >= edges[pos, dest]:
            self.position = self.destination
            self.state = "browsing"
            self.destination = None
            self.departure_time = None
        else:
            pass

    def get_position(self, time: pd.datetime, edges: np.ndarray):
        """
        Get the agent's position at a given time
        """
        # If the agent is browsing or is eating then their location must
        # be the building thier currently in
        if self.state == "browsing" or self.state == "eating":
            return self.position
        # We have to check if the agent is still transiting or has arrived
        # at their destination
        else:
            self.check_position(time, edges)
            if self.state == "browsing":
                return self.position
            else:
                travel_time = edges[self.position, self.destination]
                s = (time - self.departure_time).seconds / travel_time
                return self.position, self.destination, s


class Simulation:
    def __init__(self, nodes: dict, edges: np.ndarray, node_dict: dict,
                 prob_df: pd.DataFrame, freefood: list,
                 first_timestamp: pd.datetime, browse_lim=5,
                 delta=1, max_step=100, lifetime_ff=60, eat_lim=60,
                 verbose=True):
        """
        nodes: dict( nodeId : (x,y) )
        edges: T_{ij} matrix for all nodes in the graph
        node_dict: Dictionary containing the mapping of building => nodeID
        freefood: [(pd.Timestamp, nodeId),...]
        browse_lim: Number of minutes the agent is allowed to browse
        delta: minutes between consecutive steps
        max_step: max number of steps -> total simulation delta*max_step
        lifetime_ff: Duration free food is available in minutes
        eat_lim: Time it takes to eat if the agent found free food
        """

        self.agents = []
        # self.step = 0
        self.delta = delta
        self.max_step = max_step
        self.browse_lim = browse_lim
        self.first_timestamp = first_timestamp
        self.time = self.first_timestamp - pd.Timedelta(minutes=self.delta)
        self.lifetime_ff = int(lifetime_ff / delta)
        self.freefood_list = freefood
        self.freefood = {}
        self.nodes = nodes
        self.edges = edges
        self.node_dict = node_dict
        self.prob_df = prob_df
        self.scores = {}
        self.verbose = verbose
        self.log = pd.DataFrame()
        self.eat_lim = eat_lim

    def parse_freefood(self):
        # Get all of the unique timestamps from the log
        uniq_timestamps = self.log["datetime"].unique()
        uniq_timestamps = [pd.Timestamp(val) for val in uniq_timestamps]
        self.freefood = {val: set() for val in uniq_timestamps}

        # Go through all of the unique timestamp entries, generate the range
        # of values that the free food is around and add this to the
        # dictionary
        for ff in self.freefood_list:
            # Get the range of timestamp values for the given free food
            # instance
            t, node_id = ff
            time_seq = np.arange(start=0, stop=self.lifetime_ff, step=self.delta)
            time_vals = [t + pd.Timedelta(minutes=val) for val in time_seq]

            for timestamp in time_vals:
                self.freefood[timestamp].add(node_id)

    def add_agent(self, agent: BaseAgent):
        self.agents.append(agent)
        self.scores[agent.name] = 0
        if self.verbose:
            print("Agent " + agent.name + " successfully added")

    def predefine_log(self):
        """
        Updates the number of entries in the log based on the number of steps
        and agents in the simulation
        """
        n = len(self.agents) * self.max_step
        time_lim = self.max_step * self.delta
        self.log = pd.DataFrame(index=np.arange(n),
                                columns=["agent", "datetime", "x", "y", "score"])

        # Add all timestamps in the simulation starting at the first_datetime
        time_seq = np.arange(start=0, stop=time_lim, step=self.delta)
        timestamps = [self.first_timestamp + pd.Timedelta(minutes=val)
                      for val in time_seq]

        self.log["datetime"] = np.repeat(timestamps, len(self.agents))

        # Add the rows which define the various agents
        agent_names = [agent.name for agent in self.agents]
        self.log["agent"] = np.tile(agent_names, time_lim)

    def infer_latlong(self, source: int, dest: int, s: float):
        """
        Infers the latitude and longitude of a given position for an
        agent that is on the move assuming that we are on a plane
        versus a "great sphere"
        """

        # Let source = x1 and dest = x2, then new_point = (1-s)*x1 + x*x2
        # (i.e. a convex combination of the two points)
        source_pt = np.array(self.nodes[source])
        dest_pt = np.array(self.nodes[dest])
        med_point = ((1 - s) * source_pt) + (s * dest_pt)
        return med_point[0], med_point[1]

    def update_log(self, agent: BaseAgent, time: pd.datetime,
                   position: Union[int, tuple]):
        """
        Updates the log given a particular agent
        """
        # Get the specific location for the agent at the given step
        if agent.state == "browsing" or agent.state == "eating":
            x, y = self.nodes[position]
        else:
            source, dest, s = position
            x, y = self.infer_latlong(source, dest, s)

        # Get the rows that correspond to the given agent and time step
        vals = [x, y, self.scores[agent.name]]
        idx = self.log.index[
            (self.log["agent"] == agent.name) & (self.log["datetime"] == time)
        ].values

        self.log.loc[idx, ["x", "y", "score"]] = vals

    def next(self):

        # Update the simulation time
        self.time += pd.Timedelta(minutes=self.delta)
        for agent in self.agents:

            # The position will either be a nodeId or a a tuple (i,j,s) where
            # i,j are the nodes ID the agent is going from i to j and s
            # \in [0,1] the position of the agent between i and j
            # s = 0 <-> agent in i || s = 1 <-> agent in j
            position = agent.get_position(self.time, self.edges)

            # Update the log for the given agent
            self.update_log(agent, self.time, position)

            if position in self.nodes:
                if position in self.freefood[self.time] and agent.state != "eating":
                    # Update teh score for the agent
                    self.scores[agent.name] += 1

                    # Change the state to eating and make the appropriate
                    # decision (they will continue eating)
                    agent.state = "eating"
                    agent.make_decision(
                        self.time, self.browse_lim, self.eat_lim, self.delta,
                        self.node_dict, self.edges, self.prob_df
                    )

                    if self.verbose:
                        print("Good job " + agent.name + "!")
                # If the agent did not find food during this iteration or they
                # are eating we have to call the decision function regarding
                # what we should do next
                else:
                    agent.make_decision(
                        self.time, self.browse_lim, self.eat_lim, self.delta,
                        self.node_dict, self.edges, self.prob_df
                    )

        return self.log
