# 15.774 MIT Free Food Final Project Code
This repository contains all of the code Project X used to run our MIT free food project. In particular we have provided both the
data we used for the project as well as the scripts utilized to run the experiments.

## Data
In the *project_data* folder we provide the final parsed emails, the buildings we utilized for the graph simulations, the
spatio-temporal distribution of free food across the MIT campus for the selected buildings, and the time it takes to travel
from one building to another according to Google Maps. Anyone who is interested in verifying or extending our work can feel
free to use this data.

## Scripts
We also provide all of the scripts we used to run the experiments. To parse email data, you will need to use the *parse_all_data.py*. 
Specifically you will have to provide a directory containing all of the email files you want to parse. We assume that these files will
either be *.mbox* or *.txt* files since those were the ones we used for running our experiments. 

Once you have the free food, to run the simulation you can use the script *run_simulation.py*. This scripts assumes that you pass
a working directory which contains the time matrix to go from building to building, the clean emails which were parsed in
*parse_all_data.py*, the latitude-longitude coordinates for the buildings in the simulation, and the spatio-temporal distribution of
free food. In the script we hard-coded for our specific file names, but the code can be easily generalized to work regardless
of file name. As a warning this simulation might take a while. Specifically, for simulating 6 months at MIT, it took about two hours
to run.
