"""
Utils
=====

Imports all required packages and sets global variables for file directories.
"""

# GENERAL
import os
import math
import warnings
import pandas as pd
import geopandas as gpd
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from copy import deepcopy
from itertools import product, chain, combinations, islice
from datetime import date, datetime
import time
import pickle as pkl
import json
import geojson
from tqdm import tqdm
from multiprocess import pool
# OPTIMIZATION
import gurobipy as gp
from gurobipy import GRB
# NETWORKS
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, Point, Polygon, MultiLineString, shape
from shapely.ops import linemerge
from shapely.errors import ShapelyDeprecationWarning
# PLOTTING
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly
import plotly.graph_objs
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from plotly.offline import iplot


warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

sns.set_theme(palette='Set2')

pio.renderers.default = "browser"
tqdm.pandas()

# TODO: for publicly released version:
#  - have local directories for key input data; store template files in these as well as sample data files for example
#  - create example/test files and sample results so that users can run and compare on their computer
#       - allow scenario code to be a name that is provided by user, otherwise, generate;
#         name all relevant input/output files with the provided/generated name
#  - module documentation; deocumentation (at top) for key methods
#  - remove large, old commented out code blocks

# GLOBAL PATHS
BASE_DIR = os.path.dirname(__file__)
# input directory
INPUT_DIR = os.path.join(BASE_DIR, 'input')
# input subdirectories
FLOW_DIR = os.path.join(INPUT_DIR, 'flow')
COMM_DIR = os.path.join(INPUT_DIR, 'commodity')
GEN_DIR = os.path.join(INPUT_DIR, 'general')
LCA_DIR = os.path.join(INPUT_DIR, 'LCA')
NX_DIR = os.path.join(INPUT_DIR, 'networks')
TEA_DIR = os.path.join(INPUT_DIR, 'TEA')
RR_DIR = os.path.join(INPUT_DIR, 'railroad')
SCENARIO_DIR = os.path.join(INPUT_DIR, 'scenario')
FACILITY_DIR = os.path.join(INPUT_DIR, 'facility')
# output directory
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
# output subdirectories
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
MET_O_DIR = os.path.join(OUTPUT_DIR, 'metrics')
NODE_O_DIR = os.path.join(OUTPUT_DIR, 'nodes')
EDGE_O_DIR = os.path.join(OUTPUT_DIR, 'edges')


# GLOBAL VARS
# filename shortcuts
FILES = {2017: 'WB2017_900_Unmasked.csv', 2018: 'WB2018_900_Unmasked.csv', 2019: 'WB2019_913_Unmasked.csv'}
KM2MI = 0.62137119  # miles / km
