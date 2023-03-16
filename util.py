"""
Utils
=====

Imports all required packages and sets global variables for file directories.
"""

# GENERAL
import os
import os.path
import math
import warnings
import pandas as pd
import geopandas as gpd
import numpy as np
from copy import deepcopy
from itertools import product
from datetime import date
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
import logging
# DASH
from dash import dash, dcc, html, Input, Output, State
from flask import request, jsonify
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from flask_caching import Cache
import dash_loading_spinners as dls
import csv
import dash_auth
import traceback
from werkzeug.middleware.proxy_fix import ProxyFix # for nginx

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

sns.set_theme(palette='Set2')

pio.renderers.default = "browser"
tqdm.pandas()

# GLOBAL PATHS
# only need to alter <PERSONAL_PATH>, can take this as an input or search for it?
PERSONAL_PATH = '/Users/adrianhz/Library/CloudStorage/OneDrive-NorthwesternUniversity/Adrian Hernandez/'
# input directory
INPUT_DIR = os.path.join(PERSONAL_PATH, 'ARPA-E LOCOMOTIVES/Alpha Framework/Input Data')
# input subdirectories
CCWS_DIR = os.path.join(INPUT_DIR, 'CCWS')
COMM_DIR = os.path.join(INPUT_DIR, 'Commodity')
DEP_TAB_DIR = os.path.join(INPUT_DIR, 'Deployment Tables')
GEN_DIR = os.path.join(INPUT_DIR, 'General')
LCA_DIR = os.path.join(INPUT_DIR, 'LCA')
NX_DIR = os.path.join(INPUT_DIR, 'Networks')
TEA_DIR = os.path.join(INPUT_DIR, 'TEA')
RR_DIR = os.path.join(INPUT_DIR, 'Railroad')
SCENARIO_DIR = os.path.join(INPUT_DIR, 'Scenario')
# output directory
OUTPUT_DIR = os.path.join(PERSONAL_PATH, 'ARPA-E LOCOMOTIVES/Alpha Framework/Output Data')
# output subdirectories
FIG_O_DIR = os.path.join(OUTPUT_DIR, 'Figures')
DEP_TAB_O_DIR = os.path.join(OUTPUT_DIR, 'Deployment')
MET_O_DIR = os.path.join(OUTPUT_DIR, 'Metrics')
NODE_O_DIR = os.path.join(OUTPUT_DIR, 'Nodes')
EDGE_O_DIR = os.path.join(OUTPUT_DIR, 'Edges')

# GLOBAL VARS
# filename shortcuts
FILES = {2017: 'WB2017_900_Unmasked.csv', 2018: 'WB2018_900_Unmasked.csv', 2019: 'WB2019_913_Unmasked.csv'}
KM2MI = 0.62137119  # miles / km
