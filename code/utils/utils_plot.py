import os
import sys
import warnings
import time
import json
from natsort import natsorted
from pathlib import PureWindowsPath, PurePosixPath
import pickle
import numpy as np
import xarray as xr
import pandas as pd
import math 
import holoviews as hv
from holoviews.operation.datashader import datashade, regrid
from holoviews.util import Dynamic
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.pyplot import figure
from matplotlib.patches import Patch, Rectangle
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import colorsys
import seaborn as sns
import cmasher as cmr
from IPython.display import display, HTML
from IPython.display import Audio
from itertools import combinations

import scipy.stats as stats

def set_pub_style():
    """Sets matplotlib params."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'sans-serif'],
        'font.size': 7,                # Base font size (usually 6-8pt)
        'axes.labelsize': 7,           # Axis labels
        'axes.titlesize': 7,           # Title
        'xtick.labelsize': 5.5,
        'ytick.labelsize': 5.5,
        'legend.fontsize': 5.5,
        'axes.linewidth': 0.75,        # Thinner spines (0.5 - 0.75 is ideal)
        'xtick.major.width': 0.75,
        'ytick.major.width': 0.75,
        'axes.spines.top': False,      # Remove box
        'axes.spines.right': False,
        'pdf.fonttype': 42,            # Ensures text is editable in Illustrator
        'ps.fonttype': 42,
        'figure.dpi': 300,             # High res for preview
        'xtick.major.size': 2,
        'ytick.major.size': 2,       
        'xtick.major.pad': 1,
        'ytick.major.pad': 1,
        
    })
    
def get_asterisks(p_val):
    """Converts p-values to standard significance asterisks."""
    if p_val < 0.001: return '***'
    elif p_val < 0.01: return '**'
    elif p_val < 0.05: return '*'
    else: return 'ns'
    
def save_metadata_json(metadata_dict, output_path, title):
    #os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, f"{title.replace(' ', '_')}_caption_data.json")
    with open(file_path, 'w') as f:
        json.dump(metadata_dict, f, indent=4)

def lighten_color(color, amount=0.6):
    """
    Blends a base color with white to create a true solid light version, 
    preventing the alpha-stacking issue in dense scatter plots.
    amount: 0 is white, 1 is the original color.
    """
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mcolors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    
def add_significance_bar(ax, x1, x2, y_max, text, color='black'):
    """
    Draws a flat significance line and returns the 'roof'
    so subsequent brackets stack flawlessly.
    """
    if text == 'ns' or not text:
        return y_max        
    ymin, ymax_ax = ax.get_ylim()
    axis_range = ymax_ax - ymin  # 0.06, 0.04, 0.06
    line_gap = axis_range * 0.04       # 1. Gap from the data/previous layer to the new line
    text_offset = axis_range * 0.02    # 2. Gap between the line and the bottom of the star
    star_height = axis_range * 0.04

    y_line = y_max + line_gap
    # Draw flat horizontal line 
    ax.plot([x1, x2], [y_line, y_line], lw=0.5, c='k')   
    ax.text((x1+x2)*0.5, y_line + text_offset, text, ha='center', va='center', color='k', fontsize=5)
    return y_line + text_offset + star_height
    
def add_stat_annotation_two_sided(ax, data1, data2, x1, x2, y_max, ttest=0, paired=0):
    """
    Calculates significance and draws a flat horizontal line.
    Defaults to Mann-Whitney (ttest=0, paired=0) for safer bounded-data statistics.
    """
    # 1. Choose Test
    if (ttest==1) and (paired==0): 
        # Welch's t-test (equal_var=False)
        stat, p = stats.ttest_ind(data1, data2, alternative='two-sided') 
    elif (ttest==1) and (paired==1): 
        stat, p = stats.ttest_rel(data1, data2, alternative='two-sided')
    elif (ttest==0) and (paired==0): 
        stat, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
    elif (ttest==0) and (paired==1): 
        stat, p = stats.wilcoxon(data1, data2, alternative='two-sided')
    else:
        raise SystemExit("Stop right there! Check the test methods!")
        
    star = get_asterisks(p)
    top = add_significance_bar(ax, x1, x2, y_max, star, color='black')   
    return top
         
