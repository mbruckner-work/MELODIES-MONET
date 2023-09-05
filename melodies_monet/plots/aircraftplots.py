
# Copyright (C) 2022 National Center for Atmospheric Research and National Oceanic and Atmospheric Administration
# SPDX-License-Identifier: Apache-2.0
#
#Code to create plots for surface observations

import os
import monetio as mio
import monet as monet
import seaborn as sns
from monet.util.tools import calc_8hr_rolling_max, calc_24hr_ave
import xarray as xr
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import corrcoef
sns.set_context('paper')
from monet.plots.taylordiagram import TaylorDiagram as td
from matplotlib.colors import TwoSlopeNorm, ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from monet.util.tools import get_epa_region_bounds as get_epa_bounds 
import math
from ..plots import savefig
from .surfplots import make_24hr_regulatory,calc_24hr_ave_v1,make_8hr_regulatory,calc_8hr_rolling_max_v1,calc_default_colors,new_color_map,map_projection,get_utcoffset,make_timeseries,make_taylor,calculate_boxplot,make_boxplot


# Define a custom formatting function 
def custom_yaxis_formatter(x, pos):
    if x == int(x):
        return str(int(x))
    else:
        return str(x)

def make_spatial_bias(df, df_reg=None, column_o=None, label_o=None, column_m=None, 
                      label_m=None, ylabel = None, ptile = None, vdiff=None,
                      outname = 'plot', 
                      domain_type=None, domain_name=None, fig_dict=None, 
                      text_dict=None,debug=False):
        
    """Creates surface spatial bias plot. 
    
    Parameters
    ----------
    df : pandas.DataFrame
        model/obs paired data to plot
    df_reg : pandas.DataFrame
        model/obs paired regulatory data to plot
    column_o : str
        Column label of observation variable to plot
    label_o : str
        Name of observation variable to use in plot title 
    column_m : str
        Column label of model variable to plot
    label_m : str
        Name of model variable to use in plot title
    ylabel : str
        Title of colorbar axis
    ptile : integer
        Percentile calculation
    vdiff : float
        Min and max value to use on colorbar axis
    outname : str
        file location and name of plot (do not include .png)
    domain_type : str
        Domain type specified in input yaml file
    domain_name : str
        Domain name specified in input yaml file
    fig_dict : dictionary
        Dictionary containing information about figure
    text_dict : dictionary
        Dictionary containing information about text
    debug : boolean
        Whether to plot interactively (True) or not (False). Flag for 
        submitting jobs to supercomputer turn off interactive mode.
        
    Returns
    -------
    plot 
        surface bias plot
        
    """
    if debug == False:
        plt.ioff()
        
    def_map = dict(states=True,figsize=[10, 5])
    if fig_dict is not None:
        map_kwargs = {**def_map, **fig_dict}
    else:
        map_kwargs = def_map
        
    #If not specified use the PlateCarree projection
    if 'crs' not in map_kwargs:
        map_kwargs['crs'] = ccrs.PlateCarree()
  
    #set default text size
    def_text = dict(fontsize=20)
    if text_dict is not None:
        text_kwargs = {**def_text, **text_dict}
    else:
        text_kwargs = def_text
        
    # set ylabel to column if not specified.
    if ylabel is None:
        ylabel = column_o
     
    if ptile is None:
        ylabel = 'Mean '+ylabel
    else:
        ylabel = '{:02d}'.format(ptile)+'th percentile '+ylabel
 
    if df_reg is not None:
        # JianHe: include options for percentile calculation (set in yaml file)
        if ptile is None:
            df_mean=df_reg.groupby(['siteid'],as_index=False).mean()
        else:
            df_mean=df_reg.groupby(['siteid'],as_index=False).quantile(ptile/100.)

        #Specify val_max = vdiff. the sp_scatter_bias plot in MONET only uses the val_max value
        #and then uses -1*val_max value for the minimum.
        ax = monet.plots.sp_scatter_bias(
            df_mean, col1=column_o+'_reg', col2=column_m+'_reg', map_kwargs=map_kwargs,val_max=vdiff,
            cmap="OrangeBlue", edgecolor='k',linewidth=.8)
    else:
        # JianHe: include options for percentile calculation (set in yaml file)
        if ptile is None:
            df_mean=df.groupby(['siteid'],as_index=False).mean()
        else:
            df_mean=df.groupby(['siteid'],as_index=False).quantile(ptile/100.)
       
        #Specify val_max = vdiff. the sp_scatter_bias plot in MONET only uses the val_max value
        #and then uses -1*val_max value for the minimum.
        ax = monet.plots.sp_scatter_bias(
            df_mean, col1=column_o, col2=column_m, map_kwargs=map_kwargs,val_max=vdiff,
            cmap="OrangeBlue", edgecolor='k',linewidth=.8)
    
    if domain_type == 'all':
        latmin= 25.0
        lonmin=-130.0
        latmax= 50.0
        lonmax=-60.0
        plt.title(domain_name + ': ' + label_m + ' - ' + label_o,fontweight='bold',**text_kwargs)
    elif domain_type == 'epa_region' and domain_name is not None:
        latmin,lonmin,latmax,lonmax,acro = get_epa_bounds(index=None,acronym=domain_name)
        plt.title('EPA Region ' + domain_name + ': ' + label_m + ' - ' + label_o,fontweight='bold',**text_kwargs)
    else:
        latmin= math.floor(min(df.latitude))
        lonmin= math.floor(min(df.longitude))
        latmax= math.ceil(max(df.latitude))
        lonmax= math.ceil(max(df.longitude))
        plt.title(domain_name + ': ' + label_m + ' - ' + label_o,fontweight='bold',**text_kwargs)
    
    if 'extent' not in map_kwargs:
        map_kwargs['extent'] = [lonmin,lonmax,latmin,latmax]  
    ax.axes.set_extent(map_kwargs['extent'],crs=ccrs.PlateCarree())
    
    #Update colorbar
    f = plt.gcf()
    model_ax = f.get_axes()[0]
    cax = f.get_axes()[1]
    #get the position of the plot axis and use this to rescale nicely the color bar to the height of the plot.
    position_m = model_ax.get_position()
    position_c = cax.get_position()
    cax.set_position([position_c.x0, position_m.y0, position_c.x1 - position_c.x0, (position_m.y1-position_m.y0)*1.1])
    cax.set_ylabel(ylabel,fontweight='bold',**text_kwargs)
    cax.tick_params(labelsize=text_kwargs['fontsize']*0.8,length=10.0,width=2.0,grid_linewidth=2.0)    
    
    #plt.tight_layout(pad=0)
    savefig(outname + '.png', loc=4, logo_height=120)
    
####NEW function for adding 'altitude' variable as secondary y- axis (qzr++)
def add_yax2_altitude(ax, pairdf, altitude_yax2, text_kwargs, vmin_y2, vmax_y2): 

    """Creates secondary y-axis (altitude) for timeseries plot.
    
    Parameters
    ----------
    ax : ax
        Matplotlib ax from previous occurrences so it can overlay obs and model results on the same plot.
    pairdf : pandas.DataFrame
        Model/obs paired data to plot.
    text_kwargs : dictionary
        Dictionary containing information about text.
    altitude_yax2: dictionary
        Secondary y-axis (altitude) control options, including altitude_variable, altitude_ticks, etc.
    vmin_y2, vmax_y2: the value[0], value[1] respectively defined in filter_dict in altitude_yax2 in YAML control option 
                
    Returns
    -------
    ax : ax
        Matplotlib ax such that driver.py can iterate to overlay multiple models on the same plot.
    """
    ax2 = ax.twinx()
    
    # Fetch altitude parameters from altitude_yax2
    altitude_variable = altitude_yax2['altitude_variable']
    altitude_ticks = altitude_yax2['altitude_ticks']
    plot_kwargs_y2 = altitude_yax2.get('plot_kwargs_y2', {})
    ylabel2 = altitude_yax2.get('ylabel2', 'Altitude')
    
    # Plot altitude
    ax2.plot(pairdf.index, pairdf[altitude_variable], **plot_kwargs_y2, label=ylabel2)
    
    # Set labels, ticks, and limits
    ax2.set_ylabel(ylabel2, fontweight='bold', fontsize=text_kwargs['fontsize'], color=plot_kwargs_y2.get('color', 'g'))
    ax2.tick_params(axis='y', labelcolor=plot_kwargs_y2.get('color', 'g'), labelsize=text_kwargs['fontsize'] * 0.8)
    ax2.set_ylim(vmin_y2, vmax_y2) 
    ax2.set_xlim(ax.get_xlim())
    start_tick = max(0, vmin_y2 - altitude_ticks)
    ax2.yaxis.set_ticks(np.arange(start_tick, vmax_y2 + altitude_ticks + 1, altitude_ticks))

    # Extract the current legend and add a custom legend for the altitude line
    lines, labels = ax.get_legend_handles_labels()
    lines.append(ax2.get_lines()[0])
    labels.append(ylabel2)
    ax.legend(lines, labels, frameon=False, fontsize=text_kwargs['fontsize'], bbox_to_anchor=(1.15, 0.9), loc='center left')

    return ax

                              
####NEW vertprofile has option for both shading (for interquartile range) or box (interquartile range)-whisker (10th-90th percentile bounds) (qzr++)
def make_vertprofile(df, column=None, label=None, ax=None, bins=None, altitude_variable=None, ylabel=None,
                     vmin=None, vmax=None, 
                     domain_type=None, domain_name=None,
                     plot_dict=None, fig_dict=None, text_dict=None, debug=False, interquartile_style=None):
    """Creates altitude profile plot.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Model/obs paired data to plot.
    column : str
        Column label of variable to plot.
    label : str
        Name of variable to use in plot legend.
    ax : ax
        Matplotlib ax from previous occurrence so it can overlay obs and model results on the same plot.
    bins : int or array-like
        Bins for binning the altitude variable.
    altitude_variable: str
        The Altitude variable in the paired df e.g., 'MSL_GPS_Altitude_YANG'
    ylabel : str
        Title of y-axis.
    vmin : float
        Min value to use on y-axis.
    vmax : float
        Max value to use on y-axis.
    domain_type : str
        Domain type specified in input yaml file
    domain_name : str
        Domain name specified in input yaml file
    plot_dict : dictionary
        Dictionary containing information about plotting for each pair
        (e.g., color, linestyle, markerstyle).
    fig_dict : dictionary
        Dictionary containing information about the figure.
    text_dict : dictionary
        Dictionary containing information about text.
    debug : bool
        Whether to plot interactively (True) or not (False). Flag for submitting jobs to supercomputer turn off interactive mode.
    interquartile_style= str
        Whether the vertical profile uses shading or box style for interquartile range
        
    Returns
    -------
    ax : ax
        Matplotlib ax such that driver.py can iterate to overlay multiple models on the same plot.
    """
    if debug == False:
        plt.ioff()
    
    # First, define items for all plots
    # Set default text size
    def_text = dict(fontsize=14)
    if text_dict is not None:
        text_kwargs = {**def_text, **text_dict}
    else:
        text_kwargs = def_text
    
    # Set ylabel to column if not specified
    if ylabel is None:
        ylabel = column
    if label is not None:
        plot_dict['label'] = label
    if vmin is not None and vmax is not None:
        plot_dict['ylim'] = [vmin, vmax]
    # Scale the fontsize for the x and y labels by the text_kwargs
    plot_dict['fontsize'] = text_kwargs['fontsize'] * 0.8

      
                         
    # Then, if no plot has been created yet, create a plot and plot the obs
    if ax is None: 
        # First define the colors for the observations
        obs_dict = dict(color='k', linestyle='-', marker='*', linewidth=1.2, markersize=6.)
        if plot_dict is not None:
            # Whatever is not defined in the yaml file is filled in with the obs_dict here
            plot_kwargs = {**obs_dict, **plot_dict} 
        else:
            plot_kwargs = obs_dict
        # Create the figure
        if fig_dict is not None:
            f, ax = plt.subplots(**fig_dict)    
        else: 
            f, ax = plt.subplots(figsize=(10, 6))
       
        # Bin the altitude variable and calculate median and interquartiles
        altitude_bins = pd.cut(df[altitude_variable], bins=bins)
        # Calculate the midpoints of the altitude bins
        bin_midpoints = altitude_bins.apply(lambda x: x.mid)
        # Convert bin_midpoints to a column in the DataFrame
        df['bin_midpoints'] = bin_midpoints
        median = df.groupby(altitude_bins)[column].median()
        q1 = df.groupby(altitude_bins)[column].quantile(0.25)
        q3 = df.groupby(altitude_bins)[column].quantile(0.75)
        # Convert bin_midpoints to a numerical data type
        df['bin_midpoints'] = df['bin_midpoints'].astype(float)

        p5 = df.groupby(altitude_bins)[column].quantile(0.05)
        p10 = df.groupby(altitude_bins)[column].quantile(0.10)
        p90 = df.groupby(altitude_bins)[column].quantile(0.90)
        p95 = df.groupby(altitude_bins)[column].quantile(0.95)
        
        # Calculate the mean of bin_midpoints grouped by altitude bins
        binmidpoint = df.groupby(altitude_bins)['bin_midpoints'].mean()

        ##Plotting vertprofile starts
        plot_kwargs_fillbetween = plot_kwargs.copy()
        del plot_kwargs_fillbetween['marker']
        del plot_kwargs_fillbetween['markersize']
        del plot_kwargs_fillbetween['fontsize']

       
        # Copy the plot_kwargs outside the loop
        plot_kwargs_fillbox = plot_kwargs.copy()
        

        # Define box properties for transparency
        boxprops = {
            'facecolor': 'none',  # Transparent facecolor
        }

        # Track labels that have been added to the legend
        labels_added_to_legend = set()
        
        # Plot shaded interquartiles
        plot_kwargs_fillbetween['label'] = f"{label} (interquartile 25th-75th percentile range)"
        if interquartile_style == 'shading':
            ax.fill_betweenx(binmidpoint.values, q1.values, q3.values, alpha=0.3, **plot_kwargs_fillbetween)
        
        # For interquartile_style == 'box' case
        elif interquartile_style == 'box':
            box_data = []
            for idx in range(len(q1)):
                # Create a dictionary for each box with the required statistics
                box_data.append({
                    'med': median.values[idx],
                    'q1': q1.values[idx],
                    'q3': q3.values[idx],
                    'whislo': p10.values[idx],
                    'whishi': p90.values[idx],
                    'fliers': []
                })        
            
            # Creating a vertical boxplot for each bin
            for idx, data in enumerate(box_data):                
                # Get the color for the label from plot_kwargs or use a default color
                color = plot_kwargs_fillbox.get('color', 'black')
                                                        
                # Determine if the label should be added to the legend (only once for each label)
                legend_label = label if idx == 0 else None

                # Box properties
                boxprops = {
                    'facecolor': 'none',  # Transparent face color
                    'edgecolor': color,   # Edge color matching the median line color
                    'linewidth': 2        # Thickness of the box
                }

                # Whisker properties
                whiskerprops = {
                'color': color,       # Whisker color
                'linewidth': 2        # Thickness of the whiskers
                }

                # Median line properties
                medianprops = {
                    'color': 'none'       # Transparent color for median line
                }

                capprops = {'color': color, 'linewidth': 2} # Whisker end bar properties

                # Create the box plot
                ax.bxp([data], vert=False, positions=[binmidpoint.values[idx]],
                       widths=500, meanline=True, patch_artist=True, 
                       boxprops=boxprops, whiskerprops=whiskerprops, medianprops=medianprops,
                      capprops=capprops)

                # Optional: Add a legend entry for the box (only once)
                if legend_label and idx == 0:
                    box_legend_artist = Rectangle((0, 0), 1, 1, facecolor='none', edgecolor=color, linewidth=2)
                    ax.legend([box_legend_artist], [f"{legend_label} (box-whisker)"], loc='upper left')
            

        plot_kwargs['label'] = f"{label} (median)"
        median_line_df = pd.DataFrame(data={'median': median.values, 'binmidpoint': binmidpoint.values})
        ax = median_line_df.plot(x='median', y='binmidpoint', ax=ax, legend=True, **plot_kwargs)
    
    # If plot has been created, add to the current axes
    else:
        # This means that an axis handle already exists, so use it to plot the model output
        altitude_bins = pd.cut(df[altitude_variable], bins=bins)
        # Calculate the midpoints of the altitude bins
        bin_midpoints = altitude_bins.apply(lambda x: x.mid)
        # Convert bin_midpoints to a column in the DataFrame
        df['bin_midpoints'] = bin_midpoints
        # can be .groupby(bin_midpoints) as well (qzr)
        median = df.groupby(altitude_bins)[column].median()
        q1 = df.groupby(altitude_bins)[column].quantile(0.25)
        q3 = df.groupby(altitude_bins)[column].quantile(0.75)
        # Convert bin_midpoints to a numerical data type
        df['bin_midpoints'] = df['bin_midpoints'].astype(float)

        # Calculate the 10th, 90th, 5th, and 95th percentiles
        p10 = df.groupby(altitude_bins)[column].quantile(0.10)
        p90 = df.groupby(altitude_bins)[column].quantile(0.90)
        p5 = df.groupby(altitude_bins)[column].quantile(0.05)
        p95 = df.groupby(altitude_bins)[column].quantile(0.95)

        # Calculate the mean of bin_midpoints grouped by altitude bins
        binmidpoint = df.groupby(altitude_bins)['bin_midpoints'].mean()
          
        plot_kwargs_fillbetween = plot_dict.copy()
        del plot_kwargs_fillbetween['marker']
        del plot_kwargs_fillbetween['markersize']
        del plot_kwargs_fillbetween['fontsize']  
        
        # Copy the plot_kwargs outside the loop
        plot_kwargs_fillbox = plot_dict.copy()
        
        # Define box properties for transparency
        boxprops = {
            'facecolor': 'none',  # Transparent facecolor
        }

        # Track labels that have been added to the legend
        labels_added_to_legend = set()
        
        plot_kwargs_fillbetween['label'] = f"{label} (interquartile 25th-75th percentile range)"
        if interquartile_style == 'shading':
            ax.fill_betweenx(binmidpoint.values, q1.values, q3.values, alpha=0.3, **plot_kwargs_fillbetween)
        # For interquartile_style == 'box' case
        elif interquartile_style == 'box':
            box_data = []
            for idx in range(len(q1)):
                # Create a dictionary for each box with the required statistics
                box_data.append({
                    'med': median.values[idx],
                    'q1': q1.values[idx],
                    'q3': q3.values[idx],
                    'whislo': p10.values[idx],
                    'whishi': p90.values[idx],
                    'fliers': []
                })

            # Creating a vertical boxplot for each bin
            for idx, data in enumerate(box_data):
                # Get the color for the label from plot_kwargs or use a default color
                color = plot_kwargs_fillbox.get('color', 'black')

                # Determine if the label should be added to the legend (only once for each label)
                legend_label = label if idx == 0 else None

                # Box properties
                boxprops = {
                    'facecolor': 'none',  # Transparent face color
                    'edgecolor': color,   # Edge color matching the median line color
                    'linewidth': 2        # Thickness of the box
                }

                # Whisker properties
                whiskerprops = {
                'color': color,       # Whisker color
                'linewidth': 2        # Thickness of the whiskers
                }

                # Median line properties
                medianprops = {
                    'color': 'none'       # Transparent color for median line
                }

                capprops = {'color': color, 'linewidth': 2} # Whisker end bar properties

                # Create the box plot
                ax.bxp([data], vert=False, positions=[binmidpoint.values[idx]],
                       widths=500, meanline=True, patch_artist=True, 
                       boxprops=boxprops, whiskerprops=whiskerprops, medianprops=medianprops,
                      capprops=capprops)

                # Optional: Add a legend entry for the box (only once)
                if legend_label and idx == 0:
                    box_legend_artist = Rectangle((0, 0), 1, 1, facecolor='none', edgecolor=color, linewidth=2)
                    ax.legend([box_legend_artist], [f"{legend_label} (box-whisker)"], loc='upper left')
            
        # Plot median line
        plot_dict['label'] = f"{label} (median)"
        median_line_df = pd.DataFrame(data={'median': median.values, 'binmidpoint': binmidpoint.values})
        ax = median_line_df.plot(x='median', y='binmidpoint', ax=ax, legend=True, **plot_dict)
        
    
    if interquartile_style == 'box':
        # Add text to legend (adjust the x and y coordinates to place the text below the legend)
        plt.text(1.12, 0.7, 'Bounds of box: Interquartile range\nWhiskers: 10th and 90th percentiles', transform=ax.transAxes, fontsize=text_kwargs['fontsize']*0.8)
    # Apply the custom formatter to the y-axis (round off y-axis tick labels if after decimal its just zero)
    ax.yaxis.set_major_formatter(FuncFormatter(custom_yaxis_formatter))                     
    # Set parameters for all plots
    ax.set_ylabel('Altitude (m)', fontweight='bold', **text_kwargs) 
    ax.set_xlabel(ylabel, fontweight='bold', **text_kwargs) 
    ax.legend(frameon=False,fontsize=text_kwargs['fontsize']*0.8)
    ax.tick_params(axis='both',length=10.0,direction='inout')
    ax.tick_params(axis='both',which='minor',length=5.0,direction='out')
    #Adjust label position                         
    ax.legend(frameon=False, fontsize=text_kwargs['fontsize']*0.8, bbox_to_anchor=(1.1, 0.9), loc='center left')

    if domain_type is not None and domain_name is not None:
        if domain_type == 'epa_region':
            ax.set_title('EPA Region ' + domain_name,fontweight='bold',**text_kwargs)
        else:
            ax.set_title(domain_name,fontweight='bold',**text_kwargs)         
                
    breakpoint() #debug
                         
    return ax


##NEW Scatter Density Plot for model obs pairs (matplotlib scatter plot if fill=False or seaborn kde sactter density plot if fill= True)
def make_scatter_density_plot(df, mod_var=None, obs_var=None, ax=None, color_map='viridis', xlabel=None, ylabel=None, title=None, fill=False, vmin_x=None, vmax_x=None, vmin_y=None, vmax_y=None, **kwargs):
    
    """  
    Creates a scatter density plot for the specified column (variable) in the paired DataFrame (df).

    Parameters
    --------

    df: dataframe
        Paired DataFrame containing the model and observation data to plot
    obs_var: str
        obs variable name in mapped pairs
    mod_var: str
        model variable name in mapped pairs
    ax: Matplotlib axis from a previous occurrence to overlay obs and model results on the same plot
    color_map: str
        Colormap for the density (optional)
    xlabel: str
        Label for the x-axis (optional)
    ylabel: str
        Label for the y-axis (optional)
    title: str
        Title for the plot (optional)
    fill: bool
        Fill set to True for seaborn kde plot
    **kwargs: dict 
        Additional keyword arguments for customization

    Returns
    -------
    ax : ax
        Matplotlib ax such that driver.py can iterate to overlay multiple models on the same plot.
    """

    # Create a custom colormap based on color_map options in yaml or just use default colormap id color_map is just a string (e.g. viridis)
    # Determine the normalization based on vcenter
    vcenter = kwargs.get('vcenter', None)
    
    if vcenter is not None:
        norm = TwoSlopeNorm(vcenter=vcenter, vmin=vmin_x, vmax=vmax_x)
    else:
        norm = None  # This means we'll use a default linear normalization

    extensions = kwargs.get('extensions', None)  # Extract extensions for the colorbar
    
    # Check if the color_map key from the YAML file provides a dictionary 
    # (indicating a custom colormap) or just a string (indicating a built-in colormap like 'magma', 'viridis' etc.).
    color_map_config = color_map

    #print(f"Color Map Config: {color_map_config}") #Debugging
    
    if isinstance(color_map_config, dict):
        colors = color_map_config['colors']
        over = color_map_config.get('over', None)
        under = color_map_config.get('under', None)
        
        cmap = (mpl.colors.ListedColormap(colors)
                .with_extremes(over=over, under=under))
    else:
        cmap = plt.get_cmap(color_map_config)

    # Debug print statement to check the colormap configuration
    #print(f"Using colormap: {cmap}") #Debugging

    if isinstance(cmap, mpl.colors.ListedColormap):
        cmap = LinearSegmentedColormap.from_list("custom", cmap.colors)


    # Check if 'ax' is None and create a new subplot if needed
    if ax is None:
        fig, ax = plt.subplots()
        
    x_data = df[mod_var]
    y_data = df[obs_var]

    if fill:  # For KDE plot
        #print("Generating KDE plot...")
    
        # Check the type of the colormap and set Seaborn's palette accordingly
        if isinstance(cmap, mpl.colors.ListedColormap):
            sns.set_palette(cmap.colors)
        elif isinstance(cmap, mpl.colors.LinearSegmentedColormap):
            # If it's a LinearSegmentedColormap, extract N colors from the colormap
            N = 256
            sns.set_palette([cmap(i) for i in range(N)])
    
        # Create the KDE fill plot using seaborn
        plot = sns.kdeplot(x=x_data.dropna(), y=y_data.dropna(), cmap=cmap, norm=norm, fill=True, ax=ax, 
                           **{k: v for k, v in kwargs.items() if k in sns.kdeplot.__code__.co_varnames})
        colorbar_label = 'Density'
        
        # Get the QuadMesh object from the Axes for the colorbar and explicitly set its colormap
        mappable = ax.collections[0]
        mappable.set_cmap(cmap)
        
    else:  # For scatter plot using matplotlib
        #print("Generating scatter plot...")
        plot = plt.scatter(x_data, y_data, c=y_data, cmap=cmap, norm=norm, marker='o', 
                           **{k: v for k, v in kwargs.items() if k in plt.scatter.__code__.co_varnames})
        units = ylabel[ylabel.find("(")+1: ylabel.find(")")]
        colorbar_label = units  # Units for scatter plot
        mappable = plot

    
    # Set plot labels and titles
    if xlabel:
        plt.xlabel(xlabel, fontweight='bold')
    if ylabel:
        plt.ylabel(ylabel, fontweight='bold')
    if title:
        plt.title(title, fontweight='bold')
    if vmin_x is not None:
        plt.xlim(left=vmin_x)
    if vmax_x is not None:
        plt.xlim(right=vmax_x)
    if vmin_y is not None:
        plt.ylim(bottom=vmin_y)
    if vmax_y is not None:
        plt.ylim(top=vmax_y)

    
    # Handle the colorbar using the mappable object
    if extensions:
        cbar = plt.colorbar(mappable, extend='both', ax=ax)  # Extends the colorbar at both ends
    else:
        cbar = plt.colorbar(mappable, ax=ax)
    cbar.set_label(colorbar_label)

    plt.show()

    return ax


##NEW Violin plot test
def make_violin_plot(comb_violin, ylabel=None, vmin=None, vmax=None, outname='plot',
                     domain_type=None, domain_name=None,
                     plot_dict=None, fig_dict=None, text_dict=None, debug=False,
                     obs_label=None, model_label=None):  
    

    
    
    """Creates violin plot. 
    
    Parameters
    ----------
    comb_violin: dataframe
        dataframe containing information to create violin plot.
    ylabel : str
        Title of y-axis
    vmin : real number
        Min value to use on y-axis
    vmax : real number
        Max value to use on y-axis
    outname : str
        file location and name of plot (do not include .png)
    domain_type : str
        Domain type specified in input yaml file
    domain_name : str
        Domain name specified in input yaml file
    plot_dict : dictionary
        Dictionary containing information about plotting for each pair 
        (e.g., color, linestyle, markerstyle)   
    fig_dict : dictionary
        Dictionary containing information about figure
    text_dict : dictionary
        Dictionary containing information about text
    debug : boolean
        Whether to plot interactively (True) or not (False). Flag for 
        submitting jobs to supercomputer turn off interactive mode.
    obs_label, model_label : str
        String for obs and model label(s) for x axis labels
        
    Returns
    -------
    plot 
        violin plot
        
    """
    
    # Setup the plot aesthetics
    if fig_dict:
        plt.figure(**fig_dict)
    if not debug:
        plt.ioff()
    
    # Create the violin plot
    color = plot_dict.get('color', None) if plot_dict else None
    
    sns.violinplot(data=comb_violin, palette=color)

    
    # Setup the axes labels and limits
    # Bold x and y axis labels
    plt.xlabel(None, weight='bold')  # Removing x label but making it bold for uniformity
    plt.ylabel(ylabel, weight='bold')

    if vmin and vmax:
        plt.ylim(vmin, vmax)

    # Setup the plot title based on domain_type and domain_name
    if domain_type and domain_name:
        plt.title(f"Violin Plot for {domain_type} - {domain_name}")
    
    # Save the plot
    plt.savefig(f"{outname}.png", dpi=300, bbox_inches='tight')
    
    # If debug is on, show the plot
    if debug:
        plt.show()

    plt.close()
