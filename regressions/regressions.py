#!/usr/bin/env python
# coding: utf-8

# # Mosquito Regressions 

# The idea behind this notebook is to combine data from the Mosquito Habitat Mapper with data from another GLOBE protocol, such as air temperature or precipitation.  The goal is to provide tools for examining the relationship between the two protocols using Weighted Least Squares Regression.  

# ### Importing Required Modules

# A few Python modules and tools are required to run this script.

# In[1]:


# subroutine for designating a code block
def designate(title, section='main'):
    """Designate a code block with a title so that the code may be hidden and reopened.
    
    Arguments:
        title: str, title for code block
        section='main': str, section title
        
    Returns:
        None
    """
    
    # begin designation
    designation = ' ' * 20
    
    # if marked for user parameters
    if section == 'settings':
        
        # begin designation with indicator
        designation = '*** settings -----> '
    
    # add code designator
    designation += '^ [code] (for {}: {})'.format(section, title)
    
    # print
    print(designation)
    
    return None

# apply to itself
designate('designating hidden code blocks', 'designation')


# In[2]:


designate('importing Python system tools')

# import os and sys modules for system controls
import os
import sys

# set runtime warnings to ignore
import warnings

# import requests and json modules for making API requests
import requests
import json

# import fuzzywuzzy for fuzzy name matching
from fuzzywuzzy import fuzz

# import datetime module for manipulating date and time formats
from datetime import datetime, timedelta

# import pandas for dataframe manipulation
import pandas


# In[3]:


designate('importing Python mathematical modules')

# import numpy for math
from numpy import array, isnan
from numpy import exp, sqrt, log, log10, sign, abs
from numpy import arcsin, arcsinh, sin, cos, pi
from numpy import average, std, histogram, percentile
from numpy.random import random, choice

# import scipy for scientific computing
from scipy import stats
from scipy.optimize import curve_fit

# import sci-kit for linear regressions
from sklearn.neighbors import BallTree
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, PoissonRegressor


# In[4]:


designate('importing Python visualization modules')

# import bokeh for plotting graphs
from bokeh.plotting import figure
from bokeh.io import output_notebook, show
from bokeh.layouts import row as Row
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.models import Circle, LinearAxis, Range1d

# import ipyleaflet and branca for plotting maps
from ipyleaflet import Map, Marker, basemaps, CircleMarker, LayerGroup
from ipyleaflet import WidgetControl, ScaleControl, FullScreenControl, LayersControl
from branca.colormap import linear as Linear, StepColormap

# import iPython for javascript based notebook controls
from IPython.display import Javascript, display, FileLink

# import ipywidgets for additional widgets
from ipywidgets import Label, HTML, Button, Output, Box, VBox, HBox


# In[5]:


designate('inferring fuzzy match', 'tools')

# subroutine for fuzzy matching
def fuzzy_match(text, options):
    """Infer closest match of text from a list of options.
    
    Arguments:
        text: str, entered text
        options: list of str, the options
        
    Returns:
        str, the closest match
    """
    
    # perform fuzzy search to get closest match
    fuzzies = [(option, fuzz.ratio(text, option)) for option in options]
    fuzzies.sort(key=lambda pair: pair[1], reverse=True)
    inference = fuzzies[0][0]
    
    return inference


# In[6]:


designate('truncating field names', 'tools')

# truncate field names to first capital
def truncate_field_name(name, size=5, minimum=4, maximum=15):
    """Truncate a name to the first captial letter past the minimum.
    
    Arguments:
        name: str, the name for truncation
        size: the final size of the truncation
        minimum=4: int, minimum length of name
        maximum=15: int, maximum length of name
        
    Returns
        str, truncated name
    """
    
    # chop name at maximum and capitalize
    name = name[-maximum:]
    name = name[0].capitalize() + name[1:]
    
    # make stub starting at minimum length
    length = minimum
    stub = name[-length:]
    while not stub[0].isupper():
        
        # add to length
        length += 1
        stub = name[-length:]
        
    # only pass size
    stub = stub[:size]
        
    return stub


# In[7]:


designate('entitling a name by capitalizing', 'tools')

# entitle function to capitalize a word for a title
def make_title(word):
    """Entitle a word by capitalizing the first letter.
    
    Arguments:
        word: str
        
    Returns:
        str
    """
    
    # capitalize first letter
    word = word[0].upper() + word[1:]
    
    return word


# In[8]:


designate('resolving country name and code', 'tools')

# resolving country name and codes
def resolve_country_code(country, code):
    """Resolve the country code from given information.
    
    Arguments:
        country: str, country name as input
        code: str, country code as input
        
    Returns:
        (str, str) tuple, the country name and country code
    """
    
    # check for code
    if code:
        
        # find closest matching code
        code = fuzzy_match(code, [member for member in codes.values()])
        country = countries[code]
    
    # if no code, but a country is given
    if not code and country:
        
        # find closest matching country
        country = fuzzy_match(country, [member for member in codes.keys()])
        code = codes[country]
    
    # if there's no code, check the country
    if not code and not country:
        
        # default to all countries
        country = 'All countries'
        code = ''
    
    return country, code


# In[9]:


designate('scanning notebook for cells', 'introspection')

# scan notebook for cell information
def get_cell_info():
    """Scan the notebook and collect cell information.

    Arguments:
        None

    Returns:
        list of dicts
    """

    # open the notebook file 
    with open('regressions_ksenia_2.ipynb', 'r', encoding='utf-8') as pointer:
        
        # and read its contents
        contents = json.loads(pointer.read())

    # get all cells
    cells = contents['cells']

    return cells


# In[10]:


designate('defining global variables')

# ignore runtime warnings
warnings.filterwarnings('ignore')

# set pandas optinos
pandas.set_option("display.max_rows", None)
pandas.set_option("display.max_columns", None)

# begin optimizations list for previous optimizations
optimizations = []

# establish genera and colors
classification = ['Unknown', 'Other', 'Aedes', 'Anopheles', 'Culex']
colors = ['gray', 'green', 'crimson', 'orange', 'magenta']

# create indicator colors to be used on plots
indicators = {genus: color for genus, color in zip(classification, colors)}
indicators.update({'All': 'blue'})

# initiate regression modes
regressions = {mode: {} for mode in ('linear', 'quadratic', 'exponential', 'power', 'gaussian')}

# define cancellation message
cancellation = 'no fit achieved'

# initialize memory dictionary for latitude, longitude measurements
memory = {}

# define template for units
template = {'distance': '(km)', 'interval': '(d)', 'lag': '(d)', 'confidence': '', 'cutoff': '', 'inclusion': ''}
template.update({'mode': '', 'genus': '', 'records': '', 'pairs': '', 'coverage': ''})
template.update({'s.e.': '(larvae)', 'correlation': '', 'R^2': '', 'pvalue': '', 'equation': ''})
template.update({'slope': '(larvae/feature)', 'center': '(feature)', 'onset': '(feature)'})
template.update({'curvature': '(larvae/feature^2)', 'height': '(larvae)', 'rate': '(/feature)'})
template.update({'power': '', 'spread': '(feature^2)'})

# define units
making = lambda unit: lambda feature: unit.replace('feature', truncate_field_name(feature))
units = {field: making(unit) for field, unit in template.items()}

# make doppelganger for navigation
doppelganger = get_cell_info()


# In[11]:


designate('import status')

# print status
print('modules imported.')


# ### Notes on Navigation

# #### General Organization:
# 
# 
# - The notebook is organized in two main sections, with documentation near the top and user settings and plots in the second half.  Relevant subroutines are generally presented in the documentation sections, or at the end of the preceding section.
# 
# 
# - There are several sections that require the input of user selected parameters.  Click Apply to see the affect of changing those parameters on that section only, then Propagate to propagate the changes down the notebook.  Clicking Both will do both these actions.
# 
# #### Running Cells:
# 
# 
# - Upon loading the notebook, most plots will not be visible.  It is necessary to run all the code by selecting "Restart & Run All" from the "Kernel" menu and clicking "Restart and Run All Cells" to confirm.
# 
# 
# - This action may be performed at any time, for instance after altering the parameters or changing the code in other ways.
# 
# 
# - Alternatively, any single block of code may be rerun by highlighting the block and pressing Shift-Return.
# 
# 
# - Also, under the "Cell" menu is the option to "Run All Below" the currently selected cell, or to simply "Run Cells" that have been selected.
# 
# 
# #### Processing Indicator:
# 
# - In the upper righthand corner it says "Python 3" with a circle.  If this circle is black, it means the program is still processing.   A hollow circle indicates all processing is done.
# 
# 
# #### Collapsible Headings and Code Blocks:
# 
# - The Jupyter notebook format features collapsible code sections and headings.  An entire section may be collapsed by clicking on the downward pointing triangle at the left of the heading.  
# 
# 
# - Likewise, blocks of code are loaded hidden from view, and designated with '[code] ^'.  Click on the '[code] ^' text and select '^' from the toolbar next to "Download" to expand the code.  Blocks with user parameters to enter are marked with *** settings ---->.
# 
# 
# - All code blocks may be hidden or exposed by toggling the eye icon in the toolbar.
# 
# 
# - Large output boxes may be collapsed to scrollable window by clicking to the left, and may also be collapsed completely by double-clicking in the same area.  Clicking on the "..." will reopen the area.
# 
# 
# #### Hosting by myBinder:
# 
# 
# - This notebook is hosted by myBinder.org in order to maintain its interactivity within a browser without the user needing an established Python environment.  Unfortunately, connection with myBinder.org will break after 10 minutes of inactivity.  In order to reconnect you may use the link under "Browser Link" to reload.
# 
# 
# - The state of the notebook may be saved by clicking the leftmost cloud icon in the toolbar to the right of the Download button.  This saves the notebook to the browser.  The rightmost cloud icon can then retrieve this saved state in a newly opened copy.  Often reopening a saved version comes with all code blocks visible, so toggle this using the eye icon in the toolbar.
# 
# 
# - The following browser link will reload the notebook in case the connection is lost:
# https://mybinder.org/v2/git/https%3A%2F%2Fmattbandel%40bitbucket.org%2Fmattbandel%2Fglobe-mosquitoes-regressions.git/master?filepath=regressions.ipynb

# In[12]:


designate('looking for particular cell', 'navigation')

# function to look for cells with a particular text snippet
def seek_text_in_cell(text):
    """Look for a particular text amongst the cells.
    
    Arguments:
        text: str, the text to search for
        
    Returns:
        list of int, the cell indices.
    """
    
    # search for cells 
    indices = []
    for index, cell in enumerate(doppelganger):
        
        # search for text in source
        if any([text in line.replace("'{}'".format(text), '') for line in cell['source']]):
            
            # add to list
            indices.append(index)
            
    return indices


# In[13]:


designate('jumping to a particular cell', 'navigation')

# jump to a particular cell
def jump_to_cell(identifier):
    """Jump to a particular cell.
    
    Arguments:
        identifier: int or str
        
    Returns:
        None
    """
    
    # try to look for a string
    try:
        
        # assuming string, take first index with string
        index = seek_text_in_cell(identifier)
        
    # otherwise assume int
    except (TypeError, IndexError):
        
        # index is identifier
        index = identifier 
    
    # scroll to cell
    command = 'IPython.notebook.scroll_to_cell({})'.format(index)
    display(Javascript(command))
    
    return


# In[14]:


designate('executing cell range by text', 'navigation')

# execute cell range command
def execute_cell_range(start, finish):
    """Execute a cell range based on text snippets.
    
    Arguments:
        start: str, text from beginning cell of range
        finish: str, text from ending cell of range
        
    Returns:
        None
    """
    
    # find start and finish indices, adding 1 to be inclusive
    opening = seek_text_in_cell(start)[0] 
    closing = seek_text_in_cell(finish)[0]
    bracket = (opening, closing)
    
    # make command
    command = 'IPython.notebook.execute_cell_range' + str(bracket)
    
    # perform execution
    display(Javascript(command))
    
    return None


# In[15]:


designate('refreshing cells by relative position', 'navigation')

# execute cell range command
def refresh_cells_by_position(start, finish=None):
    """Refresh a particular cell relative to current cell.
    
    Arguments:
        start: int, the first cell offset
        finish=None: int, the second cell offset
        
    Returns:
        None
    """
    
    # make offset into a string
    stringify = lambda offset: str(offset) if offset < 0 else '+' + str(offset)
    
    # default finish to start
    finish = finish or start
    
    # make command
    command = 'IPython.notebook.execute_cell_range('
    command += 'IPython.notebook.get_selected_index()' + stringify(start) + ','
    command += 'IPython.notebook.get_selected_index()' + stringify(finish + 1) + ')'
    
    # perform execution
    display(Javascript(command))
    
    return None


# In[16]:


designate('revealing open cells', 'navigation')

# outline headers
def reveal_open_cells(cells):
    """Outline the headers and collapsed or uncollapsed state.
    
    Arguments:
        cells: dict
        
    Returns:
        list of int, the indices of visible cells
    """
    
    # search through all cells for headers
    indices = []
    visible = True
    for index, cell in enumerate(cells):
        
        # check for text
        header = False
        if any(['###' in text for text in cell['source']]):
        
            # check for header and visible state
            header = True
            visible = True
            if 'heading_collapsed' in cell['metadata'].keys(): 

                # set visible flag
                visible = not cell['metadata']['heading_collapsed']

        # if either header or visible
        if header or visible: 

            # add to indices
            indices.append(index)
            
    return indices


# In[17]:


designate('gauging cell size', 'navigation')

# measure a cell's line count and graphics
def measure_cell_info(cell):
    """Gauge a cell's line count and graphic size.
    
    Arguments:
        cell: cell dict
        
    Returns:
        (int, boolean) tuple, line count and graphic boolean
    """
    
    # check for display data
    graphic = False
    displays = [entry for entry in cell.setdefault('outputs', []) if entry['output_type'] == 'display_data']
    if len(displays) > 0:
        
        # check for many displays or one long one
        if len(displays) > 2 or '…' in displays[0]['data']['text/plain'][0]:

            # switch graphic to true
            graphic = True

    # determine total lines of text in source, 2 by default
    length = 2
    
    # determine executions
    executions = [entry for entry in cell.setdefault('outputs', []) if entry['output_type'] == 'execute_result']
    for execution in executions:
        
        # add to length
        length += execution['execution_count']
    
    # check hide-input state
    if not cell['metadata'].setdefault('hide_input', False):
        
        # add lines to source
        source = cell['source']
        for line in source:

            # split on newlines
            length += sum([int(len(line) / 100) + 1 for line in line.split('\n')])

    return length, graphic


# In[18]:


designate('bookmarking cells for screenshotting', 'navigation')

# bookmark which cells to scroll to
def bookmark_cells(cells):
    """Bookmark which cells to scroll to.

    Arguments:
        cells: list of dicts
        visibles: list of ints

    Returns:
        list of ints
    """

    # set page length criterion and initialize counters
    criterion = 15
    accumulation = criterion + 1

    # determine scroll indices
    bookmarks = []
    visibles = reveal_open_cells(cells)
    for index in visibles:

        # measure cell and add to total
        cell = cells[index]
        length, graphic = measure_cell_info(cell)
        accumulation += length
        
        # compare to criterion
        if accumulation > criterion or graphic:

            # add to scrolls and reset
            bookmarks.append(index)
            accumulation = length

            # for a graphic, make sure accumulation is already maxed
            if graphic:
                
                # add to accumulation
                accumulation = criterion + 1

    return bookmarks


# In[19]:


designate('propagating setting changes across cells', 'buttons')

# def propagate
def propagate_setting_changes(start, finish, finishii, descriptions=['Apply', 'Propagate', 'Both']):
    """Propagate changes across all code cells given by the headings.
    
    Arguments:
        start: str, top header
        finish: str, update stopping point
        finishii: str, propagate stopping point
        descriptions: list of str
        
    Returns:
        None
    """
    
    # define jump points
    cues = [(start, finish), (finish, finishii), (start, finishii)]
    
    # make buttons
    buttons = []
    buttoning = lambda start, finish: lambda _: execute_cell_range(start, finish)
    for description, cue in zip(descriptions, cues):

        # make button
        button = Button(description=description)
        button.on_click(buttoning(*cue))
        buttons.append(button)

    # display
    display(HBox(buttons))
    
    return None


# In[20]:


designate('navigating to main sections', 'buttons')

# present buttons to jump to particular parts of the notebook
def navigate_notebook():
    """Guide the user towards regression sections with buttons.
    
    Arguments:
        None
        
    Returns:
        None
    """

    # define jump points
    descriptions = ['Top', 'Settings', 'Filter', 'Weights', 'Data', 'Map']
    cues = ['# Mosquitoe Larvae Regressions', '### Setting the Parameters', '### Filtering Records']
    cues += ['### Defining the Weighting Scheme', '### Viewing the Data Table', '### Visualizing on a Map', ]
    
    # make buttons
    buttons = []
    buttoning = lambda cue: lambda _: jump_to_cell(cue)
    for description, cue in zip(descriptions, cues):

        # make button
        button = Button(description=description)
        button.on_click(buttoning(cue))
        buttons.append(button)

    # display
    display(HBox(buttons))
    
    return None


# In[21]:


designate('guiding to regression modes', 'buttons')

# present buttons to choose the regression part of the notebook
def jump_to_regression():
    """Guide the user towards regression sections with buttons.
    
    Arguments:
        None
        
    Returns:
        None
    """

    # make buttons
    buttons = []
    buttoning = lambda mode: lambda _: jump_to_cell('### ' + mode.capitalize() + ' Regression')
    for mode in regressions.keys():

        # make button
        button = Button(description=mode.capitalize())
        button.on_click(buttoning(mode))
        buttons.append(button)

    # display
    display(HBox(buttons))
    
    return None


# ### Getting Started

# - Select "Restart & Run All" from the Kernel menu, and confirm by clicking on "Restart and Run All Cells" and wait for the processing to stop (the black circle in the upper right corner next to "Python 3" will turn hollow).
# 
# 
# - Use a Settings button from a navigation menu like the one below to navigate to the Settings section.
# 
# 
# - Find the ^ [code] block marked with *** setings ----->, and open it using the "^" button in the toolbar at the top of the page.  Begin inputting your settings and apply the changes with the buttons.  Then move on to the next section and apply those settings as well.

# In[22]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# ### Background: A Quick Statistics Rundown

# #### Definition of a linear regression
# 
# A straight line can be represented geometrically in the following way:
# 
# \begin{align*}
# y=\beta_0+\beta_1 X + \varepsilon
# \end{align*}
# 
# where $X$ is the independent variable, $y$ is the dependent variable, $\beta_1$ is the slope of the line, $\beta_0$ is the intercept, and $\varepsilon$ is the error term.  Any particular line has a single value for $\beta_0$ and for $\beta_1$.  Thus the above equation describes a family of lines, each with a unique value for $\beta_0$ and for $\beta_1$. 
# 
# If $X$ represents air temperature at a sampling site, for instance, and $y$ represents the number of mosquito larvae found there, then $\beta_0$ describes the average number of larvae at zero degrees, and $\beta_1$ describes the average change in the number of larvae for every single degree increase. The equation serves as a model for the relationship between air temperature and mosquito counts.
# 
# Given a set of observations represented as points along an X and y axis, regression can estimate the values for $\beta_0$ and $\beta_1$ and give insight into a relationship between the dependent and independent variables.
# 
# #### Some notes
# 
# There are several points to make about this process:
# 
# - The family of lines must be specified beforehand.  $y=\beta_0+\beta_1 X$ for instance, only describes a family of straight lines.  The best fitting straight line may be a poor description of data with a curving relationship.  To this end, six different families (called "modes" here) are used in this notebook, each with particular characteristics.  
# 
# 
# - For some modes, the best fitting parameters may be estimated simply.  In these cases, the Ordinary Least Squares equation is sufficient to find the best fit with one calculation.  In other cases, however, the fit must be found with Weighted Least Squares, needing an initial starting guess, and several subsequent iterations to find the best fit.
# 
# 
# - The initial guess must already be somewhat close to the best fitting parameters, or there is a chance the nonlinear algorithm will find only a local best and not a global best.  There may be several "basins of attraction," and an initial guess in the wrong basin will lead to only a local best. 
# 
# 
# - In some cases, the nonlinear algorithm fails to find a fit at all, generally because the data is not distributed in a way that suggests the proposed relationship strongly enough.
# 
# 
# - In other cases, nonlinear or linear regression can produce seemingly absurd results.  For example, an entire Gaussian curve has a bell shape, but just one of the tails is a good approximation to an exponential curve.  If the data distribution does not clearly suggest a Gaussian curve, the regression may find that fitting a huge Gaussian with only its tail immersed amongst the data points gives a closer fit to the data than a more reasonably sized complete Gaussian.
# 
# #### Statistical tools
# 
# - Standard error is an excellent tool for judging how well a model fits the data.
#     1) The [standard error of estimate](https://en.wikipedia.org/wiki/Standard_error) ($s_e$) is the typical difference found between the estimates from the regression and the actual observations. In particular, it is the standard deviation of the sampling distribution.
# 
# - The best fitting line may still be a poor description of the relationship.  There are a few summary statistics given here to indicate the quality of the fit:
#     
#     1) [Pearson's correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) ($\rho$) is a unitless statistic between -1 and 1 that represents the linear correlation between X and y.  A value of 1 represents a positive linear relationship between the dependent and independent variable.  A value of 0 represents no linear correlation, and a value of -1 indictes a negative linear relationship.
#     
#     2) The [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination) ($R^2$) quantifies how much of the variabiation in y can be explained by the model. It is a unitless statistic between 0 and 1. An $R^2$ of 1 can indicate that the model fits the observations perfectly whereas a value of 0 can indicate that the model fits the data poorly. **Note** that $R^2$ will always increase with more predictors but that *does not mean* that the model is better.
#     
#     
# - Correlation and $R^2$ are excellent for understanding how well a model fits the data if we have linear data. Standard error is useful for nonlinear models. In general, these measures are mostly used for linear relationships, and are less frequently used for nonlinear modes. The goal is to fit the data as closely as possible, but not at the expense of its ability to generalize to the population as a whole.
# 
# #### Caution
# 
# - Note also that the statistics above are succeptible to a number of problems. First, if the dataset is small, then correlation and $R^2$ may not be quite accurate. Second, there is a chance that the observations are not representative of the population, and the correlation hinted at by the model is due to sampling bias. A fun explantion of this is the [Datasaurus](https://www.autodeskresearch.com/publications/samestats). All of these plots have the same summary statistics!
# ![Datasaurus](https://d2f99xq7vri1nk.cloudfront.net/DinoSequentialSmaller.gif "datasaurus")
# 
# 
# #### Hypothesis testing
# 
# - Additionally, a probability (p-value) is calculated to assess the statistical significance of the relationship between X and y. Remember that a probability ranges from 0 to 1. This [hypothesis test](http://www.biostathandbook.com/hypothesistesting.html) is structured as follows:
# 
#     1) Assert the null hypothesis ($H_0$). $H_0$ is that there is no  linear relationship between X and y.
#     
#     2) Choose a critical value ($\alpha$ -- typically set to 0.05 but is sometimes set to a value as high at 0.1 and as low as 0.001) which we will use to conduct a Hypothesis Test.
#     
#     3) Next we calculate a [test statistic](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Testing_using_Student's_t-distribution) ($t* = r\sqrt{\frac{n-2}{1-r^2}}$ where $r$ is the sample correlation coefficient and $n$ is the number of samples). Use the Test Statistic to find the corresponding p-value (use a [table](http://www.ttable.org/) or the [calculator](http://www.ttable.org/student-t-value-calculator.html)). Note that the Test Statistic will differ between hypotheses.
#     
#     4) Compare the p-value found in step 3 to the $\alpha$ value selected in step 2. A p-value less than $\alpha$ means that we reject the null hypothesis which means that there is not no linear relationship between X and y which often means that there is a linear relationship between X and y. A p-value greater than $\alpha$ means that we fail to reject the null hypothesis meaning that there is likely no linear relationship between X and y. It is important to choose a critical value before performing the study, because it is very easy to grant your study significance by calculating the p-value first and then choosing a critical value higher than the one you calculated (called ["p-hacking"](https://en.wikipedia.org/wiki/Data_dredging)).
# 
# #### Considerations
# 
# - Domain knowledge is key for translating the regression models to real situations. The model can only explain the given observations. In particular, a regression model is an interpolation model. We should be wary of [extrapolation](https://online.stat.psu.edu/stat501/lesson/12/12.8) beyond the range of the data unless it is properly justifiable -- remember to ask yourself, is this model logical?
# 
# 
# - Considering confounding variables while conducting your analysis. A statistically significant model that reasonably models a relationship may still be misleading due to confounding variables. For instance, rainier weather is usually associated with cooler temperatures. An observation of a large numbers of larvae on a day with cooler temperatures may really reflect the relationship between larvae and rainfall. In this example, precipitation would be a confounding variable.
# 
# 
# - Remember! Correlation is not causation.  A representative model that models the data well and is free from confounding variables can only describe a correlation between the dependent and independent variables.  We can not answer the question of causation with regression.
# 
# 
# #### Checking your model
# 
# First, note that the true representation of linear data is $y=\beta_0+\beta_1 X + \varepsilon$ and $\hat{y}=\hat{\beta}_0+\hat{\beta}_1 X$ is our model. We can find an estimation of the error, $\varepsilon$, by: $y-\hat{y} = \hat{\varepsilon}$ where $\hat{\varepsilon}$ is the estimated error.
# 
# 
# An excellent [diagnostic](http://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/R/R5_Correlation-Regression/R5_Correlation-Regression7.html) for linear regression is a plot of the residuals versus fitted values plot. The residuals are an estimation of the error as seen above and the fitted values are our estimated response. In the plot below, we see the residuals on the y-axis and the fitted values on the x-axis. We can learn a lot from the following figure:
# ![ConstantVar](https://www.researchgate.net/profile/John_Hodsoll/publication/318911883/figure/fig1/AS:535566758277120@1504700473370/Diagnostic-plots-of-the-residuals-plotted-against-fitted-values-of-regression-model-to.png "constantvar")
# 
#     A) Suggests that a linear regression could be a very good model.
#     
#     B) Suggests that we should use Weighted Least Squares to account for non constant variance
#     
#     C) Suggests that our data is not independent. This could suggest that some of our predictors are correlated and we should consider removing some of them. It could also suggest that our data is correlated in time or space and should be processed using time series or spatial statistics methods.
#     
#     D) Suggests we do not have a linear relationship and should consider non linear methodology.
# 
# 
# Note that Ordinary Least Squares assumes a constant variance among the errors meaning that the errors around each observation have the same standard deviation. Weighted Least Squares allows for non constant variance.
# 
# 
# For this particular study, there is an additional level of complexity because data from two protocols are being combined, and are therefore not necessarily concurrent.  To this end, observations from one protocol are paired with observations from another protocol and weighted according to their closeness in space and times.
# 
# #### To be continued + Summary
# 
# Details of the Linear and Non-Linear Weighted Least Squares algorithms used in this notebook are found in the following sections, as well as the statistical methods for measuring model quality, and specific descriptions of each regression mode.
# 
# In summary, the process will be performed as follows:
# 
# - Retrieve records from the GLOBE API, process the data, and prune off outliers.
# - Assemble associations between the two protocols, weighing them according to the chosen weighting parameters.
# - Perform Linear Weighted Least Squares on an approximate problem to get reasonable starting parameter values.
# - Evaluate the model according to standard error, Pearson's correlation coefficient, coefficient of determination, and p-value.

# ### What the code below is doing:
# 
# #### Summary
# 
# In this notebook, we are looking at the relationship between data from the MHM and other GLOBE data. In order to do so, we perform semi-complex regressions to fit a handful of different curves. See more info on which curves we use further down.
# 
# #### Under the hood
# 
# In particular, we use a non linear method to fit the curves called Non-Linear Weighted Least Square. This method requires initial guesses for the parameters of each function. These parameters are estimated by getting the coefficients from a [Poisson Regression](http://www.adasis-events.com/statistics-blog/2012/11/26/what-is-the-difference-between-linear-logistic-and-poisson-r.html) (implented using scikit-learn's [sklearn.linear_model.PoissonRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html#sklearn.linear_model.PoissonRegressor)). These parameters are then used in the Levenberg-Marquardt algorithm (an algorithm for Non-Linear Weighted Least Squares) which was impleneted using the scipy function [scipy.optimize.curve_fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html).

# In[23]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# In[24]:


designate('notching ticks in a range', 'graphing')

# notch function to get evenly spaced points in a range
def set_ticks(left, right, number=100):
    """Notch a number of ticks along a span
    
    Arguments:
        left: float, left axis boundary
        right: float, right axis boundary
        number=100: number ticks
        
    Returns:
        list of floats
    """
    
    # get chunk length
    chunk = (right - left) / number
    ticks = [left + index * chunk for index in range(number + 1)]
    
    return ticks


# In[25]:


designate('sketching functions', 'graphing')

# sketch a function
def sketch_plot(*functions, legend=None, span=(-5, 5), title="[TEMP TITLE]", xlab=["TEMP X LABEL"], ylab="[TEMP Y LABEL]"):
    """Sketch a function.
    
    Arguments:
        *functions: unpacked list of function objects
        legend=None: list of str
        span=(-5, 5): tuple of floats, the x-axis range
        
    Returns:
        None
    """
    
    # begin curve
    curve = figure(x_range=span, plot_width=300, plot_height=300, title=title)
    curve.xaxis.axis_label = xlab
    curve.yaxis.axis_label = ylab
    
    # set colors
    colors = ['red', 'green', 'blue', 'violet', 'cyan', 'orange']
    
    # set default legend
    if not legend:
        
        # set up legend
        legend = [str(index + 1) for index, _ in enumerate(functions)]
    
    # get points
    xs = set_ticks(*span)
    
    # plot functions
    for function, color, name in zip(functions, colors, legend):
        if function == arcsinh:
            # graph line
            points = [{'x': x, 'y': function(x/2)} for x in xs]
        else:
            # graph line
            points = [{'x': x, 'y': function(x)} for x in xs]
        table = ColumnDataSource(pandas.DataFrame(points))
        curve.line(source=table, x='x', y='y', color=color, line_width=1, legend_label=name)
        
    # add hover annotations and legend
    annotations = [('x', '@x'), ('y', '@y')]
    hover = HoverTool(tooltips=annotations)
    curve.add_tools(hover)
    curve.legend.location='top_left'
    
    # show the results
    output_notebook()
    show(curve)
    
    return None


# In[26]:


designate('annotating graphs', 'graphing')

# annotate graphs subroutine
def annotate_plot(graph, annotations):
    """Annotate the graph with annotations.
    
    Arguments:
        graph: bokeh graph object
        annotations: list of (str, str) tuples
        
    Returns:
        graph object
    """
    
    # set up hover summary
    summary = """
    <style>
        .bk-tooltip>div:not(:first-child) {display:none;}
    </style>
    
    """
    
    # add annotations
    for field, value in annotations:
        
        # add to summary
        summary += '<b>{}: </b> {} <br>'.format(field, value)

    # setup hovertool
    hover = HoverTool()
    hover.tooltips = summary
    graph.add_tools(hover)
            
    # set up graph legend
    graph.legend.location='top_left'
    graph.legend.click_policy='hide'
    
    return graph


# In[27]:


designate('issuing a ticket', 'regressions')

# function to issue a default ticket based on settings
def issue_ticket(settings, genus, mode):
    """Issue a blank ticket with default settings.
    
    Arguments:
        settings: dict
        genus: str
        mode: str
        
    Returns:
        dict
    """
    
    # begin ticket with all values set to zero
    ticket = {parameter: 0 for parameter, _ in units.items()}
    
    # update with settings
    ticket.update(settings)
    
    # add other default settings
    ticket.update({'genus': genus, 'mode': mode, 'pvalue': 1.0, 'equation': cancellation})
    ticket.update({'coefficients': [0] * (regressions[mode]['polynomial'] + 2)})
    ticket.update({'curve': [0] * regressions[mode]['requirement']})
    
    return ticket


# In[28]:


designate('running a regression on samples', 'regressions')

# generalized regression function
def regress(samples, ticket, spy=False, calculate=True):
    """Perform regression on the samples, based on the submission ticket.
    
    Arguments:
        samples: list of dicts, the samples
        ticket: dict, the settings
        approximation: float, approximate value for zero
        spy: boolean, verify initial regression fits with plots?
        calculate: boolean, calculate jacobian directly?
        
    Returns:
        dict, the report
    """   
    
    # resolve regression styles
    mode = ticket['mode']
    size = ticket['records']
    
    # try to run regression
    try:
    
        # get coefficients from linear model
        coefficients = get_coefficients_linear_regression(samples, mode)

        # fit non linear model
        curve = perform_nonlinear_regression(samples, mode, coefficients, calculate)

        # check the two fits against each other
        if spy:

            # verify
            check_optimized_params_with_initial(samples, mode, coefficients, curve)

        # assess the model
        assessment = check_model_fit(samples, mode, curve, size)
        ticket.update(assessment)
        
    # but skip for math errors
    except (ZeroDivisionError, TypeError, ValueError, RuntimeError):
        
        # skip
        pass
        
    return ticket


# In[29]:


designate('assessing a model', 'regressions')

# assess model fit
def check_model_fit(samples, mode, curve, size):
    """Assess the model by comparing predictions to targets
    
    Arguments:
        samples: list of dicts
        mode: str, the regression mode
        curve: list of floats, the regression parameters
        size: int, number of records
        
    Returns:
        dict
    """
    
    # make predictions using model
    matrix = [sample['x'] for sample in samples]
    weights = [sample['weight'] for sample in samples]
    truths = [sample['y'] for sample in samples]
    predictions = [regressions[mode]['function'](entry, *curve) for entry in matrix]
    
    # get validation scores
    validation = calculate_corr_coef(truths, predictions, weights, size)
    
    # create equation
    equation = regressions[mode]['equation']
    for parameter, place in zip(curve, ('β0', 'β1','β2')):
        
        # replace in equation
        equation = equation.replace(place, '{}'.format(round(float(parameter), 2)))

    # get critical points
    criticals = {name: round(float(quantity), 4) for name, quantity in zip(regressions[mode]['names'], curve)}

    # make assessment
    assessment = {'curve': curve, 'equation': equation}
    assessment.update(validation)
    assessment.update(criticals)
    
    return assessment


# In[30]:


designate('performing regression study', 'performing')

# perform regression mode on mosquitoes data
def perform(mode, associations, spy=False, calculate=True):
    """Perform a mode of regression on a set of associations.
    
    Arguments:
        mode: str, the mode of regression
        associations: list of dicts
        spy=True: boolean, observe initial parameter fits?
        calculate: boolean, calculate Jacobian directly?
        
    Returns:
        None
    """
    
    # make graph
    graph, panda = plot_scatter_plot(associations, mode, spy, calculate)

    # show the results
    output_notebook()
    show(graph)

    # get columns and add units
    columns = ['genus', 'records', 'pairs', 'coverage', 'pvalue', 'correlation', 'R^2', 's.e.', 'equation']
    columns += regressions[mode]['names']
    panda = panda[columns]
    panda.columns = [column + units[column](feature) for column in columns]

    # show panda
    display(panda)
    
    return None


# In[31]:


designate('studying all genera', 'regressions')

# function to run regression on the associations and return reports per genus
def study(associations, mode, spy=True, calculate=True):
    """Study the data under a regression mode.
    
    Arguments:
        associations: list of dicts
        mode: str, the regression mode
        spy: boolean, plot initial linear fits?
        calculate: boolean, calculate jacobian directly?
        
    Returns:
        list of dicts
    """
    
    # go through each genus, running regression
    reports = []
    for genus in ['All', 'Aedes', 'Anopheles', 'Culex', 'Other', 'Unknown']:
        
        # begin ticket
        ticket = issue_ticket(settings, genus, mode)
        
        # perform subsampling by genus and make the samples
        subset = get_subset_of_data(associations, genus)
        samples = assemble_samples(subset)
        coverage = round(len(subset) / len(data), 2)
        ticket.update({'records': len(subset), 'pairs': len(samples), 'coverage': coverage})
        
        # fit the regressor
        report = regress(samples, ticket, spy, calculate)
        reports.append(report)
            
    return reports


# In[32]:


designate('interpolating between parameters', 'debugging')

# define interpolation function
def interpolate(start, finish, number=5):
    """Interpolate between start and finish parameters.
    
    Arguments:
        start: list of floats
        finish: list of floats
        number=5: int, number of total curves
        
    Returns:
        list of lists of floats, the curves
    """
    
    # find points for each pair
    pairs = zip(start, finish)
    tuplets = []
    for first, last in pairs:
        
        # get chunk
        chunk = (last - first) / (number - 1)
        tuplet = [first + chunk * index for index in range(number)]
        tuplets.append(tuplet)
        
    # zip together sets
    curves = [curve for curve in zip(*tuplets)]
    
    return curves


# In[33]:


designate('verifying optimized model', 'debugging')

# verify with a plot the optimized model
def check_optimized_params_with_initial(samples, mode, coefficients, curve):
    """Verify the fit of the optimized model compared to the initial estimates.
    
    Arguments:
        samples: list of dicts
        mode: str, regression mode
        coefficients: list of floats, coefficients of linear model
        curve: list of floats, parameters of nonlinear model
        
    Returns:
        None
    """
    
    # define points
    independents = [sample['x'] for sample in samples]
    dependents = [sample['y'] for sample in samples]
    sizes = [sample['size'] for sample in samples]
    
    # get ticks
    ticks = set_ticks(min(independents), max(independents), 100)
    
    # make predictions
    initials = regressions[mode]['initial'](*coefficients)
    approximation = [regressions[mode]['function'](tick, *initials) for tick in ticks]
    regression = [regressions[mode]['function'](tick, *curve) for tick in ticks]

    # print comparison
    print('{} samples'.format(len(samples)))
    print('{} (coefficients)'.format([round(float(entry), 8) for entry in coefficients]))
    print('{} (initials)'.format([round(float(entry), 8) for entry in initials]))
    print('{} (curve)'.format([round(float(entry), 8) for entry in curve]))
    
    # make figure
    graph = figure()
    graph.circle(x=independents, y=dependents, size=sizes, color='gray', fill_alpha=0.05)
    
    # add lines
    graph.line(x=ticks, y=approximation, color='blue', line_width=3, legend_label='linear')
    graph.line(x=ticks, y=regression, color='green', line_width=3, legend_label='nonlinear')
    
    # show the results
    output_notebook()
    show(graph)

    return None


# ### Data Preparation

# The process begins with **calling the GLOBE API**:

# https://www.globe.gov/en/globe-data/globe-api

# In[34]:


designate('calling the api', 'api')

# call the api with protocol and country code
def query_api(protocol, code, beginning, ending, sample=False):
    """Call the api:
    
    Arguments:
        protocol: str, the protocol
        code: str, the country code
        beginning: str, the beginning date
        ending: str, the ending date
        sample=False: boolean, only get small sampling?
        
    Returns:
        list of dicts, the records
    """
    
    # default to all countries unless a code is specified
    extension = 'country/' if code else ''
    extensionii = '&countrycode=' + code if code else ''
    
    # assemble the url for the API call 
    url = 'https://api.globe.gov/search/v1/measurement/protocol/measureddate/' + extension
    url += '?protocols=' + protocol
    url += '&startdate=' + beginning 
    url += '&enddate=' + ending
    url += extensionii

    # geojson parameter toggles between formats
    url += '&geojson=FALSE'
    
    # sample parameter returns small sample set if true
    url += '&sample=' + str(sample).upper()

    # make the API call and return the raw results
    request = requests.get(url)
    raw = json.loads(request.text)
    
    return raw


# After retrieving the data, several steps are taken to prepare the data.  Initially, the data is returned in a nested structure.  It is useful to **flatten this nesting** so that all fields are readily accessible.  

# In[35]:


designate('flattening records', 'processing')

# function to flatten a nested list into a single-level structure
def flatten_dict_list(record, label=None):
    """Flatten each record into a single level.

    Arguments:
        record: dict, a record
        label: str, key from last nesting

    Returns:
        dict
    """

    # initiate dictionary
    flattened = {}

    # try to flatten the record
    try:

        # go through each field
        for field, info in record.items():

            # and flatten the smaller records found there
            flattened.update(flatten_dict_list(info, field))

    # otherwise record is a terminal entry
    except AttributeError:

        # so update the dictionary with the record
        flattened.update({label: record})

    return flattened


# Additionally, it can be useful to **abbreviate the field names** as the initial field names are often quite long.

# In[36]:


designate('abbreviating records', 'processing')

# function to abbreviate the fields of a record
def abbreviate_field_name(record, primary=True, removal=True):
    """Abbreviate certain fields in the record for easier manipulation later.
    
    Arguments:
        record: dict
        primary=True: boolean, primary record?
        removal=True: boolean, remove original fields?
        
    Returns:
        dict
    """
    
    # define abbreviations dictionary for primary records
    abbreviations = {}
    abbreviations['count'] = larvae
    abbreviations['genus'] = 'mosquitohabitatmapperGenus'
    abbreviations['source'] = 'mosquitohabitatmapperWaterSource'
    abbreviations['stage'] = 'mosquitohabitatmapperLastIdentifyStage'
    abbreviations['type'] = 'mosquitohabitatmapperWaterSourceType'
    abbreviations['measured'] = 'mosquitohabitatmapperMeasuredAt'
    abbreviations['habitat'] = 'mosquitohabitatmapperWaterSourcePhotoUrls'
    abbreviations['body'] = 'mosquitohabitatmapperLarvaFullBodyPhotoUrls'
    abbreviations['abdomen'] = 'mosquitohabitatmapperAbdomenCloseupPhotoUrls'
    
    # if a secondary record
    if not primary:

        # define abbreviations dictionary for secondary protocol
        abbreviations = {}
        abbreviations['feature'] = feature
        abbreviations['measured'] = measured

    # and each abbreviation
    for abbreviation, field in abbreviations.items():

        # copy new field from old, or None if nonexistent
        record[abbreviation] = record.setdefault(field, None)
        
        # remove original field if desired
        if removal:
            
            # remove field
            del(record[field])
    
    return record


# As all measurements are recorded in reference to UTC time, it is helpful to **convert the measurements to local times**.  This is accomplished by adjusting the hour according to the longitude.  Though this may not accurately reflect the local time in a political sense as it ignores daylight savings time and time zone boundaries, it is perhaps a more accurate measure in the astronomical sense.

# In[37]:


designate('synchronizing times with longitudes', 'processing')

# synchronize the time of measurement with the longitude for local time
def sync_time_with_long(record):
    """Synchronize the measured times with longitudes.
    
    Arguments:
        record: dict
        
    Returns:
        dict
    """

    # convert the date string to date object and normalize based on partitioning
    record['time'] = datetime.strptime(record['measured'], "%Y-%m-%dT%H:%M:%S")
    record['date'] = record['time'].date()
    
    # convert the date string to date object and correct for longitude
    zone = int(round(record['longitude'] * 24 / 360, 0))
    record['hour'] = record['time'] + timedelta(hours=zone)

    return record


# The larvae count data are initially returned as strings.  In order to analyze the data, we need to **convert the numerical strings into number data types**.  Additionally, some of the data is entered as a range (e.g., '1-25'), or as a more complicated string ('more than 100').  These strings will be converted to floats using the following rules:
# - a string such as '50' is converted to its floating point equivalent (50)
# - a range such as '1-25'is converted to its average (13)
# - a more complicated string, such as 'more than 100' is converted to its nearest number (100)

# In[38]:


designate('converting strings to numbers', 'processing')

# function to convert a string into a floating point number
def convert_str_to_float(record, field, name):
    """Translate info given as a string or range of numbers into a numerical type.
    
    Arguments:
        record: dict
        field: str, the field to get converted
        name: str, the name of the new field
        
    Returns:
        float
    """
    
    # try to convert directly
    info = record[field]
    try:
        
        # translate to float
        conversion = float(info)
        
    # otherwise
    except ValueError:
        
        # try to convert a range of values to their average
        try:
        
            # take the average, assuming a range separated by a hyphen
            first, last = info.split('-')
            first = float(first.strip())
            last = float(last.strip())
            conversion = float(first + last) / 2
            
        # otherwise
        except ValueError:
            
            # scan for digits
            digits = [character for character in info if character.isdigit()]
            conversion = ''.join(digits)
            conversion = float(conversion)
            
    
    # add new field
    record[name] = conversion
        
    return record


# Also, some steps have been taken towards **mosquito genus identification**.  The three noteworthy genera in terms of potentially carrying diseases are Aedes, Anopheles, and Culex.  If the identification process did not lead to one of these three genera, the genus is regarded as "Other."  If the identification process was not fully carried out, the genus is regarded as "Unknown."

# In[39]:


designate('identifying mosquito genera', 'processing')

# function to identify the mosquito genera based on last stage of identification
def identify_mosquito_genera(record):
    """Identify the genera from a record.
    
    Arguments:
        record: dict
        
    Returns:
        dict
    """

    # check genus
    if record['genus'] is None:

        # check last stage
        if record['stage'] in (None, 'identify'):

            # correct genus to 'Unidentified'
            record['genus'] = 'Unknown'

        # otherwise
        else:

            # correct genus to 'Other'
            record['genus'] = 'Other'
                
    return record


# Also, many of the records contain photo urls.  The **photo urls will be parsed** and the file names formatted according to a naming convention.

# In[40]:


designate('localizing latitude and longitude', 'processing')
    
# specify the location code for the photo based on its geo coordinates
def location_code_for_photo_naming(latitude, longitude):
    """Specify the location code for the photo naming convention.
    
    Arguments:
        latitude: float, the latitude
        longitude: float, the longitude
        
    Returns:
        str, the latlon code
    """
    
    # get latlon codes based on < 0 query
    latitudes = {True: 'S', False: 'N'}
    longitudes = {True: 'W', False: 'E'}
    
    # make latlon code from letter and rounded geocoordinate with 3 places
    latlon = latitudes[latitude < 0] + ('000' + str(abs(int(latitude))))[-3:]
    latlon += longitudes[longitude < 0] + ('000' + str(abs(int(longitude))))[-3:]
    
    return latlon


# In[41]:


designate('applying photo naming convention', 'processing')

# apply the naming convention to a photo url to make a file name
def photo_naming(urls, code, latitude, longitude, time):
    """Apply the naming convention to a group of urls
    
    Arguments:
        urls: list of str, the photo urls
        code: str, the photo sector code
        latitude: float, the latitude
        longitude: float, the longitude
        time: datetime object, the measurement time
        
    Returns:
        list of str, the filenames
    """
    
    # begin file name with protocol and latlon
    base = 'GLOBEMHM_' + location_code_for_photo_naming(latitude, longitude) + '_'
    
    # add the measurement time and sector code
    base += time.strftime('%Y%m%dT%H%MZ') + '_' + code
    
    # add index and unique id
    names = []
    for index, url in enumerate(urls):
        
        # add index, starting with 1
        name = base + str(index + 1)
            
        # add unique id and extension
        unique = url.split('/')[-2]
        name += '_' + unique + '.jpg'
        names.append(name)
        
    return names


# In[42]:


designate('parsing photo urls', 'processing')

# function for parsing photo urls
def parse_photo_info(record):
    """Parse photo url information.
    
    Arguments:
        record: dict
        
    Returns:
        dict
    """

    # dictionary of photo sector codes
    sectors = {'habitat': 'WS', 'body': 'FB', 'abdomen': 'AB'}

    # initialize fields for each sector and parse urls
    record['originals'] = []
    record['thumbs'] = []
    record['photos'] = []
    for field, stub in sectors.items():
        
        # split on semicolon, and keep all fragments with 'original'
        datum = record[field] or ''
        originals = [url.strip() for url in datum.split(';') if 'original' in url]
        
        # sort by the unique identifier as the number before the last slash
        originals.sort(key=lambda url: url.split('/')[-2])
        record['originals'] += originals
        
        # get the thumbnail versions
        thumbs = [url.split('original')[0] + 'small.jpg' for url in originals]
        record['thumbs'] += thumbs
        
        # apply the naming convention
        photos = photo_naming(originals, code, record['latitude'], record['longitude'], record['time'])
        record['photos'] += photos
        
    return record


# In[43]:


designate('processing records', 'processing')

# function for processing records
def process_records(records, primary=True):
    """Process all records.
    
    Arguments:
        records: list of dicts
        primary=True: boolean, primary record?
        
    Returns:
        list of dicts
    """
    
    # flatten and abbreviate all records
    records = [flatten_dict_list(record) for record in records]
    records = [abbreviate_field_name(record, primary) for record in records]
    records = [sync_time_with_long(record) for record in records]
    
    # process primary records
    if primary:
        
        # process
        records = [convert_str_to_float(record, 'count', 'larvae') for record in records]
        records = [identify_mosquito_genera(record) for record in records]
        records = [parse_photo_info(record) for record in records]
        
    # process secondary records
    if not primary:
        
        # process
        records = [convert_str_to_float(record, 'feature', feature) for record in records]
        
    return records


# Finally, it is sometimes the case that records contain **potential outliers**.  For instance, an entry of '1000000' for larvae counts is suspicous because likely no one counted one million larvae.  These data can skew analysis and dwarf the rest of the data in graphs. While there are [numerous ways](https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/) to detect and remove these outliers, the method implemented in this notebook uses the [interquartile range](https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/).
# 
# We can remove these outliers by setting a threshold for the upper quartile boundary. We then calculate the interquartile range by finding the $x$th percentile and the $100-x$th percentile and subtract the two.
# 
# We then calculate a cutoff value at 1.5 multiplied by the interquartile range and subract this value from the lower quartile and add it to the upper quartile.
# 
# Any value between the two values is considered valid and any value outside of that range is considered an outlier and is removed from the dataset.

# In[44]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# In[45]:


designate('outlier pruning', 'pruning')

# function to prune away outlying observations
def remove_outliers(records, field, threshold=75):
    """Prune away outlying observations based on the interquartile range.
    
    Arguments:
        records: list of dicts, the records
        field: str, field under inspection
        threshold: float, upper percentile; default = 75
        
    Returns:
        tuple of two lists of dicts, (pruned records, outliers)
    """

    # continually attempt pruning until there are no more records removed, but at least two remain
    outliers = []
        
    # reset number of records
    number = len(records)

    # calculate the mean and standard deviation
    values = [record[field] for record in records]
    q_l, q_u = percentile(values, 100-threshold), percentile(values, threshold)
    iqr = q_u - q_l
    cut_off = iqr * 1.5
    lower, upper = q_l - cut_off, q_u + cut_off
    
    # for each record
    for record in records:

        # set the quartiles
        record['lq'] = lower
        record['uq'] = upper

        # if the threshold is exceeded
        if record[field] < lower or record[field] > upper:

            # append to outliers
            outliers.append(record)
            
    records = [record for record in records if record[field] > lower and record[field] < upper]
        
    return records, outliers


# In[46]:


designate('sifting data through filter', 'filtering')

# function to sift data through filters
def filter_data_by_field(records, parameters, fields, functions, symbols):
    """Sift records according to parameters.
    
    Arguments:
        records: list of dicts
        parameters: list of settings
        fields: list of str
        functions: list of function objects
        symbols: list of str
        
    Returns:
        list of dicts, str
    """
    
    # begin criteria string
    criteria = ''

    # filter primaries based on parameters
    for parameter, field, function, symbol in zip(parameters, fields, functions, symbols):

        # check for None
        if parameter is not None:

            # filter
            if field in records[0].keys():
            
                # filter
                records = [record for record in records if function(record[field], parameter)]

                # add to criteria string
                criteria += '{} {} {}\n'.format(field, symbol, parameter)
                
    # sort data by date and add an index
    records.sort(key=lambda record: record['date'])
    [record.update({'index': index}) for index, record in enumerate(records)]

    return records, criteria


# In[47]:


designate('chopping data up into bins', 'histograms')

# chopping data into histogram bars
def get_bins_width_min_max_for_hist(observations, width=1, limit=1000):
    """Chop the observations from the records up into bars
    
    Arguments:
        observations: list of floats
        width=1: float, minimum width of each bar
        limit: int, maximum number of bins
        
    Returns:
        (float, float) tuple, the number of bins and the width
    """

    # adjust width until number of bins is less than a limit
    bins = limit + 1
    width = width / 10
    while bins > limit:
    
        # multiply width by 10
        width *= 10
    
        # calculate the number of histogram bins
        minimum = (int(min(observations) / width) * width) - (width * 0.5)
        maximum = (int(max(observations) / width) * (width + 1)) + (width * 0.5)
        bins = int((max(observations) - min(observations)) / width) + 1

    # readjust maximum to cover an even number of bins
    maximum = minimum + bins * width

    return bins, width, minimum, maximum


# In[48]:


designate('zooming in on best view', 'histograms')

# function to define horizontal and vertical ranges of the graph
def zoom_to_best_x_y_ranges(observations, counts, width, percent=1):
    """Zoom in on the best horizontal and vertical view ranges.
    
    Arguments:
        observations: list of float
        counts: list of counts per bin
        width: width of each bin
        percent: float, the percentile margin
        
    Returns:
        tuple of tuples of floats, the view boundaries
    """
    
    # make left and right boudaries as a width past the percentiles
    left = percentile(observations, percent) - width
    right = percentile(observations, 100 - percent) + width
    horizontal = (left, right)
    
    # make up and down boudaries based on counts
    down = 0
    up = max(counts) * 1.1
    vertical = (down, up)
    
    return horizontal, vertical


# In[49]:


designate('begin drafting a histogram', 'histograms')

# function for drafting a histogram
def draft_histogram(field, horizontal, vertical, mean, deviation):
    """Draft a histogram with beginning boundary information.
    
    Arguments:
        field: str
        horizontal: (float, float) tuple, the horizontal extent
        vertical: (float, float) tuple, the vertical extent
        mean: float, the mean of the observations
        deviation: standard deviation of the observations
        
    Returns:
        bokeh figure object
    """
    
    # create parameters dictionary for histogram labels
    parameters = {}
    parameters['title'] = 'Histogram for {}'.format(make_title(field))
    parameters['x_axis_label'] = '{}'.format(field)
    parameters['y_axis_label'] = 'observations'
    parameters['x_range'] = horizontal
    parameters['y_range'] = vertical
    parameters['plot_height'] = 400
    parameters['plot_width'] = 450
    
    # set extra axis
    starting = (horizontal[0] - mean) / deviation
    ending = (horizontal[1] - mean) / deviation
    parameters['extra_x_ranges'] = {'z-score': Range1d(start=starting, end=ending)}
    
    # initialize the bokeh graph with the parameters
    gram = figure(**parameters)
                                                       
    # label the histogram
    formats = round(mean, 2), round(deviation, 2)
    label = 'Overlay of normal distribution (mean={}, std={})'.format(*formats)
    gram.add_layout(LinearAxis(x_range_name='z-score', axis_label=label), 'above')
    
    # add annotations
    annotations = [('{}:'.format(truncate_field_name(field, 6)), '@left to @right')]
    annotations += [('Observations:', '@ys')]
    annotations += [('Z-score:', '@scores')]
    
    # activate the hover tool
    hover = HoverTool(tooltips=annotations)
    gram.add_tools(hover)
    
    return gram


# In[50]:


designate('blocking in bars on the histogram', 'histograms')

# function to block in bars on the histogram
def draw_bars(gram, counts, edges, mean, deviation):
    """Block in bars on the histogram.
    
    Arguments:
        gram: bokeh figure
        counts: list of floats, the bin counts
        edges: list of floats, the bin edges
        mean: float
        deviation: float
        
    Returns:
        bokeh figure
    """
    
    # get middle points
    middles = [(right + left) / 2 for left, right in zip(edges[:-1], edges[1:])]
    
    # calculate z-scores for all middles
    scores = [(middle - mean) / deviation for middle in middles]
    
    # accumulate the info into a table
    table = {'ys': counts, 'left': edges[:-1], 'right': edges[1:]}
    table.update({'scores': scores, 'xs': middles})
    table = ColumnDataSource(table)
    
    # set parameters for drawing the bars, indicating the source of the data
    bars = {'source': table}
    bars.update({'left': 'left', 'right': 'right', 'bottom': 0, 'top': 'ys'})
    bars.update({'line_color': 'white', 'fill_color': 'lightgreen'})
    
    # add to histogram
    gram.quad(**bars)
    
    return gram


# In[51]:


designate('normalizing observations to a normal distribution', 'histograms')

# function to produce the normalization curve
def draw_gaussian_on_bargraph(gram, counts, edges, mean, deviation):
    """Normalize the observations by drawing the normal distribution.
    
    Arguments:
        gram: bokeh figure
        counts: list of floats, the bin counts
        edges: list of floats, the bin edges
        mean: float
        deviation: float
        
    Returns:
        bokeh figure
    """
    
    # create line from z-score of -4 to 4
    scores = [tick * 0.01 - 4.0 for tick in range(801)]
    xs = [(score * deviation) + mean for score in scores]
    
    # create gaussian fucntion
    area = sum([count * (right - left) for count, left, right in zip(counts, edges[:-1], edges[1:])])
    height = area / (deviation * sqrt(2 * pi))
    normalizing = lambda x: height * exp(-(x - mean) ** 2 / (2 * deviation ** 2))
    ys = [normalizing(x) for x in xs]
    ys = [round(y, 3) for y in ys]

    # make column object
    table = ColumnDataSource({'xs': xs, 'ys': ys, 'scores': scores, 'left': xs, 'right': xs})
    summary = 'Normal Distribution'
    gram.line(source=table, x='xs', y='ys', color='blue')
    
    # draw standard lines
    for score in (-3, -2, -1, 0, 1, 2, 3):
        
        # draw std lines
        xs = [(deviation * score) + mean] * 2
        ys = [0, normalizing(xs[0])]
        table = ColumnDataSource({'xs': xs, 'ys': ys, 'scores': [score, score], 'left': xs, 'right': xs})
        gram.line(source=table, x='xs', y='ys', color='blue')
    
    return gram


# In[52]:


designate('constructing a bar graph', 'histograms')

# function for constructing a histogram
def construct_bargraph(records, field, width=1):
    """Make a histogram from the dataset.
    
    Arguments:
        record: list of dicts, the records
        field: str, the field of interest
        width=1: int, the width of each histogram bar
        
    Returns:
        bokeh figure object
    """
    
    # gather up observations
    observations = [record[field] for record in records]
    
    # separate into bins
    bins, width, minimum, maximum = get_bins_width_min_max_for_hist(observations, width)
    
    # get the counts and edges of each bin
    counts, edges = histogram(observations, bins=bins, range=(minimum, maximum))
    
    # get the zoom coordinates
    horizontal, vertical = zoom_to_best_x_y_ranges(observations, counts, width)

    # get the normal distribution, defaulting to a small deviation in case of zero
    mean = average(observations)
    deviation = max([0.000001, std(observations)])
    
    # begin histogram
    gram = draft_histogram(field, horizontal, vertical, mean, deviation)
    
    # block in bars on the histogram
    gram = draw_bars(gram, counts, edges, mean, deviation)
    
    # draw in equivalent normal distribution
    gram = draw_gaussian_on_bargraph(gram, counts, edges, mean, deviation)
    
    return gram


# ### Assembling Associations

# Because the two sets of measurements were not taken concurrently, there must be some criteria to determine when measurements from one protocol correspond to measurements from the other protocol.  The method implemented here is a weighing function that determines how strongly to weigh the association between the two data sets, based on the following parameters:
#     
# - distance: the distance in kilometers between measurements that will be granted full weight.
#     
# - interval: the time interval in days between measurements that will be granted full weight.
#     
# - lag: the time in days to anticipate an effect on mosquitoes from a secondary measurement some 
#     days before.
#     
# - confidence: the weight to grant a measurement twice the distance or interval. This determines how steeply the weighting shrinks as the intervals are surpassed.  A high confidence will grant higher weights to data outside the intervals.  A confidence of zero will have no tolerance for data slightly passed the interval.
#     
# - cutoff: the minimum weight to consider in the dataset.  A cutoff of 0.1, for instance, will only retain data if the weight is at least 0.1.
#     
# - inclusion: the maximum number of nearest secondary measurements to include for each mosquitoes measurement.
# 
# The sketch below shows several examples differing in their confidence parameter.

# In[53]:


designate('weighing function', 'weighting')

# weigh a pair of records according to the space and time between them
def weigh_record_pairs_by_space_and_time(space, time, settings):
    """Weigh the significance of a correlation based on the space and time between them
    
    Arguments:
        space: float, space in distance
        time: float, the time time in interval
        settings: dict
        
    Returns:
        float, the weight
    """
    
    # unpack settings
    distance, interval, lag = settings['distance'], settings['interval'], settings['lag']
    confidence, cutoff = settings['confidence'], settings['cutoff']
    
    # set default weight and factors to 1.0
    weight = 1.0
    factor = 1.0
    factorii = 1.0
    
    # if beyond the space, calculate the gaussian factor
    if abs(space) > distance:
        
        # default factor to 0, but calculate gaussian
        factor = 0.0
        if confidence > 0:
        
            # calculate the gaussian factor (e ^ -a d^2 = c), a = -ln (c) / d^2
            alpha = -log(confidence) / distance ** 2
            factor = exp(-alpha * (abs(space) - distance) ** 2)
            
    # if beyond time time, calculate gaussian factor
    if abs(time - lag) > interval:
        
        # default factor to 0, but calculate gaussian
        factorii = 0.0
        if confidence > 0:
            
            # calculate the gaussian factor (e ^ -a d^2 = c), a = -ln (c) / d^2
            beta = -log(confidence) / interval ** 2
            factorii = exp(-beta * (abs(time - lag) - interval) ** 2)
            
    # multiply by factors and apply cutoff
    weight *= (factor * factorii)
    weight *= int(weight >= cutoff)
    
    return weight


# In[54]:


designate('sketching weight function')

# set up faux settings
faux = {'distance': 10, 'interval': 1, 'lag': 0, 'confidence': 0.0, 'cutoff': 0.1}
fauxii = {'distance': 10, 'interval': 1, 'lag': 0, 'confidence': 0.5, 'cutoff': 0.1}
fauxiii = {'distance': 10, 'interval': 1, 'lag': 0, 'confidence': 0.8, 'cutoff': 0.1}

# make functions
weighting = lambda x: weigh_record_pairs_by_space_and_time(10, x, faux)
weightingii = lambda x: weigh_record_pairs_by_space_and_time(10, x, fauxii)
weightingiii = lambda x: weigh_record_pairs_by_space_and_time(10, x, fauxiii)

# make legend
legend = ['{}'.format(scheme['confidence']) for scheme in [faux, fauxii, fauxiii]]
title = "Weighting Scheme Example"
xlab = "Parameter"
ylab = "Weight"
# sketch
sketch_plot(weighting, weightingii, weightingiii, legend=legend, title=title, xlab=xlab, ylab=ylab)


# All the secondary measurements are accumulated in a Balltree structure that automatically places measurements close together in the tree that are close together spatially and timewise.  The tree can then be queried with the time and geolocation information from a mosquito record to find the closest secondary measurements.  The weight is calculated for each secondary measurement, and if the weight is greater than the cutoff it is included in the dataset.

# Note that potentially several secondary measurements may be associated with a single mosquitoes measurement.  This has the effect of creating more regression points than mosquito records used.  Because statistical significance tends to increase with increasing number of samples, the p-value calculation of the potential for sampling bias is performed with reference to only the number of mosquito records used.

# In[55]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# In[56]:


designate('plotting weighting scheme', 'weighting')

# function to plot weighting scheme
def plot_plateaued_weighting_scheme(field, settings):
    """Plot an example of the weighting function.
    
    Arguments:
        field: str
        settings: dict
        
    Returns:
        bokeh object
    """
    
    # set offset parameter if an interval plot
    offset = settings['lag']

    # create functions
    functions = {'distance': lambda x: weigh_record_pairs_by_space_and_time(x, offset, settings), 'interval': lambda x: weigh_record_pairs_by_space_and_time(0, x, settings)}
    
    # calculate point range
    start = (settings[field] * -5) + offset
    finish = (settings[field] * 5) + offset
    total = finish - start
    chunk = total / 1000
    
    # calculate points
    xs = [start + (number * chunk) for number in range(1001)]
    ys = [functions[field](x) for x in xs]
    table = ColumnDataSource({'xs': xs, 'ys': ys})
    
    # create annotations
    annotations = [('{}'.format(field + units[field](feature)), '@xs')]
    annotations += [('weight', '@ys')]

    # create title
    title = 'Weighting Scheme visualization based on {} of {} {}'.format(field, settings[field], units[field](feature))
    if field == interval:
        
        # add delay
        title += ' with a {} day lag'.format(offset)
    
    # create parameters dictionary for guassian
    parameters = {}
    parameters['title'] = title
    parameters['x_axis_label'] = field
    parameters['y_axis_label'] = 'weight'
    parameters['plot_height'] = 300
    parameters['plot_width'] = 450

    # make graph
    plateau = figure(**parameters)
    plateau.line(source=table, x='xs', y='ys', line_width=1)

    # activate the hover tool
    hover = HoverTool(tooltips=annotations)
    plateau.add_tools(hover)
    
    return plateau


# In[57]:


designate('haversine', 'weighting')

# haversine to calculate distance in kilometers from latitude longitudes
def haversine_get_distance_from_lat_lon(latitude, longitude, latitudeii, longitudeii):
    """Use the haversine function to calculate a distance from latitudes and longitudes.
    
    Arguments:
        latitude: float, latitude of first location
        longitude: float, longitude of first location
        latitudeii: float, latitude of second location
        longitudeii: float, longitude of second location
        
    Returns:
        float, the distance
    """
    
    # radius of the earth at the equator in kilometers
    radius = 6378.137
    
    # get distance from memory
    reference = (latitude, longitude, latitudeii, longitudeii)
    distance = memory.setdefault(reference, None)
    if distance is None:
    
        # convert to radians
        latitude = latitude * pi / 180
        latitudeii = latitudeii * pi / 180
        longitude = longitude * pi / 180
        longitudeii = longitudeii * pi / 180

        # calculate distance with haversine formula
        # d = 2r arcsin (sqrt (sin^2((lat2 - lat1) /2) + cos(lat1)cos(lat2)sin^2((lon2 - lon1) / 2)))
        radicand = sin((latitudeii - latitude) / 2) ** 2
        radicand += cos(latitude) * cos(latitudeii) * sin((longitudeii - longitude) / 2) ** 2
        distance = 2 * radius * arcsin(sqrt(radicand))
        
        # add to memory
        memory[reference] = distance
    
    return distance


# In[58]:


designate('get day difference', 'weighting')

# function to measure number of days between datetimes
def get_day_difference(time, timeii):
    """Measure number of days between two datetimes.
    
    Arguments:
        time: float, first timestamp
        timeii: float, second timestamp
        
    Returns:
        float, number of days
    """
    
    # find time delta
    delta = timeii - time
    
    # convert to days
    conversion = delta / (24 * 60 * 60)
    
    return conversion


# In[59]:


designate('measuring between records', 'weighting')

# tuple of distance and interval
def measure_dist_time_bw_records(record, recordii):
    """Measure the distance and time interval between two records.
    
    Arguments:
        record: record dict
        recordii: record dict
        
    Returns:
        (float, float) tuple, the distance and interval
    """
    
    # unpack first record
    latitude = record['latitude']
    longitude = record['longitude']
    time = record['time'].timestamp()
    
    # unpack second record
    latitudeii = recordii['latitude']
    longitudeii = recordii['longitude']
    timeii = recordii['time'].timestamp()
    
    # calculate distance and interval
    distance = haversine_get_distance_from_lat_lon(latitude, longitude, latitudeii, longitudeii)
    interval = get_day_difference(time, timeii)
    
    return distance, interval


# In[60]:


designate('adjusting to euclidean points', 'assembling')

# function for adjusting a lat long vector to euclidean points
def convert_euclidean_point(record, settings, lag=None):
    """Adjust a geotime vector to a euclidean point.
    
    Arguments:
        record: dict
        settings: dict
        lag=None: float, lag in days (overrides settings)
        
    Returns:
        euclidean vector
    """
    
    # unpack settings
    distance, interval = settings['distance'], settings['interval']
    
    # get lag value
    if not lag:
        
        # get from settings
        lag = settings['lag']
    
    # determine number of latitude chunks based on distance sensitivity
    latitude = haversine_get_distance_from_lat_lon(record['latitude'], record['longitude'], 0, record['longitude']) / distance
    
    # determine number of longitude chunks based on distance sensitivity
    longitude = haversine_get_distance_from_lat_lon(record['latitude'], record['longitude'], record['latitude'], 0) / distance
    
    # determine number of time chunks based on time sensitivity
    time = lag + get_day_difference(0, record['time'].timestamp()) / interval
    
    # form euclidean vector
    euclidean = [latitude, longitude, time]
    
    return euclidean


# In[61]:


designate('planting the tree', 'assembling')

# plant a Balltree for closest sample retrieval
def organize_records_by_space_time(records, settings):
    """Plant a Balltree to organize records by space and time.
    
    Arguments:
        records: list of dicts
        settings: dict
    
    Returns:
        sklearn Balltree object
    """
    
    # form matrix from secondary records
    matrix = array([convert_euclidean_point(record, settings) for record in records])

    # construct the balltree
    tree = BallTree(matrix, leaf_size=2, metric='euclidean')   
    
    return tree


# In[62]:


designate('querying tree', 'assembling')

# querying tree function to retrieve record indices of the closet records
def find_closest_neighbor(tree, records, settings):
    """Query the balltree for closest neighbors to each primary record.
    
    Arguments:
        tree: sklearn balltree, the tree to query
        records: list of records dict
        settings: dict
        
    Returns:
        list of list of dicts
    """
    
    # set number to not exceed number of samples
    number = min([settings['inclusion'], len(dataii)])
    
    # query the tree
    matrix = [convert_euclidean_point(record, settings, 0) for record in records]
    results = tree.query(matrix, k=number, return_distance=False)
    
    # get associated records and sort by feature size
    neighbors = [[dataii[int(entry)] for entry in indices] for indices in results]
    for cluster in neighbors:
        
        # sort by feature size
        cluster.sort(key=lambda record: record[feature], reverse=True)
    
    return neighbors


# In[63]:


designate('webbing together associations', 'assembling')

# form associations between records
def merge_datasets(settings):
    """Web together samples in the second dataset with those of the first based on parameters.
    
    Arguments:
        settings: dict
        
    Returns:
        list of tuples, the associations
    """
    
    # plant a Balltree based on euclidean distances
    tree = organize_records_by_space_time(dataii, settings)
    
    # get all the nearest neighbors for each primary record
    neighbors = find_closest_neighbor(tree, data, settings)

    # assemble associations
    associations = []
    for record, cluster in zip(data, neighbors):

        # begin association
        location = (record['latitude'], record['longitude'])
        association = {'record': record, 'associates': [], 'location': location}
        for neighbor in cluster:

            # weigh the records, only keeping those with a 1% confidence
            space, time = measure_dist_time_bw_records(record, neighbor)            
            weight = weigh_record_pairs_by_space_and_time(space, time, settings)
            if weight > 0.0:

                # add to association
                association['associates'].append({'record': neighbor, 'weight': weight})

        # append if nonzero
        if len(association['associates']) > 0:
            
            # sort associates by weight
            association['associates'].sort(key=lambda associate: associate['weight'], reverse=True)
            associations.append(association)

    # sort by highest weight
    associations.sort(key=lambda association: association['associates'][0]['weight'], reverse=True)
    
    # add reference number
    [association.update({'reference': index}) for index, association in enumerate(associations)]
    
    return associations


# In[64]:


designate('summarizing associations in a table', 'assembling')

# function to summarize associations into a data table
def summary_df(associations):
    """Summarize association information into a data table.
    
    Arguments:
        associations: list of dicts
        
    Returns:
        panda data frame
    """

    # make a table from associations
    summary = []

    # begin headers
    base = ['pair', 'point', 'weight', 'distance', 'days']
    headers = ['larvae', 'genus', 'date', 'latitude', 'longitude']
    headersii = [feature, 'date', 'latitude', 'longitude']
    parameters = [setting + '_setting' + units[setting](feature) for setting in settings.keys()]

    # get remaining headers
    remainder = [key for key in associations[0]['record'].keys() if key not in headers]
    remainderii = [key for key in associations[0]['associates'][0]['record'].keys() if key not in headersii]
    remainder.sort(key=lambda field: len(field))
    remainderii.sort(key=lambda field: len(field))

    # go through each association
    for index, association in enumerate(associations):

        # go through each associate
        record = association['record']
        for number, pair in enumerate(association['associates']):

            # begin row
            associate = pair['record']
            weight = pair['weight']
            row = [(association['reference'], number), (round(associate[feature], 2), round(record['larvae']))]
            row += [round(weight, 2)]
            row += [haversine_get_distance_from_lat_lon(record['latitude'], record['longitude'], associate['latitude'], associate['longitude'])]
            row += [get_day_difference(record['time'].timestamp(), associate['time'].timestamp())]

            # add primary record fields, secondary record fields, and parameters
            row += [record[field] for field in headers]
            row += [associate[field] for field in headersii]
            row += [value for value in settings.values()]

            # add all remaining primary and secondary record fields
            row += [record[field] for field in remainder]
            row += [associate[field] for field in remainderii]

            # add row to table
            summary.append(row)

    # construct labels, adding prefix in the case of duplicates
    labels = base + headers + headersii + parameters + remainder + remainderii
    duplicating = lambda index, label: '2nd_' + label if label in labels[:index] else label
    labels = [duplicating(index, label) for index, label in enumerate(labels)]
    
    # create dataframe from data
    summary = pandas.DataFrame.from_records(summary, columns=labels)
    
    return summary


# In[65]:


designate('subsampling by genus', 'assembling')

# subset the associations for a particular genus
def get_subset_of_data(associations, genus):
    """Sub sample the associations based on a particular genus.
    
    Arguments:
        associations: list of dicts
        genus: str, the genus name
        
    Returns:
        list of dicts
    """
    
    # take subsample
    subsample = []
    for association in associations:
        
        # check the genus
        if genus == 'All' or genus == association['record']['genus']:
            
            # add to subsample
            subsample.append(association)
            
    return subsample


# In[66]:


designate('assembling samples', 'assembling')

# assemble samples from association routine
def assemble_samples(associations):
    """Assemble samples from associations.
    
    Arguments:
        associations: list of dicts
        
    Returns:
        list of dicts
    """

    # form samples
    samples = []
    for association in associations:

        # go through each match
        record = association['record']
        for index, associate in enumerate(association['associates']):
            
            # make sample
            larvae = record['larvae']
            if larvae != larvae:
                continue
            measurement = associate['record'][feature]
            if measurement != measurement:
                continue
            weight = associate['weight']
            genus = record['genus']
            sample = {'y': larvae, 'x': measurement, 'weight': weight, 'genus': genus}
            
            # add other record info
            sample.update({'latitude': record['latitude'], 'longitude': record['longitude']})
            sample.update({'site': record['siteName'], 'organization': record['organizationName']})
            sample.update({'time': str(record['time'])})
            
            # add color and size components
            sample['color'] = indicators[genus]
            sample['size'] = int(1 + 20 * weight)
            
            # add indices to list
            sample['pair'] = (association['reference'], index)
            
            # add to list
            samples.append(sample)
            
    # sort
    samples.sort(key=lambda sample: sample['x'])
    
    return samples


# ### Ordinary Least Squares and Weighted Least Squares

# The goal is to find a relationship between the secondary measurements and the larvae counts.  The approach taken here is to use the method of Weighted Least Squares to find the relationship that best fits the data.  In the simplest case, this relationship can take the form of a straight line:
# 
# \begin{align*}
# y_i=\beta_0 + \beta_1 x_i
# \end{align*}
# 
# A quadratic relationship may also be tried:
# 
# \begin{align*}
# y_i=\beta_0 + \beta_1 x_i + \beta_2 x_i^2
# \end{align*}
# 
# The goal is to find the best $\beta$'s that minimize the distance between the estimated line and the data points for all data points (i=1,...,n).
# 
# The vector form for all of the data points is more compact:
# 
# \begin{align*}
# \begin{bmatrix}
# y_1\\
# y_2\\
# \vdots \\
# y_n
# \end{bmatrix} &= \begin{bmatrix}
# 1 & x_{1,1} & x_{1,2}\\
# 1 & x_{2,1} & x_{2,2} \\
# \vdots & \vdots & \vdots \\
# 1 & x_{n,1} & x_{n,2}
# \end{bmatrix} \begin{bmatrix}
# \beta_0\\
# \beta_1 \\
# \beta_2
# \end{bmatrix} +  \begin{bmatrix}
# \varepsilon_1\\
# \varepsilon_2 \\
# \vdots \\
# \varepsilon_n
# \end{bmatrix}
# \\
# \mathbf{y}&=\mathbf{X}\hat{\beta} + \mathbf{\varepsilon}
# \end{align*}
# 
# Each actual observation, $y_i$, will differ somewhat from that estimated by the $\hat{\beta}$'s and the particular dependent observation $x_i$. As mentioned earlier, the difference between the actual and predicted value is called a residual, $\epsilon_i$ (the estimated error):
# 
# \begin{align*}
# \mathbf{y}-\mathbf{X}\hat{\beta}=\hat{\mathbf{\varepsilon}}
# \end{align*}
# 
# The goal is to find the values for $\hat{\beta}$ that minimize the sum of all squared residuals, $\frac{1}{n}\sum_{i=1}^n{\hat{\varepsilon}_i^{2}}$, hence "ordinary least squares" ([OLS](https://en.wikipedia.org/wiki/Ordinary_least_squares)). We can derive the OLS solution as follows:
# 
# \begin{align*}
# \mathbf{y}&=\mathbf{X}\hat{\beta} + \mathbf{\varepsilon} \\
# \mathbf{\varepsilon}&=\mathbf{X}\hat{\beta} - \mathbf{y} \\
# \Rightarrow \mathbf{\varepsilon}^T\mathbf{\varepsilon} &= (\mathbf{X}\hat{\beta} - \mathbf{y})^T(\mathbf{X}\hat{\beta} - \mathbf{y}) \tag{Least squares} \\
# &= \mathbf{y}^T\mathbf{y}-2\hat{\beta}^T\mathbf{X}^T\mathbf{y}+ \hat{\beta}^T\mathbf{X}^T\mathbf{X}\hat{\beta} \\
# \Rightarrow \frac{\delta}{\delta\hat{\beta}} &= -2\mathbf{X}^T\mathbf{y}+ 2\mathbf{X}^T\mathbf{X}\hat{\beta} = 0 \tag{Minimize} \\
# \mathbf{X}^T\mathbf{y} &= \mathbf{X}^T\mathbf{X}\hat{\beta} \tag{Reorder} \\
# \Rightarrow \hat{\beta} &= (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y} \tag{Solution}
# \end{align*}
# 
# In "weighted least squares",the errors do not have the same variance and therefore the [variance covatiance matrix](https://en.wikipedia.org/wiki/Covariance_matrix) are weighted differently, therefore the weighted sum of squares is minimized, $\frac{1}{n}\sum_{i=1}^n{w_i\hat{\epsilon_i^{2}}}$. The $\hat{\beta}$ for weighted least squares would be found as follows:
# 
# 
# \begin{align*}
# \mathbf{\varepsilon}^T\mathbf{W}\mathbf{\varepsilon} &= (\mathbf{X}\hat{\beta} - \mathbf{y})^T\mathbf{W}(\mathbf{X}\hat{\beta} - \mathbf{y}) \tag{Least squares} \\
# &= \mathbf{y}^T\mathbf{W}\mathbf{y}-\mathbf{y}^T\mathbf{W}\mathbf{X}\hat{\beta} - \hat{\beta}^T\mathbf{X}^T\mathbf{W}\mathbf{y}+\hat{\beta}^T\mathbf{X}^T\mathbf{W}\mathbf{X}\hat{\beta} \\
# \Rightarrow \frac{\delta}{\delta\hat{\beta}} &= -2\mathbf{X}^T\mathbf{W}\mathbf{y} + 2 \mathbf{X}^T\mathbf{W}\mathbf{X}\hat{\beta} = 0 \tag{Minimize}\\
# \mathbf{X}^T\mathbf{W}\mathbf{y} &= \mathbf{X}^T\mathbf{W}\mathbf{X}\hat{\beta} \tag{Reorder} \\
# \Rightarrow \hat{\beta} &= (\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{W}\mathbf{y} \tag{Solution}
# \end{align*}

# In[67]:


designate('approximating fit with linear least squares', 'regressions')

def get_coefficients_linear_regression(samples, mode):
    """Approximate the regression fit to use for initial starting points.
    
    Arguments:
        samples: list of dicts
        mode: str, regression mode
        
    Returns:
        list of floats, the coefficients
    """
    
    # calculate weighted averages
    weights = [sample['weight'] for sample in samples]
    mean = average([sample['x'] for sample in samples], weights=weights)
    meanii = average([sample['y'] for sample in samples], weights=weights)
    
    # create matrix from samples
    transforming = regressions[mode]['independent']
    matrix = array([transforming(sample['x'], mean) for sample in samples]).reshape(-1, 1)
    
    # create targets from samples
    transforming = regressions[mode]['dependent']
    targets = array([transforming(sample['y'], meanii) for sample in samples]).reshape(-1, 1)
    
    # create list of weights
    weights = array([sample['weight'] for sample in samples])
    
    # create polynomial to fit to higher order polynomials
    fitter = PolynomialFeatures(degree=regressions[mode]['polynomial'])
    polynomial = fitter.fit_transform(matrix)
    
    # fit regression model, keeping intercept in coefficients vector
    model = PoissonRegressor(fit_intercept=False).fit(polynomial, targets, weights)#PoissonRegressor(fit_intercept=False).fit(polynomial, targets, weights)
    coeffs = model.coef_
    coefficients = [float(entry) for entry in coeffs]
    
    # add weighted means to coefficients
    coefficients += [mean, meanii]
    
    return coefficients


# In[68]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# ### Nonlinear Least Squares

# As hinted at above, the linear method can fit a variety of shapes beyond straight lines.  In fact any curve with the general form:
# 
# \begin{align*}
# y=\beta_0+\beta_1f_1(x)+\beta_2f_2(x)...
# \end{align*}
# 
# can be solved through linear least squares, with the requirement that each $\beta$ is a constant parameter.  The various functions $f(x)$ can be anything, as long as the $\beta$'s are outside the functions themselves.  Consider, however, an exponential relationship:
# 
# \begin{align*}
# y=\beta_0 + \beta_1e^{\beta_2x}
# \end{align*}
# 
# In this case, $\beta_0$ and $\beta_1$ is outside the exponential function, but $\beta_2$ is not.  Finding the best $\beta_2$ is not solvable via linear least squares.  The [nonlinear alternative](https://www.itl.nist.gov/div898/handbook/pmd/section1/pmd142.htm) is similar with these key differences:
# - the nonlinear problem must begin with an initial estimate of the parameters that is fairly good (otherwise it may only find local solutions).
# - the nonlinear problem cannot be solved exactly, but must be approached asymptotically better estimates.
# 
# #### Less interested in the math:
# Please see this website if you are less interested in the math: [link](https://physics.nyu.edu/pine/pymanual/html/chap8/chap8_fitting.html#nonlinear-fitting)
# 
# #### More interested in the math:
# 
# The goal is the same as in the linear case.  Consider a function $y = f(x; \hat{\beta})$ that might be a good model for the relationship between mosquito larvae counts and the secondary data:
# 
# In the linear least squares case, this model took the strictly linear form:
# 
# \begin{align*}
# \mathbf{y}=\mathbf{X}\hat{\beta}
# \end{align*}
# 
# but that restriction is lifted here.  Consider a model with two unknown parameter, $\beta_0$ and $\beta_1$.  Just as in the linear case, there will be some residual $\epsilon_i$ between the actual observations $y_i$ and the predictions of the model:
# 
# \begin{align*}
# \mathbf{y}-f(X; \hat{\beta})=\hat{\varepsilon}
# \end{align*}
# 
# As before, the goal is to find the set of $\beta$'s that minimize $S = \sum_{i=1}^{n}\varepsilon^2$. The minimum value of S occurs when the gradient is zero:
# 
# \begin{align*}
# \frac{\delta S}{\delta \beta_j} = 2 \sum_i \varepsilon_i \frac{\delta \varepsilon_i}{\delta \beta_j} = 0
# \end{align*}
# 
# Note that in a non linear system, the gradient equations do not have a closed solution which is why an initial guess is necessary. The parameters are then refined iteratively as follows:
# 
# \begin{align*}
# \beta_j \approx \beta_j^{k+1} = \beta_j^k + \delta \beta_j
# \end{align*}
# where $k$ is the number of iterations and $\delta \beta$ is known as the shift vector.
# 
# At each iteration, the model is approximately linearized using a [Taylor Polynomial](https://tutorial.math.lamar.edu/classes/calcii/taylorseries.aspx#:~:text=the%20n%20th%20degree%20Taylor%20polynomial%20is%20just%20the%20partial,polynomial%20for%20a%20given%20n%20.) expansion around $\beta^k$:
# \begin{align*}
# f(x_i;\beta)\approx f(x_i;\beta^k)+\sum_j J_{ij}\Delta \beta_j
# \end{align*}
# where $J_{ij} = -\frac{\delta \varepsilon_i}{\delta\beta_j}$ is the $i,j$th element of the [Jacobian](https://mathworld.wolfram.com/Jacobian.html).
# 
# Therefore the residuals are:
# \begin{align*}
# \Delta y_i = y_i - f(x_i, \beta^k)
# \varepsilon_i = \Delta y_i - \sum_j J_{ij}\Delta \beta_j
# \end{align*}
# 
# Plugging everything into the gradient equation we get:
# \begin{align*}
# -2 \sum_i J_{ij}(\Delta y_i - \sum_\ell J_{i\ell}\Delta \beta_\ell) = 0
# \end{align*}
# 
# Which gives us a new set of [normal equations](https://mathworld.wolfram.com/NormalEquation.html):
# \begin{align*}
# (\mathbf{J}^T\mathbf{J})\Delta\beta = \mathbf{J}^T\Delta\mathbf{y}
# \end{align*}
# which we can then use to find:
# \begin{align*}
# \Delta\beta\approx(\mathbf{J}^T\mathbf{J})^{-1}\mathbf{J}^T\Delta\mathbf{y}
# \end{align*}
# 
# Note, that for [**Weighted Nonlinear least squares**](http://www2.imm.dtu.dk/pubdb/edoc/imm2804.pdf), the normal equatios are:
# \begin{align*}
# (\mathbf{J}^TW\mathbf{J})\Delta\beta = \mathbf{J}^TW\Delta\mathbf{y}
# \end{align*}
# 
# There are multiple algorithms for solving the nonlinear least squares problem. The best known algorithm is the [Gauss-Newton algorithm](http://fourier.eng.hmc.edu/e176/lectures/NM/node36.html). In this notebook, the [Levenberg-Marquardt algorithm](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm) is implemented.

# In[69]:


designate('tighening with nonlinear least squares', 'regressions')

# perform nonlinear regression
def perform_nonlinear_regression(samples, mode, coefficients, calculate=True, direct=False, limit=1000):
    """Tighten a regression model to a nonlinear function.
    
    Arguments:
        samples: list of dicts, the samples
        mode: dict
        coefficients: list of floats, the coefficients from linear approximation
        calculate: boolean, calculate jacobian directly?
        direct: boolean, use coefficients directly as initials?
        limit: int, max number of iteractions
        
    Returns:
        list of float, the parameters
    """
    
    # assemble the matrix of secondary measurements
    matrix = array([sample['x'] for sample in samples])
    
    # get list of targets, taking logarithm if appropriate
    targets = array([sample['y'] for sample in samples])
    
    # create list of weights as if they are the sqrt of standard deviations
    sigmas = array([1 / sqrt(sample['weight']) for sample in samples])
    
    # set up the curve
    function = regressions[mode]['function']
    
    # get initial values
    initials = coefficients
    if not direct:
        
        # get initial values from coefficients
        initials = regressions[mode]['initial'](*coefficients)

    # calculate jacobian
    jacobian = None
#     if calculate:
        
#         # get jacobian from function
#         jacobian = regressions[mode]['jacobian']

    # fit the curve
    curve, _ = curve_fit(function, matrix, targets, p0=initials, sigma=sigmas, maxfev=limit, jac=jacobian)
    
    # raise value error if nans are there
    if any([isnan(entry) for entry in curve]):
        
        # raise value error
        raise ValueError
    
    return curve


# In[70]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# ### Measuring Significance

# We can check how well our model fits the data by checking the correlation between the actual observations and the fitted model. While the correlation will ideally be close to 1 indicating a perfect positive relationship between the variables, we also want the slope of the line to be 1. So, the equation that we would like to see is: $y_{observed} = y_{predicted}$ (i.e. a 1 to 1 relationship). [Note this also applies to $R^2$](https://en.wikipedia.org/wiki/Coefficient_of_determination).
# 
# <!-- First, we need to find the mean of the observations: -->
# <!-- \begin{align*}
# \overline{y_{ob}}=\frac{\sum_{i=1}^{n}{y_{ob, i}}}{n}
# \end{align*} -->
# <!-- 
# and the mean of the predictions:
# \begin{align*}
# \overline{y_{pr}}=\frac{\sum_{i=1}^{n}{y_{pr, i}}}{n}
# \end{align*} -->
# It is important to determine the strength of the relationships found through weighted regression.  The strategy here will be to use Pearson's weighted correlation coefficient to measure the strength of the relationship, and to calculate a p-value as a measure of its statistical significance.
# 
# Caluclate the weighed mean, $\overline{y}$ for the observations $y_i$ with regard to the weights $w_i$:
# 
# \begin{align*}
# \overline{y}=\dfrac{\sum{w_iy_i}}{\sum{w_i}}
# \end{align*}
# 
# Do the same for the predicted values $z_i$ found from the regression model:
# 
# \begin{align*}
# z_i &=f(x_i,\vec{\beta}) \\
# \overline{z} &= \dfrac{\sum{w_iz_i}}{\sum{w_i}}
# \end{align*}

# Calculate the weighted variances for both the observations and predictions:
# 
# \begin{align*}
# s_{y} &= \dfrac{\sum{w_i(y_i-\overline{y})^2}}{\sum{w_i}} \\
# s_{z} &= \dfrac{\sum{w_i(z_i-\overline{z})^2}}{\sum{w_i}}
# \end{align*}
# 
# We can then find the weighted correlation coefficient as follows:
# \begin{align*}
# \rho_{yz} = \frac{s_{yz}}{\sqrt{s_ys_z}}
# \end{align*}
# 
# Calculate the standard error of the estimate, which shows what the standard deviation of our model is.  The squared version is the very sum of squares minimized during least squares.  Roughly two-thirds of observations should fall within one standard error from the regression line:
# 
# \begin{align*}
# s_{e}=\sqrt{\dfrac{\sum{w_i(y_i-z_i)^2}}{\sum{w_i}}}
# \end{align*}

# In[71]:


designate('calculating variance', 'regressions')

# calculate the variance
def find_variance(targets, predictions, weights, formula, formulaii):
    """Calculate the variance among weighted samples:
    
    Arguments:
        targets: list of float
        predictions: list of float
        weights: list of float
        formula: function object
        formulaii: function object
        
    Returns:
        float
    """
    
    # calculate weighted means
    mean = average(targets, weights=weights)
    meanii = average(predictions, weights=weights)
    
    # create zipper
    zipper = zip(weights, targets, predictions)
    
    # calculate variance
    variance = 0.0
    for weight, target, prediction in zipper:
        
        # add term to variance
        term = weight * formula(mean, meanii, target, prediction) * formulaii(mean, meanii, target, prediction)
        variance += term
        
    # divide by sum of weights and convert to float
    variance = variance / len(targets)#sum(weights)
    variance = float(variance)

    return variance


# The coefficient of determination ($R^2$) is the proportion of variance in the dependent variable that is predictable from the independent variable:
# 
# The Pearson correlation coefficient ($\rho_{yz}$) is calculated as the ratio of covariance to the variances:
# 
# \begin{align*}
# \rho_{yz}=\frac{s_{y_{ob}y_{pr}}}{\sqrt{s_{y_{ob}}s_{y_{pr}}}}
# \end{align*}
# 
# 
# It is always possible that a correlation found in a study is the result of sampling bias.  The likelihood of the correlation being due to chance sampling is described by Student's t-distribution:
# 
# \begin{align*}
# T(x)=\dfrac{\Gamma(\frac{n-1}{2})}{\sqrt{(n-2)\pi}\Gamma({\frac{n-2}{2}})}(1+\frac{x^2}{n-2})^{-\frac{n-1}{2}}
# \end{align*}
# 
# where $n$ is the number of samples in the study, and $\Gamma$ is the extension of the factorial function to nonintegers:
# 
# \begin{align*}
# \Gamma(n)=(n-1)!=\int_{0}^{\infty}x^{n-1}e^{-x}dx
# \end{align*}
# 
# The Student's t-distribution is similar in shape to a Gaussian normal distribution.  In fact, as the number of samples increases, it makes a better and better approximation to the normal distribution:
# 
# \begin{align*}
# \lim_{n\to\infty}T(x)=\dfrac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}x^2}
# \end{align*}

# In[72]:


designate('sketching t-distribution')

# make normal distribution
normal = lambda x: (1 / sqrt(2 * pi)) * exp(-(x ** 2 / 2))

# sketch t-distribution
legend = ['t, n=3', 't, n=8', 'normal']
title = "Probability Distribution Function Example"
xlab = "Support (x)"
ylab = "Density"

sketch_plot(lambda x: stats.t(1).pdf(x), lambda x:stats.t(5).pdf(x), normal,
       legend=legend, title=title, xlab=xlab, ylab=ylab
      )


# Just as the z-score measures the distance from the center of a normal distribution, a t-value measures the distance from the center of a t-distribution.  In both cases, the relative likelihood of a result decreases away from the center.  The t-value is calculated from the Pearson correlation and number of samples as follows:
# 
# \begin{align*}
# t=\dfrac{\rho\sqrt{n-2}}{\sqrt{1-r^2}}
# \end{align*}
# 
# Finally, the p-value represents the probability of getting a particular t-value through sampling bias alone.  It is defined as the area under the t-distribution curve with magnitude greater than the t-value, and can be calculated with reference to the area under the curve beween -t and t:
# 
# center.  The t-value is calculated from the Pearson correlation and number of samples as follows:
# 
# \begin{align*}
# p=1-\int_{-t}^{t}T(x)dx
# \end{align*}
# 
# Generally, a critical p-value is chosen prior to a study, and a correlation with a p-value less than this critical value is regarded as statistically significant, because it is unlikely to have gotten such a correlation through chance sampling alone.
# 
# Note: In the case of this study, the number of samples used for the p-value calculation is based only on the number of mosquito records used, even though the regression model, weighted variances, and correlation are based on potentially several secondary measurements per mosquito record.  This is because the p-value is sensitive to the number of unique measurements, and it is debatable whether one mosquito record paired with two secondary records constitutes one unique measurement or two.  Underestimating the number of unique samples will tend to overestimate the p-value, and hence underestimate the statistical significance of the regression model.  This is considered preferable to overestimating the number of unique measurements, thereby inflating the statistical significance.

# References: 
# 
# https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
# 
# https://en.wikipedia.org/wiki/Coefficient_of_determination
# 
# https://en.wikipedia.org/wiki/Student%27s_t-distribution

# In[73]:


designate('validating model', 'regressions')

# function for calculating pearson correlation coefficient
def calculate_corr_coef(truths, predictions, weights, size):
    """Calculate the correlation coefficient and pvalue significance from truths and predictions
    
    Arguments:
        truths: list of floats
        predictions: list of floats
        weights: list of floats
        size: int, number of records involved
        
    Returns:
        (float, float) tuple, correlation and bias
    """
    
    # create variance formulas
    formula = lambda mean, meanii, target, prediction: target - mean
    formulaii = lambda mean, meanii, target, prediction: prediction - meanii
    formulaiii = lambda mean, meanii, target, prediction: target - prediction
    
    # calculate variances
    variance = find_variance(truths, predictions, weights, formula, formula)
    varianceii = find_variance(truths, predictions, weights, formulaii, formulaii)
    covariance = find_variance(truths, predictions, weights, formula, formulaii)
    error = find_variance(truths, predictions, weights, formulaiii, formulaiii)
        
    # calculate correlation
    pearson = covariance / sqrt(variance * varianceii)

    # calculate ttest value, using the cumulative distribution function of Student's t
    test = pearson * sqrt(size - 2) / sqrt(1 - pearson ** 2)
    distribution = stats.t(size - 2)
    bias = 2 * (1 - distribution.cdf(abs(test))) # or 1 - 2 * (distribution.cdf(abs(test)) - 0.5)
    
    # remove nans
    removing = lambda x, y: y if isnan(x) else x
    pearson = removing(pearson, 0.0)
    bias = removing(bias, 1.0)
    
    # make validation report
    validation = {'correlation': round(pearson, 4), 'R^2': round((1 - (error / variance)), 4)}
    validation.update({'pvalue': round(bias, 4), 's.e.': round(sqrt(error), 2)})
    
    return validation


# In[74]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# ### Linear Mode Description

# The simplest relationship to look for is a linear one, where the goal is to fit the data to the closest straight line given by:
# 
# $y=\beta_0+\beta_1x$
# 
# where:
# 
# - $y$ is the predicted mosquito larvae count
# - $x$ is the secondary measurement
# - $\beta_0$ is the y-intercept
# - $\beta_1$ is the slope of the line

# In[75]:


designate('defining linear regression')

# linear regression
linear = regressions['linear']
linear['requirement'] = 2
linear['polynomial'] = 1
linear['independent'] = lambda x, b: x
linear['dependent'] = lambda y, d: y
linear['initial'] = lambda b, m, c, d: (m / b, m)
linear['function'] = lambda x, b, m: m * x + b
linear['equation'] = 'y = β0+β1 x'
linear['names'] = ['onset', 'slope']

# define jacobian
# linear['dydc'] = lambda x, c, m: -m
# linear['dydm'] = lambda x, c, m: x - c
# linear['gradient'] = lambda x, c, m: [linear[slope](x, c, m) for slope in ('dydc', 'dydm')]
# linear['jacobian'] = lambda xs, c, m: array([linear['gradient'](x, c, m) for x in xs], dtype=float)

# make versions with different parameters
versions = [(1, 1), (-1, 0.5), (2, -2)]
legend = ['y = {}x+{}'.format(*version) if version[1] > 0 else 'y = {}x{}'.format(*version) for version in versions]

# make functions
making = lambda version: lambda x: linear['function'](x, *version)
linears = [making(version) for version in versions]
title = "Line Plot Example"
xlab = "x"
ylab = "y"
sketch_plot(*linears, legend=legend, title=title, xlab=xlab, ylab=ylab)


# Note that if $m$ is positive, the line grows towards the right, whereas if $m$ is negative, the line grows towards the left.

# <!-- With the following transformation:
#    
# \begin{align*}
# c=\frac{-b}{m}
# \end{align*}
# 
# the line is in polynomial form:
# 
# \begin{align*}
# y=mx+b
# \end{align*}
# 
# Note that the two forms differ in that $c$ represents the x-intercept where the number of larvae is zero, while $b$ represents the y-intercept where the secondary measurement is zero.  The second form is considered standard and is a more convenient form for solving the regression problem.  However, the first form is presented here because the x-intercept is a more useful parameter.  It is generally of interest to know at what secondary measurment there are zero larvae, but the number of larvae at a measurement of zero is arbitrary as the range of the secondary measurement will vary widely.  $c$ is called an "onset" here to avoid confusion with the y-intercept.
# 
# Fitting this line to the dataset is exactly solvable through linear least squares.  However, for the sake of completeness, the gradient equations that make up the Jacobian matrix for solving through nonlinear least squares are as follows:
# 
# \begin{align*}
# y &= m(x-c) \\
# j_c &= \frac{\delta y}{\delta c} = -m \\
# j_m &= \frac{\delta y}{\delta m} = x-c
# \end{align*} -->

# In[76]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# ### Quadratic Mode Description

# However, it may be that a curved relationship is more appropriate.  The simplest curve to try is a parabola:
# 
# \begin{align*}
# y &= k(x-c)^2+h = kx^2-2kcx+kc^2+h \\
# y &= \beta_0 + \beta_1 x + \beta_2 x^2 \tag{standard statistics form}
# \end{align*}
# 
# where:
#     
# - $y$ is the predicted mosquito larvae count
# - $x$ is the secondary measurement
# - $c$ is the center of the parabola
# - $h$ is the height of the parabola at the center
# - $k$ is the curvature of the parabola

# In[77]:


designate('defining quadratic regressions')

# quadratic regression
quadratic = regressions['quadratic']
quadratic['requirement'] = 3
quadratic['polynomial'] = 2
quadratic['independent'] = lambda x, c: x
quadratic['dependent'] = lambda y, d: y
quadratic['initial'] = lambda b, m, k, c, d: (-m / (2 * k), b - m ** 2 / (4 * k), k)
quadratic['function'] = lambda x, a,b,c: a+b*x+c*x**2#k*x**2 - 2*k*x*c + k*c**2 + h
quadratic['equation'] = 'y = β0+β1 x+β2 x^2'
quadratic['names'] = ['center', 'height', 'curvature']

# define jacobian
# quadratic['dydc'] = lambda x, c, h, k: -2 * k * (x - c)
# quadratic['dydh'] = lambda x, c, h, k: 1
# quadratic['dydk'] = lambda x, c, h, k: (x - c) ** 2
# quadratic['gradient'] = lambda x, c, h, k: [quadratic[slope](x, c, h, k) for slope in ('dydc', 'dydh', 'dydk')]
# quadratic['jacobian'] = lambda xs, c, h, k: array([quadratic['gradient'](x, c, h, k) for x in xs], dtype=float)

# make versions with different parameters
versions = [(1, 3, 0.1), (-1, 0.5, 0.3), (2, 1, -0.2)]
legend = ['β0={}, β1={}, β2={}'.format(*version) for version in versions]

# make functions
making = lambda version: lambda x: quadratic['function'](x, *version)
quadratics = [making(version) for version in versions]
title = "Quadratic Plot Example"
xlab = "x"
ylab = "y"
sketch_plot(*quadratics, legend=legend, title=title, xlab=xlab, ylab=ylab)


# Note that if the curvature is positive, the parabola has a minimum, whereas if the curvature is negative, the parabola has a maximum.

# <!-- With the following transformations:
# 
# \begin{align*}
# c &= \dfrac{-m}{2k} \\
# h &= b-{\dfrac{m^2}{4k}}
# \end{align*}
# 
# the equation takes on a new form:
# 
# \begin{align*}
# y=kx^2+mx+b
# \end{align*}
# 
# This polynomial is exactly solvable through linear least squares.  The noninear approach requires that the Jacobian be defined for each parameter:
# 
# \begin{align*}
# y&=k(x-c)^2+h \\
# j_c&=\dfrac{\delta y}{\delta c}=-2k(x-c) \\
# j_h&=\dfrac{\delta y}{\delta h}=1 \\
# j_k&=\dfrac{\delta y}{\delta k}=(x-c)^2
# \end{align*} -->

# In[78]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# ### Exponential Mode Description

# On the other hand, quadratic models tend to predict negative larvae outside the region near the center.  An exponential relationship solves this problem, as it makes no predictions below zero.  It is described like this:
# 
# \begin{align*}
# y=e^{\beta_0+\beta_1 x}
# \end{align*}

# In[79]:


designate('defining exponential regression')

# exponential regression
exponential = regressions['exponential']
exponential['requirement'] = 2
exponential['polynomial'] = 1
exponential['independent'] = lambda x, c: x
exponential['dependent'] = lambda y, d: y#arcsinh(y)
exponential['initial'] = lambda b, m, c, d: (m / b, m)
exponential['function'] = lambda x, m, b: exp(m*x + b)
exponential['equation'] = 'y = e^(β0 + β1* x)'
exponential['names'] = ['onset', 'rate']

# define jacobian
# exponential['dydc'] = lambda x, c, r: -r * exp(r * (x - c))
# exponential['dydr'] = lambda x, c, r: (x - c) * exp(r * (x - c))
# exponential['gradient'] = lambda x, c, r: [exponential[slope](x, c, r) for slope in ('dydc', 'dydr')]
# exponential['jacobian'] = lambda xs, c, r: array([exponential['gradient'](x, c, r) for x in xs], dtype=float)

# make versions with different parameters
versions = [(2, 1), (-1, 0.2), (-2, -0.5)]
legend = ['β0={}, β1={}'.format(*version) for version in versions]

# make functions
making = lambda version: lambda x: exponential['function'](x, *version)
exponentials = [making(version) for version in versions]
title = "Exponential Plot Example"
xlab = "x"
ylab = "y"
sketch_plot(*exponentials, legend=legend, title=title, xlab=xlab, ylab=ylab, span=(-5, 5))


# Note that a positive $r$ means a curve that grows to the right, and a negative $r$ means a curve that grows to the left.  Also note that no value on either curve is below zero.

# Note that the equation:
# 
# \begin{align*}
# y=e^{\beta_0+\beta_1 x}
# \end{align*}
# 
# can be transformed to a linear function by taking the natural log:
# 
# \begin{align*}
# \ln(y)=\beta_0+\beta_1 x
# \end{align*}

# In[80]:


designate('sketching logarithmic approximation')

# sketch functions
legend = ['natural log', 'hyperbolic']
title = "Log and Hyperbolic Plot Example"
xlab = "x"
ylab = "y"
sketch_plot(log, arcsinh, legend=legend, title=title, xlab=xlab, ylab=ylab)


# Note that the natural log has no value at zero, but gets increasingly negative.  The inverse hyperbolic sine approximation sets the value at zero to zero, and is therefore a poor approximation to the natural log near zero.  However, the inverse hyperbolic sine becomes an increasingly better approximation to the natural log with increasing values.  Also, the inverse hyperbolic sine is conveniently defined for negative values as well, whereas the natural log is undefined for negative values.

# In[81]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# ### Power Mode Description

# Exponential relationships are however limited to sharply increasing or sharply decreasing curves.  A power law relationship can model less steep curves:
# 
# \begin{align*}
# y=\beta_0(x-c)^{\beta_1}
# \end{align*}
# 
# where:
#     
# - $y$ is the predicted mosquito count
# - $x$ is the secondary observation
# - $c$ is the onset at zero larvae
# - $\beta_0$ is the height in larvae at one unit passed the onset
# - $\beta_1$ is the power to which the secondary measurement is raised

# In[82]:


designate('defining power law regression')

# power law regression
power = regressions['power']
power['requirement'] = 3
power['polynomial'] = 1
power['independent'] = lambda x, c: arcsinh(x - c)
power['dependent'] = lambda y, d: arcsinh(y)
power['initial'] = lambda b, m, c, d: (c, exp(b), m)
power['function'] = lambda x, c, h, p: sign(p * (x -c)) * h * abs(x - c) ** abs(p)
power['equation'] = 'y = β0  * (x - c)^β1'
power['names'] = ['onset', 'height', 'power']

# define jacobian
# power['dydc'] = lambda x, c, h, p: -c * h * p * abs(x - c) ** (abs(p) - 1)
# power['dydh'] = lambda x, c, h, p: sign(p * (x -c)) * abs(x - c) ** abs(p)
# power['dydp'] = lambda x, c, h, p: sign(x - c) * log(abs(x - c)) * h * abs(x - c) ** abs(p)
# power['gradient'] = lambda x, c, h, p: [power[slope](x, c, h, p) for slope in ('dydc', 'dydh', 'dydp')]
# power['jacobian'] = lambda xs, c, h, p: array([power['gradient'](x, c, h, p) for x in xs], dtype=float)

# make versions with different parameters
versions = [(1, 1, 0.5), (0, 0.5, 0.2), (-1, 1, -0.3)]
legend = ['c={}, β0 ={}, β1={}'.format(*version) for version in versions]

# make functions
making = lambda version: lambda x: power['function'](x, *version)
powers = [making(version) for version in versions]
title = "Power Law Plot Example"
xlab = "x"
ylab = "y"
sketch_plot(*powers, legend=legend, title=title, xlab=xlab, ylab=ylab)


# Note that a negative value of $\beta_1$ produces a curve growing in the opposite direction.  This behavior requires a modification described below.

# As with exponential regression, taking the natural log of both sides yields a linear equation:
# 
# \begin{align*}
# \ln(y)=\ln(\beta_0)+\beta_1\ln(x-c)
# \end{align*}
# 
# This describes a line with slope $p$ and intercept $\ln(\beta_0)$.  However, a value for $c$ is also necessary.  As an approximation, this will be the weighted mean of all secondary measurements:
# 
# \begin{align*}
# c=\frac{\sum{w_ix_i}}{\sum{w_i}}
# \end{align*}
# 
# As mentioned in the description for exponential regression, the natural log is not defined for either zero or negative values.  As there are plenty of zero larvae observations, as well as potentially zero or negative secondary measurements, the natural logs on both sides of the equation must be approximated.  The hyperbolic approximation will be used, because it produces values for both zero and negative measurements:
# 
# \begin{align*}
# \ln(y)\approx\ln\biggl(\dfrac{y}{2}+\sqrt{1+\dfrac{y^2}{4}}\biggr)
# \end{align*}
# 
# With these two estimates in place, the regression can be solved exactly using linear least squares to get intial values for $h$ and $p$.  These initial values begin the process of nonlinear least squares to tighten the fit of all three parameters.
# 
# It may happen that the slope $p$ solved for with linear regression is negative, because larvae counts may increase for smaller values of the secondary measurement.  This poses a problem for the original power law, because a negative power is radically different in form than a positive power.  In fact, taking a noninteger negative $(x-c)$ value to any kind of power is undefined.  Both of these problems can be solved with the following adjustment:
# 
# \begin{align*}
# y=sgn(p(x-c))\cdot h\lvert x-c\rvert^{\lvert p\rvert}
# \end{align*}
# 
# The absolute values and sign function create the symmetric form seen in the diagram above, allowing the regression to fit a curve pointing in either direction.  From the initial conditions, a tighter fit can be found with the help of the Jacobian equations:
# 
# \begin{align*}
# j_c &=\dfrac{\delta y}{\delta c}=-chp\lvert x-c\rvert^{\lvert p\rvert-1} \\
# j_h &=\dfrac{\delta y}{\delta h}=sgn(p(x-c))\cdot \lvert x-c\rvert^{\lvert p\rvert} \\
# j_p &=\dfrac{\delta y}{\delta p}=sgn(x-c)\cdot h \cdot\ln\lvert x-c\rvert\lvert x-c\rvert^{\lvert p\rvert} \\
# \end{align*}

# In[83]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# ### Gaussian Mode Description

# Alternatively, it may be appropriate to search for a peak value of larvae at a secondary value with a relationship like this:
# 
# \begin{align*}
# y=\beta_0e^{-(x-\beta_1)^2/2\beta_2}
# \end{align*}
# 
# where:
# 
# - $y$ is the predicted mosquito count
# - $x$ is the secondary observation
# - $c$ is the center, equivalent to the mean in a normal distribution
# - $h$ is the height at the center
# - $s$ is the spread, equivalent to the variance in a normal distribution

# In[84]:


designate('defining gaussian regression')

# gaussian law regression
gaussian = regressions['gaussian']
gaussian['requirement'] = 3
gaussian['polynomial'] = 2
gaussian['independent'] = lambda x, c: x
gaussian['dependent'] = lambda y, d: y # arcsinh(y)
gaussian['initial'] = lambda b, m, k, c, d: (-m / (2 * k), exp(b - m ** 2 / (4 * k)), -1 / (2 * k))
gaussian['function'] = lambda x, c, h, s: h * exp(-(x - c) ** 2 / (2 * s))
gaussian['equation'] = 'y = β0 * e^(-(x - β1)^2 / 2 * c)'
gaussian['names'] = ['center', 'height', 'spread']

# define jacobian
# gaussian['dydc'] = lambda x, c, h, s: (h * (x - c) / s) * exp(-(x - c) ** 2 / (2 * s))
# gaussian['dydh'] = lambda x, c, h, s: exp(-(x - c) ** 2 / (2 * s))
# gaussian['dyds'] = lambda x, c, h, s: (h * (x - c) ** 2 / (2 * s ** 2)) * exp(-(x - c) ** 2 / (2 * s))
# gaussian['gradient'] = lambda x, c, h, s: [gaussian[slope](x, c, h, s) for slope in ('dydc', 'dydh', 'dyds')]
# gaussian['jacobian'] = lambda xs, c, h, s: array([gaussian['gradient'](x, c, h, s) for x in xs], dtype=float)

# make versions with different parameters
versions = [(1, 1, 1), (-1, 0.5, 4), (2, 2, -40)]
legend = ['β2={}, β0={}, β2={}'.format(*version) for version in versions] # c={}, h={}, s={}

# make functions
making = lambda version: lambda x: gaussian['function'](x, *version)
gaussians = [making(version) for version in versions]
title = "Gaussian (Normal) Plot Example"
xlab = "x"
ylab = "y"
sketch_plot(*gaussians, legend=legend, title=title, xlab=xlab, ylab=ylab)


# Note that this family of curves also includes those that curve upwards, marked by a negative value for the spread.  An upward curving "Gaussian" indicates that a curve with a minimum that grows to either side is a better fit for the data set.

# With the following transformations:
# 
# \begin{align*}
# \beta_2 &=\dfrac{-m}{2k} \\
# \beta_0 &=e^{b-m^2/4k} \\
# \beta_1 &=\dfrac{-1}{2k}
# \end{align*}
# 
# the equation takes on a new form:
# 
# \begin{align*}
# y=e^{kx^2+mx+b}
# \end{align*}
# 
# Taking the natural logarithm of both sides:
# 
# \begin{align*}
# \ln{(y)}=kx^2+mx+b
# \end{align*}
# 
# creates a polynomial, and would be solvable through linear least squares if not for observations of zero larvae that render the logarithm undefined.  The strategy here is to use the inverse hyperbolic sine approximation and solve for the initial parameters:
# 
# \begin{align*}
# \ln(y)\approx\ln\biggl(\dfrac{y}{2}+\sqrt{1+\dfrac{y^2}{4}}\biggr)
# \end{align*}
# 
# These parameters are then fed into nonlinear least squares for a tighter fit using the transformation equations above. With the associated Jacobian equations, the nonlinear problem can be solved:
# 
# \begin{align*}
# y &=he^{-(x-\beta_2)^2/2\beta_1} \\
# j_{\beta_2} &=\dfrac{\delta y}{\delta \beta_2}=\dfrac{h(x-\beta_2)}{\beta_1}e^{-(x-\beta_2)^2/2\beta_1} \\
# j_{\beta_0} &=\dfrac{\delta y}{\delta \beta_0}=e^{-(x-\beta_2)^2/2\beta_1} \\
# j_{\beta_1} &=\dfrac{\delta y}{\delta \beta_1}=\dfrac{h(x-\beta_2)^2}{2\beta_1^2}e^{-(x-\beta_2)^2/2\beta_2}
# \end{align*}

# In[85]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# In[86]:


designate('preparing scatter graph', 'scatters')

# preparing scatter graph
def prepare_scatter_plot(associations, mode):
    """Prepare the scatter graph.
    
    Arguments:
        associations: list of dicts
        mode: str, the regression mode
        
    Returns:
        bokeh graph object
    """
    
    # get samples
    samples = assemble_samples(associations)
    
    # get sample collections
    primaries = [sample['y'] for sample in samples]
    secondaries = [sample['x'] for sample in samples]
        
    # begin the plot
    entitling = lambda word: word[0].upper() + word[1:]
    title = '{} Regression of Larvae Counts Predicted by {} in {}'.format(entitling(mode), entitling(feature), country)
    parameters = {'title': title, 'x_axis_label': feature, 'y_axis_label': 'larvae counts'}
    parameters.update({'plot_width': 900, 'plot_height': 400})

    # set y range based on range of primary data
    extent = max(primaries) - min(primaries)
    parameters['y_range'] = [min(primaries) - 0.05 * extent, max(primaries) + 0.05 * extent]
    
    # set x range based on range of secondary data    extent = max(primaries) - min(primaries)
    extent = max(secondaries) - min(secondaries)
    parameters['x_range'] = [min(secondaries) - 0.05 * extent, max(secondaries) + 0.05 * extent]
    
    # begin graph
    graph = figure(**parameters)
    
    return graph


# In[87]:


designate('tracing lines on the graph', 'scatters')

# tracing lines on the graph
def draw_function(function, graph, ticks, genus, width=2, style='solid', offset=0):
    """Trace a function on the graph at particular tickmarks and other characteristics.
    
    Arguments:
        function: function object
        graph: bokeh graph object
        ticks: list of floats
        genus: str
        width=2: int
        style='solid': str
        offset=0: float, offset of line
        
    Returns:
        bokeh graph object
    """
            
    # apply function to get the ticks and filter for large values
    points = [{'x': tick, 'y': function(tick) + offset, 'genus': genus, 'weight': 'NA'} for tick in ticks]
    [point.update({'latitude': 'NA', 'longitude': 'NA', 'time': 'NA'}) for point in points]
    [point.update({'pair': 'NA', 'site': 'NA', 'organization': 'NA'}) for point in points]
    points = [point for point in points if abs(point['y']) < 1000]
    table = ColumnDataSource(pandas.DataFrame(points))
    
    # check for len
    if len(points) > 0 and any([point['y'] != 0.0 for point in points]):
    
        # trace line on graph
        color = indicators[genus]
        line = {'source': table, 'x': 'x', 'y': 'y', 'color': color, 'line_width': width}
        line.update({'legend_label': genus, 'line_dash': style})
        graph.line(**line)
    
    return graph


# In[88]:


designate('splotching markers on the graph', 'scatters')

# splotching markers on the graph
def add_points(samples, graph, genus):
    """Splotch samples onto the graph as markers, based on a genus for color.
    
    Arguments:
        samples: list of dicts
        graph: bokeh graph obeject
        genus: str, the genus
        
    Returns:
        bokeh graph object
    """
    
    # add circle marker
    table = ColumnDataSource(pandas.DataFrame(samples))
    parameters = {'source': table, 'x': 'x', 'y': 'y', 'size': 'size'}
    parameters.update({'fill_color': 'color', 'line_width': 1, 'line_color': 'color'})
    parameters.update({'fill_alpha': 0.2, 'legend_label': genus})
    graph.circle(**parameters)
    
    return graph


# In[89]:


designate('presenting a genus report on the graph', 'scatters')

# presenting a report on the graph
def add_report(report, graph, associations, ticks):
    """Present a report on the graph about the associations at certain tick marks.
    
    Arguments:
        report: dict
        associations: list of dicts
        ticks: list of floate
        graph: bokeh graph object
        
    Returns:
        bokeh graph object
    """
    
    # get samples
    genus = report['genus']
    mode = report['mode']
    subset = get_subset_of_data(associations, genus)
    samples = assemble_samples(subset)
    
    # add markers
    if len(samples) > 0:
        
        # add markers
        graph = add_points(samples, graph, genus)
        
    # add lines
    if report['equation'] != cancellation :
        
        # add the regression line
        width = 1 + int((1 - report['pvalue']) * 3)
        function = lambda x: regressions[mode]['function'](x, *report['curve'])
        graph = draw_function(function, graph, ticks, genus, width=width)

        # add the error lines
        error = report['s.e.']
        graph = draw_function(function, graph, ticks, genus, width=width, style='dashed', offset=error)
        graph = draw_function(function, graph, ticks, genus, width=width, style='dashed', offset=-error)

        # add the extrapolation lines
        extent = max(ticks) - min(ticks)
        extrapolation = [tick - extent for tick in ticks]
        extrapolationii = [tick + extent for tick in ticks]
        graph = draw_function(function, graph, extrapolation, genus, style='dotted')
        graph = draw_function(function, graph, extrapolationii, genus, style='dotted')

    return graph


# In[90]:


designate('plotting scatter graphs', 'scatters')

# function for plotting a scatter graph
def plot_scatter_plot(associations, mode, spy=False, calculate=True):
    """Plot a scatter graph based on a regression style.
    
    Arguments:
        associations: list of dicts
        mode: str ('linear, 'quadratic', 'exponential', 'power', 'logistic', 'gaussian')
        spy=True: boolean, observe initial linear regression fit?
        calculate: boolean, calculate jacobian directly?
        
    Returns
        bokeh graph object
    """
    
    # get regression results
    reports = study(associations, mode, spy, calculate)
    
    # begin graph
    graph = prepare_scatter_plot(associations, mode)
    
    # get ticks for regression lines
    ticks = set_ticks(graph.x_range.start, graph.x_range.end, 1000)

    # add each report to the graph
    for report in reports:
    
        # add to graph
        graph = add_report(report, graph, associations, ticks)
    
    # add annotations
    annotations = [('Genus', '@genus'), ('Larvae', '@y')]
    annotations += [(truncate_field_name(feature), '@x'), ('Weight', '@weight'), ('Pair', '@pair')]
    annotations += [('Latitude', '@latitude'), ('Longitude', '@longitude'), ('Time', '@time')]
    graph = annotate_plot(graph, annotations)
            
    # make the panda
    pandas.set_option('max_colwidth', 60)
    panda = pandas.DataFrame(reports)

    return graph, panda


# ### Table of GLOBE Countries

# The follow is a list of all GLOBE supporting countries and their country codes.

# In[91]:


designate('extracting all GLOBE countries')

# retrieve list of GLOBE countries and country codes from the API
url = 'https://api.globe.gov/search/dev/country/all/'
request = requests.get(url)
raw = json.loads(request.text)
countries = [record for record in raw['results']]

# create codes reference for later
codes = {record['countryName']: record['id'] for record in countries}
countries = {record['id']: record['countryName'] for record in countries}

# make table
table = [item for item in codes.items()]
table.sort(key=lambda item: item[0])

# print table
print('{} GLOBE countries:'.format(len(table)))
table


# ### Setting the Parameters

# Set the main geographic and time parameters in the box below.  You may specify a country name or a country code, with the code overriding the name when both are given.  Refer to the Table of GLOBE Countries above to see all countries involved in the GLOBE program.  Misspelled names and codes will be matched to the closest options available.  If both code and country name are left blank, it will default to all countries.  Also, the beginning date will default to Earth Day 1995 if left blank, and the ending date will default to the current date if left blank.  
# 
# Set the name of the secondary protocol to use as the independent variable of the study.  A list of the protocols in the mosquito habitat mapper bundle are listed for reference.  A close match to the name is generally sufficient.
# 
# Set the name of the particular feature of interest.  Currently this notebook only supports numerical features, so choose a feature with numerical values as opposed to categorical names. (For example, a field like 'mosquitohabitatmapperWaterSource' is not appropriate for the regression analyses in this notebook because the values are discrete source names instead of a continuum of values).  It may be difficult to know the name of this feature offhand.  In this case, you may use the Peruse button to see some sample records.  A close match to the name should be sufficient.
# 
# Also set the name of the time of measurement field, as this may vary depending on the protocol.  Usually 'measured' is sufficient to find the right one.  
# 
# You may click the Apply button to verify your choices.  Clicking the Propagate button will propagate the changes to the rest of the notebook.  Clicking Both will both apply the new parameters and propagate the changes throughout the notebook.

# In[92]:


designate('table of secondary protocols')

# get from protocols file
with open('protocols.txt', 'r') as pointer:
    
    # get all protocols
    protocols = [protocol.strip('\n') for protocol in pointer.readlines()]
    protocols = [protocol for protocol in protocols if 'X' in protocol]
    protocols = [protocol.strip('X').strip() for protocol in protocols]

# print as list
print('{} mosquito bundle protocols:'.format(len(protocols)))
protocols


# In[93]:


designate('setting the country, date range, and secondary protocol', 'settings')

# set the country name, defaulting to All countries if blank
country = 'United States'

# set the country code, defaulting to country name if left blank and overriding otherwise
code = ''

# set beginning date in 'YYYY-mm-dd' format, defaulting to 1995-04-22
beginning = '2016-01-01'

# set ending date in 'YYYY-mm-dd' format, defaulting to today's date
ending = ''

# set secondary protocol
secondary = 'dissolvedoxygen'

# set secondary protocol feature
feature = 'dissolvedoxygensDissolvedOxygenViaKitMgl'

# set secondary protocal measured time field ('measured' is usually close enough)
measured = 'measured'


# In[94]:


designate('applying setting changes or propagating throughout notebook')

# propagate changes
propagate_setting_changes('### Setting the Parameters', '### Retrieving Records from the API', '### Optimizing Parameters')


# In[95]:


designate('resolving user settings')

# define primary protocol name and larvae field
mosquitoes = 'mosquito_habitat_mapper'
larvae = 'mosquitohabitatmapperLarvaeCount'

# resolve country and code to default values
country, code = resolve_country_code(country, code)

# default beginning to first day of GLOBE and ending to current date if unspecified
beginning = beginning or '1995-04-22'
ending = ending or str(datetime.now().date())

# make api call to get number of records
print('\nprimary protocol: {}'.format(mosquitoes))
print('checking number of records...')
raw = query_api(mosquitoes, code, beginning, ending, sample=True)
count = raw['count']
print('{} {} records from {} ({}), from {} to {}'.format(count, mosquitoes, country, code, beginning, ending))

# infer closest match
perusal = [0]
secondary = fuzzy_match(secondary, protocols)
print('\nsecondary protocol: {}'.format(secondary))

# make the API call and get the examples and record count
print('checking number of records...')
raw = query_api(secondary, code, beginning, ending, sample=True)
examples = [record for record in raw['results']]
count = raw['count']
print('{} records for {} from {} ({}), {} to {}'.format(count, secondary, country, code, beginning, ending))

# assert there must be more records
assert count > 0, '* Error * Unfortunately no records were returned for that protocol.'

# infer best matches
fields = [key for key in examples[0]['data'].keys()]
feature = fuzzy_match(feature, fields)
measured = fuzzy_match(measured, fields)

# print for inspection
print('\nsecondary feature: {}'.format(feature))
print('secondary time field: {}'.format(measured))


# In[96]:


designate('perusing secondary records')

# peruse through the ten sample records
def view_ten_records(_):
    """Peruse through example records.
    
    Arguments:
        None
        
    Returns:
        None
    """
    
    # refresh this cell
    refresh_cells_by_position(0)
    
    # get last number
    last = perusal[-1]
    
    # advance
    after = last + 1
    if after > 9:
        
        # return to beginning
        after = 0
        
    # append to numberings
    perusal.append(after)

    # print record
    print('record {} of 10:'.format(last))
    print(json.dumps(examples[last]['data'], indent=1))
    
    return None

# create button
button = Button(description="Peruse")
display(button)
button.on_click(view_ten_records)


# In[97]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# ### Retrieving Records from the API

# Click Retrieve to retrieve the data from the GLOBE API.  Clicking Propagate afterward will recalculate the notebook with the new dataset.

# In[98]:


designate('applying setting changes or propagating throughout notebook')

# propagate button
labels = ['Retrieve', 'Propagate', 'Both']
propagate_setting_changes('### Retrieving Records from the API', '### Pruning Outliers', '### Optimizing Parameters', labels)


# In[99]:


designate('retrieving and processing records from the api')

# get the mosquitoes records from the 'results' field, and prune off nulls
print('\nmaking api request...')
raw = query_api(mosquitoes, code, beginning, ending)
count = len(raw['results'])
results = [record for record in raw['results'] if record['data'][larvae] is not None]
results = process_records(results, primary=True)
formats = (len(results), mosquitoes, count, country, beginning, ending)
print('{} valid {} records (of {} total) from {}, {} to {}'.format(*formats))

# get the secondary protocol records and prune off nulls
print('\nmaking api request...')
raw = query_api(secondary, code, beginning, ending)
count = len(raw['results'])
resultsii = [record for record in raw['results'] if record['data'][feature] is not None]

# try to process data, may encounter error
try:

    # process data
    resultsii = process_records(resultsii, primary=False)
    formats = (len(resultsii), secondary, count, country, beginning, ending)
    print('{} valid {} records (of {} total) from {}, {} to {}'.format(*formats))
    
# but if trouble
except ValueError:
    
    # raise error
    message = '* Error! * Having trouble processing the data.  Is the requested field a numberical one?'
    raise Exception(message)

    
# raise assertion error if zero records
assert len(results) > 0, '* Error! * No valid {} records in the specified range'.format(mosquitoes)
assert len(resultsii) > 0, '* Error! * No valid {} records in the specified range'.format(secondary)


# In[100]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# ### Pruning Outliers

# You may wish to remove datapoints that seem suspiciously like outliers. Observations will be removed from the study if they fall outside the interquartile range.

# In[101]:


designate('setting upper quartile', 'settings')

# set upper quartile (max 100, min 0)
threshold = 85
thresholdii = 85


# In[102]:


designate('applying setting changes or propagating throughout notebook')

# propagate button
propagate_setting_changes('### Pruning Outliers', '### Filtering Records', '### Optimizing Parameters')


# In[103]:


designate('pruning away outliers')

# add default quartile field
for record in results + resultsii:
    
    # add default score field
    record['lq'] = -999.0
    record['uq'] = -999.0

# prune data
authentics, outliers = remove_outliers(results, 'larvae', threshold)
authenticsii, outliersii = remove_outliers(resultsii, feature, thresholdii)

# report
zipper = zip((authentics, authenticsii), (outliers, outliersii), (mosquitoes, secondary), ('larvae', feature))
for records, prunes, protocol, field in zipper:
    
    # report each outlier
    print('\n\n' + field + ' where lower quartile  and upper quartile  ')
    for outlier in prunes:

        # print
        print(outlier[field], outlier['lq'], outlier['uq'])

    # report total
    print('\n{} observations removed'.format(len(prunes)))
    print('{} {} records after removing outliers'.format(len(records), protocol))


# In[104]:


designate('drawing histograms')

# construct histograms
gram = construct_bargraph(authentics, 'larvae', width=5)
gramii = construct_bargraph(authenticsii, feature, width=1)

# display plots
output_notebook()
show(Row(gram, gramii))


# In[105]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# ### Filtering Records

# You may further filter the data if desired.  Smaller datasets render more quickly, for instance.  Set the criteria and click the Apply button to perform the search.  Clicking Propagate will propagate the changes down the notebook.  You may set a parameter to None to ignore any filtering.

# In[106]:


designate('setting filter parameters', 'settings')

# set the specific genera of interest ['Anopheles', 'Aedes', 'Culex', 'Unknown', 'Other']
# None defaults to all genera
genera = ['Anopheles', 'Aedes', 'Culex', 'Unknown', 'Other']

# set fewest and most larvae counts or leave as None
fewest = None
most = None

# set the inital or final date ranges (in 'YYYY-mm-dd' format), or leave as None
initial = None
final = None

# set the latitude boundaries, or leave as None
south = None
north = None

# set the longitude boundaries, or leave as None
west = None
east = None


# In[107]:


designate('applying setting changes or propagating throughout notebook')

# propagate changes
propagate_setting_changes('### Filtering Records', '### Defining the Weighting Scheme', '### Optimizing Parameters')


# In[108]:


designate('filtering records')

# set records to data
records = [record for record in authentics]
recordsii = [record for record in authenticsii]

# make parameters and fields list
parameters = [genera, fewest, most, initial, final, south, north, west, east]
fields = ['genus', 'larvae', 'larvae', 'date', 'date', 'latitude', 'latitude', 'longitude', 'longitude']

# make comparison functions list
within = lambda value, setting: value in setting
after = lambda value, setting: value >= datetime.strptime(str(setting), "%Y-%m-%d").date()
before = lambda value, setting: value <= datetime.strptime(str(setting), "%Y-%m-%d").date()
greater = lambda value, setting: value >= setting
lesser = lambda value, setting: value <= setting

# make associated functions
functions = [within, greater, lesser, after, before, greater, lesser, greater, lesser]
symbols = ['in', '>', '<', '>', '<', '>', '<', '>', '<']

# filter primaries
data, criteria = filter_data_by_field(authentics, parameters, fields, functions, symbols)
formats = (len(data), len(authentics), mosquitoes, criteria)
print('\n{} of {} {} records meeting criteria:\n\n{}'.format(*formats))

# filter secondaries
dataii, criteria = filter_data_by_field(authenticsii, parameters, fields, functions, symbols)
formats = (len(dataii), len(authenticsii), secondary, criteria)
print('\n{} of {} {} records meeting criteria:\n\n{}'.format(*formats))

# set genera to classification by default
genera = genera or classification


# In[109]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# ### Defining the Weighting Scheme

# Because the two sets of measurements were not taken concurrently, there must be some criteria to determine when measurements from one protocol correspond to measurements from the other protocol.  The strategy implemented here is using a weighting function that determines how strongly to weigh the association, based on the following parameters:
#     
# - distance: the distance in kilometers between measurements that will be granted full weight.
#     
# - interval: the time interval in days between measurements that will be granted full weight.
#     
# - lag: the time in days to anticipate an effect on mosquitoes from a secondary measurement some 
#     days before.
#     
# - confidence: the weight to grant a measurement twice the distance or interval. This determines how steeply the weighting shrinks as the intervals are surpassed.  A high confidence will grant higher weights to data outside the intervals.  A confidence of zero will have no tolerance for data slightly passed the interval.
#     
# - cutoff: the minimum weight to consider in the dataset.  A cutoff of 0.1, for instance, will only retain data if the weight is at least 0.1.
#     
# - inclusion: the maximum number of nearest secondary measurements to include for each mosquitos measurement.
# 
# If you have tried the optimizer at the bottom of this notebook, you may use the Remember button to bring up a list of those results.

# In[110]:


designate('setting the weighting scheme', 'settings')

# set the distance sensitivity in kilometers
distance = 50

# set time interval sensitivity in days
interval = 1

# set the expected time lag in days
lag = 0

# weight for twice the interval
confidence = 0.8

# minimum weight to include
cutoff = 0.1

# number of nearest neighboring points to include
inclusion = 5


# In[111]:


designate('remembering optimization paramters button')

# reset parameters function
def remember(_):
    """Remeber previous obtimization results.
    
    Arguments:
        None
        
    Returns:
        None
    """
    
    # refresh
    refresh_cells_by_position(0)
    
    # view last panda
    try:
        
        # view last panda
        display(optimizations[-1])
    
    # otherwise
    except IndexError:
        
        # message
        print('no optimizations generated yet')
    
    return None

# create button
button = Button(description="Remember")
display(button)
button.on_click(remember)


# In[112]:


designate('applying setting changes or propagating throughout notebook')

# propagate changes
propagate_setting_changes('### Defining the Weighting Scheme', '### Linear Regression', '### Optimizing Parameters')


# In[113]:


designate('plotting weighting scheme')

# create settings dictionary
settings = {'distance': distance, 'interval': interval, 'lag': lag, 'confidence': confidence}
settings.update({'cutoff': cutoff, 'inclusion': inclusion})

# create plateaus
plateau = plot_plateaued_weighting_scheme('distance', settings)
plateauii = plot_plateaued_weighting_scheme('interval', settings)

# display plots
output_notebook()
show(Row(plateau, plateauii))


# The histograms below show the distribution of weights amongst the data set, and the distribution of secondary-measurement pairs found.

# In[114]:


designate('assembling associations')

# make associations
print('\nassembling associations...')
associations = merge_datasets(settings)
coverage = round(len(associations) / len(data), 2)
print('{} {} records used, {} of {} records pulled'.format(len(associations), mosquitoes, coverage, len(data)))

# raise assertion error
assert len(associations) > 1, '* Error! * Not enough records retrieved for regression' 

# make summary table
summary = summary_df(associations)

# construct histograms
records = [{'weight': association['associates'][0]['weight']} for association in associations]
gram = construct_bargraph(records, 'weight', width=0.1)

# construct histogram
records = [{'pairs': len(association['associates'])} for association in associations]
gramii = construct_bargraph(records, 'pairs', width=1)

# display
output_notebook()
show(Row([gram, gramii]))


# In[115]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# ## Curve Fitting and Table Legend
# 
# Once the weighting characteristics have been chosen, propagating the changes will perform each mode of regression on each genus of larvae from the protocol.  The graph indicates the results of the regression in the following way:
# - The solid lines represents the best fitting lines for each genus.
# - The dashed lines above and below represent one standared error away from the best fit.
# - A thicker line indicates a lower probability of sampling bias (lower p-value).
# - The dotted lines represent exptrapolation of the model beyond the data range.
# - The genus is colored as indicated in the legend.  Clicking in the legend will hide the regression for that genus.
# - Each marker is colored according to genus and sized according to its weight.
# 
# The validation results of each study are presented in a table with the following fields:
# - genus: the mosquito genus of the study.
# - records: the number of mosquito records involved in the model.
# - pairs: the number of mosquito-secondary paired records.
# - coverage: the fraction of mosquito records involved compared to pulled records.
# - pvalue: testing the hypothesis $H_0: \rho=0$ (correlation is 0) vs $H_1: \rho \neq 0$ (correlation is not zero).
# - correlation: the correlation between the true observations and the model [why?](https://analyse-it.com/docs/user-guide/fit-model/linear/predicted-actual-plot#:~:text=Predicted%20against%20actual%20Y%20plot,line%2C%20with%20narrow%20confidence%20bands.&text=Points%20that%20are%20vertically%20distant%20from%20the%20line%20represent%20possible%20outliers.)
# - $R^2$: the coefficient of determination which is the proportion of variance in the dependent variable that is predictable from the independent variable
# - s.e: the standard error of estimate (residuals) between truths and predictions, in units of larvae.
# - equation: the regression equation for the model.
# - mode specific parameters that have been fit by the regression

# ### Linear Curve Fit

# In[116]:


designate('performing linear regression')

perform('linear', associations)


# In[117]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# ### Quadratic Curve Fit

# In[118]:


designate('performing quadratic curve fit')

# perform
perform('quadratic', associations)


# In[119]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# ### Exponential Curve Fit

# In[120]:


designate('performing exponential curve fit')

# perform
perform('exponential', associations)


# In[121]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# ### Power Curve Fit

# In[122]:


designate('performing power law curve fit')

# perform
perform('power', associations)


# In[123]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# ### Gaussian Curve Fit

# In[124]:


designate('performing gaussian curve fit')

# perform
perform('gaussian', associations)


# In[125]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# ### Viewing the Data Table

# The data table of associations may be viewed below by clicking the View button.  Each row in the table begins with a pair of indices, with the first representing the primary mosquito record and the second representing the seconday record.  Each primary record may have multiple secondary records.  The strength of the association is given in the weights column, followed by the distance and time interval between them.  Thereafter are all the fields of both the primary and secondary records.

# In[126]:


designate('viewing data table button')

# function to examine record
def view(_):
    """Examine secondary protocol record.
    
    Arguments:
        None
        
    Returns:
        None
    """

    # display in output
    print('displaying...')
    display(summary)
    
    return None

# create button
button = Button(description="View")
button.on_click(view)
display(button)


# In[127]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# ### Exporting Data to CSV

# It may be desirable to export the data to a csv file.  Click the Export button to export the data.  You will get a link to download the csv file.

# In[128]:


designate('exporting data to csv button')

# create push button linked to output
exporter = Output()
button = Button(description='Export')
display(button, exporter)

# function to export to zip file
def export(_):
    """Export the data as a csv file.
    
    Arguments:
        None
        
    Returns:
        None
    """
    
    #write dataframe to file
    name = 'mosquitoes_' + secondary + '_' + str(int(datetime.now().timestamp())) + '.csv'
    summary.to_csv(name)  
    
    # make link
    link = FileLink(name)
    
    # add to output
    exporter.append_display_data(link)

    return None

# add button click command
button.on_click(export)


# In[129]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# ### Visualizing on a Map

# Click the map button to generate a map of all sampling locations.

# In[130]:


designate('viewing the map button')

# reset parameters function
def illustrate(_):
    """Illustrate sample locations in a map.
    
    Arguments:
        None
        
    Returns:
        None
    """
    
    # clear output
    execute_cell_range('### Visualizing on a Map', '### Optimizing Parameters')
    
    # construct the map
    chart = construct_map_at_central_geo_loc()
    
    # add the markers
    chart = populate_map(chart)
    
    # add the controls
    chart = add_map_controls(chart)
    
    # display the chart
    display(chart)
    
    return None

# create button
button = Button(description="Map")
display(button)
button.on_click(illustrate)


# The map is broken into 4 layers which may be selected from the Layers box underneath the full screen control on the left side of the map.  The layers may be toggled on and off here.  They are described as follows:

# 1) associated primaries: this layer contains markers for all mosquito habitat observations that have been associated with secondary protocol measurements.  The markers are colored according to genus as indicated by the map legend in the upper right.  The size of the marker reflects the size of the larvae count.  The opacity of the marker reflects the highest weight amongst secondary associations.  Clicking the marker will bring up a summary table and highlight the associated secondary measurements

# 2) associated secondaries: this layer contains markers for all secondary protocol observations that have been associated with mosquito measurements.  The markers are colored according the scale displayed in the lower right corner of the map.  The opacity of the marker reflects the highest weight amongst mosquito associations.  Clicking the marker will bring up a summary table and highlight the associated mosquito measurements.  Where multiple measurements have occured at the same longitude, latitude coordinates, the marker with the highest weight is displayed.

# 3) unassociated primaries: this layer contains markers for all mosquito habitat observations that have not been associated with secondary protocol measurements.  The markers are colored tan.  The size of the marker reflects the size of the larvae count.  Clicking the marker will bring up a summary table.

# 4) unassociated secondaries: this layer contains markers for all secondary protocol observations that have not been associated with mosquito measurements.  The markers are colored according the scale displayed in the lower right corner of the map.  Clicking the marker will bring up a summary table.  Where multiple measurements have occured at the same longitude, latitude coordinates, the marker with the highest secondary measurement is displayed.

# In[131]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# In[132]:


designate('chiseling a range into even numbers', 'map')

# chisel subroutine to round numbers of a range
def chisel(low, high):
    """Chisel a rounded range from a precise range.
    
    Arguments:
        low: float, low point in range
        high: float, high point in range
        
    Returns:
        (float, float) tuple, the rounded range
    """
    
    # find the mean of the two values
    mean = (low + high) / 2
    
    # find the order of magnitude of the mean
    magnitude = int(log10(mean))
    
    # get first and second 
    first = int(low / 10 ** magnitude)
    second = int(high / 10 ** magnitude) + 1
    
    # get all divisions
    divisions = [(10 ** magnitude) * number for number in range(first, second + 1)]
    
    # make half division if length is low
    if len(divisions) < 5:
        
        # create halves
        halves = [(entry + entryii) /2 for entry, entryii in zip(divisions[:-1], divisions[1:])]
        divisions += halves
        
    # sort
    divisions.sort()
    
    return divisions


# In[133]:


designate('mixing colors', 'map')

# mix colors according to a gradient
def mix_colors(measurement, bracket, spectrum):
    """Mix the color for the measurment according to measurment bracket and color spectrum.
    
    Arguments:
        measurement: float, the measurement
        bracket: tuple of floats, low and high measurement range
        specturm: tuple of int tuples, the low and high rgb color endpoints
        
    Returns:
        str, hexadecimal color
    """
    
    # truncate measurement to lowest of highest of range
    low, high = bracket
    measurement = max([measurement, low])
    measurement = min([measurement, high])
    
    # determine fraction of range
    fraction = (measurement - low) / (high - low)
    
    # define mixing function
    mixing = lambda index: int(spectrum[0][index] + fraction * (spectrum[1][index] - spectrum[0][index]))
    
    # mix colors
    red, green, blue = [mixing(index) for index in range(3)]
    
    # define encoding function to get last two digits of hexadecimal encoding
    encoding = lambda intensity: hex(intensity).replace('x', '0')[-2:]
    
    # make the specific hue
    color = '#' + encoding(red) + encoding(green) + encoding(blue)

    return color


# In[134]:


designate('shrinking records to unique locations', 'map')

# shrink the secondary dataset to unique members per location
def get_unique_record_per_loc_secondary_data(records, criterion):
    """Shrink the records set to the unique record at each latitude longitude, based on a criterion field.
    
    Arguments:
        records: list of dicts
        criterion: str, field to sort by
        
    Returns:
        list of dicts
    """
    
    # add each record to a geocoordinates dictionary
    locations = {}
    for record in records:
        
        # each record is a contender for the maximum criterion
        contender = record
        location = (record['latitude'], record['longitude'])
        highest = locations.setdefault(location, None)
        if highest:
            
            # replace with record higher in the criterion
            contenders = [highest, contender]
            contenders.sort(key=lambda entry: entry[criterion], reverse=True)
            contender = contenders[0]
            
        # replace with condender
        locations[location] = contender
        
    # get the shrunken list
    uniques = [member for member in locations.values()]
    
    return uniques


# In[135]:


designate('code for stipling map with samples', 'map')

# stiple function to add a circle marker layer to the map
def add_points_to_map(records, weights, field, fill, line, thickness, radius, opacity, click, name=''):
    """Stipple the map with a layer of markers.
    
    Arguments:
        records: list of dicts, the records
        weights: list of floats, the weights
        field: str, the name of the appropriate field
        fill: function object, to determine fill color
        line: function object, to determine line color
        thickness: int, circle outline thickness
        radius: function object, to determine radius
        opacity: function object, to determine opacity as a function of weight
        click: function object, to act upon a click
        name: str, name of layer
        
    Returns:
        pyleaflet layer object
    """

    # create marker layer
    markers = []
    for record, weight in zip(records, weights):

        # unpack record
        latitude = record['latitude']
        longitude = record['longitude']
        date = record['date']
        measurement = record[field]
        
        # make circle marker
        circle = CircleMarker()
        circle.location = (latitude, longitude)
        
        # marker outline attributes
        circle.weight = thickness
        circle.color = line(record)
        circle.radius = radius(record)

        # set fill color attributes
        circle.fill = True
        circle.fill_opacity = opacity(weight)
        circle.fill_color = fill(record)

        # add click function
        circle.on_click(click)
        
        # annotate marker with popup label
        formats = (round(weight, 3), date, round(latitude, 3), round(longitude, 3), field, int(measurement))
        message = 'Weight: {}, Date: {}, Latitude: {}, Longitude: {}, {}: {}'.format(*formats)
        circle.popup = HTML(message)

        # add to markers layer
        markers.append(circle)

    # make marker layer
    layer = LayerGroup(layers=markers, name=name)
    
    return layer


# In[136]:


designate('fetching closest record subroutine', 'map')

def get_closest_record(associations, latitude, longitude):
    """Fetch the closest record from the list of records based on latitude and longitude
    
    Arguments:
        associations: list of dicts
        latitude: float, latitude coordinate
        longitude: float, longitude coordinate
        
    Returns:
        dict, the closest record
    """
    
    # calculate the distances to all records
    distances = []
    for association in associations:
        
        # calculate squared distance
        geocoordinates = association['location']
        distance = (latitude - geocoordinates[0]) ** 2 + (longitude - geocoordinates[1]) ** 2
        distances.append((association, distance))
        
    # sort by distance and choose closest
    distances.sort(key=lambda pair: pair[1])
    closest = distances[0][0]
    
    return closest


# In[137]:


designate('exhibiting secondary markers on map subroutine', 'map')

def plot_secondary_data_on_map(coordinates, chart, associations, field, fill, line, thickness, radius, opacity, click):
    """Exhibit a record's associated records.
    
    Arguments:
        coordinates: (float, float) tuple, the latitude and longitude
        chart: ipyleaflets Map object
        associations: list of dicts
        field: str, the name of the appropriate field
        fill: function object, to determine fill color
        line: function object, to determine line color
        thickness: int, circle outline thickness
        radius: function object, to determine radius
        opacity: function object, to determine opacity as a function of weight
        click: function object, to act upon a click

    Returns:
        None
    """
    
    # remove last set of associates
    chart.layers = chart.layers[:5]
    
    # fetch the index of the closest association
    association = get_closest_record(associations, *coordinates)
    
    # create marker layer
    records = [associate['record'] for associate in association['associates']]
    weights = [associate['weight'] for associate in association['associates']]
    layer = add_points_to_map(records, weights, field, fill, line, thickness, radius, opacity, click)
    
    # add layer
    chart.add_layer(layer)
    
    return None


# In[138]:


designate('constructing map', 'map')

# begin the map at a central latitude and longitude
def construct_map_at_central_geo_loc():
    """Construct the map at a central geolocation.
    
    Arguments:
        None
        
    Returns:
        ipyleaflet map object
    """

    # print status
    print('constructing map...')

    # get central latitude
    latitudes = [record['latitude'] for record in data]
    latitude = (max(latitudes) + min(latitudes)) / 2

    # get central longitude
    longitudes = [record['longitude'] for record in data]
    longitude = (max(longitudes) + min(longitudes)) / 2

    # set up map with topographical basemap zoomed in on center
    chart = Map(basemap=basemaps.Esri.WorldTopoMap, center=(latitude, longitude), zoom=5)
    
    return chart


# In[139]:


designate('populating map with markers', 'map')

# populate map with markers
def populate_map(chart):
    """Populate the chart with markers.
    
    Arguments:
        chart: ipyleaflet map
        
    Returns:
        ipyleaflet map
    """

    # create unassociated secondary marker layer
    print('marking unassociated secondaries...')
    indices = [association['record']['index'] for association in mirror]
    records = [record for record in uniques if record['index'] not in indices]
    weights = [0.0 for record in records]
    parameters = [records, weights, fieldii, fillii, lineii, thicknessii, radiusii]
    parameters += [empty, nothing, 'unassociated secondaries']
    layer = add_points_to_map(*parameters)
    chart.add_layer(layer)

    # create unassociated primaries layer
    print('marking unassociated primaries...')
    indices = [association['record']['index'] for association in associations]
    records = [record for record in data if record['index'] not in indices]
    weights = [0.0 for record in records]
    parameters = [records, weights, field, tan, tan, thickness, radius]
    parameters += [empty, nothing, 'unassociated primaries']
    layer = add_points_to_map(*parameters)
    chart.add_layer(layer)

    # create associated secondary marker layer
    print('marking associated secondaries...')
    records = [association['record'] for association in heavies]
    weights = [association['associates'][0]['weight'] for association in heavies]
    parametersii = [chart, mirror, 'larvae', fill, black, thickness, radius, highlight, nothing]
    clickingii = lambda **event: plot_secondary_data_on_map(event['coordinates'], *parametersii)
    parameters = [records, weights, fieldii, fillii, lineii, thicknessii, radiusii, opacityii]
    parameters += [clickingii, 'associated secondaries']
    layer = add_points_to_map(*parameters)
    chart.add_layer(layer)

    # create primary marker layer
    print('marking associated primaries...')
    records = [association['record'] for association in associations]
    weights = [association['associates'][0]['weight'] for association in associations]
    parametersii = [chart, associations, feature, fillii, black, thicknessii, radiusii, highlight, nothing]
    clicking = lambda **event: plot_secondary_data_on_map(event['coordinates'], *parametersii)
    parameters = [records, weights, field, fill, line, thickness, radius, opacity]
    parameters += [clicking, 'associated primaries']
    layer = add_points_to_map(*parameters)
    chart.add_layer(layer)
    
    return chart


# In[140]:


designate('enhancing map with controls', 'map')

# enhance the map with control
def add_map_controls(chart):
    """Enhance the map with controls and legends.
    
    Arguments:
        chart: ipyleaflet map object
        
    Returns:
        ipyleaflet map object
    """

    # add full screen button and map scale
    chart.add_control(FullScreenControl())
    chart.add_control(ScaleControl(position='topright'))

    # add genus legend
    labels = [Label(value = r'\(\color{' + 'black' +'} {' + 'Genera:'  + '}\)')]
    labels += [Label(value = r'\(\color{' + indicators[genus] +'} {' + genus  + '}\)') for genus in classification]
    legend = VBox(labels)

    # send to output
    out = Output(layout={'border': '1px solid blue', 'transparency': '50%'})
    with out:

        # display
        display(legend)

    # add to map
    control = WidgetControl(widget=out, position='topright')
    chart.add_control(control)

    # add colormap legend
    colors = [mix_colors(division, bracket, spectrum) for division in divisions]
    colormap = StepColormap(colors, divisions, vmin=bracket[0], vmax=bracket[1], caption=feature)
    out = Output(layout={'border': '1px solid blue', 'transparency': '80%', 'height': '50px', 'overflow': 'scroll'})
    text = Label(value = r'\(\color{' + 'blue' +'} {' + feature  + '}\)')
    with out:

        # display
        display(text, colormap)

    # add to map
    control = WidgetControl(widget=out, position='bottomright')
    chart.add_control(control)

    # add layers control
    control = LayersControl(position='topleft')
    chart.add_control(control)
    
    return chart


# In[141]:


designate('making mirror of associations')

# go through primary record
mirror = {}
for association in associations:
    
    # and each set of associated secondary records
    record = association['record']
    for associate in association['associates']:
         
        # skip if weight is zero 
        weight = associate['weight']
        if associate['weight'] > 0.0:
            
            # add default entry
            index = record['index']
            if index not in mirror.keys():
                
                # begin an entry
                location = (associate['record']['latitude'], associate['record']['longitude'])
                entry = {'record': associate['record'], 'associates': [], 'location': location}
                mirror[index] = entry
                
            # populate entry and sort
            mirror[index]['associates'].append({'record': record, 'weight': weight})
            mirror[index]['associates'].sort(key=lambda associate: associate['weight'], reverse=True)
            
# make into list and sort
mirror = [value for value in mirror.values()]
mirror.sort(key=lambda association: association['associates'][0]['weight'], reverse=True)

# shrink to uniques
uniques = get_unique_record_per_loc_secondary_data(dataii, feature)


# In[142]:


designate('shrinking mirror to highest weight per location')

# construct records
unshrunk = []
for index, association in enumerate(mirror):
    
    # construct faux records
    record = association['record']
    faux = {'index': index, 'latitude': record['latitude'], 'longitude': record['longitude']}
    faux['weight'] = association['associates'][0]['weight']
    unshrunk.append(faux)
    
# shrink records
shrunken = get_unique_record_per_loc_secondary_data(unshrunk, 'weight')
indices = [record['index'] for record in shrunken]
heavies = [mirror[index] for index in indices]


# In[143]:


designate('describing markers')

# get 5th and 95th percentile bracket
measurements = [association['record'][feature] for association in mirror]
low = percentile(measurements, 5)
high = percentile(measurements, 95)
divisions = chisel(low, high)
bracket = (divisions[0], divisions[-1])

# define color spectrum as dark and light rgb tuples
dark = (100, 50, 150)
light = (0, 255, 255)
spectrum = (dark, light)

# define minimum and maximum of larvae counts
measurements = [association['record']['larvae'] for association in associations]
minimal = min(measurements)
maximal = max(measurements)

# create standin functions
black = lambda record: 'black'
tan = lambda record: 'tan'
yellow = lambda record: 'yellow'
nothing = lambda **event: None
empty = lambda weight: 0.0
highlight = lambda weight: 1.0

# create functions for primary markers
field = 'larvae'
fill = lambda record: indicators[record['genus']]
line = fill
thickness = 2
radius = lambda record: min([25, 5 + int(0.1 * record['larvae'])])
opacity = lambda weight: 0.1 + max([weight - 0.1, 0])

# create functions for secondary markers
fieldii = feature
fillii = lambda record: mix_colors(record[feature], bracket, spectrum)
lineii = fillii
thicknessii = 2
radiusii = lambda record: 20
opacityii = lambda weight: 0.1 + max([weight - 0.1, 0])


# ### Optimizing Parameters

# Because there are many parameters involved in the association scheme, it would be nice to be able to test multiple combinations quickly.  In the following table you may input lists of parameters to try.  The optimization function will attempt to find the combination with highest correlation coefficient or lowest bias depending on the criteria set.  To save time at the expense of thoroughness, the optimizer begins at several random combinations and climbs towards the top based on the correlations of similar parameter combinations. 
# 
# In the graph that follows, the size indicates the correlation, and the color indicate the bias.  The two axes are chosen that seem to most affect the criterion.  

# In[144]:


designate('setting optimizer ranges', 'settings')

# set distances to try in kilometers
distances = [10, 25, 50]

# set time intervals to try in days
intervals = [0.5, 1, 5]

# set lag times to try in days
lags = [0, 2]

# set confidences in percents
confidences = [0.2, 0.8]

# set cutoffs in percents
cutoffs = [0.1, 0.5]

# set inclusions
inclusions = [5, 10]

# regression mode ('linear', 'quadratic', 'exponential', 'power', 'gaussian')
mode = 'linear'

# larvae genus ('All', 'Aedes', 'Anopheles', 'Culex', 'Other', 'Unknown')
genus = 'All'

# set criterion ('correlation', 'pvalue')
criterion = 'pvalue'


# In[145]:


designate('applying the parameters button')

# reset parameters function
def update_params_with_new_settings(_):
    """Update the parameters according to new settings.
    
    Arguments:
        None
        
    Returns:
        None
    """
    
    # refresh cells 
    execute_cell_range('### Optimizing Parameters', '### Thank You!')
    
    return None

# create button
button = Button(description="Apply")
display(button)
button.on_click(update_params_with_new_settings)

# infer mode and genus
mode = fuzzy_match(mode, regressions.keys())
genus = fuzzy_match(genus, indicators.keys())
criterion = fuzzy_match(criterion, ('correlation', 'pvalue'))

# update optimizers
optimizers = [distances, intervals, lags, confidences, cutoffs, inclusions]
[optimizer.sort() for optimizer in optimizers]
knobs = ['distance', 'interval', 'lag', 'confidence', 'cutoff', 'inclusion']

# sorting by pearson means taking the biggest, but sorting by pvalue means taking the biggest complement
conditionals = {'correlation': lambda pair: pair[1]['correlation'], 'pvalue': lambda pair: 1 - pair[1]['pvalue']}
condition = conditionals[criterion]

# generate parameters list
styles = ['mode', 'genus', 'criterion']
choices = [mode, genus, criterion]
selections = ['{}: {}\n'.format(knob, optimizer) for knob, optimizer in zip(styles + knobs, choices + optimizers)]

# print status
print('parameters selected:\n')
print(''.join(selections))


# In[146]:


designate('optimize button')

# function to optimize
def optimize(_):
    """Run the optimization sequence.
    
    Arguments:
        None
        
    Returns:
        None
    """
    
    # run optimization
    scores, panda = find_highest_corr_coef()
    
    # make graph
    graph = visualize_corr_scores(scores)
    
    # display graph
    output_notebook()
    show(graph)
    
    # display panda
    display(panda.head(10))
    
    return None

# create button
button = Button(description="Optimize")
button.on_click(optimize)
display(button)


# In[147]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# In[148]:


designate('seeding random starts', 'optimizer')

# making random seeds
def seed(number=5, size=50):
    """Create random parameter seeds that are far from each other
    
    Arguments:
        number=5: int, number of seeds
        size=20: int, size of random pool
        
    Returns:
        list of (float) tuples
    """
    
    # get all lengths of optimizers
    optimizers = [distances, intervals, lags, confidences, cutoffs, inclusions]
    lengths = [len(optimizer) for optimizer in optimizers]
    
    # make pool
    pool = []
    for _ in range(size):
        
        # make a random parameter grab
        trial = tuple([choice([entry for entry in range(length)]) for length in lengths])
        pool.append(trial)
        
    # calculate average euclidean distances and pick furthest
    choices = [pool[choice(len(pool))]]
    pool = [member for member in pool if member not in choices]
    for _ in range(number - 1):
        
        # calculate distances
        averages = []
        for trial in pool:
            
            # calculate average squared distance from all seeds
            euclidizing = lambda a, b: sum([(c - d) ** 2 for c, d in zip(a, b)])
            mean = sum([euclidizing(trial, trialii) for trialii in choices]) / len(choices)
            averages.append((trial, mean))
                            
        # sort and add to choices
        averages.sort(key=lambda pair: pair[1], reverse=True)
        choices.append(averages[0][0])
        pool = [member for member in pool if member not in choices]
        
    # get seeds
    seeds = [tuple(optimizer[entry] for optimizer, entry in zip(optimizers, trial)) for trial in choices]
    
    return seeds


# In[149]:


designate('analyzing a parameter set', 'optimizer')

# analyze regression for one parameter set
def analyze_regression_one_param_set(ticket):
    """Analyze one round of regression after defining the grid.
    
    Arguments:
        ticket: dict
        
    Returns:
        regression scores
    """

    # assemble associations and samples
    associations = merge_datasets(ticket)
    subset = get_subset_of_data(associations, ticket['genus'])
    samples = assemble_samples(subset)
    
    # add lengths to ticket
    ticket['records'] = len(subset)
    ticket['pairs'] = len(samples)
    ticket['coverage'] = round(len(subset) / len(data), 2)
    
    # perform regression
    score = regress(samples, ticket, spy=True)
    
    return score


# In[150]:


designate('cranking through regressions', 'optimizer')

# function to crank through regressions
def crank_through_regressions(seed, scores):
    """Crank through one set of regressions, given a seed
    
    Arguments:
        seed: list of floats, the settings
        scores: dict
        
    Returns
        (dict, int) tuple (scores, count)
    """
    
    # start from each startpoint
    current = tuple(seed)
    delta = 1.0
    optimum = 0.0
    count = 0
    while delta > 0.0:

        # go through each super optimizer
        for index, optimizer in enumerate(optimizers):

            # go through each member
            trials = []
            for member in optimizer:

                # replace in current
                trial = [entry for entry in current]
                trial[index] = member
                trial = tuple(trial)

                # optimize if not already done 
                if trial not in scores.keys():

                    # print status
                    count += 1
                    print('.', end="")
                    if count % 10 == 0:

                        # print
                        print('({})'.format(count), end='')

                    # create ticket
                    ticket = {knob: setting for knob, setting in zip(knobs, trial)}
                    ticket = issue_ticket(ticket, genus, mode)

                    # analyze
                    scores[trial] = analyze_regression_one_param_set(ticket)

                # append
                trials.append(trial)

                # check all scores and sort
                correlations = [(trial, scores[trial]) for trial in trials]
                correlations.sort(key=condition, reverse=True)
                current = correlations[0][0]

                # calculate detla and reset optimum
                delta = condition(correlations[0]) - optimum
                optimum = max([condition(correlations[0]), optimum])
                
    # exit count
    print('')
                
    return scores


# In[151]:


designate('climbing parameters', 'optimizer')

# function to find maximum
def find_highest_corr_coef():
    """Climb to the highest correlation coefficient.
    
    Arguments:
        None
        
    Returns:
        (list of lists, list of lists, dataframe) tuple, (optimizers, scores, panda)
    """

    # start at midpoints and try neighboring combinations
    seeds = 5
    scores = {}
    print('optimizing {} mode for genus {} by {} (using {} seeds)'.format(mode, genus, criterion, seeds))
    for start in seed(seeds):
        
        # crank through optimization
        scores = crank_through_regressions(start, scores)
        
    # sort scores
    scores = [item for item in scores.items()]
    scores.sort(key=condition, reverse=True)
    reports = [report for _, report in scores]
    
    # make panda
    panda = pandas.DataFrame(reports)
    columns = knobs
    columns += ['correlation', 'pvalue', 'records']
    panda = panda[columns]
    panda.columns = [column + units[column](feature) for column in columns]
    
    # add to optimizations
    optimizations.append(panda)
    
    return reports, panda


# In[152]:


designate('orienting plot axes', 'optimizer')

# function to orient the scores along most variant axes
def orient_scores_and_find_highest_scoring_on_each_axis(scores):
    """Orient the scores along most variant axis, and prune to the highest scoring along each.
    
    Arguments:
        scores: list of dicts
        
    Returns:
        (list of dicts, str, str) tuple, the pruned scores and axes names.
    """
    
    # rewrite condition to get triggered only by score
    evaluating = lambda score: condition((None, score))
    
    # collect all maximum correlations by optimizer setting
    maxes = [{} for optimizer in optimizers]
    for score in scores:
        
        # go through each optimizer
        for index, optimizer in enumerate(optimizers):
        
            # add max score
            setting = score[knobs[index]]
            maxes[index][setting] = max([maxes[index].setdefault(setting, 0.0), evaluating(score)])
            
    # get axes with highest difference between max and min
    differences = [(index, (max(members.values()) - min(members.values()))) for index, members in enumerate(maxes)]
    differences.sort(key=lambda pair: pair[1], reverse=True)
    horizontal = knobs[differences[0][0]]
    vertical = knobs[differences[1][0]]
    
    # only plot the highest pearson or lowest bias at any point
    filterer = {}
    for score in scores:
        
        # get pair of coordinates
        faux = {'correlation': 0.0, 'pvalue': 1.0}
        coordinates = (score[horizontal], score[vertical])
        if evaluating(score) >= evaluating(filterer.setdefault(coordinates, faux)):
            
            # replace score
            filterer[coordinates] = score
    
    # create sample attributes
    scores = [score for score in filterer.values()]
    scores.sort(key=evaluating)
    
    return scores, horizontal, vertical


# In[153]:


designate('visualizing correlations', 'optimizer')

# function to scatter graph correlations
def visualize_corr_scores(scores):
    """Visualize the different correlation scores in parameter space.
    
    Arguments:
        optimizers: list of lists of parameters
        scores: list of dicts, regression reports
        
    Returns:
        bokeh graph object
    """
    
    # get the top scores and the axes 
    scores, horizontal, vertical = orient_scores_and_find_highest_scoring_on_each_axis(scores)

    # begin the plot with labels and size
    parameters = {}
    parameters['title'] = 'Predicting larvae counts by {} using {} regression'.format(feature, mode)
    parameters['x_axis_label'] = horizontal + ' {}'.format(units[horizontal](feature))
    parameters['y_axis_label'] = vertical + ' {}'.format(units[vertical](feature))
    parameters['plot_width'] = 800
    parameters['plot_height'] = 400
    graph = figure(**parameters)

    # make annotation labels
    labels = knobs + ['correlation', 'pvalue', 'records']
    annotations = [(label, '@' + label) for label in labels]
    graph = annotate_plot(graph, annotations)
    
    # create ponts, relating size to pearson and color to bias
    blue = (0, 0, 255)
    cyan = (0, 255, 255)
    bracket = (0.5, 1.0)
    xs = [score[horizontal] for score in scores]
    ys = [score[vertical] for score in scores]
    sizes = [10 + int(100 * score['correlation']) for score in scores]
    colors = [mix_colors(1 - score['pvalue'], bracket, (blue, cyan)) for score in scores]
    
    # make columns from scores and labels, suplement with graph attributes
    columns = {label: [round(score[label], 2) for score in scores] for label in labels}
    columns.update({'xs': xs, 'ys': ys, 'sizes': sizes, 'colors': colors})
    source = ColumnDataSource(columns)
    
    # add sample markers
    parameters = {'source': source, 'x': 'xs', 'y': 'ys', 'size': 'sizes', 'fill_color': 'colors'}
    parameters.update({'line_width': 1, 'line_color': 'black', 'fill_alpha': 1.0})
    graph.circle(**parameters)
    
    return graph


# ### Scrutinizing All Combinations

# Alternatively you may use brute force to try all parameter combinations.  This will take a while, but skips shortcuts.

# In[154]:


designate('scrutinize button')

# function to scrutinize
def scrutinize_all_combinations(_):
    """Run the optimization sequence.
    
    Arguments:
        None
        
    Returns:
        None
    """
    
    # run optimization
    scores, panda = optimize()
    
    # make graph
    graph = visualize_corr_scores(scores)
    
    # display graph
    output_notebook()
    show(graph)
    
    # display panda
    display(panda.head(10))
    
    return None

# create button
button = Button(description="Scrutinize")
button.on_click(scrutinize_all_combinations)
display(button)


# In[155]:


designate('combing through all combinations', 'scrutinizer')

# function to find maximum
def optimize():
    """Scrutinize all combinations.
    
    Arguments:
        None
        
    Returns:
        (list of lists, list of lists, dataframe) tuple, (optimizers, scores, panda)
    """

    # make all combinations
    combinations = [[]]
    for optimizer in optimizers:

        # add to combinations
        combinations = [combination + [member] for combination in combinations for member in optimizer]

    # go through all combinations
    combinations = [tuple(combination) for combination in combinations]
    print('{} combinations'.format(len(combinations)))
    print('optimizing by {} '.format(criterion), end="")
    scores = {}
    count=0
    for combination in combinations:
        
        # print status
        count += 1
        print('.', end="")
        if count % 10 == 0:

            # print
            print('({})'.format(count), end='')
        
        # analyze
        ticket = {knob: setting for knob, setting in zip(knobs, combination)}
        ticket = issue_ticket(ticket, genus, mode)
        scores[combination] = analyze_regression_one_param_set(ticket)
  
    # collect scores
    scores = [item for item in scores.items()]
    scores.sort(key=condition, reverse=True)
    scores = [value for _, value in scores]

    # make panda
    panda = pandas.DataFrame(scores)
    columns = knobs
    columns += ['correlation', 'pvalue', 'records']
    panda = panda[columns]
    panda.columns = [column + units[column](feature) for column in columns]
    
    # add to optimizations
    optimizations.append(panda)
    
    return scores, panda


# In[156]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()


# ### Thank You!

# Thanks for taking this notebook for a spin.  Please feal free to direct questions, issues, or other feedback to Matthew Bandel at matthew.bandel@ssaihq.com

# In[157]:


designate('navigation buttons')

# set two navigation buttons
navigate_notebook()
jump_to_regression()

