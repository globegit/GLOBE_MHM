#!/usr/bin/env python
# coding: utf-8

# # Mosquito Larvae in my Country

# Under the GLOBE project, citizen scientists around the world have been monitoring mosquitoes.  In particular, they have been counting larvae and attempting to identify the specific genera involved.  This data is available from the GLOBE API.  The goal of this notebook is to demonstrate how to download the data from the API using Python code, and how to create some useful visualizations.  The hope is for this notebook to be both an exploratory tool of the data at hand, as well as to serve as a tutorial for analyzing the GLOBE data in general.

# ### Importing Required Modules

# A few Python modules are required to run this script.

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
    designation = ' ' * 15
    
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


designate('importing Python modules')

# import os and sys modules for system controls
import os
import sys

# import requests and json modules for making API requests
import requests
import json

# set runtime warnings to ignore
import warnings

# import fuzzywuzzy for fuzzy name matching
from fuzzywuzzy import fuzz

# import datetime module for manipulating date and time formats
from datetime import datetime, timedelta

# import numpy module and math module for mathematical functions
from numpy import sqrt, pi, average, std, histogram

# import pandas for date table manipulation
import pandas

# import zipfile to create zip files
from zipfile import ZipFile


# In[3]:


designate('importing Python plotting modules')

# import bokeh for plotting graphs
from bokeh.plotting import figure
from bokeh.io import output_notebook, show
from bokeh.models import HoverTool, ColumnDataSource, Whisker, TeeHead
from bokeh.models.formatters import DatetimeTickFormatter

# import ipyleaflet for plotting maps
from ipyleaflet import Map, Marker, basemaps, CircleMarker, LayerGroup
from ipyleaflet import WidgetControl, ScaleControl, FullScreenControl

# import iPython for javascript based notebook controls
from IPython.display import Javascript, display, Image, FileLink

# import ipywidgets for additional widgets
from ipywidgets import Label, HTML, Button, Layout, Output, HBox, VBox, Box


# In[4]:


designate('inferring fuzzy match', 'tools')

# subroutine for fuzzy matching
def infer(text, options):
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


# In[5]:


designate('truncating field names', 'tools')

# truncate field names to first capital
def truncate(name, size=5, minimum=4, maximum=15):
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


# In[6]:


designate('entitling a name by capitalizing', 'tools')

# entitle function to capitalize a word for a title
def entitle(word):
    """Entitle a word by capitalizing the first letter.
    
    Arguments:
        word: str
        
    Returns:
        str
    """
    
    # capitalize first letter
    word = word[0].upper() + word[1:]
    
    return word


# In[7]:


designate('resolving country name and code', 'tools')

# resolving country name and codes
def resolve(country, code):
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
        code = infer(code, [member for member in codes.values()])
        country = countries[code]
    
    # if no code, but a country is given
    if not code and country:
        
        # find closest matching country
        country = infer(country, [member for member in codes.keys()])
        code = codes[country]
    
    # if there's no code, check the country
    if not code and not country:
        
        # default to all countries
        country = 'All countries'
        code = ''
    
    return country, code


# In[8]:


designate('scanning notebook for cells', 'introspection')

# scan notebook for cell information
def scan():
    """Scan the notebook and collect cell information.

    Arguments:
        None

    Returns:
        list of dicts
    """

    # open the notebook file 
    with open('mosquitoes.ipynb', 'r', encoding='utf-8') as pointer:
        
        # and read its contents
        contents = json.loads(pointer.read())

    # get all cells
    cells = contents['cells']

    return cells


# In[9]:


designate('defining global variables')

# ignore runtime warnings
warnings.filterwarnings('ignore')

# set pandas optinos
pandas.set_option("display.max_rows", None)
pandas.set_option("display.max_columns", None)

# make doppelganger for navigation
doppelganger = scan()


# In[10]:


designate('import status')

# print status
print('modules imported.')


# ### Notes on Navigation

# Running Cells:
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
# Processing Indicator:
# 
# - In the upper righthand corner it says "Python 3" with a circle.  If this circle is black, it means the program is still processing.   A hollow circle indicates all processing is done.
# 
# 
# Collapsible Headings and Code Blocks:
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
# Hosting by myBinder:
# 
# 
# - This notebook is hosted by myBinder.org in order to maintain its interactivity within a browser without the user needing an established Python environment.  Unfortunately, connection with myBinder.org will break after 10 minutes of inactivity.  In order to reconnect you may use the link under "Browser Link" to reload.
# 
# 
# - The state of the notebook may be saved by clicking the leftmost cloud icon in the toolbar to the right of the Download button.  This saves the notebook to the browser.  The rightmost cloud icon can then retrieve this saved state in a newly opened copy.  Often reopening a saved version comes with all code blocks visible, so toggle this using the eye icon in the toolbar.
# 
# 
# - The following browser link will reload the notebook in case the connection is lost:
# https://mybinder.org/v2/git/https%3A%2F%2Fmattbandel%40bitbucket.org%2Fmattbandel%2Fglobe-mosquitoes-notebook.git/master?filepath=mosquitoes.ipynb

# In[11]:


designate('looking for particular cell', 'navigation')

# function to look for cells with a particular text snippet
def look(text):
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


# In[12]:


designate('jumping to a particular cell', 'navigation')

# jump to a particular cell
def jump(identifier):
    """Jump to a particular cell.
    
    Arguments:
        identifier: int or str
        
    Returns:
        None
    """
    
    # try to look for a string
    try:
        
        # assuming string, take first index with string
        index = look(identifier)
        
    # otherwise assume int
    except (TypeError, IndexError):
        
        # index is identifier
        index = identifier 
    
    # scroll to cell
    command = 'IPython.notebook.scroll_to_cell({})'.format(index)
    display(Javascript(command))
    
    return


# In[13]:


designate('executing cell range by text', 'navigation')

# execute cell range command
def execute(start, finish):
    """Execute a cell range based on text snippets.
    
    Arguments:
        start: str, text from beginning cell of range
        finish: str, text from ending cell of range
        
    Returns:
        None
    """
    
    # find start and finish indices, adding 1 to be inclusive
    opening = look(start)[0] 
    closing = look(finish)[0]
    bracket = (opening, closing)
    
    # make command
    command = 'IPython.notebook.execute_cell_range' + str(bracket)
    
    # perform execution
    display(Javascript(command))
    
    return None


# In[14]:


designate('refreshing cells by relative position', 'navigation')

# execute cell range command
def refresh(start, finish=None):
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


# In[15]:


designate('revealing open cells', 'navigation')

# outline headers
def reveal(cells):
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


# In[16]:


designate('gauging cell size', 'navigation')

# measure a cell's line count and graphics
def gauge(cell):
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


# In[17]:


designate('bookmarking cells for screenshotting', 'navigation')

# bookmark which cells to scroll to
def bookmark(cells):
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
    visibles = reveal(cells)
    for index in visibles:

        # measure cell and add to total
        cell = cells[index]
        length, graphic = gauge(cell)
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


# In[18]:


designate('describing cell contents', 'navigation')

# describe the cells
def describe(*numbers):
    """Describe the list of cells by printing cell summaries.

    Arguments:
        numbers: unpacked list of ints

    Returns:
        None
    """

    # get cells and analyze
    visibles = reveal(doppelganger)
    bookmarks = bookmark(doppelganger)

    # print cell metadata
    for index, cell in enumerate(doppelganger):

        # construct stars to mark visible and bookmark statuses
        stars = '' + '*' * (int(index in visibles) + int(index in bookmarks))

        # check in numbers
        if len(numbers) < 1 or index in numbers:
        
            # print metadata
            print(' \n{} cell {}:'.format(stars, index))
            print(cell['cell_type'])
            print(cell['source'][0][:100])
            print(cell['metadata'])
            print([key for key in cell.keys()])
            if 'outputs' in cell.keys():

                # print outputs
                print('outputs:')
                for entry in cell['outputs']:

                    # print keys
                    print('\t {}, {}'.format(entry['output_type'], [key for key in entry.keys()]))

    return None


# In[19]:


designate('propagating setting changes across cells', 'buttons')

# def propagate
def propagate(start, finish, finishii, descriptions=['Apply', 'Propagate', 'Both']):
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
    buttoning = lambda start, finish: lambda _: execute(start, finish)
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
def navigate():
    """Guide the user towards regression sections with buttons.
    
    Arguments:
        None
        
    Returns:
        None
    """

    # define jump points
    descriptions = ['Top', 'Settings', 'Filter', 'Graphs', 'Sources', 'Map', 'Bottom']
    cues = ['# Mosquito Larvae in my Country', '### Setting the Parameters', '### Filtering Records']
    cues += ['### Plotting Larvae Counts over Time', '### Plotting Larvae Counts by Water Source']
    cues += ['### Plotting Sampling Locations on a Map', '### Thank You!']
    
    # make buttons
    buttons = []
    buttoning = lambda cue: lambda _: jump(cue)
    for description, cue in zip(descriptions, cues):

        # make button
        button = Button(description=description, layout=Layout(width='140px'))
        button.on_click(buttoning(cue))
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

# In[21]:


designate('navigation buttons')

# set navigation buttons
navigate()


# ### Table of GLOBE Countries

# The follow is a list of all GLOBE supporting countries and their country codes.

# In[22]:


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

# Set the parameters for desired country and date range here.

# In[23]:


designate('setting the country and date range', 'settings')

# set the desired country name (refer to table above for exact spelling)
country = 'Thailand'

# or set the country code (the country code will override the name, unless it is set to None or '')
code = ''

# set beginning and ending dates in 'YYYY-mm-dd' format
beginning = '2019-01-01'
ending = '2020-01-01'


# Press Apply to apply the changes to this section, then Propagate to propagate the changes down the notebook.  Clicking Both will perform both these actions.

# In[24]:


designate('applying setting changes or propagating throughout notebook')

# propagate changes
propagate('### Setting the Parameters', '### Making the API Call', '### Exporting to PDF')


# In[25]:


designate('calling the api', 'api')

# call the api with protocol and country code
def call(protocol, code, beginning, ending, sample=False):
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


# In[26]:


designate('resolving user settings')

# define primary protocol name and larvae field
mosquitoes = 'mosquito_habitat_mapper'
larvae = 'mosquitohabitatmapperLarvaeCount'

# resolve country and code to default values
country, code = resolve(country, code)

# default beginning to first day of GLOBE and ending to current date if unspecified
beginning = beginning or '1995-04-22'
ending = ending or str(datetime.now().date())

# make api call to get number of records
print('checking number of records...')
raw = call(mosquitoes, code, beginning, ending, sample=True)
count = raw['count']
print('{} {} records from {} ({}), from {} to {}'.format(count, mosquitoes, country, code, beginning, ending))


# In[27]:


designate('navigation buttons')

# set navigation buttons
navigate()


# ### Making the API Call

# Assemble the url for the API call and collect the resulting records.  The API is also available at the following link:
# 
# https://www.globe.gov/en/globe-data/globe-api

# In[28]:


designate('retrieving mosquito habitat data from the API')

# make api call
print('making request...')
raw = call(mosquitoes, code, beginning, ending, sample=False)
results = [record for record in raw['results']]

# report number of records found
print('{} records from {}'.format(len(results), country))


# In[29]:


designate('checking records length')

# raise assertion error on zero records
try:
    
    # assert length > 0
    assert len(results) > 0
    
# otherwise
except AssertionError:
    
    # raise the error with a message
    message = '* Error! * No records returned'
    raise Exception(message)


# In[30]:


designate('navigation buttons')

# set navigation buttons
navigate()


# ### Simplifying the Records

# The data is returned as a nested structure, as can be viewed here.

# In[31]:


designate('viewing example original record')

# view first record
results[0]


# It is useful to flatten this structure so that all fields are available at the top level of the record.  The following defines a recursive flattening function.

# In[32]:


designate('record flattening records', 'processing')

# function to flatten a nested list into a single-level structure
def flatten(record, label=None):
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
            flattened.update(flatten(info, field))

    # otherwise record is a terminal entry
    except AttributeError:

        # so update the dictionary with the record
        flattened.update({label: record})

    return flattened


# Flatten all results for convenience.

# In[33]:


designate('for flattening all records')

# flatten all results
flats = [flatten(record) for record in results]

# print status
print('records flattened.')


# Additionally, it can be useful to abbreviate the fields of interest as the initial field names are often quite long.

# In[34]:


designate('abbreviating fields')

# define abbreviations dictionary
abbreviations = {}
abbreviations['count'] = 'mosquitohabitatmapperLarvaeCount'
abbreviations['genus'] = 'mosquitohabitatmapperGenus'
abbreviations['source'] = 'mosquitohabitatmapperWaterSource'
abbreviations['stage'] = 'mosquitohabitatmapperLastIdentifyStage'
abbreviations['type'] = 'mosquitohabitatmapperWaterSourceType'
abbreviations['measured'] = 'mosquitohabitatmapperMeasuredAt'
abbreviations['habitat'] = 'mosquitohabitatmapperWaterSourcePhotoUrls'
abbreviations['body'] = 'mosquitohabitatmapperLarvaFullBodyPhotoUrls'
abbreviations['abdomen'] = 'mosquitohabitatmapperAbdomenCloseupPhotoUrls'

# for each record
for record in flats:
    
    # and each abbreviation
    for abbreviation, field in abbreviations.items():
        
        # copy new field from old, or None if nonexistent
        record[abbreviation] = record.setdefault(field, None)
        del(record[field])
        
# print status
print('fields abbreviated.')


# It is also useful to convert the time field into a "datetime" object for later processing.

# In[35]:


designate('converting times')

# convert each record
for record in flats:

    # convert the date string to date object and normalize based on partitioning
    record['time'] = datetime.strptime(record['measured'], "%Y-%m-%dT%H:%M:%S")
    record['date'] = record['time'].date()
    
    # convert the date string to date object and correct for longitude
    zone = int(round(record['longitude'] * 24 / 360, 0))
    record['hour'] = record['time'] + timedelta(hours=zone)
    
# status
print('times converted.')


# Also, some steps have been taken towards mosquito genus identification.  The three noteworthy genera in terms of potentially carrying disease are Aedes, Anopheles, and Culex.  If the identification process did not lead to one of these three genera, the genus is regarded as "Other."  If the identification process was not fully carried out, the genus is regarded as "Unknown."

# In[36]:


designate('parsing mosquito genera')

# correct genus based on last stage
for record in flats:
    
    # check genus
    if record['genus'] is None:
        
        # check last stage
        if record['stage'] in (None, u'identify'):
            
            # correct genus to 'Unidentified'
            record['genus'] = 'Unknown'
            
        # otherwise
        else:
            
            # correct genus to 'Other'
            record['genus'] = 'Other'

# establish genera and colors
classification = ['Unknown', 'Other', 'Aedes', 'Anopheles', 'Culex']
colors = ['gray', 'lightgreen', 'crimson', 'orange', 'magenta']

# create indicator colors to be used on plots
indicators = {genus: color for genus, color in zip(classification, colors)}
indicators.update({None: 'lightgreen'})

# status
print('genera parsed.')


# Finally, the photo urls will be parsed here.

# In[37]:


designate('constructing latlon string', 'processing')
    
# specify the location code for the photo based on its geo coordinates
def localize(latitude, longitude):
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


# In[38]:


designate('applying naming convention subroutine')

# apply the naming convention to a photo url to make a file name
def apply(urls, code, latitude, longitude, time):
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
    base = 'GLOBEMHM_' + localize(latitude, longitude) + '_'
    
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


# In[39]:


designate('parsing photo urls')

# dictionary of photo sector codes
sectors = {'habitat': 'WS', 'body': 'FB', 'abdomen': 'AB'}

# for each record
for record in flats:

    # initialize fields for each sector and parse urls
    record['originals'] = []
    record['thumbs'] = []
    record['photos'] = []
    for field, code in sectors.items():
        
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
        photos = apply(originals, code, record['latitude'], record['longitude'], record['time'])
        record['photos'] += photos
        
# status
print('photo urls parsed.')


# The final record format can be viewed here.

# In[40]:


designate('viewing example flattened and abbreviated record')

# view first record
flats[0]


# ### Removing Null Results

# It is often the case that there is no valid entry in the field of interest.  Filter out these records so that only valid records remain.

# In[41]:


designate('removing null results')

# split nulls from the records so that only valid records are kept for analysis
nulls = [record for record in flats if record['count'] is None]
valids = [record for record in flats if record['count'] is not None]

# raise assertion error on zero record
try:
    
    # assert length > 0
    assert len(valids) > 0
    
# otherwise
except AssertionError:
    
    # raise the error with a message
    message = '* Error! * No valid records returned'
    raise Exception(message)
    
# sort records by time
valids.sort(key=lambda record: str(record['time']))

# count the number of these records
print('{} valid records'.format(len(valids)))


# ### Converting to Numbers

# The larvae count data are initially returned as strings.  In order to analyze the data, these strings must be converted into numbers.  Additionally, some of the data is entered as a range (e.g., '1-25'), or as a more complicated string ('more than 100').  The following function converts each of these cases to a number by:
# - converting a string, such as '50' to its floating point equivalent (50)
# - converting a range, such as '1-25' to its average (13)
# - converting a more complicated string, such as 'more than 100' to its nearest number (100)

# In[42]:


designate('string conversion subroutine')

# function to convert a string into a floating point number
def convert(info):
    """Translate info given as a string or range of numbers into a numerical type.
    
    Arguments:
        info: string
        
    Returns:
        float
    """
    
    # try to convert directly
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
        
    return conversion


# Convert all larvae counts to numbers.

# In[43]:


designate('converting all strings to numbers')

# for each record
for record in valids:
    
    # add the new field
    record['larvae'] = convert(record['count'])
    
# print status
print('strings converted.')


# ### Pruning Suspicious Outliers

# It is sometimes the case that records contain suspicous data.  For instance, an entry of '1000000' for larvae counts is suspicous because likely no one counted one million larvae.  These data can skew analysis and dwarf the rest of the data in graphs.  
# 
# The approach taken here is to calculate a "z-score" for each record.  The z-score measures how many standard deviations the observation is from the mean of all observations.  A highly negative or positive z-score, for instance 20, indicates an observation 20 standard deviations away from the mean.  In a normal distribution, 99% of observations are found within 3 standard deviations, so an abnormally high z-score indicates a highly unlikely observation.  
# 
# The following function prunes away likely outliers by calculating z-scores and removing those above a threshold.  With the outliers removed, new z-scores are calculated and the process continues until no more records get removed.

# In[44]:


designate('outlier pruning subroutine')

# function to prune away outlying observations
def prune(records, field, threshold=75):
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


# Set the z-score threshold.

# In[45]:


designate('setting z-score threshold', 'settings')

# set z-score threshold, the number of standard deviations allowed
threshold = 85


# In[46]:


designate('applying setting changes or propagating throughout notebook')

# propagate changes
propagate('### Pruning Suspicious Outliers', '### Filtering Records', '### Exporting to PDF')


# Prune away the outliers.

# In[47]:


designate('pruning away outliers')

# only prune valid authentics if more than two records
authentics = valids
outliers = []
for record in authentics:
    
    # set zscores to 1
    record['score'] = 1.0

# find outliers
if len(authentics) > 2:
    
    # prune outliers
    authentics, outliers = prune(authentics, threshold)
    
# report each outlier
print('z-score  larvae count')
for outlier in outliers:
    
    # print
    print(round(outlier['score'], 5), outlier['larvae'])
    
# report total
print('\n{} observations removed'.format(len(outliers)))
print('{} records after removing outliers'.format(len(authentics)))


# Construct a histogram of the larvae count size to examine the distribution.  There are zooming tools at the right as well as a hover tool for inspecting individual columns.

# In[48]:


designate('gathering histogram data')

# gather up larvae observations
observations = [record['larvae'] for record in authentics]

# set the width of each bar in larvae counts
width = 5

# calculate the number of histogram bins
minimum = min(observations)
maximum = max(observations)
bins = int((maximum - minimum) / width) + 1

# readjust maximum to cover an even number of bins
maximum = minimum + bins * width

# use numpy to get the counts and edges of each bin
counts, edges = histogram(observations, bins=bins, range=(minimum, maximum))

# calculate z-scores for each histogram bin
scores = [round((edge - average(observations)) / std(observations), 2) for edge in edges[:-1]]

# accumulate the info into a table
table = ColumnDataSource({'counts': counts, 'left': edges[:-1], 'right': edges[1:], 'scores': scores})


# In[49]:


designate('setting overall histogram parameters')

# create parameters dictionary for histogram labels
parameters = {}
parameters['title'] = 'Histogram of Larve Counts in {}'.format(country)
parameters['x_axis_label'] = 'Larvae Counts'
parameters['y_axis_label'] = 'Number of Observations'

# add plot size parameters
parameters['plot_height'] = 400
parameters['plot_width'] = 600

# add initial zoom range at 5 standard deviations
parameters['x_range'] = (0, int(std(observations)) * 5)

# set the height of the graph to extend past the highest error bound
parameters['y_range'] = (0, max(counts) + 10)

# initialize the bokeh graph with the parameters
gram = figure(**parameters)


# In[50]:


designate('setting the bar parameters and annotations')

# set parameters for drawing the bars, indicating the source of the data
bars = {}
bars['source'] = table

# use the column headers to indicate the outline of each bar
bars['bottom'] = 0
bars['top'] = 'counts'
bars['left'] = 'left'
bars['right'] = 'right'

# set the colors
bars['line_color'] = 'white'
bars['fill_color'] = 'lightgreen'

# draw the bars with the quad plotting function
gram.quad(**bars)

# create annotations for the hover tool, where the @ is used in strings to refer to lists of data
annotations = []
annotations += [('Larvae Counts Interval:', '(@left - @right)')]
annotations += [('Number of Observations', '@counts')]
annotations += [('Z-score', '@scores')]

# activate the hover tool
hover = HoverTool(tooltips=annotations)
gram.add_tools(hover)


# In[51]:


designate('displaying the histogram')

# display
output_notebook()
show(gram)


# In[52]:


designate('navigation buttons')

# set navigation buttons
navigate()


# ### Filtering Records

# You may further filter the data if desired.  Smaller datasets render more quickly, for instance.  Set the criteria and click Apply to perform the filtering. You may set a parameter to None to avoid filtering by that parameter.

# In[53]:


designate('setting filter parameters', 'settings')

# set the specific genera of interest ['Anopheles', 'Aedes', 'Culex', 'Unknown', 'Other']
# None defaults to all genera
genera = ['Anopheles', 'Aedes', 'Culex', 'Unknown', 'Other']

# set minimum and maximum larvae counts or leave as None
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


# In[54]:


designate('applying setting changes or propagating throughout notebook')

# propagate changes
propagate('### Filtering Records', '### Grouping Observations by Time', '### Exporting to PDF')


# In[55]:


designate('sifting data through filter', 'filtering')

# function to sift data through filters
def sift(records, parameters, fields, functions, symbols):
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


# In[56]:


designate('filtering records')

# set records to data
records = [record for record in authentics]

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
symbols = ['in', '>=', '<=', '>=', '<=', '>=', '<=', '>=', '<=']

# filter primaries
data, criteria = sift(authentics, parameters, fields, functions, symbols)
formats = (len(data), len(authentics), mosquitoes, criteria)
print('\n{} of {} {} records meeting criteria:\n\n{}'.format(*formats))

# set genera to classification by default
genera = genera or classification


# In[57]:


designate('navigation buttons')

# set navigation buttons
navigate()


# ### Grouping Observations by Time

# It may be useful to group the data by some time interval to study long term trends.  Set the desired time interval in order to group the records.  Possible values are 'year', 'month', 'week', and 'day'.

# In[58]:


designate('setting the time interval', 'settings')

# set time interval
interval = 'week'


# Click the Apply button to refresh the interval.

# In[59]:


designate('applying setting changes or propagating throughout notebook')

# propagate changes
bookmarks = ['### Grouping Observations by Time', '### Plotting Larvae Counts over Time']
bookmarks += ['### Grouping Observations by Time of Day']
propagate(*bookmarks)


# Group records into boxes based on the chosen interval.

# In[60]:


designate('partitioning data around time interval')

# create normalization functions to round dates to the first of the month or year
normalize = {}
normalize['day'] = lambda date: date
normalize['week'] = lambda date: date
normalize['month'] = lambda date: date.replace(day=1)
normalize['year'] = lambda date: date.replace(day=1).replace(month=1)

# resolve function to pick the relevant date
resolving = lambda first, second: datetime.strptime(str(first or second), "%Y-%m-%d").date()
                                                   
# zoom in on initial and final times from beginning and ending
initial = resolving(initial, beginning)
final = resolving(final, ending)

# normalize initial and final dates
initial = normalize[interval](initial)
final = normalize[interval](final)

# create incrementation functions to advance one time partition
increment = {}
increment['day'] = lambda box: box + timedelta(days=1)
increment['week'] = lambda box: box + timedelta(weeks=1)
increment['month'] = lambda box: (box + timedelta(weeks=5)).replace(day=1)
increment['year'] = lambda box: (box + timedelta(weeks=53)).replace(day=1)

# create boxes for each month
boxes = {}
box = initial
while box <= final:
    
    # add box
    boxes[box] = []
    
    # increment to get next box
    box = increment[interval](box)


# In[61]:


designate('sorting records into boxes')    
    
# for each record
for record in data:
    
    # normalize the date record for the nearest box
    box = normalize[interval](record['date'])
    
    # fit into appropriate box
    unboxed = True
    while unboxed:
        
        # try directly adding
        try:
            
            # add to boxes
            boxes[box].append(record)
            unboxed = False
            
        # otherwise (if partitioning is in weeks)
        except KeyError:
            
            # try using box based on previous day
            box = box - timedelta(days=1)
    
# convert boxes to list and sort
boxes = [item for item in boxes.items()]
boxes.sort()

# status
print('\ndata sorted into {} boxes by {}.'.format(len(boxes), interval))


# In[62]:


designate('navigation buttons')

# set navigation buttons
navigate()


# ### Plotting Larvae Counts over Time

# The following plots average larvae counts over time, grouped according to the selected time interval.  Standard deviations are also calculated and indicated with error bars.

# In[63]:


designate('delineating bar graph bars subroutine')
    
# outline bar graph bars
def delineate(boxes, increment, table, categorical=False, scale=1):
    """Delineate the graph bars.
    
    Arguments:
        boxes: list of (number, list) tuples
        increment: function to get rightmost point from left
        table: dict
        categorical=False: boolean, categorical variable?
        scale: float, scale for whiskers
        
    Returns:
        dict
    """
    
    # make samples column for number of samples in each box
    table['size'] = [len(box[1]) for box in boxes]
    
    # make left column from data or from data indices if categorical
    table['left'] = [(box[0], float(index))[int(categorical)] for index, box in enumerate(boxes)]
    
    # make right and middle columns from increment function
    table['right'] = [increment(left) for left in table['left']]
    table['middle'] = [left + (right - left) / 2 for left, right in zip(table['left'], table['right'])]
    
    # calculate averages and standard deviations, using [0.0] for default if empty
    table['average'] = [average([record['larvae'] for record in box[1]] or [0.0]) for box in boxes]
    table['deviation'] = [std([record['larvae'] for record in box[1]] or [0.0]) for box in boxes]
    
    # calculte upper and lower bounds for the error bars, using 1 /10 the standard deviation
    table['upper'] = [height + (error / scale) for height, error in zip(table['average'], table['deviation'])]
    table['lower'] = [height - (error / scale) for height, error in zip(table['average'], table['deviation'])]
    
    # set bottom at 0 and top at the average counts
    table['bottom'] = [0.0 for box in boxes]
    table['top'] = [entry for entry in table['average']]
    
    return table


# In[64]:


designate('parameterizing bars and error bars subroutine')

# parameterize bars and error bar whiskers
def parameterize(table, genus=None):
    """Parameterize the bar and error bar details for graphing.
    
    Arguments:
        table: dict, the data
        genus: str, the genus
        
    Returns:
        dict, dict of bar parameters, whisker parameters
    """

    # set the parameters for the bars in reference to the data table
    color = indicators[genus]
    bar = ({'source': table, 'x': 'middle', 'y': 'top', 'line_color': color, 'line_width': 4})
    whisker = {'source': table, 'x': 'middle', 'line_color': color, 'line_width': 3, 'line_dash': 'dashed'}
    whiskerii = whisker.copy()
    whisker.update({'y': 'lower'})
    whiskerii.update({'y': 'upper'})
    whiskers = [whisker, whiskerii]
    
    # update with legend if available
    if genus:
        
        # update bars and whiskers
        bar.update({'legend_label': genus})
        [whisker.update({'legend_label': genus}) for whisker in whiskers]

    return bar, whiskers


# In[65]:


designate('gathering the dataset')

# delineate columns
table = delineate(boxes, increment[interval], {})

# get maximum height
height = max(table['upper'])

# pack into a source
table = ColumnDataSource(table)

# get bar and whisker
bar, whiskers = parameterize(table)


# In[66]:


designate('setting the graph parameters')

# set the graph label parameters
parameters = {}
parameters['title'] = 'Average Larvae Counts per {} for {}'.format(interval, country)
parameters['x_axis_label'] = 'Date'
parameters['y_axis_label'] = 'Average Larvae Count'

# set the x-axis format as a datetime
parameters['x_axis_type'] = 'datetime'

# set the size of the graph parameters
parameters['plot_height'] = 400
parameters['plot_width'] = 800

# set the height of the graph to extend past the highest error bound
parameters['y_range'] = (0, height + 10)

# Add annotations for the hover tool
annotations = []
annotations += [('Time Interval', '@left{%F} - @right{%F}')]
annotations += [('Number of Observations', '@size')]
annotations += [('Average Larvae Counts', '@average')]
annotations += [('Standard Deviation', '@deviation')]

# activate hover tool
hover = HoverTool(tooltips=annotations, formatters={'left': 'datetime', 'right': 'datetime'})


# In[67]:


designate('drawing the graph')

# draw the graph
graph = figure(**parameters)
graph.line(**bar)
graph.circle(**bar)
[graph.line(**whisker) for whisker in whiskers]
graph.add_tools(hover)

# show plot
output_notebook()
show(graph)


# Note: Each point is potentially the average of several observations.  The standard deviation is indicated by the dotted lines above and below the main plots.

# In[68]:


designate('navigation buttons')

# set navigation buttons
navigate()


# ### Plotting Larvae Counts per Genus

# In addition to counting mosquito larvae, citizen scientists are asked to identify the genus of the larvae they find. There are three genera in particular, Aedes, Anopheles, and Culex that are known to be potential hosts for diseases. If one of these three genera cannot be identified, the genus is regarded as Other. And if the identification step is not performed, the genus is regarded as Unknown.  The following plots average larvae counts over time as above, but separated by genus.

# In[69]:


designate('stacking bars subroutine')

# stack collection of bars on top of each other
def stack(tables):
    """Stack data tables on top of each other.
    
    Arguments:
        tables: list of dicts
        
    Returns:
        list of dicts, stacked tables
    """
    
    # default previous to zeroes
    previous = [0.0] * len(tables[0]['average'])
    
    # adjust heights on each table
    for table in tables:
        
        # raise bar heights according to previous bar heights
        table['bottom'] = previous
        table['top'] = [top + entry for top, entry in zip(table['top'], previous)]

        # raise error bars by same amount
        table['upper'] = [upper + entry for upper, entry in zip(table['upper'], previous)]
        table['lower'] = [lower + entry for lower, entry in zip(table['lower'], previous)]

        # update the new starting heights and maximum height
        previous = [entry for entry in table['top']]
        
    return tables


# In[70]:


designate('gathering the datasets')

# make data table for each genus
tables = []
for genus in genera:
    
    # retain only the subset of specific genera
    subset = [(box[0], [record for record in box[1] if record['genus'] == genus]) for box in boxes]

    # begin data columns with genus information
    table = {}
    table['genus'] = [genus] * len(subset)
    
    # delineate columns
    table = delineate(subset, increment[interval], table)
    tables.append(table)
    
# calculate maximum height of upper bars
height = max([max(table['upper']) for table in tables])

# create columns objects
tables = [ColumnDataSource(table) for table in tables]

# gather bar and whisker parameters
gathering = [parameterize(table, genus) for table, genus in zip(tables, genera)]


# In[71]:


designate('setting the graph parameters')

# set the graph label parameters
parameters = {}
parameters['title'] = 'Average Larvae Counts per {} for {} per Genus'.format(interval, country)
parameters['x_axis_label'] = 'Date'
parameters['y_axis_label'] = 'Stacked Average Larvae Counts'

# set the x-axis format as a datetime
parameters['x_axis_type'] = 'datetime'

# set the y-axis range
parameters['y_range'] = (0, height + 10)

# set the size of the graph parameters
parameters['plot_height'] = 400
parameters['plot_width'] = 800

# Add annotations for the hover tool
annotations = []
annotations += [('Genus', '@genus')]
annotations += [('Time Interval', '@left{%F} - @right{%F}')]
annotations += [('Number of Observations', '@size')]
annotations += [('Average Larvae Counts', '@average')]
annotations += [('Standard Deviation', '@deviation')]

# make hover tool
hover = HoverTool(tooltips=annotations, formatters={'left': 'datetime', 'right': 'datetime'})


# In[72]:


designate('drawing the graph')

# initialize the graph and draw the bars
graph = figure(**parameters)
[graph.line(**bar) for bar, _ in gathering]
[graph.circle(**bar) for bar, _ in gathering]
[[graph.line(**whisker) for whisker in whiskers] for _, whiskers in gathering]
#[graph.add_layout(Whisker(**whisker)) for _, whisker in gathering]
graph.add_tools(hover)
graph.legend.click_policy='hide'

# show plot
output_notebook()
show(graph)


# Note: The correspondance between this graph and the above graph will only be approximate, because the first graph is plotting the average of all samples, whereas this graph is plotting the average per genus.  The sum of averages may differ from the average of sums.  The 'Other' category represents genus identified but not in the three of note, whereas the 'Unknown' category represents genus identification that has not been carred out.

# In[73]:


designate('navigation buttons')

# set navigation buttons
navigate()


# ### Grouping Observations by Time of Day

# Instead, it may be useful to group the data by time of day.  Set the desired number of hours to use for grouping the records.  Possible values are 1, 2, 3, 4, 6, 12, and 24.
# 

# In[74]:


designate('setting the daypart interval', 'settings')

# set length of hours interval
hours = 1


# Click the button below to refresh the daypart interval, grouping records into appropriate boxes.

# In[75]:


designate('applying setting changes or propagating throughout notebook')

# propagate changes
bookmarks = ['### Grouping Observations by Time of Day', '### Plotting Larvae Counts by Water Source']
bookmarks += ['### Plotting Larvae by Counts Water Source']
propagate(*bookmarks)


# In[76]:


designate('making boxes for each daypart interval')

# use new year's as dummy date
midnight = datetime(2020, 1, 1, 0)

# create boxes for each month, starting with dummy date
boxesii = {}
box = midnight
while box < midnight + timedelta(days=1):
    
    # add box
    boxesii[box] = []
    
    # increment to get next box
    box += timedelta(hours=hours)


# In[77]:


designate('sorting records into boxes')

# for each record
for record in data:
    
    # center on dummy date of 1/1/2020 for datetime operations
    box = record['hour'].replace(year=2020, month=1, day=1, minute=0)
    
    # fit into appropriate box
    unboxed = True
    while unboxed:
        
        # try directly adding
        try:
            
            # add to boxes
            boxesii[box].append(record)
            unboxed = False
            
        # otherwise
        except KeyError:
            
            # try using box based on previous hour
            box = box - timedelta(hours=1)
    
# convert boxes to list and sort
boxesii = [item for item in boxesii.items()]
boxesii.sort()

# status
print('\ndata sorted into {} boxes by {} hours each.'.format(len(boxesii), hours))


# The following plots this data.

# In[78]:


designate('gathering the datasets')

# make data table for each genus
tables = []
for genus in genera:
    
    # retain only the subset of specific genera
    subset = [(box[0], [record for record in box[1] if record['genus'] == genus]) for box in boxesii]

    # begin data columns with genus information
    table = {}
    table['genus'] = [genus] * len(subset)
    
    # delineate columns
    incrementing = lambda time: time + timedelta(hours=hours)
    table = delineate(subset, incrementing, table)
    tables.append(table)
    
# calculate maximum height of upper bars
height = max([max(table['upper']) for table in tables])

# create columns objects
tables = [ColumnDataSource(table) for table in tables]

# gather bar and whisker parameters
gathering = [parameterize(table, genus) for table, genus in zip(tables, genera)]


# In[79]:


designate('setting the graph parameters')

# set the graph label bars
parameters = {}
parameters['title'] = 'Average Larvae Counts per {} hours for {} per Genus'.format(hours, country)
parameters['x_axis_label'] = 'Time (solar local)'
parameters['y_axis_label'] = 'Stacked Average Larvae Counts'

# set the x-axis format as a datetime
parameters['x_axis_type'] = 'datetime'

# set the size of the graph bars
parameters['plot_height'] = 400
parameters['plot_width'] = 800

# set height
parameters['y_range'] = (0, height + 10)
    
# Add annotations for the hover tool
annotations = []
annotations += [('Genus', '@genus')]
annotations += [('Number of Observations', '@size')]
annotations += [('Average Larvae Counts', '@average')]
annotations += [('Standard Deviation', '@deviation')]

# activate hover tool
hover = HoverTool(tooltips=annotations, formatters={'left': 'datetime', 'right': 'datetime'})


# In[80]:


designate('drawing the graph')

# initialize the graph and draw the bars
graph = figure(**parameters)
graph.xaxis.formatter = DatetimeTickFormatter(days="%H:%M", hours="%H:%M")
[graph.line(**bar) for bar, _ in gathering]
[graph.circle(**bar) for bar, _ in gathering]
[[graph.line(**whisker) for whisker in whiskers] for _, whiskers in gathering]
#[graph.add_layout(Whisker(**whisker)) for _, whisker in gathering]
graph.add_tools(hover)
graph.legend.click_policy='hide'

# show plot
output_notebook()
show(graph)


# Note: Local times have been inferred by reference to the location's longitude (called local solar time).  This may differ slightly from local time due to unusual time zone borders and daylight savings time adjustments.

# In[81]:


designate('navigation buttons')

# set navigation buttons
navigate()


# ### Plotting Larvae Counts by Water Source

# It may be interesting to know what water sources tend to have the highest larve counts.  Organize the data by water source instead of date.

# In[82]:


designate('organizing larvae counts by water source')

# gather all sources
sources = list(set([str(record['source']) for record in data]))

# collect data by source
collection = {source: [] for source in sources}
for record in data:
               
    # add to collection
    source = record['source']
    collection[source].append(record)
               
# sort by number of observations
collection = [item for item in collection.items()]
collection.sort(key=lambda item: len(item[1]), reverse=True)


# In[83]:


designate('gathering the datasets')

# make data table for each genus
tables = []
for genus in genera:
    
    # retain only the subset of specific genera
    subset = [(box[0], [record for record in box[1] if record['genus'] == genus]) for box in collection]

    # begin data columns with genus information
    table = {}
    table['genus'] = [genus] * len(subset)
    
    # delineate columns
    incrementing = lambda index: index + 1.0
    table = delineate(subset, incrementing, table, categorical=True)
    tables.append(table)
    
# calculate maximum height of upper bars
height = max([max(table['upper']) for table in tables])

# create columns objects
tables = [ColumnDataSource(table) for table in tables]

# gather bar and whisker parameters
gathering = [parameterize(table, genus) for table, genus in zip(tables, genera)]


# In[84]:


designate('setting the graph parameters')

# set the graph label parameters
parameters = {}
parameters['title'] = 'Average Larvae Counts per Water Source per Genus for {}'.format(country)
parameters['y_axis_label'] = 'Stacked Average Larvae Counts'
parameters['x_axis_label'] = 'Water Source (Number of Observations in Parentheses)'

# set the x-axis as souces
parameters['x_range'] = ['{} ({})'.format(source, len(observations)) for source, observations in collection]

# set the y range
parameters['y_range'] = (0, height + 10)
                     
# set the size of the graph parameters
parameters['plot_height'] = 600
parameters['plot_width'] = 800
parameters['min_border_left'] = 150

# Add annotations for the hover tool
annotations = []
annotations += [('Genus', '@genus')]
annotations += [('Number of Observations', '@size')]
annotations += [('Average Larvae Counts', '@average')]
annotations += [('Standard Deviation', '@deviation')]

# activate hover tool
hover = HoverTool(tooltips=annotations)


# In[85]:


designate('drawing the graph')

# initialize the graph and draw the bars
graph = figure(**parameters)
graph.xaxis.major_label_orientation = pi/4
graph.xaxis.major_label_text_font_size = '10pt'
[graph.line(**bar) for bar, _ in gathering]
[graph.circle(**bar) for bar, _ in gathering]
[[graph.line(**whisker) for whisker in whiskers] for _, whiskers in gathering]
#[graph.add_layout(Whisker(**whisker)) for _, whisker in gathering]
graph.add_tools(hover)
graph.legend.click_policy='hide'

# show plot
output_notebook()
show(graph)


# In[86]:


designate('navigation buttons')

# set navigation buttons
navigate()


# ### Plotting Larvae Counts by Source Type

# Alternatively, the water sources could be grouped by water source type.

# In[87]:


designate('organizing larvae counts by water source type')

# gather all sources
types = list(set([str(record['type']) for record in data]))

# collect data by source
collectionii = {source: [] for source in types}
for record in data:
               
    # add to collection
    source = record['type']
    collectionii[source].append(record)
               
# sort by number of observations
collectionii = [item for item in collectionii.items()]
collectionii.sort(key=lambda item: len(item[1]), reverse=True)


# In[88]:


designate('gathering the datasets')

# make data table for each genus
tables = []
for genus in genera:
    
    # retain only the subset of specific genera
    subset = [(box[0], [record for record in box[1] if record['genus'] == genus]) for box in collectionii]

    # begin data columns with genus information
    table = {}
    table['genus'] = [genus] * len(subset)
    
    # delineate columns
    incrementing = lambda index: index + 1.0
    table = delineate(subset, incrementing, table, categorical=True)
    tables.append(table)
    
# calculate maximum height of upper bars
height = max([max(table['upper']) for table in tables])

# create columns objects
tables = [ColumnDataSource(table) for table in tables]

# gather bar and whisker parameters
gathering = [parameterize(table, genus) for table, genus in zip(tables, genera)]


# In[89]:


designate('setting the graph parameters')

# set the graph label parameters
parameters = {}
parameters['title'] = 'Average Larvae Counts per Water Source Type per Genus for {}'.format(country)
parameters['y_axis_label'] = 'Stacked Average Larvae Counts'
parameters['x_axis_label'] = 'Water Source Type'

# set the x-axis as souces
parameters['x_range'] = ['{} ({})'.format(source, len(observations)) for source, observations in collectionii]

# set y-range
parameters['y_range'] = (0, height + 10)

# set the size of the graph parameters
parameters['plot_height'] = 600
parameters['plot_width'] = 800
parameters['min_border_left'] = 150

# Add annotations for the hover tool
annotations = []
annotations += [('Genus', '@genus')]
annotations += [('Number of Observations', '@size')]
annotations += [('Average Larvae Counts', '@average')]
annotations += [('Standard Deviation', '@deviation')]

# activate hover tool
hover = HoverTool(tooltips=annotations)


# In[90]:


designate('drawing the graph')

# initialize the graph and draw the bars
graph = figure(**parameters)
graph.xaxis.major_label_orientation = pi/4
graph.xaxis.major_label_text_font_size = '10pt'
[graph.line(**bar) for bar, _ in gathering]
[graph.circle(**bar) for bar, _ in gathering]
[[graph.line(**whisker) for whisker in whiskers] for _, whiskers in gathering]
#[graph.add_layout(Whisker(**whisker)) for _, whisker in gathering]
graph.add_tools(hover)
graph.legend.click_policy='hide'

# show plot
output_notebook()
show(graph)


# In[91]:


designate('navigation buttons')

# set navigation buttons
navigate()


# ### Plotting Sampling Locations on a Map

# The following plots all sample locations on a geographical map.  This may take a little time depending on the size of the data set.  The +/- are zoom controls, and the box below that enters full screen mode.  Clicking on a sample will bring up a sample summary as well as all associated photos.

# In[92]:


designate('fetching closest record subroutine')

def fetch(latitude, longitude):
    """Fetch the closest record based on latitude and longitude
    
    Arguments:
        latitude: float, latitude coordinate
        longitude: float, longitude coordinate
        
    Returns:
        dict, the closest record
    """
    
    # calculate the distances to all records
    distances = []
    for index, record in enumerate(data):
        
        # calculate squared distance
        distance = (latitude - record['latitude']) ** 2 + (longitude - record['longitude']) ** 2
        
        # append to list
        distances.append((index, distance))
        
    # sort by distance and choose closest
    distances.sort(key=lambda pair: pair[1])
    closest = distances[0][0]
    
    return closest


# In[93]:


designate('exhibiting photos on map subroutine')

def exhibit(coordinates, chart, flags, thumbs=True):
    """Exhibit a record's photos on the map and add to photos list.
    
    Arguments:
        coordinates: (float, float) tuple, the latitude and longitude
        chart: ipyleaflets Map object
        photos: list of photos so far
        thumbs=True: boolean, use thumbnails?
    
    Returns:
        None
    """

    # fetch the closest record and add to flags
    index = fetch(*coordinates)
    flags.append(index)
    record = data[index]
    
    # gather all photo urls depending on thumbnail setting
    photos = record['thumbs'] if thumbs else record['originals']
    
    # send to output
    layout = {'border': '1px solid blue', 'transparency': '50%', 'overflow_y': 'scroll', 'height': '200px'}
    exhibitor = Output(layout=layout)
    with exhibitor:
            
        # remove last set of photos
        chart.controls = chart.controls[:5]
        
        # for each url
        for photo in photos:

            # paste the photo on the map
            display(Image(photo, width=200, unconfined=True))

    # add to map
    control = WidgetControl(widget=exhibitor, position='bottomleft')
    chart.add_control(control)
    
    return None


# In[94]:


designate('initializing map')

# print status
print('constructing map...')

# get central latitude
latitudes = [record['latitude'] for record in data]
latitude = (max(latitudes) + min(latitudes)) / 2

# get central longitude
longitudes = [record['longitude'] for record in data]
longitude = (max(longitudes) + min(longitudes)) / 2

# set up map with topographical basemap zoomed in on center
chart = Map(basemap=basemaps.Esri.WorldTopoMap, center=(latitude, longitude), zoom=5, double_click_zoom=False)

# organize samples biggest first so smaller samples will be on top
samples = [record for record in data]
samples.sort(key=lambda record: record['score'], reverse=True)

# begin flags list and record list
flags = []


# In[95]:


designate('creating sample markers')

# create marker layer
markers = []
for record in samples:
    
    # unpack record
    latitude = record['latitude']
    longitude = record['longitude']
    larvae = record['larvae']
    genus = record['genus']
    date = record['date']
    score = record['score']
    
    # add sample marker
    circle = CircleMarker()
    circle.location = (latitude, longitude)
    
    # set color attributes
    circle.weight = 1
    circle.color = 'black'
    circle.fill = True
    circle.opacity = 0.8
    circle.fill_color = indicators[genus]
    
    # set radius as a function of z-score (relative larvae count)
    circle.radius = int(sqrt(score) + 1) * 5
    
    # exhibit photo on map for click
    exhibiting = lambda **event: exhibit(event['coordinates'], chart, flags, thumbs=True)
    circle.on_click(exhibiting)
    
    # annotate marker with popup label
    formats = (int(larvae), genus, date, latitude, longitude)
    message = 'Larvae: {}, Genus: {}, Date: {}, Latitude: {}, Longitude: {}'.format(*formats)
    circle.popup = HTML(message)
    
    # add to markers layer
    markers.append(circle)


# In[96]:


designate('constructing map')

# add marker layer
group = LayerGroup(layers=markers)
chart.add_layer(group)

# add full screen control
chart.add_control(FullScreenControl())

# add genus legend
labels = [Label(value = r'\(\color{' + 'black' +'} {' + 'Genera:'  + '}\)')]
labels += [Label(value = r'\(\color{' + indicators[genus] +'} {' + genus  + '}\)') for genus in genera]
legend = VBox(labels)

# send to output
cartographer = Output(layout={'border': '1px solid blue', 'transparency': '50%'})
with cartographer:
    
    # display legent
    display(legend)

# add to map
control = WidgetControl(widget=cartographer, position='topright')
chart.add_control(control)

# add full screen button and map scale
chart.add_control(ScaleControl(position='topleft'))

# display the map
chart


# Note: Clicking on a sample will reveal a sample summary and will display the associated photos in a scrollable box.  

# In[97]:


designate('navigation buttons')

# set navigation buttons
navigate()


# ### Viewing Photos

# Clicking on the View button below will view the first fifteen photos in a scrollable window.  Clicking on the left side of the window will expand the view.  The export option in the next section may be used to download the full set of photos.

# In[98]:


designate('summarizing record subroutine')

# print a record summary
def summarize(record):
    """Summarize the record.
    
    Arguments:
        record: dict
        
    Returns:
        None
    """
    
    # print record summary
    print(' ')
    print(' ')
    print('date: {}'.format(record['date']))
    print('latitude: {}'.format(record['latitude']))
    print('longitude: {}'.format(record['longitude']))
    print('larvae count: {}'.format(record['larvae']))
    print('genus: {}'.format(record['genus']))
    
    return None


# In[99]:


designate('segregating records subroutine')

# function to segregate records
def segregate():
    """Segregate the records into those flagged and those remaining.
    
    Arguments:
        flags: list of ints, the flagged records
        
    Returns:
        list of ints
    """
    
    # get chosen records and remaining records, linking them to their indices
    chosen = [number for index, number in enumerate(flags) if flags.index(number) == index]
    remainder = [number for number, _ in enumerate(data) if number not in chosen]
    ids = chosen + remainder
    
    return ids


# In[100]:


designate('viewing photos')

# add button click command
scrapbook = Output()
button = Button(description='View')
display(button, scrapbook)

# function to export to csv
def view(_):
    """Retrieve all flagged photos and print them to screen.
    
    Arguments:
        None
        
    Returns:
        None
    """
    
    # get ids list
    ids = segregate()
    
    # count total number of photos
    lengths = [len(data[number]['originals']) for number in ids]
    total = sum(lengths)
    
    # find cutoff point
    quantity = 0
    cutoff = 0
    while quantity < 15 and cutoff < len(ids):
        
        # increment index
        cutoff += 1
        quantity = sum(lengths[:cutoff])
        
    # print
    print('displaying {} photos of {} total photos in {} records'.format(quantity, total, len(ids)))
    
    # go through top fifty records
    for index in ids[:cutoff]:
        
        # check through all photos
        record = data[index]
        summarize(record)
        for original, photo in zip(record['originals'], record['photos']):

            # display photo
            print(' ')
            print('url: {}'.format(original))
            print('file: {}'.format(photo))
            display(Image(original, width=800))

    return None

# add click 
button.on_click(view)


# In[101]:


designate('navigation buttons')

# set navigation buttons
navigate()


# ### Exporting Photos

# The full set of flagged photos may be exported to a zip file.  This may take a few minutes.

# In[102]:


designate('exporting photos to zip file')

# function to export to zip file
def archive(_):
    """Archive all photos into a zip drive.
    
    Arguments:
        None
        
    Returns:
        None
    """

    # begin zip file and go through records
    album = ZipFile('mosquitoes_photos.zip', 'w')
    print('{} records:'.format(len(data)), end='')
    for record in data:
        
        # upload photos
        print('.', end='')
        for photo, original in zip(record['photos'], record['originals']):

            # get original and pass to file
            request = requests.get(original, stream=True)
            with open(photo, 'wb') as pointer: [pointer.write(chunk) for chunk in request]
            album.write(photo)
            os.remove(photo)
                    
    # close zip file and make link
    print('')
    album.close()
    link = FileLink('mosquitoes_photos.zip')
    archiver.append_display_data(link)

    return None

# create push button linked to output
archiver = Output()
button = Button(description='Zip')
display(button, archiver)
button.on_click(archive)


# In[103]:


designate('navigation buttons')

# set navigation buttons
navigate()


# ### Viewing the Records

# You may view the records searched for above here.

# In[104]:


designate('viewing the searched records')

# create push button linked to output
window = Output()
button = Button(description='View')
display(button, window)

# view records
def view(_):
    """View all records.
    
    Arguments:
        None
        
    Returns:
        None
    """

    # get chosen records and remaining records, linking them to their indices
    chosen = [number for index, number in enumerate(flags) if flags.index(number) == index]
    remainder = [number for number, _ in enumerate(data) if number not in chosen]

    # list records
    for index in chosen + remainder:

        # print to screen
        print('\nrecord {} of {}:'.format(index, len(data)))
        features = [item for item in data[index].items()]
        features.sort(key=lambda feature: feature[0])
        features.sort(key=lambda feature: len(feature[0]))
        for field, datum in features:

            # print each field
            print('{}: {}'.format(field, datum))
        
# add button click command
button.on_click(view)


# In[105]:


designate('navigation buttons')

# set navigation buttons
navigate()


# ### Viewing the Data Table

# The full data set may be viewed as a table (organized by date).

# In[106]:


designate('viewing the data table')

# set options for viewing full columns
pandas.set_option("display.max_rows", None)
pandas.set_option("display.max_columns", None)

# create dataframe from data
panda = pandas.DataFrame(data)

# reorder columns according to length of field
columns = [field for field in panda.columns]
columns.sort(key=lambda field: field)
columns.sort(key=lambda field: len(field))
panda = panda[columns]

# display in output
panda


# In[107]:


designate('navigation buttons')

# set navigation buttons
navigate()


# ### Exporting Data to CSV

# It may be desirable to export the data to a csv file.  Click the Export button to export the data.  You will get a link to download the csv file.

# In[108]:


designate('exporting data to csv')

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

    # create dataframe from data
    panda = pandas.DataFrame(data)
    
    # reorder columns according to length of field
    columns = [field for field in panda.columns]
    columns.sort(key=lambda field: field)
    columns.sort(key=lambda field: len(field))
    panda = panda[columns]
    
    #write dataframe to file
    panda.to_csv("mosquitoes.csv")  
    
    # make link
    link = FileLink('mosquitoes.csv')
    
    # add to output
    exporter.append_display_data(link)

    return None

# add button click command
button.on_click(export)


# In[109]:


designate('navigation buttons')

# set navigation buttons
navigate()


# ### Exporting to PDF

# The notebook in static form may exported as a pdf by clicking the Generate button below.  Landscape mode tends to better preserve the formatting.

# In[110]:


designate('generating pdf')

# function to generate pdf
def generate(_):
    """Export the data into a csv file.

    Arguments:
        None

    Returns:
        None
    """

    # make pdf
    display(Javascript('window.print()'))
    
    return None

# add button click command
button = Button(description='Generate')
display(button)
button.on_click(generate)


# In[111]:


designate('navigation buttons')

# set navigation buttons
navigate()


# ### Thank You!

# Please feel free to direct inquires or comments to Matthew Bandel at matthew.bandel@ssaihg.com

# In[112]:


designate('button for navigation')

# button for navigation
navigate()

