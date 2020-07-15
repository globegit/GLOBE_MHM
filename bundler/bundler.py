#!/usr/bin/env python
# coding: utf-8

# # Mosquito Bundler

# The purpose of this notebook is to allow the user to select a data range and polygon on a map to retrieve all the data from the mosquito bundle protocols for the selection.

# Select "Restart & Run All" from the "Kernel" menu and confirm "Restart and Run All Cells."

# Open any *** settings -----> sections with the ^ button in the tool bar above.

# ### Importing Modules and Tools

# The following code imports needed modules and subroutines.

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


designate('importing Python modules')

# import os and sys modules for system controls
import os
import sys

# import requests and json modules for making API requests
import requests
import json

# set runtime warnings to ignore
import warnings

# import datetime module for manipulating date and time formats
from datetime import datetime

# import iPython for javascript based notebook controls
from IPython.display import Javascript, display, FileLink

# import ipywidgets for additional widgets
from ipywidgets import Button

# import ipyleaflet for the map
from ipyleaflet import Map, DrawControl, basemaps, GeoJSON

# import geopy to get country from coordinates
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderQueryError

# import pandas to create csv files and display tables
import pandas

# import zipfile to create zip files
from zipfile import ZipFile


# In[3]:


designate('defining global variables')

# disable runtime warnings
warnings.filterwarnings('ignore')

# set pandas optinos
pandas.set_option("display.max_rows", None)
pandas.set_option("display.max_columns", None)

# default link
link = None


# In[4]:


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
    with open('bundler.ipynb', 'r', encoding='utf-8') as pointer:
        
        # and read its contents
        contents = json.loads(pointer.read())

    # get all cells
    cells = contents['cells']

    return cells


# In[5]:


designate('looking for particular cell', 'navigation')

# function to look for cells with a particular text snippet
def look(text):
    """Look for a particular text amongst the cells.
    
    Arguments:
        text: str, the text to search for
        
    Returns:
        list of int, the cell indices.
    """
    
    # get cells
    cells = scan()
    
    # search for cells 
    indices = []
    for index, cell in enumerate(cells):
        
        # search for text in source
        if any([text in line for line in cell['source']]):
            
            # but disregard if text is in quotes
            if all(["'{}'".format(text) not in line for line in cell['source']]):
            
                # add to list
                indices.append(index)
            
    return indices


# In[6]:


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


# In[7]:


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


# In[8]:


designate('ridding polygons', 'mapping')

# function to clear old polygons with drawing of newone
def rid(self, action, geo_json):
    """Rid the map of previous polygons, keeping only the one drawn.
    
    Arguments:
        self: self
        action: action
        geo_json: dict
        
    Returns:
        None
    """
    
    # clear polygons and rectanges from draw control
    chart.controls[-1].clear_polygons()
    chart.controls[-1].clear_rectangles()
    
    # remove all previous layers
    chart.layers = chart.layers[:1]
    
    # add polygon to chart
    chart.add_layer(GeoJSON(data=geo_json))
        
    return None


# In[9]:


designate('forcing rectanges', 'mapping')

# force a polygon on the map
def force(south, north, west, east):
    """Force a rectangle onto the map.
    
    Arguments:
        south: float
        north: float
        west: float
        east: float
        
    Returns:
        None
    """
    
    # check for values
    if all([cardinal is not None for cardinal in [south, north, west, east]]):
        
        # construct coordinates
        coordinates = [[[west, south], [west, north], [east, north], [east, south], [west, south]]]
    
        # construct geo_json
        geo_json = {'type': 'Feature'}
        geo_json['properties'] = {'style': chart.controls[-1].rectangle['shapeOptions']}
        geo_json['geometry'] = {'type': 'Polygon', 'coordinates': coordinates}
                
        # add rectangle to chart
        chart.add_layer(GeoJSON(data=geo_json))
    
    return None


# In[10]:


designate('locating country from geocoordinates', 'mapping')

# locate function
def locate(longitude, latitude):
    """Locate the country from a geocoordinate pair:
    
    Arguments:
        latitude: float
        longitude: float
        
    Returns:
        str
    """

    # try to get country
    country = ''
    try:
        
        # get country name
        locator = Nominatim(user_agent="GLOBE")
        query = ', '.join([str(latitude), str(longitude)])
        location = locator.reverse(query, language='english')
        country = location.raw['address']['country']
        country = '{}'.format(country.encode('ascii', errors='ignore').decode())
        
    # otherwise
    except (KeyError, GeocoderQueryError):
        
        # pass
        pass
    
    return country


# In[11]:


designate('retrieving the surrounding polygon', 'mapping')

# getting polygon from map routine
def surround(chart):
    """Get the polygon surrounding the area on the map
    
    Arguments:
        chart: ipyleaflet chart
        
    Returns:
        list of points, polygon
    """

    # try to retrieve the polygon
    try:

        # get polygon
        polygon = chart.layers[1].data['geometry']['coordinates'][0]

    # unless it is not available
    except IndexError:
        
        # set to message
        polygon = ['no polygon drawn yet, please draw one on the map']
        
    return polygon


# In[12]:


designate('record flattening', 'processing')

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


# In[13]:


designate('calling the api subroutine', 'api')

# call the api with protocol and country code
def call(protocol, beginning, ending, polygon):
    """Call the api:
    
    Arguments:
        protocol: str, the protocol
        beginning: str, the beginning date
        ending: str, the ending date
        polygon: list of points
        
    Returns:
        list of dicts, the records
    """
    
    # assemble the url for the API call 
    url = 'https://api.globe.gov/search/v1/measurement/protocol/measureddate/polygon/geojson/'
    url += '?protocols=' + protocol
    url += '&startdate=' + beginning 
    url += '&enddate=' + ending
    
    # begin with first point, and continue
    coordinates = '%5B%5B' + str(polygon[0][0]) + '%2C%20' + str(polygon[0][1])
    for point in polygon[1:]:
    
        # add points
        coordinates += '%5D%2C%20%5B' + str(point[0]) + '%2C%20' + str(point[1])
        
    # end with cap
    coordinates += '%5D%5D'
    url += '&coordinates=' + coordinates

    # geojson parameter toggles between formats
    url += '&geojson=FALSE'
    
    # sample parameter returns small sample set if true
    url += '&sample=FALSE'
    
    # make the API call and return the raw results
    request = requests.get(url)
    raw = json.loads(request.text)
    
    return raw


# In[14]:


designate('collecting results from all protocols', 'api')

# collecting results
def collect(polygon):
    """Collect all data from within dates and polygon on map.
    
    Arguments:
        polygon: list of points
        
    Returns:
        list of panda frames
    """
    
    # check polygon
    assert polygon[0] != str(polygon[0]), 'no polygon drawn on map yet'
    
    # set protocol list
    mosquitoes = 'mosquito_habitat_mapper'
    protocols = [mosquitoes] + secondaries
    
    # begin zipfile
    date = str(datetime.now().date()).replace('-', '')
    now = str(int(datetime.now().timestamp()))
    bundle = 'mosquitoes_bundle_' + date + '_' + now +'.zip'
    album = ZipFile(bundle, 'w')

    # collect results
    for protocol in protocols:

        # make call
        print('\nmaking request from {}...'.format(protocol))
        raw = call(protocol, beginning, ending, polygon)
        result = raw['results']
        length = len(result)
        message = '{} results returned from {}.\n'.format(length, protocol)
        summary.append(message)
        print(message)

        # flatten all records
        records = [flatten(record) for record in result]
        panda = pandas.DataFrame(records)
        
        # write dataframe to file
        name = protocol + '_' + date + '_' + now + '.csv'
        panda.to_csv(name)  
        album.write(name)
        
        # delete file
        os.remove(name)
        
        # display sample
        display(panda.head(5))
        
    # create summary
    path = 'summary_' + date + '_' + now + '.txt'
    with open(path, 'w') as pointer:
        
        # write summary file
        pointer.writelines(summary)
        
    # add to album and remove from main directory
    album.write(path)
    os.remove(path)
        
    # make link
    link = FileLink(bundle)

    return link


# In[108]:


designate('import status')

# print status
print('modules imported.')


# ### MyBinder Link

# This notebook is available at the following link hosted by MyBinder:

# https://mybinder.org/v2/git/https%3A%2F%2Fmattbandel%40bitbucket.org%2Fmattbandel%2Fglobe-mosquitoes-bundler.git/master?filepath=bundler.ipynb

# ### Mosquito Bundle Protocols

# Data from the following GLOBE protocols are considered part of the mosquitoes bundle.

# In[16]:


designate('table of mosquito bundle protocols')

# get from secondaries file
with open('protocols.txt', 'r') as pointer:
    
    # get all secondaries
    secondaries = [protocol.strip('\n') for protocol in pointer.readlines()]
    secondaries = [protocol for protocol in secondaries if 'X' in protocol]
    secondaries = [protocol.strip('X').strip() for protocol in secondaries]

# print as list
print('{} mosquito bundle protocols:'.format(len(secondaries)))
secondaries


# ### Setting the Geographic Area

# Set latitude and longitude boundaries for the study area and click Apply.

# In[111]:


designate('setting the latitude and longitude', 'settings')

# set the latitude and longitude boundaries
south = None
north = None
west = None
east = None


# In[112]:


designate('button to apply rectangle')

# function to retrieve all data
def retrieve(_):
    """Retrieve the data from the api.
    
    Arguments:
        None
        
    Returns:
        None
    """

    # refresh cells
    execute('### Setting the Geographic Area', '### Setting the Date Range')
    
    return None

# create button
button = Button(description="Apply")
button.on_click(retrieve)
display(button)


# Or use the pentagon tool or rectangle tool to draw a polygon on the map.

# In[113]:


designate('constructing map')

# print status
print('constructing map...')

# set up map with topographical basemap
chart = Map(basemap=basemaps.Esri.WorldTopoMap, center=(0, 0), zoom=3)

# initiate draw control
control = DrawControl()

# specify polygon
control.polygon = {'shapeOptions': {'fillColor': '#00ff00', 'color': '#ffffff', 'fillOpacity': 0.1}}
control.polygon['shapeOptions'].update({'color': '#ffffff', 'weight': 4, 'opacity': 0.5, 'stroke': True})
control.polygon['drawError'] = {'color': '#dd253b', 'message': 'Oops!'}
control.polygon['allowIntersection'] = False

# specify rectange marker
control.rectangle = {'shapeOptions': {'fillColor': '#00ff00', 'color': '#ffffff', 'fillOpacity': 0.1}}
control.rectangle['shapeOptions'].update({'color': '#ffffff', 'weight': 4, 'opacity': 0.5, 'stroke': True})

# remove default polyline and circlemarker
control.polyline = {}
control.circlemarker = {}

# specify clear function
control.on_draw(rid)

# add draw control
chart.add_control(control)

# force a rectange onto the map
force(south, north, west, east)

# display chart
chart


# ### Setting the Date Range

# Set the date range below.  Leaving the beginning date blank will default to Earth Day 1995.  Leaving the ending date blank will default to today's date.

# In[103]:


designate('setting date range', 'settings')

# set beginning and ending dates in 'YYYY-mm-dd' format, None for ending date defaults to now
beginning = '2019-01-01'
ending = '2020-01-01'


# ### Retrieving the Data

# Press the Retrieve button believe to retrieve the data.  A link will appear to a zip file below.

# In[104]:


designate('resolving default dates')

# default beginning to 1995 and ending to current date if unspecified
beginning = beginning or '1995-04-22'
ending = ending or str(datetime.now().date())

# begin summary
today = datetime.now().date()
clock = datetime.now().time().replace(microsecond=0)
message = 'Summary of Mosquito Bundle Query at {} on {}:\n'.format(clock, today)
summary = [message]
print(message)

# print date range
message = 'date range: {} to {}\n'.format(beginning, ending)
summary.append(message)
print(message)

# retrieve polygon from map
message = 'polygon:\n'
polygon = surround(chart)
summary.append(message)
print(message)
for pair in polygon:

    # print and add to summary
    country = locate(pair[0], pair[1])
    message = '{}: {}'.format(pair, country)
    summary.append(message + '\n')
    print(message)


# In[105]:


designate('button to retrieve data')

# function to retrieve all data
def retrieve(_):
    """Retrieve the data from the api.
    
    Arguments:
        None
        
    Returns:
        None
    """

    # refresh cells
    execute('### Setting the Date Range', '### Thanks!')
    
    return None

# create button
button = Button(description="Retrieve")
button.on_click(retrieve)
display(button)


# In[106]:


designate('collecting data')

# collect data
link = collect(polygon)


# In[107]:


designate('displaying link')

# attemmpt to display link
if link:
    
    # display last link
    display(link)


# ### Thanks!

# Please feel free to direct questions or feedback to Matthew Bandel at matthew.bandel@ssaihq.com
