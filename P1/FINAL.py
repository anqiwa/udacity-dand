import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

osm_path = open(r'E:\P3\sample.osm','rb')

nodes_path = 'nodes.csv'
node_tags_path = 'node_tags.csv'
ways_path = 'ways.csv'
way_nodes_path = 'way_nodes.csv'
way_tags_path = 'way_tags.csv'

LOWER_COLON = re.compile(r'^([a-z]|_)+:([a-z]|_)+')
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

# Make sure the fields order in the csvs matches the column order in the sql table schema
NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']

def shape_tag(el, tag): 
    tag = {
        'id'   : el.attrib['id'],
        'key'  : tag.attrib['k'],
        'value': tag.attrib['v'],
        'type' : 'regular'
    }
    
    if LOWER_COLON.match(tag['key']):
        tag['type'], _, tag['key'] = tag['key'].partition(':')
        
    return tag

def shape_way_node(el, i, nd):
    return {
        'id'       : el.attrib['id'],
        'node_id'  : nd.attrib['ref'],
        'position' : i
    }

def shape_element(el, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS,
                  problem_chars=PROBLEMCHARS, default_tag_type='regular'):
    """Clean and shape node or way XML element to Python dict"""

    node_attribs = {}
    way_attribs = {}
    way_nodes = []
    tags = []  # Handle secondary tags the same way for both node and way elements

    # YOUR CODE HERE
    tags = [shape_tag(el, t) for t in el.iter('tag')]

    if el.tag == 'node':
        node_attribs = {f: el.attrib[f] for f in node_attr_fields}
        
        return {'node': node_attribs, 'node_tags': tags}
        
    elif el.tag == 'way':
        way_attribs = {f: el.attrib[f] for f in way_attr_fields}
        
        way_nodes = [shape_way_node(el, i, nd) 
                     for i, nd 
                     in enumerate(el.iter('nd'))]
   
        return {'way': way_attribs, 'way_nodes': way_nodes, 'way_tags': tags}

def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag"""

    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()
