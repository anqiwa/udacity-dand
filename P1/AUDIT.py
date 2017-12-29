import xml.etree.cElementTree as ET
import re
from collections import defaultdict 
import pprint

osm_file = open(r'C:\Users\PC2016\Desktop\P3\honolulu_hawaii.osm','rb')
incorrect_phone = defaultdict(set)
incorrect_postcode = defaultdict(set)
incorrect_state = defaultdict(int)
incorrect_street = defaultdict(set)

def audit_phone_number():
    phone_re = re.compile(r'^\d$')
    for event, elem in ET.iterparse(osm_file, events=('start',)):
        if elem.tag == 'way' or elem.tag == 'node':
            for tag in elem.iter('tag'):
                if 'phone' in tag.attrib['k']:
                    if not phone_re.search(tag.attrib['v']) or len(tag.attrib['v'] == 11):
                        incorrect_phone[tag.attrib['k']].add((tag.attrib['v']))
    return incorrect_phone

def audit_postcode_type():
    osm_file.seek(0)
    for event, elem in ET.iterparse(osm_file, events=('start',)):
        if elem.tag == 'node' or elem.tag == 'tag':
            for tag in elem.iter('tag'):
                if tag.attrib['k']=='addr:postcode' or tag.attrib['k']=='postal_code':
                    if len(tag.attrib['v']) != 5:
                        incorrect_postcode[tag.attrib['k']].add(tag.attrib['v'])
    return incorrect_postcode

def audit_state_name():
    osm_file.seek(0)
    for event, elem in ET.iterparse(osm_file, events=('start',)):
        if elem.tag == 'way' or elem.tag == 'node':
            for tag in elem.iter('tag'):
                if tag.attrib['k']=='addr:state':
                     if tag.attrib['v'] != 'HI':
                         incorrect_state[tag.attrib['v']] += 1
    return incorrect_state

def audit_street_type():
    osm_file.seek(0)
    expected = ['Street','Avenue','Boulevard','Drive','Court','Place','Parkway',
            'Highway','Circle','Lane','Road','Loop','Way','Walk','Square',
            'Trail','Commons','Place','Terrace','Promenade']
    for event, elem in ET.iterparse(osm_file, events=('start',)):
        if elem.tag == 'way' or elem.tag == 'node':
            for tag in elem.iter('tag'):
                if tag.attrib['k']=='addr:street':
                    street_split = tag.attrib['v'].split(' ')
                    if street_split[-1] not in expected:
                        incorrect_street[street_split[-1]].add(tag.attrib['v'])
    return incorrect_street

print("\nIncorrect Phone Numbers:")
pprint.pprint(audit_phone_number())
print("\nIncorrect Postcodes: ")
pprint.pprint(audit_postcode_type())
print("\nIncorrect State Names:")
pprint.pprint(audit_state_name())
print("\nIncorrect Streets:")
pprint.pprint(audit_street_type())
                        
    
