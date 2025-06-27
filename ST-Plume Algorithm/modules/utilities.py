# Handles input and output functionality
import logging
import pandas as pd
import geojson
import csv
import json
import os
import shapefile
import xml.etree.ElementTree as ET

######################################################################################################################################
#  Data Load and Validation Functions
#
# ####################################################################################################################################

def load_csv(filepath, required_columns):
    """Load a CSV file and validate required columns."""
    df = pd.read_csv(filepath, usecols=required_columns)  # ✅ Load only necessary columns 
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")
    return df

def load_config(filepath):
    """Load configuration file."""
    with open(filepath, "r") as file:
        return json.load(file)

def setup_logging(log_file, log_level):
    """Set up logging."""
    logging.basicConfig(
        filename=log_file,
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def validate_config(config):
    """Validate the configuration file."""
    required_keys = ["input_files", "output_directory", "parameters"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
        
def validate_output(output_format):
    valid_formats = ["csv"] # Later if required , "geojson", "shapefile", "xml"
    if output_format not in valid_formats:
        raise ValueError(f"Unsupported output format: {output_format}. Supported formats: {valid_formats}")

######################################################################################################################################
#  Data Export Functions
#
# ####################################################################################################################################

''''''
def write_csv(filepath, data, headers):
    """Write data to a CSV file."""
    df = pd.DataFrame(data, columns=headers)
    df.to_csv(filepath, index=False)

'''
Plume and Event Export

'''
def exportPlumesEvents(plumes, greater, epsilons, output_format, output_directory):
    if output_format == "csv":
        selected = extractPlumesCSV(plumes, greater)
        extractEventsCSV(selected, epsilons, output_directory)
    '''
    elif output_format == "geojson":
        write_geojson(output_file, plumes, events_dict)
    elif output_format == "shapefile":
        write_shapefile(str(output_file).replace(".shapefile", ""), plumes, events_dict)
    elif output_format == "xml":
        write_xml(output_file, plumes, events_dict)
    '''

def extractPlumesCSV(plumes, greater):
    selectedPlumes = {}
    with open('Output/Plumes.csv', 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        headers = ['PlumeId', 'TemporalEpsilon', 'Parent', 'Children', 'EventCount'] # Can add plume coordinates 'PointCoordinates'
        writer.writerow(headers)

        for plume_id, plume in plumes.items():
            if plume.eventcount >= greater:

                
                #coordinates = [[event.x, event.y, event.z] for event in plume.eventlist]
                row = [plume.id, plume.temporal, plume.parent, plume.children, plume.eventcount] #coordinates
                writer.writerow(row)
                selectedPlumes[plume_id] = plume
    return selectedPlumes

def extractEventsCSV(plumes, epsilons, output_dir):
    event_files = {}  # Dictionary to store file handles for each epsilon

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open a CSV writer for each epsilon and store them in a dictionary
    for eps in epsilons:
        filename = os.path.join(output_dir, f'Events{eps}.csv')
        file = open(filename, 'w', newline='')
        writer = csv.writer(file)
        
        # Write headers
        headers = ['Event Id', 'Event Date', 'X', 'Y', 'Z', 'Zoneid', 'PlumeID', 'SuperPlume']
        writer.writerow(headers)
        
        # Store writer in the dictionary
        event_files[eps] = (file, writer)

    # Iterate through plumes and sort events into corresponding epsilon files
    for plume_id, plume in plumes.items():
        eps = plume.temporal  # Determine which epsilon this plume belongs to
        if eps not in event_files:
            continue  # Skip if the epsilon was not predefined
        
        _, writer = event_files[eps]

        parent_id = None
        if plume.parent:  
            parent_tuple = next(iter(plume.parent))  # Extract the first (parent_id, event_count) tuple
            parent_id = parent_tuple[0]
        # Process events in the plume

        for event in plume.eventlist:
            row = [event.id, event.eventdate, event.x, event.y, event.z, event.zoneid, plume.id, parent_id]
            writer.writerow(row)

    # Close all CSV files
    for file, _ in event_files.values():
        file.close()


######################################################################################################################################
#  Debugging Functions
#
# ####################################################################################################################################
def export_sample_zone(zones, zone_id=None, output_folder="Output", output_file="sample_zone.json"):
    """Exports a specific Zone object to JSON for debugging. If no ID is provided, exports the first Zone."""
    if not zones:
        print("No zones available to export.")
        return
    
    if zone_id and zone_id in zones:
        sample_zone = zones[zone_id]
    else:
        sample_zone = next(iter(zones.values()))  # Get the first Zone object as default
    
    zone_data = {
        "id": sample_zone.id,
        "zidx": sample_zone.zidx,
        "eventcount": sample_zone.eventcount,
        "events": [
            {
                "id": event.id,
                "eventdate": str(event.eventdate),
                "x": event.x,
                "y": event.y,
                "zoneid": event.zoneid
            }
            for event in sample_zone.eventset
        ],
        "neighbors": [neighbor.id for neighbor in sample_zone.neighbors]
    }
    
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_file)
    
    with open(output_path, "w") as outfile:
        json.dump(zone_data, outfile, indent=4)
    
    print(f"Zone object (ID: {sample_zone.id}) exported to {output_path}")



######################################################################################################################################
#  Extra Export Functions for later
#
# ####################################################################################################################################
'''Can be Implemented Later if necessary
def write_geojson(filepath, plumes, events_dict):
    """Write plumes to a GeoJSON file."""
    features = []
    for plume in plumes:
        coordinates = [[event.x, event.y] for event in get_event_objects(plume.eventlist, events_dict)]
        feature = geojson.Feature(
            geometry=geojson.MultiPoint(coordinates),
            properties={
                "PlumeID": plume.id,
                "EventCount": plume.eventcount,
                "StartDate": plume.start_date,
                "EndDate": plume.end_date
            }
        )
        features.append(feature)
    with open(filepath, "w") as outfile:
        geojson.dump(geojson.FeatureCollection(features), outfile)

def write_shapefile(filepath, plumes, events_dict):
    """Write plumes to a Shapefile."""
    shp_writer = shapefile.Writer(filepath)
    shp_writer.field("PlumeID", "N")
    shp_writer.field("EventCount", "N")
    shp_writer.field("StartDate", "C")
    shp_writer.field("EndDate", "C")
    for plume in plumes:
        for event in get_event_objects(plume.eventlist, events_dict):
            shp_writer.point(event.x, event.y)
            shp_writer.record(plume.id, plume.eventcount, plume.start_date, plume.end_date)
    shp_writer.close()

def write_xml(filepath, plumes, events_dict):
    """Write plumes to an XML file."""
    root = ET.Element("Plumes")
    for plume in plumes:
        plume_elem = ET.SubElement(root, "Plume", id=str(plume.id))
        ET.SubElement(plume_elem, "EventCount").text = str(plume.eventcount)
        ET.SubElement(plume_elem, "StartDate").text = str(plume.start_date)
        ET.SubElement(plume_elem, "EndDate").text = str(plume.end_date)
        for event in get_event_objects(plume.eventlist, events_dict):
            event_elem = ET.SubElement(plume_elem, "Event")
            ET.SubElement(event_elem, "X").text = str(event.x)
            ET.SubElement(event_elem, "Y").text = str(event.y)
            ET.SubElement(event_elem, "ZoneID").text = str(event.zoneid)
    tree = ET.ElementTree(root)
    tree.write(filepath, encoding="utf-8", xml_declaration=True)

'''




    





