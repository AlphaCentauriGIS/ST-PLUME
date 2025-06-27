# Handles data preprocessing and validation
import pandas as pd
import numpy as np
from .utilities import load_csv
from .classes import Event, Zone

import numpy as np

def build_EventsZones(event_df, zone_df, zone_links, base_spatial_eps, temporal_eps):
    """Build events & zones, assign events, and calculate KDTree for each zone."""
    # Ensure EventDate is datetime
    event_df['EventDate'] = pd.to_datetime(event_df['EventDate'])
    event_df['z'] = (event_df['EventDate'] - event_df['EventDate'].min()).dt.days
    zone_df = zone_df.set_index("ZoneID")

    # Setup Zone dictionary
    zones = {}

    # Process incoming events
    for row in event_df.itertuples():
        # Create event object
        event = Event(row.EventID, row.EventDate, row.X, row.Y, row.z, row.ZoneID, base_spatial_eps,temporal_eps)

        # If zone already exists, add event
        if event.zoneid in zones:
            zones[event.zoneid].add_event(event)
            event.set_zidx(zones[event.zoneid].zidx)

        # If zone does not exist, create it
        else:
            if event.zoneid in zone_df.index:
                zone_record = zone_df.loc[event.zoneid]

                if isinstance(zone_record, pd.Series):  # Single row case - Should only ever be single rows, unless there are duplicate zoneids
                    zone = Zone(zone_record.name, zone_record['ZIDX']) 
                else:  # Multiple rows case
                    zone_record = zone_record.iloc[0]  # Pick first row
                    zone = Zone(zone_record.name, zone_record['ZIDX'])
                
                zone.add_event(event)
                event.set_zidx(zone.zidx)
                zones[zone.id] = zone
            else:
                print(f"Warning: ZoneID {event.zoneid} not found in zone_df. Event skipped.")

    # Only update zones that exist
    for zone_id, zone in zones.items():
        if zone_id in zone_links:
            for neighbor_id, distance in zone_links[zone_id]:
                if neighbor_id in zones:  # Only add valid neighbors
                    zone.add_neighbor(neighbor_id, distance)

        # Sort neighbors by distance (ascending order)
        if len(zone.neighborDistances) > 1:
            sorted_indices = np.argsort(zone.neighborDistances)
            zone.linkedZones = zone.linkedZones[sorted_indices]
            zone.neighborDistances = zone.neighborDistances[sorted_indices]

        # Only build KDTree if the zone has more than 1 event
        if zone.eventcount > 1:
            zone.build_kdtree()

    return zones


def load_zone_links(zone_link_file):
    """Load zone links as bidirectional with distance to neighbor."""
    df = load_csv(zone_link_file, required_columns=["ZoneID", "LinkID", "LinkDist"])

    # Initialize empty dictionary
    zone_links = {}

    for row in df.itertuples():
        src, nbr, dist = int(row.ZoneID), int(row.LinkID), float(row.LinkDist)

        # Add forward link
        if src not in zone_links:
            zone_links[src] = []
        zone_links[src].append((nbr, dist))

        # Add reverse link (bidirectional)
        if nbr not in zone_links:
            zone_links[nbr] = []
        zone_links[nbr].append((src, dist))  # Ensure bidirectional linking

    return zone_links
