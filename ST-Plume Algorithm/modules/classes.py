from scipy.spatial import KDTree
import heapq
import numpy as np

class Zone:
    def __init__(self, id, zidx):
        """Zone object with properties for analysis."""
        self.id = int(id)                               # Zone Identification - Integer
        self.zidx = zidx                                # Zone Index Metric - Double  Diveide by /2.0?
        self.eventcount = 0                             # Count of Events within Zone - Integer
        self.eventset = []                              # List of Event Objects - Events
        self.kdtree = None                              # KDTree for events
        self.linkedZones = np.array([], dtype=int)      # Array of Neighboring Zone IDs - Integer Array 
        self.neighborDistances = np.array([], dtype=float) # Array of Neighboring Zone Distances - Double Array

    def add_event(self, event):
        """Adds an event to this zone and updates count."""
        self.eventset.append(event) 
        self.eventcount += 1

    def add_neighbor(self, link_id, distance):
        """Adds a linked zone with distance."""
        self.linkedZones = np.append(self.linkedZones, int(link_id))
        self.neighborDistances = np.append(self.neighborDistances, float(distance))

    def build_kdtree(self):
        """Returns KDTree, building it only if necessary."""
        if self.kdtree is None and self.eventset:
            points = [(event.x, event.y) for event in self.eventset]
            self.kdtree = KDTree(points)
        return self.kdtree

class Plume:
    def __init__(self, id, tep, event):
        """Plume object for grouping related events."""
        self.id = id                            # Plume Identification - Integer
        self.temporal = tep                     # Plume Temporal Epsilon - Integer (Days)
        self.parent = set()                    # Parent Plume ID - Integer
        self.children = set()                   # Set of Child Plume IDs - Integer
        self.eventcount = 1                     # Number of events allocated to plume (initialized = 1)
        self.eventlist = [event]                # List of event ids allocated to plume - Integer List

        # Rolling Temporal Density Tracking
        # Two heaps for efficient median zidx tracking 
        self.zidx_lower = []  # Max-Heap (negative values for heapq)
        self.zidx_upper = []  # Min-Heap
        self.min_date = event.eventdate
        self.max_date = event.eventdate
        self.timespan = max(1, abs((self.max_date-self.min_date).days))

        self.insert_zidx(event.zidx)
        #self.threshold = 


    def add_event(self, event):
        """Adds an event ID to the plume, updating the event count and date range."""
        self.eventlist.append(event) 
        self.eventcount += 1
        self.insert_zidx(event.zidx)

        if event.eventdate < self.min_date:
            self.min_date = event.eventdate
            self.update_timespan()

        if event.eventdate > self.max_date:
            self.max_date = event.eventdate
            self.update_timespan()

    
    def add_childplume(self, plume_id, eventcount):
        plumedesc= (plume_id, eventcount)
        self.children.add(plumedesc)

    def add_parentplume(self, plume_id, eventcount):
        plumedesc= (plume_id, eventcount)
        self.parent.add(plumedesc)

    def update_timespan(self):
        self.timespan = max(1, abs((self.max_date-self.min_date).days))

    def insert_zidx(self, zidx):
        """Insert Zidx into heaps and balance them."""
        if not self.zidx_lower or zidx <= -self.zidx_lower[0]:
            heapq.heappush(self.zidx_lower, -zidx)
        else:
            heapq.heappush(self.zidx_upper, zidx)

        # Balance the heaps to ensure median computation
        if len(self.zidx_lower) > len(self.zidx_upper) + 1:
            heapq.heappush(self.zidx_upper, -heapq.heappop(self.zidx_lower))
        elif len(self.zidx_upper) > len(self.zidx_lower):
            heapq.heappush(self.zidx_lower, -heapq.heappop(self.zidx_upper))

    def get_median_zidx(self):
        """Retrieve the median Zidx in O(1) time."""

        if len(self.zidx_lower) > len(self.zidx_upper):
            return -self.zidx_lower[0]  # Max-Heap root (negative sign reversal)
        elif len(self.zidx_lower) < len(self.zidx_upper):
            return self.zidx_upper[0]  # Min-Heap root
        else:
            return (-self.zidx_lower[0] + self.zidx_upper[0]) / 2  # Average of two heaps

    
class Event:
    def __init__(self, id, eventdate, x, y, z, zoneid, base_spatial_eps, temporal_eps):
        """Event object representing an individual spatial-temporal event."""
        self.id = id                                        # Event Identification - Integer
        self.assigned = dict.fromkeys(sorted(temporal_eps, reverse=True), 0)      # Dictionary of Plume Allocations in different temporal epsilons                                   
        #self.SuperPlume = dict.fromkeys(sorted(temporal_eps, reverse=True), 0)  # Dictionary of Super Plume Categorization, linking plumes across space/time
        self.eventdate = eventdate                          # Event Date - DateTime
        self.x = x                                          # Event X Coordinate - Double
        self.y = y                                          # Event Y Coordinate - Double
        self.z = z                                          # Event Z value is the number of days from the first event date
        self.zoneid = int(zoneid)                           # Associated Zone ID - Integer
        self.zidx = 0
        self.spatial_eps = base_spatial_eps

    def set_plumeid(self, eps, plume_id):
        self.assigned[eps] = plume_id

    def set_zidx(self, zidx):
        self.zidx = zidx
        self.spatial_eps *= zidx
        



