# Core spatio-temporal plume detection algorithm
from .classes import Plume
import logging
import queue
import time
from math import sqrt, exp

class Model:
    def __init__(self, zones, spatial_eps, temporal_eps, TempDens = 0, psi = 0.5, omega = 0.5, TDecay = 0.05, Scaling = 10, useMinNbr = 0, maxMinNbr = 5):
        self.s_eps = spatial_eps
        self.max_eps = (self.s_eps * max(zone.zidx for zone in zones.values())) + 1
        self.LinkedZones_Threshold = 1.5 * self.max_eps
        self.t_eps = temporal_eps
        self.zones = zones
        self.plumes = {}
        self.plumecounter = 1
        self.visited = {}
        self.temporalDens = TempDens
        self.psi = psi
        self.omega = omega
        self.temporaldecay = TDecay
        self.k = Scaling
        self.UseMinNeighbors = useMinNbr
        self.baseMin = 1
        self.MaxMin = maxMinNbr


    def pluming(self):
        """Process zones, events into plumes."""
        for _, zone in self.zones.items(): 
            #print(f'Processing {zone.id}')
            for event in zone.eventset:  # Directly iterate over eventset list
                active_eps, active_plumes = self.initialize_active_plumes(event)

                if active_eps:  
                    self.visit(event, active_eps)   # Mark event as visited for each active epsilon
                    neighbor_queue = queue.Queue()  # Initialize neighbor queue
                    neighbor_queue.put([event, active_eps]) # Put the initiating event into the queue

                    while not neighbor_queue.empty():
                        e, eps = neighbor_queue.get() # Pull the event and its epsilons (active or considered) from the next tuple in the queue
                        st_neighbors = self.getNeighbors(e, eps)
                                               
                        if st_neighbors:
                            checkpass = True      
                            if self.UseMinNeighbors == 1:
                                nbrs_needed = int(self.baseMin + (self.MaxMin - self.baseMin) * ((e.zidx-1) / 2))
                                if len(st_neighbors) < nbrs_needed:
                                    checkpass = False

                            if checkpass: 
                                for neighbor in st_neighbors:
                                    neighbor_event, neighbor_epsilons = neighbor[0], neighbor[1]
                                    self.process_neighbor_event(neighbor_event, neighbor_epsilons, active_plumes, neighbor_queue)
                        
                    # Queue Finished, now add active plumes to our all plume list
                    for tep, plume in active_plumes.items():
                        self.plumes[plume.id] = plume  

                    # Create Plume Links
                    self.set_plumeLinks(event)

        return self.plumes
 
    ######################################################################################################################################
    # Event-Plume Main loop processing overhead                                                                                          #
    #                                                                                                                                    #
    ######################################################################################################################################
    def initialize_active_plumes(self, event):
        # Determine unprocessed epsilons
        active_eps = [t for t in self.t_eps if event.assigned[t] == 0]
        # Create a set of plumes, one for each epsilon, that we will process with this event as the initiator
        active_plumes = {
            t: Plume(self.plumecounter + i, t, event) for i, t in enumerate(active_eps)
        }
        # Assign the new plume ids to the event assigned dictionary
        for i, t in enumerate(active_eps):
            event.assigned[t] = self.plumecounter + i  

        # Increase plume counter for future ids
        self.plumecounter += len(active_eps)

        # Return the active epsilons and plumes
        return active_eps, active_plumes
    
    def set_plumeLinks(self, event):
        ordered_eps = list(event.assigned.keys())  # Preserve descending order (e.g., [45, 21])

        for idx, eps in enumerate(ordered_eps):
            plumeid = event.assigned[eps]
            cur_plume = self.plumes[plumeid]

            # Process Parent of Current Plume (Higher Epsilon First)
            if idx + 1 < len(ordered_eps):  # If there is a next smaller epsilon
                next_ep = ordered_eps[idx + 1]  # 21-day (child) follows 45-day (parent)
                child_id = event.assigned[next_ep]
                child_plume = self.plumes[child_id]

                cur_plume.add_childplume(child_id, child_plume.eventcount)
                child_plume.add_parentplume(plumeid, cur_plume.eventcount)

            # Process Child of Current Plume (Lower Epsilon Becomes Child)
            if idx > 0:  # If there was a previous larger epsilon
                prev_ep = ordered_eps[idx - 1]  # 45-day (parent) comes before 21-day
                parent_id = event.assigned[prev_ep]
                parent_plume = self.plumes[parent_id]

                cur_plume.add_parentplume(parent_id, parent_plume.eventcount)
                parent_plume.add_childplume(plumeid, cur_plume.eventcount)

        return
    

    ######################################################################################################################################
    # Event Processing Functions                                                                                                         #
    #                                                                                                                                    #
    ######################################################################################################################################
    
    def visit(self, event, eps):
        if event.id in self.visited:
            self.visited[event.id].extend(eps)                                                          
        else:
            self.visited[event.id] = eps
        return
    
    def neighbor_visit(self, event, eps):
        if event.id in self.visited:
            filtered_eps = [tnep for tnep in eps if tnep not in self.visited[event.id]]
            if filtered_eps:
                self.visited[event.id].extend(filtered_eps)
                return filtered_eps
            else:
                return []
        else:
            self.visited[event.id] = eps  # Mark event as visited with these epsilons
            return eps
    
    """
        Processes a neighbor event to determine relevant temporal epsilons and assign it to active plumes.

        Parameters:
            event (Event): The neighboring event being processed.
            temporal_eps (list): The list of temporal epsilons for this neighbor event.
            active_plumes (dict): Dictionary of active plumes indexed by temporal epsilon.
            neighbor_queue (Queue): The queue for processing neighbor events.
        """
    
    def process_neighbor_event(self, event, temporal_eps, active_plumes, neighbor_queue):
        # Determine which temporal epsilons should be considered
        considered_eps = self.neighbor_visit(event, temporal_eps)

        # Process the event for the relevant epsilons
        if considered_eps:  # Only proceed if there are new epsilons to process
            final_eps = []

            for epsilon in considered_eps:
                # Determine if we consider this neighbor, (Temporal Density) or some other exclusionary rule
                if self.temporalDens == 1:             
                    event_TemporalScore = self.event_ITRscore(event)
                    plume_Threshold = self.plumeTemporalThreshold(event, active_plumes[epsilon])

                else:
                    event_TemporalScore = 1
                    plume_Threshold = 1
                

                if event.assigned[epsilon] == 0 and event_TemporalScore >= plume_Threshold:
                    active_plumes[epsilon].add_event(event)
                    event.set_plumeid(epsilon, active_plumes[epsilon].id)  # Assign the plume
                    final_eps.append(epsilon)
            
            if final_eps:
                neighbor_queue.put([event, final_eps])  # Queue the event for neighbor processing



    ######################################################################################################################################
    # Space Time Neighbor Filtering Functions                                                                                            #
    #                                                                                                                                    #
    ######################################################################################################################################
    def getNeighbors(self, event, eps):
        space_neighbors = self.space_filter(event)                # KD Tree searches for all neighbors in current zone and neighbor zones
        #print(f'Space Neighbors: {[neighbor.id for neighbor in space_neighbors]}')
        ST_neighbors = self.temporal_filter(event, space_neighbors, eps)         # Filter spatial neighbors for temporal epslions
        return ST_neighbors

    """Filter temporal neighbors using the considered epsilons."""
    def temporal_filter(self, event, space_neighbors, consider_eps):
        neighbors = []
        for neighbor in space_neighbors:
            if event.id == neighbor.id:
                continue
            else:
                temporal_dist = abs((event.eventdate - neighbor.eventdate).days)
                neighbor_eps = [ep for ep in consider_eps if temporal_dist <= ep]
                if neighbor_eps:
                    neighbors.append((neighbor, neighbor_eps))
        return neighbors

    """Find neighbors across a zone and its linked zones using KD-Trees,
        falling back to a linear scan if a KD-Tree is not available."""

    def space_filter(self, event):
        zone = self.zones[event.zoneid]
        # Check the event's own zone: use KD-Tree if available, else do a linear scan.
        neighbors = self.process_zone_neighbors(event, zone)
        
        # Check linked neighbor zones.
        for idx, nzone_id in enumerate(zone.linkedZones):
            nzone = self.zones[nzone_id]
            nzone_dist = zone.neighborDistances[idx]  # Distance to neighbor zone

            if nzone_dist <= self.LinkedZones_Threshold:
                neighbors.extend(self.process_zone_neighbors(event, nzone))
        return neighbors
    
    def process_zone_neighbors(self, event, zone):
        neighbors = []
        query_point = [event.x, event.y]
        if zone.kdtree:
            if event.zoneid == zone.id:
                event_indices = zone.kdtree.query_ball_point(query_point, event.zidx)
            else:
                event_indices = zone.kdtree.query_ball_point(query_point, self.max_eps)
            for i in event_indices:
                nbr = zone.eventset[i]
                actual_dist = self.within_distance(event, nbr)
                if actual_dist <= event.spatial_eps or actual_dist <= nbr.spatial_eps:
                    neighbors.append(nbr)
        else:
            for evt in zone.eventset:
                if abs(event.x - evt.x) > self.max_eps or abs(event.y - evt.y) > self.max_eps:
                    continue  # Skip points that are obviously too far before computing sqrt()
    
                actual_dist = self.within_distance(event, evt)
                if actual_dist <= event.spatial_eps or actual_dist <= evt.spatial_eps:
                    neighbors.append(evt)
        return neighbors


    # Euclidean Distance for Zones without KDTrees
    def within_distance(self, evt1, evt2):
        return sqrt((evt1.x - evt2.x) ** 2 + (evt1.y - evt2.y) ** 2)

    ######################################################################################################################################
    #   Event/Plume Temporal Interaction Calculations for event inclusion/exclusion
    #
    ######################################################################################################################################

    def event_ITRscore(self, event):
        zidx = event.zidx
        space_neighbors = self.space_filter(event)

        temporal_density = 0
        min_time = float('inf')
        max_time = float('-inf')

        event_date = event.eventdate  # Avoid repeated attribute lookup

        for neighbor in space_neighbors:
            neighbor_date = neighbor.eventdate
            temporal_density += exp(-self.temporaldecay * abs((event_date - neighbor_date).days))

            # Use min/max functions directly
            min_time = min(min_time, neighbor_date)
            max_time = max(max_time, neighbor_date)

        time_span = max(1, (max_time - min_time).days)  # Avoid division by zero
        return zidx + (temporal_density / time_span)

    
    def plumeTemporalThreshold(self, event, plume):
        zidx = plume.get_median_zidx()
        N = plume.eventcount
        scaling_factor = N / (N + self.k) # Test 7 for now (k)
        rtd = sum(exp(-self.temporaldecay * abs((event.eventdate - plume_event.eventdate).days)) for plume_event in plume.eventlist)
        return self.psi * (1 / zidx) + self.omega * scaling_factor * (rtd / plume.timespan)







