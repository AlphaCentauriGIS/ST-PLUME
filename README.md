# ST-PLUME
Space-Time Plume Algorithm to identify dynamic places

Point clustering analysis commonly assumes events occur in an empty space and, therefore, ignores geospatial features where events take place.
This research sets forth a new idea: relational density to emphasize that density is not an absolute measure but a relative measure of the spatial structure of geospatial features in which events occur.
Based on relational density, we developed a new algorithm, Space-Time Plume, to detect and track spatial event clusters as plumes of smoke over time. 

Distinguished from conventional density-based clustering algorithms, ST-Plume adapts the spatial reachability to the underlying spatial structure in a zone and other zone-based parameters, as well as multiple temporal intervals to capture the temporal dynamics of events in hierarchies of plumes.
The algorithm tracks the plumes’ progression, identifies their spatial and temporal relationships, and shows how event-driven places emerge, evolve, and disappear.

A case study of crime events in Dallas, Texas, USA, demonstrated the algorithm's performance and its potential to represent and compute criminogenic places.

We revised the metaball algorithm and Perlin noise perturbations to visualize the plumes, their structures, and evolutions over space and time. 
Compared to the popular ST-DBSCAN algorithm, ST-Plume is competitive in computational efficiency but can represent and compute dynamic places with deeper geographic insights
