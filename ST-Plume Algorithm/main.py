# Main Pipeline script for STPlume Project
# Runs the entire program considering input files and config

import os
import time
import logging
from pathlib import Path
from modules.preprocessing import *
from modules.st_plume import Model
from modules.utilities import *

if __name__ == "__main__":
    try:
        # Load configuration file
        config = load_config("config.json")

        # Set up logging
        log_file = config["logging"]["log_file"]
        log_level = getattr(logging, config["logging"]["level"].upper(), logging.INFO)
        setup_logging(log_file, log_level)

        logging.info("Starting STPlume pipeline...")

        # File paths
        event_file = config["input_files"]["event_file"]
        zone_file = config["input_files"]["zone_file"]
        zone_link_file = config["input_files"]["zone_link_file"]
        output_directory = Path(config["output_directory"])
        output_directory.mkdir(parents=True, exist_ok=True)
        output_format = config.get("output_format", "csv").lower()

        # Parameters
        temporal_epsilons = config["parameters"]["temporal_epsilons"]
        base_spatial_epsilon = config["parameters"]["spatial_epsilon"]
        greater = config["parameters"]["ExportPlumesWithEventCountGreaterThan"]
        UseTemporalDensity = config["parameters"]["UseTemporalDensityFilter"]
        psi = config["parameters"]["TemporalDensityPSI"]
        omega = config["parameters"]["TemporalDensityOmega"]
        scaling = config["parameters"]["TemporalDensityScaling"]
        decay = config["parameters"]["TemporalDensityDecay"]
        UseMinNbrs = config["parameters"]["UseDynamicMinNeighbors"]
        MaxMinNbrs = config["parameters"]["MaxMinNeighbors"]

        # Validate output format
        validate_output(output_format)        

        # Load and preprocess data
        logging.info("Loading and preprocessing data...")

        # Load Event Data
        logging.info("Loading Events")
        event_df = load_csv(event_file, required_columns=["EventID", "EventDate", "X", "Y", "ZoneID"])
        print(f"Event Data loaded successfully! {len(event_df)} rows")
        
        # Load Zones & Links
        logging.info("Loading Zones and Zone Links Reference")
        zone_df = load_csv(zone_file, required_columns=["ZoneID", "ZIDX"])
        print(f" Zone Data loaded successfully! {len(zone_df)} rows")

        zone_links = load_zone_links(zone_link_file)
        print(f"Zone Links loaded successfully! {len(zone_links)} rows")

        # Build Events & Zones using Events, Zone Reference, and Zone Links
        logging.info("Building Zones")
        zones = build_EventsZones(event_df, zone_df, zone_links, base_spatial_epsilon, temporal_epsilons)  
        print(f" Zones built successfully! {len(zones)} Zones")
        logging.info(f'Total Zones created: {len(zones)}')

        # Run ST-Plume
        logging.info("Running ST-Plume Model")
        start_time = time.time()
        model = Model(zones, base_spatial_epsilon, temporal_epsilons, UseTemporalDensity, psi, omega, decay, scaling, UseMinNbrs, MaxMinNbrs)
        plumes = model.pluming()  # ✅ Ensure `pluming()` returns `self.plumes`
        end_time = time.time()

        print(f"Runtime: {end_time - start_time:.4f} seconds")
        # Ensure plumes exist before saving outputs
        logging.info(f"Saving {len(model.plumes)} plumes to output...")
        exportPlumesEvents(plumes, greater, temporal_epsilons, output_format, output_directory)
        logging.info("Pipeline completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        print(f"❌ An error occurred. Check the log file for details: {log_file}")
