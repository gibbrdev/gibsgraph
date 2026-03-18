"""Build GibsGraph training pairs for construction/AEC domain.

Sources:
  1. bSDD (buildingSMART Data Dictionary) — product classifications via API
  2. Brick Schema — building systems ontology (HVAC, electrical, etc.)
  3. BOT (Building Topology Ontology) — W3C spatial topology

Graph patterns:
  (:Building)-[:HAS_STOREY]->(:Storey)
  (:Space)-[:BOUNDED_BY]->(:Wall)
  (:Equipment)-[:IS_PART_OF]->(:System)
  (:Product)-[:CLASSIFIED_AS]->(:Classification)
  (:Material)-[:HAS_PROPERTY]->(:Property)

Construction data is sparse but the pipeline is ready for when EU digital
building requirements force data digitization.

Usage:
    python data/scripts/build_construction_pairs.py
    python data/scripts/build_construction_pairs.py --stats-only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

DEFAULT_OUTPUT = Path("data/training/construction_pairs.jsonl")
CACHE_DIR = Path("data/raw/construction")

# bSDD API
BSDD_API = "https://api.bsdd.buildingsmart.org/api"
BSDD_SEARCH = f"{BSDD_API}/SearchList/v2"
BSDD_CLASS = f"{BSDD_API}/Class/v1"
BSDD_DICTIONARY = f"{BSDD_API}/Dictionary/v1"

# Key bSDD dictionaries (IFC, construction products)
BSDD_DICTIONARIES = [
    "https://identifier.buildingsmart.org/uri/buildingsmart/ifc/4.3",
]

# Edge weights for construction domain
EDGE_WEIGHTS: dict[str, float] = {
    "HAS_STOREY": 0.9,
    "HAS_SPACE": 0.85,
    "CONTAINS_ELEMENT": 0.8,
    "BOUNDED_BY": 0.75,
    "ADJACENT_TO": 0.7,
    "CONNECTED_TO": 0.65,
    "HAS_PROPERTY": 0.6,
    "CLASSIFIED_AS": 0.7,
    "MADE_OF": 0.65,
    "PART_OF": 0.8,
    "IS_A": 0.9,
    "HAS_PORT": 0.5,
    "FEEDS": 0.7,
    "IS_FED_BY": 0.7,
    "HAS_POINT": 0.6,
    "IS_LOCATION_OF": 0.75,
    "SERVES": 0.7,
}

# ── BOT (Building Topology Ontology) — hardcoded from W3C spec ──
# https://w3id.org/bot — small, stable, canonical
BOT_CLASSES = [
    {"id": "bot_Building", "label": "Building", "description": "An independent unit of the built environment with a characteristic spatial structure."},
    {"id": "bot_Storey", "label": "Storey", "description": "A level part of a building."},
    {"id": "bot_Space", "label": "Space", "description": "A limited three-dimensional extent defined physically or notionally."},
    {"id": "bot_Element", "label": "BuildingElement", "description": "A constituent of a construction entity with a characteristic technical function, form, or position."},
    {"id": "bot_Zone", "label": "Zone", "description": "A part of the physical or virtual world that is inherently both located in this world and has a 3D spatial extent."},
    {"id": "bot_Interface", "label": "Interface", "description": "The surface where two building elements or a building element and an opening meet."},
    {"id": "bot_Site", "label": "Site", "description": "An area containing one or more buildings."},
]

BOT_RELATIONSHIPS = [
    {"source": "bot_Site", "target": "bot_Building", "type": "HAS_BUILDING"},
    {"source": "bot_Building", "target": "bot_Storey", "type": "HAS_STOREY"},
    {"source": "bot_Building", "target": "bot_Space", "type": "HAS_SPACE"},
    {"source": "bot_Storey", "target": "bot_Space", "type": "HAS_SPACE"},
    {"source": "bot_Space", "target": "bot_Element", "type": "CONTAINS_ELEMENT"},
    {"source": "bot_Space", "target": "bot_Element", "type": "ADJACENT_ELEMENT"},
    {"source": "bot_Space", "target": "bot_Space", "type": "ADJACENT_TO"},
    {"source": "bot_Element", "target": "bot_Element", "type": "CONNECTED_TO"},
    {"source": "bot_Element", "target": "bot_Interface", "type": "HAS_INTERFACE"},
    {"source": "bot_Zone", "target": "bot_Space", "type": "HAS_SPACE"},
    {"source": "bot_Zone", "target": "bot_Element", "type": "CONTAINS_ELEMENT"},
]

# ── Brick Schema core classes — hardcoded from brickschema.org ──
# ~50 most important classes from the full ~1000
BRICK_CLASSES = [
    # HVAC
    {"id": "brick_AHU", "label": "AirHandlingUnit", "kind": "Equipment", "description": "Assembly consisting of sections containing fans, heating/cooling coils, filters, etc."},
    {"id": "brick_VAV", "label": "VariableAirVolume", "kind": "Equipment", "description": "Terminal unit that regulates airflow to a space."},
    {"id": "brick_Chiller", "label": "Chiller", "kind": "Equipment", "description": "Equipment that removes heat from a liquid via vapor-compression or absorption refrigeration."},
    {"id": "brick_Boiler", "label": "Boiler", "kind": "Equipment", "description": "Equipment that heats water or produces steam for space/process heating."},
    {"id": "brick_Pump", "label": "Pump", "kind": "Equipment", "description": "Equipment that moves fluid by mechanical action."},
    {"id": "brick_Fan", "label": "Fan", "kind": "Equipment", "description": "Equipment that produces airflow by rotating blades."},
    {"id": "brick_HeatExchanger", "label": "HeatExchanger", "kind": "Equipment", "description": "Equipment that transfers heat between two fluids without mixing."},
    {"id": "brick_CoolingTower", "label": "CoolingTower", "kind": "Equipment", "description": "Equipment that rejects heat from condenser water to the atmosphere."},
    {"id": "brick_Damper", "label": "Damper", "kind": "Equipment", "description": "Device that regulates airflow in a duct or opening."},
    {"id": "brick_Valve", "label": "Valve", "kind": "Equipment", "description": "Device that regulates fluid flow in a pipe."},
    {"id": "brick_Filter", "label": "Filter", "kind": "Equipment", "description": "Device that removes contaminants from air or fluid."},
    {"id": "brick_Coil", "label": "Coil", "kind": "Equipment", "description": "Heat transfer element in an air handling unit."},
    # Electrical
    {"id": "brick_ElectricalMeter", "label": "ElectricalMeter", "kind": "Equipment", "description": "Device that measures electrical energy consumption."},
    {"id": "brick_Transformer", "label": "Transformer", "kind": "Equipment", "description": "Device that transfers electrical energy between circuits."},
    {"id": "brick_Switchgear", "label": "Switchgear", "kind": "Equipment", "description": "Electrical disconnect and protection equipment."},
    {"id": "brick_PV", "label": "PhotovoltaicPanel", "kind": "Equipment", "description": "Device that converts sunlight to electricity."},
    # Lighting
    {"id": "brick_Luminaire", "label": "Luminaire", "kind": "Equipment", "description": "Complete lighting unit with lamp, housing, and controls."},
    {"id": "brick_OccupancySensor", "label": "OccupancySensor", "kind": "Sensor", "description": "Sensor that detects human presence in a space."},
    # Locations
    {"id": "brick_Building", "label": "Building", "kind": "Location", "description": "A structure with walls and a roof."},
    {"id": "brick_Floor", "label": "Floor", "kind": "Location", "description": "A level within a building."},
    {"id": "brick_Room", "label": "Room", "kind": "Location", "description": "An enclosed space within a building."},
    {"id": "brick_Wing", "label": "Wing", "kind": "Location", "description": "A section of a building extending from the main structure."},
    # Systems
    {"id": "brick_HVAC_System", "label": "HvacSystem", "kind": "System", "description": "System that provides heating, ventilation, and air conditioning."},
    {"id": "brick_Hot_Water_System", "label": "HotWaterSystem", "kind": "System", "description": "System that heats and distributes hot water."},
    {"id": "brick_Chilled_Water_System", "label": "ChilledWaterSystem", "kind": "System", "description": "System that chills and distributes cold water for cooling."},
    {"id": "brick_Electrical_System", "label": "ElectricalSystem", "kind": "System", "description": "System that distributes electrical power."},
    {"id": "brick_Lighting_System", "label": "LightingSystem", "kind": "System", "description": "System that provides artificial illumination."},
    {"id": "brick_Fire_Safety_System", "label": "FireSafetySystem", "kind": "System", "description": "System for fire detection, alarm, and suppression."},
    # Points (sensor/setpoint/command)
    {"id": "brick_Temperature_Sensor", "label": "TemperatureSensor", "kind": "Point", "description": "Sensor that measures temperature."},
    {"id": "brick_Humidity_Sensor", "label": "HumiditySensor", "kind": "Point", "description": "Sensor that measures relative humidity."},
    {"id": "brick_CO2_Sensor", "label": "Co2Sensor", "kind": "Point", "description": "Sensor that measures CO2 concentration."},
    {"id": "brick_Pressure_Sensor", "label": "PressureSensor", "kind": "Point", "description": "Sensor that measures fluid or air pressure."},
    {"id": "brick_Flow_Sensor", "label": "FlowSensor", "kind": "Point", "description": "Sensor that measures fluid flow rate."},
    {"id": "brick_Temperature_Setpoint", "label": "TemperatureSetpoint", "kind": "Point", "description": "Desired temperature setting for a space or system."},
    {"id": "brick_Damper_Command", "label": "DamperCommand", "kind": "Point", "description": "Command signal controlling a damper position."},
    {"id": "brick_Valve_Command", "label": "ValveCommand", "kind": "Point", "description": "Command signal controlling a valve position."},
]

BRICK_RELATIONSHIPS = [
    # Equipment → System
    {"source": "brick_AHU", "target": "brick_HVAC_System", "type": "PART_OF"},
    {"source": "brick_VAV", "target": "brick_HVAC_System", "type": "PART_OF"},
    {"source": "brick_Chiller", "target": "brick_Chilled_Water_System", "type": "PART_OF"},
    {"source": "brick_Boiler", "target": "brick_Hot_Water_System", "type": "PART_OF"},
    {"source": "brick_CoolingTower", "target": "brick_Chilled_Water_System", "type": "PART_OF"},
    {"source": "brick_Luminaire", "target": "brick_Lighting_System", "type": "PART_OF"},
    {"source": "brick_Transformer", "target": "brick_Electrical_System", "type": "PART_OF"},
    {"source": "brick_Switchgear", "target": "brick_Electrical_System", "type": "PART_OF"},
    {"source": "brick_PV", "target": "brick_Electrical_System", "type": "PART_OF"},
    # Equipment composition
    {"source": "brick_Fan", "target": "brick_AHU", "type": "PART_OF"},
    {"source": "brick_Coil", "target": "brick_AHU", "type": "PART_OF"},
    {"source": "brick_Filter", "target": "brick_AHU", "type": "PART_OF"},
    {"source": "brick_Damper", "target": "brick_AHU", "type": "PART_OF"},
    {"source": "brick_Damper", "target": "brick_VAV", "type": "PART_OF"},
    {"source": "brick_Coil", "target": "brick_VAV", "type": "PART_OF"},
    # Feeds relationships
    {"source": "brick_AHU", "target": "brick_VAV", "type": "FEEDS"},
    {"source": "brick_Chiller", "target": "brick_AHU", "type": "FEEDS"},
    {"source": "brick_Boiler", "target": "brick_AHU", "type": "FEEDS"},
    {"source": "brick_CoolingTower", "target": "brick_Chiller", "type": "FEEDS"},
    {"source": "brick_Pump", "target": "brick_Chiller", "type": "FEEDS"},
    {"source": "brick_Pump", "target": "brick_Boiler", "type": "FEEDS"},
    # Location hierarchy
    {"source": "brick_Building", "target": "brick_Floor", "type": "HAS_PART"},
    {"source": "brick_Floor", "target": "brick_Room", "type": "HAS_PART"},
    {"source": "brick_Building", "target": "brick_Wing", "type": "HAS_PART"},
    {"source": "brick_Wing", "target": "brick_Floor", "type": "HAS_PART"},
    # Equipment → Location
    {"source": "brick_VAV", "target": "brick_Room", "type": "SERVES"},
    {"source": "brick_AHU", "target": "brick_Floor", "type": "SERVES"},
    {"source": "brick_Luminaire", "target": "brick_Room", "type": "SERVES"},
    # Points → Equipment
    {"source": "brick_Temperature_Sensor", "target": "brick_AHU", "type": "IS_POINT_OF"},
    {"source": "brick_Temperature_Sensor", "target": "brick_VAV", "type": "IS_POINT_OF"},
    {"source": "brick_Temperature_Sensor", "target": "brick_Room", "type": "IS_POINT_OF"},
    {"source": "brick_Humidity_Sensor", "target": "brick_AHU", "type": "IS_POINT_OF"},
    {"source": "brick_CO2_Sensor", "target": "brick_Room", "type": "IS_POINT_OF"},
    {"source": "brick_Pressure_Sensor", "target": "brick_AHU", "type": "IS_POINT_OF"},
    {"source": "brick_Flow_Sensor", "target": "brick_Chiller", "type": "IS_POINT_OF"},
    {"source": "brick_Temperature_Setpoint", "target": "brick_VAV", "type": "IS_POINT_OF"},
    {"source": "brick_Damper_Command", "target": "brick_Damper", "type": "IS_POINT_OF"},
    {"source": "brick_Valve_Command", "target": "brick_Valve", "type": "IS_POINT_OF"},
    {"source": "brick_OccupancySensor", "target": "brick_Room", "type": "IS_POINT_OF"},
    # Sensor → Location
    {"source": "brick_ElectricalMeter", "target": "brick_Building", "type": "IS_POINT_OF"},
    {"source": "brick_ElectricalMeter", "target": "brick_Floor", "type": "IS_POINT_OF"},
]

# ── IFC core classes (most common in building models) ──
IFC_CLASSES = [
    {"id": "ifc_IfcWall", "label": "Wall", "kind": "BuildingElement", "description": "Vertical building element enclosing or dividing spaces."},
    {"id": "ifc_IfcSlab", "label": "Slab", "kind": "BuildingElement", "description": "Horizontal building element (floor, roof, landing)."},
    {"id": "ifc_IfcBeam", "label": "Beam", "kind": "BuildingElement", "description": "Horizontal structural member carrying loads."},
    {"id": "ifc_IfcColumn", "label": "Column", "kind": "BuildingElement", "description": "Vertical structural member carrying loads."},
    {"id": "ifc_IfcDoor", "label": "Door", "kind": "BuildingElement", "description": "Opening element for passage between spaces."},
    {"id": "ifc_IfcWindow", "label": "Window", "kind": "BuildingElement", "description": "Opening element for light and ventilation."},
    {"id": "ifc_IfcRoof", "label": "Roof", "kind": "BuildingElement", "description": "Element covering the top of a building."},
    {"id": "ifc_IfcStair", "label": "Stair", "kind": "BuildingElement", "description": "Vertical circulation element between storeys."},
    {"id": "ifc_IfcRamp", "label": "Ramp", "kind": "BuildingElement", "description": "Inclined element for accessibility."},
    {"id": "ifc_IfcCurtainWall", "label": "CurtainWall", "kind": "BuildingElement", "description": "Non-load-bearing facade element."},
    {"id": "ifc_IfcRailing", "label": "Railing", "kind": "BuildingElement", "description": "Barrier along edges for safety."},
    {"id": "ifc_IfcCovering", "label": "Covering", "kind": "BuildingElement", "description": "Finish applied to a surface (flooring, ceiling, cladding)."},
    {"id": "ifc_IfcPipe", "label": "Pipe", "kind": "DistributionElement", "description": "Cylindrical element for fluid distribution."},
    {"id": "ifc_IfcDuct", "label": "Duct", "kind": "DistributionElement", "description": "Enclosed channel for air distribution."},
    {"id": "ifc_IfcCableCarrier", "label": "CableCarrier", "kind": "DistributionElement", "description": "Element carrying electrical cables."},
    # Spatial
    {"id": "ifc_IfcSite", "label": "Site", "kind": "SpatialElement", "description": "Area of land on which the project is located."},
    {"id": "ifc_IfcBuilding", "label": "Building", "kind": "SpatialElement", "description": "Construction that provides shelter."},
    {"id": "ifc_IfcBuildingStorey", "label": "BuildingStorey", "kind": "SpatialElement", "description": "A level within a building."},
    {"id": "ifc_IfcSpace", "label": "Space", "kind": "SpatialElement", "description": "An area or volume bounded by physical or virtual boundaries."},
]

IFC_RELATIONSHIPS = [
    # Spatial hierarchy
    {"source": "ifc_IfcSite", "target": "ifc_IfcBuilding", "type": "CONTAINS"},
    {"source": "ifc_IfcBuilding", "target": "ifc_IfcBuildingStorey", "type": "CONTAINS"},
    {"source": "ifc_IfcBuildingStorey", "target": "ifc_IfcSpace", "type": "CONTAINS"},
    # Space boundaries
    {"source": "ifc_IfcSpace", "target": "ifc_IfcWall", "type": "BOUNDED_BY"},
    {"source": "ifc_IfcSpace", "target": "ifc_IfcSlab", "type": "BOUNDED_BY"},
    {"source": "ifc_IfcSpace", "target": "ifc_IfcDoor", "type": "BOUNDED_BY"},
    {"source": "ifc_IfcSpace", "target": "ifc_IfcWindow", "type": "BOUNDED_BY"},
    # Element containment
    {"source": "ifc_IfcBuildingStorey", "target": "ifc_IfcWall", "type": "CONTAINS_ELEMENT"},
    {"source": "ifc_IfcBuildingStorey", "target": "ifc_IfcColumn", "type": "CONTAINS_ELEMENT"},
    {"source": "ifc_IfcBuildingStorey", "target": "ifc_IfcBeam", "type": "CONTAINS_ELEMENT"},
    {"source": "ifc_IfcBuildingStorey", "target": "ifc_IfcSlab", "type": "CONTAINS_ELEMENT"},
    # Openings in elements
    {"source": "ifc_IfcWall", "target": "ifc_IfcDoor", "type": "HAS_OPENING"},
    {"source": "ifc_IfcWall", "target": "ifc_IfcWindow", "type": "HAS_OPENING"},
    # Connections
    {"source": "ifc_IfcWall", "target": "ifc_IfcWall", "type": "CONNECTED_TO"},
    {"source": "ifc_IfcWall", "target": "ifc_IfcSlab", "type": "CONNECTED_TO"},
    {"source": "ifc_IfcBeam", "target": "ifc_IfcColumn", "type": "CONNECTED_TO"},
    # Distribution
    {"source": "ifc_IfcBuildingStorey", "target": "ifc_IfcPipe", "type": "CONTAINS_ELEMENT"},
    {"source": "ifc_IfcBuildingStorey", "target": "ifc_IfcDuct", "type": "CONTAINS_ELEMENT"},
    {"source": "ifc_IfcPipe", "target": "ifc_IfcPipe", "type": "CONNECTED_TO"},
    {"source": "ifc_IfcDuct", "target": "ifc_IfcDuct", "type": "CONNECTED_TO"},
    # Covering
    {"source": "ifc_IfcSlab", "target": "ifc_IfcCovering", "type": "HAS_COVERING"},
    {"source": "ifc_IfcWall", "target": "ifc_IfcCovering", "type": "HAS_COVERING"},
    # Vertical circulation
    {"source": "ifc_IfcBuildingStorey", "target": "ifc_IfcStair", "type": "CONTAINS_ELEMENT"},
    {"source": "ifc_IfcBuildingStorey", "target": "ifc_IfcRamp", "type": "CONTAINS_ELEMENT"},
    {"source": "ifc_IfcStair", "target": "ifc_IfcRailing", "type": "HAS_COMPONENT"},
]

# ── NL-to-graph training examples for construction ──
# Realistic natural language queries mapped to expected graph patterns
NL_CONSTRUCTION_EXAMPLES = [
    {
        "input": "What spaces are on floor 3?",
        "nodes": [
            {"id": "floor_3", "label": "BuildingStorey", "props": {"name": "Floor 3", "elevation": 9.0}},
            {"id": "office_301", "label": "Space", "props": {"name": "Office 301", "area": 25.0}},
            {"id": "office_302", "label": "Space", "props": {"name": "Office 302", "area": 30.0}},
            {"id": "corridor_3", "label": "Space", "props": {"name": "Corridor 3", "area": 45.0}},
        ],
        "edges": [
            {"source": "floor_3", "target": "office_301", "type": "HAS_SPACE"},
            {"source": "floor_3", "target": "office_302", "type": "HAS_SPACE"},
            {"source": "floor_3", "target": "corridor_3", "type": "HAS_SPACE"},
        ],
    },
    {
        "input": "Which HVAC equipment serves the meeting room?",
        "nodes": [
            {"id": "meeting_room", "label": "Room", "props": {"name": "Meeting Room A", "area": 40.0}},
            {"id": "vav_mr", "label": "VariableAirVolume", "props": {"name": "VAV-MR-01"}},
            {"id": "ahu_1", "label": "AirHandlingUnit", "props": {"name": "AHU-01"}},
            {"id": "temp_sensor", "label": "TemperatureSensor", "props": {"name": "TS-MR-01"}},
        ],
        "edges": [
            {"source": "vav_mr", "target": "meeting_room", "type": "SERVES"},
            {"source": "ahu_1", "target": "vav_mr", "type": "FEEDS"},
            {"source": "temp_sensor", "target": "vav_mr", "type": "IS_POINT_OF"},
        ],
    },
    {
        "input": "Show me the structural elements supporting floor 5",
        "nodes": [
            {"id": "floor_5", "label": "BuildingStorey", "props": {"name": "Floor 5"}},
            {"id": "slab_5", "label": "Slab", "props": {"name": "Slab-05", "thickness": 0.25}},
            {"id": "beam_5a", "label": "Beam", "props": {"name": "B-05-A", "span": 8.0}},
            {"id": "beam_5b", "label": "Beam", "props": {"name": "B-05-B", "span": 6.0}},
            {"id": "col_5a", "label": "Column", "props": {"name": "C-05-A"}},
            {"id": "col_5b", "label": "Column", "props": {"name": "C-05-B"}},
        ],
        "edges": [
            {"source": "floor_5", "target": "slab_5", "type": "CONTAINS_ELEMENT"},
            {"source": "floor_5", "target": "beam_5a", "type": "CONTAINS_ELEMENT"},
            {"source": "floor_5", "target": "beam_5b", "type": "CONTAINS_ELEMENT"},
            {"source": "beam_5a", "target": "col_5a", "type": "CONNECTED_TO"},
            {"source": "beam_5b", "target": "col_5b", "type": "CONNECTED_TO"},
            {"source": "slab_5", "target": "beam_5a", "type": "CONNECTED_TO"},
        ],
    },
    {
        "input": "What walls bound conference room B?",
        "nodes": [
            {"id": "conf_b", "label": "Space", "props": {"name": "Conference Room B"}},
            {"id": "wall_n", "label": "Wall", "props": {"name": "W-N-42", "material": "Concrete"}},
            {"id": "wall_s", "label": "Wall", "props": {"name": "W-S-42", "material": "Drywall"}},
            {"id": "wall_e", "label": "Wall", "props": {"name": "W-E-42", "material": "Glass"}},
            {"id": "door_b", "label": "Door", "props": {"name": "D-42", "width": 0.9}},
        ],
        "edges": [
            {"source": "conf_b", "target": "wall_n", "type": "BOUNDED_BY"},
            {"source": "conf_b", "target": "wall_s", "type": "BOUNDED_BY"},
            {"source": "conf_b", "target": "wall_e", "type": "BOUNDED_BY"},
            {"source": "conf_b", "target": "door_b", "type": "BOUNDED_BY"},
            {"source": "wall_e", "target": "door_b", "type": "HAS_OPENING"},
        ],
    },
    {
        "input": "Trace the chilled water loop from the cooling tower to the AHU",
        "nodes": [
            {"id": "ct_1", "label": "CoolingTower", "props": {"name": "CT-01"}},
            {"id": "chiller_1", "label": "Chiller", "props": {"name": "CH-01", "capacity_kw": 500}},
            {"id": "pump_chw", "label": "Pump", "props": {"name": "P-CHW-01"}},
            {"id": "ahu_2", "label": "AirHandlingUnit", "props": {"name": "AHU-02"}},
            {"id": "chw_system", "label": "ChilledWaterSystem", "props": {"name": "CHW System"}},
        ],
        "edges": [
            {"source": "ct_1", "target": "chiller_1", "type": "FEEDS"},
            {"source": "pump_chw", "target": "chiller_1", "type": "FEEDS"},
            {"source": "chiller_1", "target": "ahu_2", "type": "FEEDS"},
            {"source": "chiller_1", "target": "chw_system", "type": "PART_OF"},
            {"source": "ct_1", "target": "chw_system", "type": "PART_OF"},
            {"source": "pump_chw", "target": "chw_system", "type": "PART_OF"},
        ],
    },
    {
        "input": "What sensors monitor the data center room?",
        "nodes": [
            {"id": "dc_room", "label": "Room", "props": {"name": "Data Center", "area": 200.0}},
            {"id": "ts_dc", "label": "TemperatureSensor", "props": {"name": "TS-DC-01"}},
            {"id": "hs_dc", "label": "HumiditySensor", "props": {"name": "HS-DC-01"}},
            {"id": "occ_dc", "label": "OccupancySensor", "props": {"name": "OCC-DC-01"}},
            {"id": "em_dc", "label": "ElectricalMeter", "props": {"name": "EM-DC-01"}},
        ],
        "edges": [
            {"source": "ts_dc", "target": "dc_room", "type": "IS_POINT_OF"},
            {"source": "hs_dc", "target": "dc_room", "type": "IS_POINT_OF"},
            {"source": "occ_dc", "target": "dc_room", "type": "IS_POINT_OF"},
            {"source": "em_dc", "target": "dc_room", "type": "IS_POINT_OF"},
        ],
    },
    {
        "input": "Show the building hierarchy from site to spaces",
        "nodes": [
            {"id": "site_1", "label": "Site", "props": {"name": "Campus North"}},
            {"id": "bldg_a", "label": "Building", "props": {"name": "Building A"}},
            {"id": "floor_1", "label": "BuildingStorey", "props": {"name": "Floor 1"}},
            {"id": "floor_2", "label": "BuildingStorey", "props": {"name": "Floor 2"}},
            {"id": "lobby", "label": "Space", "props": {"name": "Lobby"}},
            {"id": "office_201", "label": "Space", "props": {"name": "Office 201"}},
        ],
        "edges": [
            {"source": "site_1", "target": "bldg_a", "type": "HAS_BUILDING"},
            {"source": "bldg_a", "target": "floor_1", "type": "HAS_STOREY"},
            {"source": "bldg_a", "target": "floor_2", "type": "HAS_STOREY"},
            {"source": "floor_1", "target": "lobby", "type": "HAS_SPACE"},
            {"source": "floor_2", "target": "office_201", "type": "HAS_SPACE"},
        ],
    },
    {
        "input": "Which fire safety systems protect floor 2?",
        "nodes": [
            {"id": "floor_2b", "label": "BuildingStorey", "props": {"name": "Floor 2"}},
            {"id": "fire_sys", "label": "FireSafetySystem", "props": {"name": "Fire System F2"}},
            {"id": "smoke_det", "label": "TemperatureSensor", "props": {"name": "SD-F2-01", "sensor_type": "smoke"}},
            {"id": "sprinkler", "label": "Sprinkler", "props": {"name": "SPR-F2-01"}},
        ],
        "edges": [
            {"source": "fire_sys", "target": "floor_2b", "type": "SERVES"},
            {"source": "smoke_det", "target": "floor_2b", "type": "IS_POINT_OF"},
            {"source": "sprinkler", "target": "fire_sys", "type": "PART_OF"},
        ],
    },
    {
        "input": "What is the energy distribution path for wing B?",
        "nodes": [
            {"id": "wing_b", "label": "Wing", "props": {"name": "Wing B"}},
            {"id": "elec_sys", "label": "ElectricalSystem", "props": {"name": "Electrical Main"}},
            {"id": "transformer_b", "label": "Transformer", "props": {"name": "TX-B-01"}},
            {"id": "switchgear_b", "label": "Switchgear", "props": {"name": "SG-B-01"}},
            {"id": "meter_b", "label": "ElectricalMeter", "props": {"name": "EM-B-01"}},
        ],
        "edges": [
            {"source": "transformer_b", "target": "elec_sys", "type": "PART_OF"},
            {"source": "switchgear_b", "target": "elec_sys", "type": "PART_OF"},
            {"source": "transformer_b", "target": "switchgear_b", "type": "FEEDS"},
            {"source": "meter_b", "target": "wing_b", "type": "IS_POINT_OF"},
            {"source": "elec_sys", "target": "wing_b", "type": "SERVES"},
        ],
    },
    {
        "input": "Show pipe connections in the plant room",
        "nodes": [
            {"id": "plant_room", "label": "Space", "props": {"name": "Plant Room"}},
            {"id": "pipe_supply", "label": "Pipe", "props": {"name": "P-SUP-01", "diameter": 150}},
            {"id": "pipe_return", "label": "Pipe", "props": {"name": "P-RET-01", "diameter": 150}},
            {"id": "pump_1", "label": "Pump", "props": {"name": "P-01"}},
            {"id": "valve_1", "label": "Valve", "props": {"name": "V-01"}},
        ],
        "edges": [
            {"source": "plant_room", "target": "pipe_supply", "type": "CONTAINS_ELEMENT"},
            {"source": "plant_room", "target": "pipe_return", "type": "CONTAINS_ELEMENT"},
            {"source": "pipe_supply", "target": "pipe_return", "type": "CONNECTED_TO"},
            {"source": "pump_1", "target": "pipe_supply", "type": "CONNECTED_TO"},
            {"source": "valve_1", "target": "pipe_return", "type": "CONNECTED_TO"},
        ],
    },
]


def build_ontology_pairs(
    classes: list[dict],
    relationships: list[dict],
    source_name: str,
) -> list[dict]:
    """Convert ontology classes + relationships into training pairs."""
    class_map = {c["id"]: c for c in classes}

    # Build edge maps
    forward: dict[str, list[dict]] = {}
    reverse: dict[str, list[dict]] = {}
    for rel in relationships:
        forward.setdefault(rel["source"], []).append(rel)
        reverse.setdefault(rel["target"], []).append(rel)

    pairs = []
    for cls in classes:
        cls_id = cls["id"]
        label = cls["label"]
        description = cls.get("description", "")
        kind = cls.get("kind", "")

        input_text = f"{label}: {description}" if description else label

        fwd = forward.get(cls_id, [])
        rev = reverse.get(cls_id, [])

        edges = []
        target_nodes = []
        for rel in fwd:
            tgt = class_map.get(rel["target"])
            if not tgt:
                continue
            edges.append({
                "source": cls_id,
                "target": rel["target"],
                "type": rel["type"],
                "weight": EDGE_WEIGHTS.get(rel["type"], 0.5),
            })
            target_nodes.append({
                "id": rel["target"],
                "label": tgt["label"],
                "properties": {"description": tgt.get("description", ""), "kind": tgt.get("kind", "")},
            })

        reverse_edges = []
        for rel in rev:
            reverse_edges.append({
                "source": rel["source"],
                "target": cls_id,
                "type": rel["type"],
                "weight": EDGE_WEIGHTS.get(rel["type"], 0.5),
            })

        pair = {
            "id": cls_id,
            "input_text": input_text,
            "metadata": {
                "domain": "construction",
                "source": source_name,
                "class_label": label,
                "kind": kind,
            },
            "expected_graph": {
                "source_node": {
                    "id": cls_id,
                    "label": label,
                    "properties": {
                        "description": description,
                        "kind": kind,
                    },
                },
                "target_nodes": target_nodes,
                "edges": edges,
                "reverse_edges": reverse_edges,
            },
            "quality": {
                "edge_count": len(edges),
                "reverse_edge_count": len(reverse_edges),
                "total_edges": len(edges) + len(reverse_edges),
                "text_length": len(input_text),
                "verified_source": source_name,
            },
        }
        pairs.append(pair)

    return pairs


def build_nl_pairs(examples: list[dict]) -> list[dict]:
    """Convert NL-to-graph examples into training pairs."""
    pairs = []
    for i, ex in enumerate(examples):
        nodes = ex["nodes"]
        edges_raw = ex["edges"]

        source_node = nodes[0]
        target_nodes = nodes[1:]

        edges = []
        for e in edges_raw:
            edges.append({
                "source": e["source"],
                "target": e["target"],
                "type": e["type"],
                "weight": EDGE_WEIGHTS.get(e["type"], 0.5),
            })

        pair = {
            "id": f"construction_nl_{i:03d}",
            "input_text": ex["input"],
            "metadata": {
                "domain": "construction",
                "source": "nl_construction",
                "query_type": "natural_language",
            },
            "expected_graph": {
                "source_node": {
                    "id": source_node["id"],
                    "label": source_node["label"],
                    "properties": source_node.get("props", {}),
                },
                "target_nodes": [
                    {
                        "id": n["id"],
                        "label": n["label"],
                        "properties": n.get("props", {}),
                    }
                    for n in target_nodes
                ],
                "edges": edges,
                "reverse_edges": [],
            },
            "quality": {
                "edge_count": len(edges),
                "reverse_edge_count": 0,
                "total_edges": len(edges),
                "text_length": len(ex["input"]),
                "verified_source": "expert_curated",
            },
        }
        pairs.append(pair)

    return pairs


def fetch_bsdd_classes(cache_dir: Path, max_classes: int = 200) -> list[dict]:
    """Fetch construction product classifications from bSDD API."""
    cache_file = cache_dir / "bsdd_classes.json"
    if cache_file.exists():
        print("  Using cached bSDD data")
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)

    print("  Fetching bSDD classifications...")
    cache_dir.mkdir(parents=True, exist_ok=True)

    classes: list[dict] = []
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        # Search for common construction terms
        search_terms = [
            "wall", "floor", "roof", "beam", "column", "slab",
            "door", "window", "pipe", "duct", "insulation",
            "concrete", "steel", "timber", "brick", "glass",
            "foundation", "stair", "ramp", "elevator",
        ]

        for term in search_terms:
            if len(classes) >= max_classes:
                break
            try:
                resp = client.get(
                    BSDD_SEARCH,
                    params={"SearchText": term, "take": 20},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get("classes", data.get("results", [])):
                        cls_name = item.get("name", "")
                        cls_uri = item.get("uri", item.get("namespaceUri", ""))
                        if cls_name and len(classes) < max_classes:
                            classes.append({
                                "id": f"bsdd_{cls_name.replace(' ', '_').lower()[:40]}",
                                "label": cls_name.replace(" ", ""),
                                "kind": "ConstructionProduct",
                                "description": item.get("definition", cls_name),
                                "uri": cls_uri,
                            })
                time.sleep(0.5)  # Be nice to the API
            except Exception as e:
                print(f"  Warning: bSDD search for '{term}' failed: {e}")
                continue

    # Deduplicate by label
    seen = set()
    unique = []
    for c in classes:
        if c["label"] not in seen:
            seen.add(c["label"])
            unique.append(c)
    classes = unique

    # Cache
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2, ensure_ascii=False)

    print(f"  Fetched {len(classes)} bSDD classes")
    return classes


def build_all_construction_pairs(cache_dir: Path, max_bsdd: int = 200) -> list[dict]:
    """Build all construction training pairs from BOT + Brick + IFC + bSDD + NL."""
    all_pairs: list[dict] = []

    # 1. BOT ontology
    print("\n  Building BOT (Building Topology) pairs...")
    bot_pairs = build_ontology_pairs(BOT_CLASSES, BOT_RELATIONSHIPS, "bot_ontology")
    all_pairs.extend(bot_pairs)
    print(f"    {len(bot_pairs)} pairs from BOT")

    # 2. Brick Schema
    print("  Building Brick Schema pairs...")
    brick_pairs = build_ontology_pairs(BRICK_CLASSES, BRICK_RELATIONSHIPS, "brick_schema")
    all_pairs.extend(brick_pairs)
    print(f"    {len(brick_pairs)} pairs from Brick")

    # 3. IFC core classes
    print("  Building IFC pairs...")
    ifc_pairs = build_ontology_pairs(IFC_CLASSES, IFC_RELATIONSHIPS, "ifc_schema")
    all_pairs.extend(ifc_pairs)
    print(f"    {len(ifc_pairs)} pairs from IFC")

    # 4. NL-to-graph examples
    print("  Building NL-to-graph construction pairs...")
    nl_pairs = build_nl_pairs(NL_CONSTRUCTION_EXAMPLES)
    all_pairs.extend(nl_pairs)
    print(f"    {len(nl_pairs)} pairs from NL examples")

    # 5. bSDD API (if available)
    try:
        print("  Fetching bSDD classifications...")
        bsdd_classes = fetch_bsdd_classes(cache_dir, max_classes=max_bsdd)
        if bsdd_classes:
            # Build simple classification pairs (no relationships between bSDD items)
            for cls in bsdd_classes:
                pair = {
                    "id": cls["id"],
                    "input_text": f"{cls['label']}: {cls['description']}",
                    "metadata": {
                        "domain": "construction",
                        "source": "bsdd",
                        "class_label": cls["label"],
                        "kind": cls.get("kind", ""),
                    },
                    "expected_graph": {
                        "source_node": {
                            "id": cls["id"],
                            "label": cls["label"],
                            "properties": {
                                "description": cls["description"],
                                "kind": cls.get("kind", ""),
                                "uri": cls.get("uri", ""),
                            },
                        },
                        "target_nodes": [],
                        "edges": [],
                        "reverse_edges": [],
                    },
                    "quality": {
                        "edge_count": 0,
                        "reverse_edge_count": 0,
                        "total_edges": 0,
                        "text_length": len(f"{cls['label']}: {cls['description']}"),
                        "verified_source": "bsdd_api",
                    },
                }
                all_pairs.append(pair)
            print(f"    {len(bsdd_classes)} pairs from bSDD")
    except Exception as e:
        print(f"  Warning: bSDD fetch failed: {e}. Continuing without bSDD data.")

    return all_pairs


def print_stats(pairs: list[dict]) -> None:
    """Print dataset statistics."""
    total = len(pairs)
    if total == 0:
        print("No pairs generated.")
        return

    by_source: dict[str, int] = {}
    by_label: dict[str, int] = {}
    by_edge_type: dict[str, int] = {}

    total_fwd = 0
    total_rev = 0

    for p in pairs:
        src = p["metadata"].get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1

        label = p["expected_graph"]["source_node"]["label"]
        by_label[label] = by_label.get(label, 0) + 1

        total_fwd += p["quality"]["edge_count"]
        total_rev += p["quality"]["reverse_edge_count"]

        for e in p["expected_graph"]["edges"]:
            et = e["type"]
            by_edge_type[et] = by_edge_type.get(et, 0) + 1
        for e in p["expected_graph"]["reverse_edges"]:
            et = e["type"]
            by_edge_type[et] = by_edge_type.get(et, 0) + 1

    avg_edges = (total_fwd + total_rev) / total if total else 0

    print(f"\n{'=' * 60}")
    print("CONSTRUCTION TRAINING PAIR STATISTICS")
    print(f"{'=' * 60}")
    print(f"  Total pairs:          {total}")
    print(f"  Total forward edges:  {total_fwd}")
    print(f"  Total reverse edges:  {total_rev}")
    print(f"  Avg edges per pair:   {avg_edges:.1f}")

    print("\n  By source:")
    for src, count in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"    {src}: {count}")

    print(f"\n  Node labels ({len(by_label)}):")
    for label, count in sorted(by_label.items(), key=lambda x: -x[1])[:20]:
        print(f"    {label}: {count}")

    if by_edge_type:
        print(f"\n  Edge types ({len(by_edge_type)}):")
        for et, count in sorted(by_edge_type.items(), key=lambda x: -x[1])[:15]:
            print(f"    {et}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build construction training pairs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-bsdd", type=int, default=200)
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    print("Building construction training pairs (BOT + Brick + IFC + bSDD + NL)...")

    pairs = build_all_construction_pairs(CACHE_DIR, max_bsdd=args.max_bsdd)
    print_stats(pairs)

    if args.stats_only:
        return

    # Write
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nWritten {len(pairs)} training pairs to {args.output}")
    print(f"File size: {args.output.stat().st_size / 1024:.1f} KB")
    print("\nSources: BOT (W3C CC-BY), Brick Schema (BSD), IFC (buildingSMART), bSDD (CC-BY)")


if __name__ == "__main__":
    main()
