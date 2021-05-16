import os

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_PATH = os.path.join(PROJECT_PATH, 'dataset')

WSS_PATH = os.path.join(DATASET_PATH, 'wss')
HORIZON_DB_FILE_PATH = os.path.join(WSS_PATH, 'chorizon.csv')
COMPONENT_DB_FILE_PATH = os.path.join(WSS_PATH, 'component.csv')

DB_OM = 'Organic Matter concentration'
DB_COMP_ID = 'Component ID'
DB_MU_ID = 'Map Unit ID'

SPATIAL_DATA_PATH = os.path.join(WSS_PATH, 'spatial')
SPATIAL_DBF_FILE_PATH = os.path.join(SPATIAL_DATA_PATH, 'gsmsoilmu_a_al.dbf')
SPATIAL_SHP_FILE_PATH = os.path.join(SPATIAL_DATA_PATH, 'gsmsoilmu_a_al.shp')
SPATIAL_SHX_FILE_PATH = os.path.join(SPATIAL_DATA_PATH, 'gsmsoilmu_a_al.shx')

YANDEX_MAPS_MAX_WIDTH = 650
YANDEX_MAPS_MAX_HEIGHT = 450
