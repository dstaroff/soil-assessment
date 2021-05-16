import numpy as np
import pandas as pd

from util import Const


def get_soil_data():
    horizon = load_horizon_table()
    component = load_component_table()

    remove_nan_from_table(horizon)
    remove_nan_from_table(component)

    data = component.join(
        horizon,
        how='inner',
    ).copy()

    del horizon, component

    data.sort_values(
        by=[Const.DB_MU_ID, Const.DB_OM],
        ascending=False,
        inplace=True,
    )
    data.drop_duplicates(subset=[Const.DB_MU_ID], keep='first', inplace=True)
    data.reset_index(drop=True, inplace=True)

    return data


def load_horizon_table():
    horizon = pd.read_csv(
        Const.HORIZON_DB_FILE_PATH,
        usecols=['cokey', 'om_r'],
    )
    horizon.rename(
        columns={
            'cokey': Const.DB_COMP_ID,
            'om_r': Const.DB_OM,
        },
        inplace=True,
    )

    horizon[Const.DB_COMP_ID] = horizon[Const.DB_COMP_ID].astype(np.uint32)
    horizon[Const.DB_OM] = horizon[Const.DB_OM].astype(np.float32)
    horizon[Const.DB_OM] = horizon[Const.DB_OM] / 100
    horizon.set_index(Const.DB_COMP_ID, inplace=True)

    return horizon


def load_component_table():
    component = pd.read_csv(
        Const.COMPONENT_DB_FILE_PATH,
        usecols=['cokey', 'mukey'],
    )
    component.rename(
        columns={
            'cokey': Const.DB_COMP_ID,
            'mukey': Const.DB_MU_ID,
        },
        inplace=True,
    )

    component[Const.DB_COMP_ID] = component[Const.DB_COMP_ID].astype(np.uint32)
    component[Const.DB_MU_ID] = component[Const.DB_MU_ID].astype(np.uint32)
    component.set_index(Const.DB_COMP_ID, inplace=True)

    return component


def remove_nan_from_table(table):
    table.dropna(inplace=True)
    table.replace('', np.nan, inplace=True)
    table.dropna(inplace=True)
