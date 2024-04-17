import os
import re
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.types import Integer, Float, String, TIMESTAMP

import config
import config_treatment_effect as config_te

# Filter C,H,E,V, discrete, continuous columns 
ch_disc_p, ev_disc_p = re.compile('[CH][0-9][A-Z]*_'), re.compile('[EV][0-9][A-Z]*_')
c_cont_p, h_cont_p = re.compile('Containment'), re.compile('Health')
e_cont_p, v_cont_p = re.compile('Economic'), re.compile('Vaccine')

is_ch_disc = lambda col: ch_disc_p.search(col) is not None
is_ev_disc = lambda col: ev_disc_p.search(col) is not None
is_ch_cont = lambda col: (c_cont_p.search(col) is not None) or (h_cont_p.search(col) is not None)
is_ev_cont = lambda col: (e_cont_p.search(col) is not None) or (v_cont_p.search(col) is not None)
_filter_func_map = {True: {True: is_ch_disc, False: is_ch_cont}, False: {True: is_ev_disc, False: is_ev_cont}}

def filter_treatment(columns, is_CH=True, is_discrete=True):
    f = _filter_func_map[is_CH][is_discrete]
    return list(filter(f, columns))

def get_columns(category, all_data_columns_map, lag):
    keys = config.node_category_to_keys[category]
    func = lambda x,l:x
    if callable(keys[0]):
        func, keys = keys
    columns = np.concatenate([all_data_columns_map[k] for k in keys])
    return np.unique(
        np.concatenate([[func(col,l+1) for l in range(lag)] for col in columns])
    ).tolist()

def filter_data_to_column_map(data_to_column_map):
    for k,v in data_to_column_map.items():
        for k2,v2 in v.items():
            v2_dict = None
            if len(v2)==0: continue
            elif type(v2) == dict: continue
            if len(v2)>1: 
                if type(v2[1]) == dict:
                    v2,v2_dict = v2
            v2_filtered = list(filter(lambda col: all([c not in col for c in config.columns_to_remove]), v2))
            if v2_dict is None: data_to_column_map[k][k2] = v2_filtered
            else: data_to_column_map[k][k2] = (v2_filtered, v2_dict)
    return data_to_column_map


def load_artifacts(national_artifacts_dir):
    # Load artifacts (data, estimands, model)
    with open(os.path.join(national_artifacts_dir, 'subnational_processed_final_var_only_df.pkl'), 'rb') as f:
        subnational_processed_final_var_only_df = pickle.load(f)

    # Load other artifacts (node related)
    with open(os.path.join(national_artifacts_dir, 'node_artifacts.pkl'), 'rb') as f:
        node_artifacts = pickle.load(f)

    # (column_names_to_node_name, column_to_node_map, node_to_column_map,
    # outcome_ts_ahead_columns_map, treatment_type_to_artifacts, node_prefix_to_count) = node_artifacts

    treatment_type_to_artifacts = node_artifacts[4]
    for treat_type,type_artifacts_dict in treatment_type_to_artifacts.items():
        for dtype,_ in type_artifacts_dict.items():
            # 1. load model
            with open(os.path.join(national_artifacts_dir, f'causal_model_A{dtype}_{treat_type}.pkl'), 'rb') as f:
                treatment_type_to_artifacts[treat_type][dtype]['model'] = pickle.load(f)
            # Check that data matches
            model = treatment_type_to_artifacts[treat_type][dtype]['model']
            assert (model._data == subnational_processed_final_var_only_df).all().all(), 'Data does not match data in the loaded model. Please re-generate estimands.'

            # 2. load estimands
            treatment_type_to_artifacts[treat_type][dtype]['estimands'] = {}
            for estimand_name in np.unique(config_te.estimand_names):
                with open(os.path.join(national_artifacts_dir, f'{estimand_name}_A{dtype}_{treat_type}.pkl'), 'rb') as f:
                    treatment_type_to_artifacts[treat_type][dtype]['estimands'][estimand_name] = pickle.load(f)

    return subnational_processed_final_var_only_df, node_artifacts, treatment_type_to_artifacts


# Save results
def _save_results(results, path=None):
    results_info = pd.MultiIndex.from_tuples([x[0] for x in results]).to_frame()
    results_info.reset_index(drop=True, inplace=True)
    results_info.columns = ['wave_ix', 'Lwave_i', 'wave_i', \
                            'outcome_ix', 'Yi', 'outcome_i', 'ts_ahead', 'model', \
                            'treat_type', 'dtype', 'treatment_ix', 'Ai', 'treatment_i', \
                            'estimand_name', 'estimator_name', 'identified_estimand', \
                            'mediator_ix', 'Mi', 'mediator_i']
    
    def get_values(x):
        if x[0] is None: return [None, None]
        return [x[0], x[0].value]

    estimand_df = pd.DataFrame(
        [get_values(x) for _,x in results],
        columns=['estimand_object', 'estimand_estimate']
    )

    results_df = pd.concat((results_info, estimand_df), axis=1)

    if path is not None:
        with open(path, 'wb') as f:
            pickle.dump(results_df, f)

    return results_df

## Functions for loading data to databse

# Process columns
def _filter(col):
    replace_to_replaced = {'_': ['.', ',', ';', ':', ' ', '-'],
                           '': ['$', '(', ')']}
    for k,v in replace_to_replaced.items():
        for c in v: 
            col = col.replace(c, k)
    return col

def _preprocess(df, columns_to_drop):
    # Initial preprocessing
    df_ = df.copy()
    df_.drop(columns=columns_to_drop, inplace=True) # drop duplicate columns

    df_.columns = [_filter(col) for col in df_.columns]
    # columns_text = ', '.join(columns)
    return df_#, columns, columns_text

def _process_data_column_map(data_to_column_map):
    for k1,v1 in data_to_column_map.items():
        for k2,v2 in v1.items():
            if type(v2) == list: 
                data_to_column_map[k1][k2] = [_filter(col) for col in v2]
            elif type(v2) == tuple:
                data_to_column_map[k1][k2] = ([_filter(col) for col in v2[0]], v2[1])
            elif type(list(v2.values())[0]) == list: # dictionary with values being lists
                data_to_column_map[k1][k2] = {
                    _filter(k): [_filter(col) for col in v]  for k,v in v2.items()
                }
            else:
                data_to_column_map[k1][k2] = {
                    _filter(k): _filter(v)  for k,v in v2.items()
                } 

    return data_to_column_map

def _get_sql_type_map(df_):
    # Map dtypes to SQL types
    dtype_to_sql_type = {
        pd.Int64Dtype(): Integer,
        np.dtype('int64'): Integer,
        pd.Float64Dtype(): Float,
        np.dtype('float64'): Float,
        np.dtype('O'): String,
        np.dtype('<M8[ns]'): TIMESTAMP,
    }

    def _get_sql_type(dtype):
        Type = dtype_to_sql_type[dtype]
        if Type == TIMESTAMP: return Type(0)
        return Type()

    dtype_map = {k: _get_sql_type(v) for k,v in df_.dtypes.items()}
    for k,v in df_.dtypes.items():
        if v==String: del dtype_map[k]

    return dtype_map

def _load_data_to_db(df_, table_name, db_path, dtype_map):
    # Establish connection
    engine = create_engine(f'sqlite:///{db_path}', echo=False)

    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE if exists {table_name};")) # drop table, then rewrite

    df_.to_sql(name=table_name, con=engine, index=False,
               dtype=dtype_map)
    return engine

def load_data_to_db(df, columns_to_drop, table_name, db_path):
    df_ = _preprocess(df, columns_to_drop)
    dtype_map = _get_sql_type_map(df_)
    engine = _load_data_to_db(df_, table_name, db_path, dtype_map)

    return df_, engine
