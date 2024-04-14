""" 
Columns to remove based on:
- high rates of missingness
- if high overlap/correlation with another policy indicator, only keep one(s) that are most general
- if already summarized in policy index 

Note:
- Some containment and health policies have differentiation by vaccine passport. 
  We've included Mv -> Ach. 
"""
import numpy as np


columns_to_remove = [
    'C1M_School.closing',
    'C2M_Workplace.closing',
    'C3M_Cancel.public.events',
    'C4M_Restrictions.on.gatherings',
    'C5M_Close.public.transport',
    'C6M_Stay.at.home.requirements',
    'C7M_Restrictions.on.internal.movement',
    'C8EV_International.travel.controls', # Containment correlated, summarized in C&H policy index
    'E1_Income.support',
    'E2_Debt.contract.relief', # Summarized in Economic policy index
    'H1_Public.information.campaigns',
    'H2_Testing.policy',
    'H3_Contact.tracing', # Contained in C&H policy index
    'E3_Fiscal.measures', # Remaining were missing, high overlap with an included indicator, or low correlation.
    'GovernmentResponseIndex_Average',
    'StringencyIndex_Average', # Highly correlated with Containment&Health index
    'E4_International.support', 
    'H4_Emergency.investment.in.healthcare',
    'H5_Investment.in.vaccines',
    'V1_Vaccine.Prioritisation..summary.',
    'V2A_Vaccine.Availability..summary.',
    # 'V2B_Vaccine.age.eligibility.availability.age.floor..general.population.summary.',
    'V2C_Vaccine.age.eligibility.availability.age.floor..at.risk.summary.',
    'V2D_Medically..clinically.vulnerable..Non.elderly.',
    'V2E_Education',
    'V2F_Frontline.workers...non.healthcare.',
    'V2G_Frontline.workers...healthcare.',
    'V3_Vaccine.Financial.Support..summary.',
    'H6M_Facial.Coverings', #other, low correlation
    'H8M_Protection.of.elderly.people', 
    'parks_percent_change_from_baseline',
    'residential_percent_change_from_baseline',
    # 'transit_stations_percent_change_from_baseline',
    # 'retail_and_recreation_percent_change_from_baseline',
    # 'grocery_and_pharmacy_percent_change_from_baseline',
    # 'workplaces_percent_change_from_baseline'
]

flags_to_include = [
    'OxCGRT_H7_Flag',
]

# Maps node categories (e.g. L, A, T, M) to lists of keys to index into all_data_columns_map
# Note: lagged categories have functions as well, to map columns to their lagged values
node_category_to_keys = dict(
    tconfounders_disc = ['OxCGRT_policy_columns'],
    tconfounders_cont = ['OxCGRT_policy_index_columns'],
    treatments_disc = ['OxCGRT_policy_intervention_columns_discrete'],
    treatments_cont = ['OxCGRT_policy_intervention_columns_continuous'],
    baselines = ['Demographic_pct_columns', 'Demographic_Median_income', \
                 'Misc_weighted_population_density', 'Misc_political', \
                 'Hospital_static_per_100_000_columns'],
    confounders = ['Hospital_dynamic_per_100_000_columns', 'Temporal_temporal', 'Wave_wave'],
    mediators_mobility = ['Mobility_mobility'],
    mediators_vax = ['OxCGRT_vax_status_columns'],
    lagged_mobility_confounders = (lambda x,l:f'{x}_{l}', ['Mobility_mobility']),
    lagged_outcome_confounders = (lambda x,l: f'{x}_{l}', ['Mobility_mobility']), 
    outcomes = ['OxCGRT_outcome_ts_ahead_columns'], 
    outcomes_raw = ['OxCGRT_outcome_columns', 'Mobility_mobility'], # used to create lagged values, not for final model
)

node_category_to_prefix = dict(
    tconfounders_disc = 'Sdisc',
    tconfounders_cont = 'Scont',
    treatments_disc = 'Adisc',
    treatments_cont = 'Acont',
    baselines = 'B',
    confounders = 'L',
    mediators_mobility = 'Mm',
    mediators_vax = 'Mv',
    lagged_mobility_confounders = 'Lm',
    lagged_outcome_confounders = 'LY', 
    outcomes = 'Y',
)
node_prefix_to_category = dict(zip(node_category_to_prefix.values(), node_category_to_prefix.keys()))

node_category_to_dummy_boolean = dict(
    tconfounders_disc = False,
    tconfounders_cont = False,
    treatments_disc = True,
    treatments_cont = True,
    baselines = False,
    confounders = False,
    mediators_mobility = True,
    mediators_vax = True,
    lagged_mobility_confounders = False,
    lagged_outcome_confounders = False,
    outcomes = True,
)

# node prefixes to remove, if looking at effect of e.g. Acont_ch on outcome
treatment_type_to_node_prefixes_to_remove = dict(
    ch=dict(
        cont=['Mv', 'Adisc_ev', 'Adisc_ch', 'Acont_ev'],
        disc=['Mv', 'Adisc_ev', 'Acont_ch', 'Acont_ev'],
    ),
    ev=dict(
        cont=['Mm', 'Adisc_ch', 'Adisc_ev', 'Acont_ch'],
        disc=['Mm', 'Adisc_ch', 'Acont_ev', 'Acont_ch'],
    )
)

treatment_type_to_mediator = dict(
    ch='Mm',
    ev='Mv'
)

# scaled control, scaled treated, unscaled control, unscaled treated
treatment_dtype_to_control_treated_values = dict(
    disc=(0,1,0,1),
    cont=(0,0.05,0,5)
)


# Full set of edges 
edges = [
    ('B', 'Adisc_ch'), ('B', 'Acont_ch'), ('B', 'Adisc_ev'), ('B', 'Acont_ev'), ('B', 'Mm'), ('B', 'Mv'), ('B', 'Y'), 
    ('L', 'Adisc_ch'), ('L', 'Acont_ch'), ('L', 'Adisc_ev'), ('L', 'Acont_ev'), ('L', 'Mm'), ('L', 'Mv'), ('L', 'Y'), 
    ('Lm', 'Adisc_ch'), ('Lm', 'Acont_ch'), ('Lm', 'Adisc_ev'), ('Lm', 'Acont_ev'),
    ('LY', 'Adisc_ch'), ('LY', 'Acont_ch'), ('LY', 'Adisc_ev'), ('LY', 'Acont_ev'), ('LY', 'Y'), 
    ('Sdisc_ch', 'Adisc_ch'), ('Sdisc_ch', 'Acont_ch'), ('Sdisc_ch', 'Adisc_ev'), ('Sdisc_ch', 'Acont_ev'), ('Sdisc_ch', 'Mm'), ('Sdisc_ch', 'Y'), 
    ('Sdisc_ev', 'Adisc_ch'), ('Sdisc_ev', 'Acont_ch'), ('Sdisc_ev', 'Adisc_ev'), ('Sdisc_ev', 'Acont_ev'), ('Sdisc_ev', 'Mv'), ('Sdisc_ev', 'Y'), 
    ('Scont_ch', 'Adisc_ch'), ('Scont_ch', 'Acont_ch'), ('Scont_ch', 'Adisc_ev'), ('Scont_ch', 'Acont_ev'), ('Scont_ch', 'Mm'), ('Scont_ch', 'Y'), 
    ('Scont_ev', 'Adisc_ch'), ('Scont_ev', 'Acont_ch'), ('Scont_ev', 'Adisc_ev'), ('Scont_ev', 'Acont_ev'), ('Scont_ev', 'Mv'), ('Scont_ev', 'Y'), 
    ('Adisc_ch', 'Mm'), ('Adisc_ch', 'Y'), 
    ('Acont_ch', 'Mm'), ('Acont_ch', 'Y'), 
    ('Adisc_ev', 'Mv'), ('Adisc_ev', 'Y'), 
    ('Acont_ev', 'Mv'), ('Acont_ev', 'Y'), 
    ('Mv', 'Adisc_ch'), ('Mv', 'Acont_ch'), ('Mv', 'Y'),
    ('Mm', 'Y')
]

