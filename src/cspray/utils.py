from pyspark.sql import DataFrame
from pyspark import StorageLevel
import h5py


import os

def get_gene_table(species='hsapiens'):
    """
    Download gene reference table from Ensembl.
    
    Args:
        species: Species identifier (e.g., 'hsapiens', 'mmusculus', 'rnorvegicus')
    
    Returns:
        DataFrame with ensembl_gene_id and external_gene_name columns
    """
    from pybiomart import Dataset
    print(f"Querying Ensembl for {species}...")
    dataset = Dataset(name=f'{species}_gene_ensembl',
                    host='http://www.ensembl.org')
    
    gene_table = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name'])
    
    # Rename BioMart's human-readable column names to code-friendly names
    gene_table.rename(columns={
        'Gene stable ID': 'gene_id',
        'Gene name': 'gene_name'
    }, inplace=True)
    
    print(f"Retrieved {len(gene_table)} genes for {species}")

    return gene_table

def get_storage_level(level_str):
    return getattr(StorageLevel, level_str)

def materialize(df: DataFrame) -> None:
    """
    Small helper to force the cache to be filled.
    """
    df.foreach(lambda _: None)   # no‑op, trigger evaluation

# def _h5ad_status_checks(paths):
#     main_csr_status = True
#     raw_csr_status  = True 
#     raw_status = True
#     fallback_status = True
#     for p in paths:
#         file = h5py.File(p, 'r')
#         main_fail=False # temp...
        
#         if file['X'].attrs['encoding-type']!='csr_matrix':
#             main_csr_status = False # one of the mains does not have csr_matrix
#             main_fail = True
#         if 'raw' in file.keys():
#             if file['raw']['X'].attrs['encoding-type']!='csr_matrix':
#                 raw_csr_status = False # raw not csr in this one - at least one raw fails
#                 if main_fail:
#                     fallback_status = False # main for this path not having csr and raw not having csr either - fallback_fail in at least one file
#         else:
#             raw_status = False

#     return raw_status, raw_csr_status, main_csr_status, fallback_status

# I need to add a check - does ['data'] exist in 'X' or does an erro get thronw
# - could do try/except on file['X']['data'].shape[0] (same for raw) and fail if any issues... 

# check does X have keys (in default and raw)
def _h5ad_status_checks(paths):
    main_csr_status = []
    raw_csr_status  = []
    raw_status = []
    fallback_status = []
    for p in paths:
        file = h5py.File(p, 'r')
        main_csr_status.append(file['X'].attrs['encoding-type']=='csr_matrix')
        if 'raw' in file.keys():
            raw_status.append(True)
            raw_csr_status.append(file['raw']['X'].attrs['encoding-type']=='csr_matrix')
        else:
            raw_status.append(False)
            raw_csr_status.append(False)
        fallback_status.append(
            (raw_csr_status[-1] or ( (not raw_status[-1]) and main_csr_status[-1]))
        )
    all_raw_csr = all(raw_status)
    all_main_csr = all(main_csr_status)
    all_fallback_csr = all(fallback_status)

    # get paths (basedon indices) where each array has False entries

    # Identify indices where each status is False
    raw_csr_status_false_paths = [paths[i] for i, val in enumerate(raw_csr_status) if not val]
    main_csr_status_false_paths = [paths[i] for i, val in enumerate(main_csr_status) if not val]
    fallback_status_false_paths = [paths[i] for i, val in enumerate(fallback_status) if not val]

    statuses = {
        'raw_csr': {
            'bool': all_raw_csr,
            'paths': raw_csr_status_false_paths
        },
        'main_csr': {
            'bool': all_main_csr,
            'paths': main_csr_status_false_paths
        },
        'fallback': {
            'bool': all_fallback_csr,
            'paths': fallback_status_false_paths
        }
    }
    return statuses
    

def h5ad_format_check(df, from_raw, fallback_default):
    statuses = _h5ad_status_checks(list(df.toPandas().file_path.values))
    print("h5ad status output : ", statuses)
    if from_raw:
        if fallback_default:
            if not statuses['fallback']['bool']:
                raise ValueError(f"at least one file has an X in main thats falling back to from raw without a CSR format (CSC on Roadmap) ;  bad files: {statuses['fallback']['paths']}")
        else:
            if not statuses['raw_csr']['bool']:
                raise ValueError(f"Not all h5ad files have a raw group with a CSR format (CSC on Roadmap). Can try fallback_default=False to use the raw data instead on files ; bad files: {statuses['raw_csr']['paths']}")
    else:
        if not statuses['main_csr']['bool']:
            raise ValueError(f"Not all h5ad files have a default X with a CSR format (CSC on Roadmap). Can try from_raw=True to use the raw data instead on files ; bad files: {statuses['main_csr']['paths']}")



# def h5ad_format_check(df, from_raw, fallback_default):
#     raw_status, raw_csr_status, main_csr_status, fallback_status = _h5ad_status_checks(list(df.toPandas().file_path.values))
#     print("h5ad status output : ", raw_status, raw_csr_status, main_csr_status, fallback_status)
#     if from_raw:
#         if not raw_status: # not all files have raw
#             if not fallback_default: # no fallback and missing raws...
#                 raise ValueError("Not all h5ad files have a raw group. Set fallback_default=True to use the default data instead (not from_raw). Or ensure your h5ad files have a raw group.")
#             else: # missing a raw but was fallback an option
#                 if not fallback_status:
#                     raise ValueError("Not all h5ad files have a default X with a CSR format (CSC on Roadmap). Set fallback_default=False to use the raw data instead.")
#                 # else: # I am falling back to default, does main have csr...
#                 #     if not main_csr_status:
#                 #         raise ValueError("Falling back to default, but not all h5ad files have a default X with a CSR format (CSC on Roadmap)")
#         else: # all did have raw
#             if not raw_csr_status:
#                 raise ValueError("Not all h5ad files have a raw csr matrix (CSC on roadmap). Set fallback_default=True to use the default data instead (not from_raw). Or ensure your h5ad files have a raw csr matrix.")
#     else:
#         if not main_csr_status:
#             raise ValueError("Not all h5ad files have a default X with a CSR format (CSC on Roadmap) - you could try from_raw to use raw data if available in the h5ad files")
                