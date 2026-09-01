# Import and process annotation files from BDF (made after SSPlab ran a first version avec the coding pipeline)

#%%
import os

import duckdb

S3_PREFIX = "s3://projet-budget-famille/data/annotations_vague1_2026"
PATTERNS = ("carnets_papier", "p1")

con = duckdb.connect(database=":memory:")

con.execute(
    f"""
CREATE SECRET secret_s3 (
    TYPE S3,
    KEY_ID '{os.environ["AWS_ACCESS_KEY_ID"]}',
    SECRET '{os.environ["AWS_SECRET_ACCESS_KEY"]}',
    ENDPOINT '{os.environ["AWS_S3_ENDPOINT"]}',
    SESSION_TOKEN '{os.environ["AWS_SESSION_TOKEN"]}',
    REGION 'us-east-1',
    URL_STYLE 'path',
    SCOPE 's3://projet-budget-famille/'
);
"""
)

#%%
# Keep the files whose name contains any of PATTERNS (a file matching several
# patterns is only listed once, so it is not stacked twice)
all_files = [
    row[0]
    for row in con.execute(
        f"select file from glob('{S3_PREFIX}/*.csv') order by file"
    ).fetchall()
]
files = [f for f in all_files if any(p in os.path.basename(f) for p in PATTERNS)]

print(f"{len(files)}/{len(all_files)} files selected:")
for file in files:
    print(f"  {os.path.basename(file)}")

#%%
# union_by_name: carnets and tickets files do not share the exact same columns
# (DATE_SAISIE / JOUR_COLL / NUM_CARN vs SAISIE / ID_TICK / code_mag), missing
# ones are filled with NULL. filename keeps track of each row's origin.
# nullstr='NA': MONT_DEP (110 rows) and INTERNET (104) hold the literal string
# "NA", which would type both columns as VARCHAR and break the budget
# aggregation downstream. The empty string has to be listed explicitly:
# nullstr *replaces* the default marker list rather than extending it, so
# nullstr='NA' alone would stop reading empty fields as NULL.
annotations = con.execute(
    """
    select * from read_csv(
        $files, union_by_name = true, filename = true, nullstr = ['NA', '']
    )
    """,
    {"files": files},
).df()

print(f"{annotations.shape[0]} rows, {annotations.shape[1]} columns")
annotations.head()

#%%
# Row count per imported file, to check the stacking against the source files
rows_per_file = con.execute(
    """
    select
        coalesce(regexp_extract(filename, '[^/]+$'), 'TOTAL') as file,
        count(*) as n
    from annotations
    group by rollup (filename)
    order by grouping(filename), file
    """
).df()
print(rows_per_file.to_string(index=False))

#%%
# Missing values in the two code columns. Beyond true NULLs, the LLM/regex steps
# write sentinel strings ("N/A", "Reprise manuelle") that pandas and DuckDB treat
# as ordinary values: anything not matching the COICOP shape 99[.9]* counts as
# missing too. DuckDB reads the `annotations` DataFrame in place (replacement
# scan, no copy), so keeping the data in pandas costs nothing here.
COICOP_RE = "^[0-9]{2}([.][0-9])*$"


def missing_counts(column: str) -> str:
    """NULL / sentinel counts for one code column, as a SQL select."""
    return f"""
        select
            '{column}' as column_name,
            count(*) as n,
            count(*) filter ({column} is null) as nulls,
            count(*) filter (
                {column} is not null
                and not regexp_matches({column}, '{COICOP_RE}')
            ) as sentinels,
            round(100.0 * count(*) filter (
                {column} is null
                or not regexp_matches({column}, '{COICOP_RE}')
            ) / count(*), 3) as pct_missing
        from annotations
    """


missing = con.execute(
    f"{missing_counts('code')} union all {missing_counts('predicted_code')}"
).df()
print(missing.to_string(index=False))

#%%
# Which sentinel values exactly, and which step produced them?
sentinels = con.execute(
    f"""
    select
        coalesce(predicted_code, '<NULL>') as predicted_code,
        coalesce(prediction_source, '<NULL>') as prediction_source,
        count(*) as n
    from annotations
    where predicted_code is null
       or not regexp_matches(predicted_code, '{COICOP_RE}')
    group by all
    order by n desc
    """
).df()
print(sentinels.to_string(index=False))

#%%
# Per-file breakdown, to check the missingness is not concentrated in one file
per_file = con.execute(
    f"""
    select
        regexp_extract(filename, '[^/]+$') as file,
        count(*) as n,
        count(*) filter (
            code is null or not regexp_matches(code, '{COICOP_RE}')
        ) as code_missing,
        count(*) filter (
            predicted_code is null
            or not regexp_matches(predicted_code, '{COICOP_RE}')
        ) as predicted_code_missing
    from annotations
    group by all
    order by file
    """
).df()
print(per_file.to_string(index=False))

# %%

#%%
# Level-4 accuracy, overall and broken down by the level-1 division of the
# annotated code. In this nomenclature every level past the first is one digit
# ("01" -> "01.1" -> "01.1.3" -> "01.1.3.1"), so level 4 is exactly the first 8
# characters and level 1 the first 2.
#
# Sentinels ("N/A", "Reprise manuelle") are kept and counted as errors: the
# pipeline failing to answer is a failure. `no_prediction` reports how many of
# them each row group carries, so the two can be told apart.
#
# `accuracy_at_annotation_level` truncates both codes to the depth of the
# annotation instead of a fixed 4 levels: 2 605 annotations stop before level 4,
# and for 600 of them the prediction agrees over the annotation's whole length
# but is *more precise*, which the strict level-4 comparison counts as wrong.
accuracy = con.execute(
    f"""
    select
        coalesce(code[:2], 'TOTAL') as coicop_level_1,
        count(*) as n,
        count(*) filter (
            predicted_code is null
            or not regexp_matches(predicted_code, '{COICOP_RE}')
        ) as no_prediction,
        round(avg((code[:8] = predicted_code[:8])::int), 4) as accuracy_level_4,
        round(avg((
            code[:least(len(code), 8)] = predicted_code[:least(len(code), 8)]
        )::int), 4) as accuracy_at_annotation_level
    from annotations
    group by rollup (code[:2])
    order by grouping(code[:2]), coicop_level_1
    """
).df()
print(accuracy.to_string(index=False))

# %%

#%%
# Export the stacked annotations back to S3, shaped to be fed to the
# codification pipeline as a new input file.
#
# "Added by the codification pipeline" is derived, not hardcoded: the union of the
# column names of the source files under data/workflow_inputs/ is the native BDF
# schema, so anything else was produced by the codification run or by the manual
# verification that followed, and gets the "_previous" suffix.
#
# `product` is the one exception: it holds the original NAT_DEP label (equal on
# 2 019 of the 2 030 rows joined against BDF_data_tickets_appli_20260527_a_codif),
# so it is renamed back to NAT_DEP, the text column the pipeline reads by default.
import re

S3_WORKFLOW_INPUTS = "s3://projet-budget-famille/data/workflow_inputs"
EXPORT_PATH = f"{S3_PREFIX}/as_new_input/annotations_vague1_2026_a_codif.parquet"

# Canonical names build-datasets/main.py creates or overwrites in prediction mode.
# Any of these left in the input file would be read as ground truth (`code`,
# `coicop` are only defaulted when absent) or silently clobbered (`id`).
PIPELINE_RESERVED = {
    "raw_product", "l_pr_product", "s_pr_product", "source", "annee", "code",
    "coicop", "shop", "shop_type_name", "budget", "id", "n_obs",
    "_source_input_file",
}

native_columns = set(
    con.execute(
        f"""
        select column_name
        from (describe from read_csv_auto(
            '{S3_WORKFLOW_INPUTS}/*_a_codif.csv',
            delim = ';', nullstr = 'NA',
            types = {{'ID_SABIANE': 'VARCHAR'}}, union_by_name = true
        ))
        """
    ).df()["column_name"]
)

renaming = {
    col: ("NAT_DEP" if col == "product" else f"{col}_previous")
    for col in annotations.columns
    if col == "product" or col not in native_columns
}
export = annotations.rename(columns=renaming)

print(f"{len(renaming)} colonnes renommées :")
for old, new in renaming.items():
    print(f"  {old:<20} -> {new}")

#%%
# Compatibility of the column names with the pipeline input contract
# (build-datasets/main.py: build_observations / load_input_file).
invalid_names = [c for c in export.columns if not re.fullmatch(r"\w+", c)]
reserved_left = sorted(set(export.columns) & PIPELINE_RESERVED)

checks = {
    "text column NAT_DEP present (-p text_column=NAT_DEP)": "NAT_DEP" in export.columns,
    "shop column MAG_DEP present (-p shop_column=MAG_DEP)": "MAG_DEP" in export.columns,
    "budget column MONT_DEP present (-p budget_column=MONT_DEP)": "MONT_DEP"
    in export.columns,
    "no name colliding with the pipeline's canonical ones": not reserved_left,
    "names are plain identifiers (no space, accent, dot)": not invalid_names,
    "no duplicated name": not export.columns.duplicated().any(),
}
for label, ok in checks.items():
    print(f"  [{'OK' if ok else 'KO'}] {label}")
if reserved_left:
    print(f"  -> colliding: {reserved_left}")
if invalid_names:
    print(f"  -> invalid: {invalid_names}")

#%%
# ID_SABIANE is forced to VARCHAR when the pipeline reads a CSV input; parquet
# carries its own types, so the cast has to be made explicit here.
assert all(checks.values()), "column names not compatible with the pipeline input"

con.execute(
    f"""
    copy (select * replace (ID_SABIANE::varchar as ID_SABIANE) from export)
    to '{EXPORT_PATH}' (format parquet)
    """
)
print(f"{len(export)} lignes exportées vers {EXPORT_PATH}")
