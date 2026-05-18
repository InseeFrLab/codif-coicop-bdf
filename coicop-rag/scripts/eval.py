# %%

# Environment --------------------------------------

import os
# os.chdir("..")
os.getcwd()
# os.chdir("coicop-rag")
import re
import duckdb
import pandas as pd
import yaml


from coicop_rag.eval.metrics import (
  truncate_code, 
  compute_hierarchical_metrics, 
  calculate_accuracy_at_level, 
  print_metrics_report,
  analyze_error_sources,
  print_error_analysis,
  export_metrics_to_list
)

from coicop_rag.eval.metrics import (
  calculate_accuracy_at_level, 
)

from coicop_rag.utils import merge_eval_and_retreived

pd.reset_option("display.max_colwidth")
pd.set_option('display.max_rows', None)

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

threshold_confidence = config["eval"]["threshold_confidence"]
retrieval_size = config["retrieval"]["size"]
prune = config["eval"]["prune"]

con = duckdb.connect(database=":memory:")

s3_path_predictions = "s3://projet-budget-famille/data/rag/predictions_20260213_125554.parquet"
s3_path_retrieved_codes = "s3://projet-budget-famille/data/rag/retrieved_codes_20260213_125554.parquet"
query_definition = f"SELECT * FROM read_parquet('{s3_path_predictions}')"
df_eval = con.sql(query_definition).to_df()

df_retrieved_codes = con.sql(f"SELECT * FROM read_parquet('{s3_path_retrieved_codes}')").to_df()


# rules = load_rules(config["eval"]["rules_path"])

records = merge_eval_and_retreived(
    df_eval=df_eval,
    retrieved_codes=df_retrieved_codes,
    retrieval_size=config["retrieval"]["size"],
    code_name="code_tprune" if prune else "code",
    col_retrieved_codes_name="list_retrieved_codes",
)

# ----------------------------------------------
# Error analyses at level 4  / all raws

(
    overall_accuracy,
    result_list,
    retrieval_accuracy,
    generation_accuracy_when_retrieved,
    label_in_retrieved_list
) = calculate_accuracy_at_level(
    records=records,
    predicted_col="code_predict_tprune" if prune else "code_predict",
    label_col="code_tprune" if prune else "code",
    level=4,
    retrieved_col='list_retrieved_codes'
)

# Errors for each type of product (level1)

label_col = "code_tprune"

product_types = set()
for record in records_rag:
    if label_col in record and record[label_col]:
        code = str(record[label_col])
        if len(code) >= 2:
            product_types.add(code[:2])

product_types = sorted(product_types)

for product_type in product_types:
    filtered_records = [
      r for r in records 
      if label_col in r and r[label_col] and str(r[label_col]).startswith(product_type)
      ]
    (
        overall_acc,
        result_list,
        retrieval_acc,
        generation_acc_when_retrieved,
        label_in_retrieved_list
    ) = calculate_accuracy_at_level(
        filtered_records,
        predicted_col="code_predict_tprune" if prune else "code_predict",
        label_col="code_tprune" if prune else "code",
        level=4,
        retrieved_col='list_retrieved_codes'
    )
    print(f"{product_type} : {overall_acc:.2} (n = {len(filtered_records)})")















errors_list = [x for x, m in zip(records_rag, result_list) if not m]
print(f"Number of errors : {len(errors_list)} (on a total of {len(records_rag)})")

errors_list_high_confidence = [x for x in errors_list if x["confidence"] > threshold_confidence]
print(f"""
  Number of errors despite high confidence (>{threshold_confidence}) : {len(errors_list_high_confidence)})
  (on a total of {len(errors_list)} errors)
""")

errors_special_codes = [x for x in errors_list if (x["code"][:2] in ("98","99"))]
n_errors = len(errors_list)
n_errors_special_codes = len(errors_special_codes)
n_errors_special_codes/n_errors
print(f"""
  Number of errors due to special BDF codes (98, 99) : {n_errors_special_codes})
  (on a total of {len(errors_list)} errors ==> proportion = {round(100 * n_errors_special_codes/n_errors, 1)}%)
""")

codable_products = [x for x in records_rag if x["codable"]]
len(codable_products)
errors_amg_codable_products = [x for x in errors_list if x["codable"]]
5
special_codes_amg_codable_errors = [x for x in errors_list if ((x["code"][:2] in ("98","99")) and (x["codable"]))]
n_special_codes_amg_codable_errors = len(special_codes_amg_codable_errors)
pct_specialcodes_amg_codable_errors = n_special_codes_amg_codable_errors/len(errors_amg_codable_products)
print(f"""
  Number of errors due to special BDF codes (98, 99) : {n_special_codes_amg_codable_errors})
  (on a total of {len(errors_amg_codable_products)} "codable" errors ==> proportion = {round(100 * pct_specialcodes_amg_codable_errors, 1)}%)
""")

errors_normal_codes_codable = [x for x in errors_list if (x["code"][:2] not in ("98", "99") and x["codable"])]
len(errors_normal_codes_codable)
errors_normal_codes_codable_too_precise = [
  x for x in errors_normal_codes_codable
  if (x["code_predict_tprune"] and x["code_predict_tprune"].startswith(x["code_tprune"]))
]
len(errors_normal_codes_codable_too_precise)

print(f"""
  Number of errors due to overprecise predictions amongst codable errors : {len(errors_normal_codes_codable_too_precise)} among normal codes (total of {len(errors_normal_codes_codable)}))
  proportion = {round(100 * len(errors_normal_codes_codable_too_precise)/len(errors_normal_codes_codable), 1)}%)
""")



pd.DataFrame(errors_list)[
  ["product", "shop", "code_tprune", "code_predict_tprune","confidence", "in_retrieved", "list_retrieved_codes"]
].sample(5)

[m for m in errors_list if m["product"]=="cadre"]


# Sample des erreurs de niveau 4 parmi les codables


import merge_eval_and_retreived, calculate_accuracy_at_level
import random

def get_tricky_errors(
  sample_size: int,
  df_eval=df_eval,
  retrieved_codes=df_retrieved_codes,
  retrieval_size=config["retrieval"]["size"],
  code_name="code_tprune" if prune else "code",
  col_retrieved_codes_name="list_retrieved_codes",
  prune=config["eval"]["prune"],
  level=4,

):

    records = merge_eval_and_retreived(
        df_eval=df_eval,
        retrieved_codes=df_retrieved_codes,
        retrieval_size=config["retrieval"]["size"],
        code_name="code_tprune" if prune else "code",
        col_retrieved_codes_name="list_retrieved_codes",
    )
  
    (
      overall_accuracy,
      result_list,
      retrieval_accuracy,
      generation_accuracy_when_retrieved,
      label_in_retrieved_list
    ) = calculate_accuracy_at_level(
      records=records,
      predicted_col="code_predict_tprune" if prune else "code_predict",
      label_col="code_tprune" if prune else "code",
      level=level,
      retrieved_col='list_retrieved_codes'
    )

    errors_list = [x for x, m in zip(records, result_list) if not m]
    codable_errors = [x for x in errors_list if x["codable"]]
    real_errors = [x for x in codable_errors if (x["code"][:2] not in ("98", "99"))]

    keys_to_keep = [
      "product", "shop", "code", 
      "code_predict", "confidence", "budget", 
      "in_retrieved"
    ]
    sample_size = max(sample_size, len(real_errors))
    real_errors_sample = random.sample(real_errors, sample_size)

    res = []
    for e in real_errors_sample:
        res.append({k: e.get(k) for k in keys_to_keep})

    res = pd.DataFrame(res)
    return res
# ----------------------------------------------

df_eval.columns
df_eval["good_pred"].mean()
df_eval["parsed"].value_counts()
df_eval["parsed"].dtype
df_eval["codable"].dtype
df_eval["codable"].value_counts()
df_eval["code_predict"]
df_eval["code"].isna().sum()
df_eval["code_predict"].isna().sum()
df_eval["good_pred"].isna().sum()
df_eval.loc[df_eval["code_predict"].isna()]

truncate_code("01.2.3.0.7.000", level=5)
truncate_code(None, level=5)

accuracy, results = calculate_accuracy_at_level(
    df_eval.to_dict('records'),
    "code_predict",
    "code",
    4
)

accuracy, results = calculate_accuracy_at_level(
    df_eval[df_eval["confidence"]>0.7].to_dict('records'),
    "code_predict",
    "code",
    4
)


df_eval["result"] = results

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_eval, x='confidence', hue="result", common_norm=False, fill=True, alpha=0.3)
plt.title("Distribution de l'indice de confiance par résultat de prédiction")
plt.xlabel("Indice de confiance (confidence_0)")
plt.ylabel("Densité")
plt.legend(title='Résultat', labels=['Faux (False)', 'Vrai (True)'])
plt.grid(True, alpha=0.3)
plt.show()
output_path = "distribution_confidence_par_resultat.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")


df_eval.groupby("result")["confidence"].mean()

print(
        df_eval
          .loc[
            ~df_eval["result"], 
            ["product", "shop", "code", "code_predict","confidence"]
            ]
          .sort_values(by="confidence", ascending=False)
          .head(20)
)

print(
        df_eval
          .loc[
            ~df_eval["result"], 
            ["product", "shop", "code", "code_predict","confidence"]
            ]
          .sample(20)
)
pd.reset_option("display.max_colwidth")
str(df_eval.loc[df_eval["product"] == "billets avion", "reasons"].to_string(index=False))


# %%
import mlflow
mlflow.set_tracking_uri("https://projet-budget-famille-mlflow.user.lab.sspcloud.fr/")
mlflow.search_experiments()