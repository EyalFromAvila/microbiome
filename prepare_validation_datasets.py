"""
Prepare the validation datasets (Franzosa et al. 2019, Yachida et al. 2019) for the pipeline.

The grader builds each cohort's ground truth from that cohort's OWN files: every microbiome and
metabolome feature, standard-scaled, then PCA. The pipeline fits its scaler and PCA per cohort as
well, so the features are kept exactly as given. They are NOT restricted to the MetaCardis taxa
or metabolites: dropping features (or adding zero-filled ones) would change the PCs the pipeline
predicts and make them differ from the grader's.

What this script does change is only the format:
1. Sample IDs: the first column of every table becomes the 'SampleID' index (stripped strings),
   and all three tables are put on the union of the sample IDs. A sample absent from an omic
   file becomes an all-NaN row there, i.e. a sample missing that omic, so no sample is lost.
2. Feature tables: values are coerced to numbers; features with no value at all are dropped
   (the grader cannot use them either). Nothing else is changed.
3. Metadata: renamed to the columns metadata_features() in snf_neighbours.py reads:
     Study.Group -> PATGROUPFINAL_C, Gender -> GENDER (Male 0, Female 1), BMI -> BMI_C,
     Age -> AGE, CENTER_C = cohort name.
   Unknown values stay NaN; metadata_features() fills them with the median and adds a
   "_missing" flag, which is more honest than a made-up value (BMI 0, "male").

Input:  <additional-dir>/<cohort>/{microbiome,metabolome,metadata}.csv
Output: <output-base>/<cohort>/{microbiome,metabolome,metadata}.csv
"""

import argparse
from pathlib import Path

import pandas as pd

COHORTS = ("franzosa", "yachida")

# Metadata columns read by snf_neighbours.metadata_features(), and the source names they may have.
METADATA_SOURCES = {
    "PATGROUPFINAL_C": ("PATGROUPFINAL_C", "Study.Group", "Study_Group", "study_group"),
    "GENDER": ("GENDER", "Gender", "gender", "Sex", "sex"),
    "BMI_C": ("BMI_C", "BMI", "bmi"),
    "AGE": ("AGE", "Age", "age"),
}
GENDER_CODES = {"male": 0.0, "m": 0.0, "0": 0.0, "0.0": 0.0, "female": 1.0, "f": 1.0, "1": 1.0, "1.0": 1.0}


def read_table(path):
    table = pd.read_csv(path, index_col=0)
    table.index = table.index.astype(str).str.strip()
    table.index.name = "SampleID"
    if table.index.duplicated().any():
        raise ValueError(f"duplicate sample IDs in {path}")
    return table


def first_present(table, candidates):
    return next((column for column in candidates if column in table.columns), None)


def prepare_features(table, ids):
    """Same features and values, on the union of sample IDs; non-numeric entries become NaN."""
    numeric = table.reindex(ids).apply(pd.to_numeric, errors="coerce")
    return numeric.dropna(axis=1, how="all")


def prepare_metadata(table, ids, cohort):
    table = table.reindex(ids)
    out = pd.DataFrame(index=ids)
    for target, candidates in METADATA_SOURCES.items():
        source = first_present(table, candidates)
        out[target] = table[source] if source else pd.NA
    out["GENDER"] = out["GENDER"].astype(str).str.strip().str.lower().map(GENDER_CODES)
    out["BMI_C"] = pd.to_numeric(out["BMI_C"], errors="coerce")
    out["AGE"] = pd.to_numeric(out["AGE"], errors="coerce")
    out["CENTER_C"] = cohort.capitalize()
    out.index.name = "SampleID"
    return out


def process_cohort(cohort, input_dir, output_dir):
    microbiome = read_table(input_dir / "microbiome.csv")
    metabolome = read_table(input_dir / "metabolome.csv")
    metadata = read_table(input_dir / "metadata.csv")
    ids = microbiome.index.union(metabolome.index, sort=False).union(metadata.index, sort=False)
    ids.name = "SampleID"

    mb = prepare_features(microbiome, ids)
    mt = prepare_features(metabolome, ids)
    meta = prepare_metadata(metadata, ids, cohort)

    output_dir.mkdir(parents=True, exist_ok=True)
    mb.to_csv(output_dir / "microbiome.csv")
    mt.to_csv(output_dir / "metabolome.csv")
    meta.to_csv(output_dir / "metadata.csv")

    mb_missing, mt_missing = mb.isna().all(axis=1), mt.isna().all(axis=1)
    return {
        "Cohort": cohort.capitalize(),
        "Samples": len(ids),
        "Complete": int((~mb_missing & ~mt_missing).sum()),
        "Miss MB": int(mb_missing.sum()),
        "Miss MT": int(mt_missing.sum()),
        "Miss both": int((mb_missing & mt_missing).sum()),
        "MB features": f"{mb.shape[1]} of {microbiome.shape[1]}",
        "MT features": f"{mt.shape[1]} of {metabolome.shape[1]}",
        "Absent from a file": int(sum(len(ids.difference(t.index)) for t in (microbiome, metabolome, metadata))),
        "Groups": meta["PATGROUPFINAL_C"].value_counts(dropna=False).to_dict(),
        "Metadata NaN": meta[["GENDER", "BMI_C", "AGE"]].isna().sum().to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description="Format the Franzosa and Yachida cohorts for the pipeline (features unchanged).")
    parser.add_argument("--additional-dir", default="additional_datasets", help="folder with one sub-folder of raw files per cohort")
    parser.add_argument("--output-base", default="prepared_datasets", help="folder for the prepared cohorts")
    args = parser.parse_args()

    rows = []
    for cohort in COHORTS:
        input_dir = Path(args.additional_dir) / cohort
        if not input_dir.exists():
            print(f"Warning: {input_dir} does not exist; skipping {cohort}.")
            continue
        rows.append(process_cohort(cohort, input_dir, Path(args.output_base) / cohort))

    if rows:
        pd.set_option("display.width", 250)
        pd.set_option("display.max_colwidth", 80)
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"Prepared cohorts saved under {Path(args.output_base).resolve()}/")


if __name__ == "__main__":
    main()
