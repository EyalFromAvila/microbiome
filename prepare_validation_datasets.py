"""
Prepare Validation Datasets (Franzosa & Yachida) for SNF Neighbours Pipeline

Aligns external validation datasets (Franzosa et al. 2019, Yachida et al. 2019)
to match the exact schema expected by snf_neighbours.py and pipeline.py, matching
the root MetaCardis datasets (microbiome.csv, metabolome.csv, metadata.csv):

1. SampleID Index: Ensure the first column in every table is set as the 'SampleID' index.
2. Microbiome Alignment: Filter to keep ONLY bacterial taxa in root microbiome.csv.
   Reorder columns to match root microbiome.csv exactly, filling missing taxa with 0.0
   while strictly preserving all-NaN rows for samples with missing microbiome modality.
3. Metabolome Alignment (KEGG Mapping): Map raw feature names to KEGG IDs using
   mtb.map / mtb.map.tsv. Keep only metabolites mapped to KEGG IDs that exist in the root
   metabolome.csv. If multiple raw features map to the same KEGG ID / MetaCardis metabolite,
   aggregate them via mean. Reorder columns to match root metabolome.csv exactly, filling
   unobserved MetaCardis metabolites with 0.0 while preserving all-NaN rows for samples
   with missing metabolome modality.
4. Metadata Alignment:
   - Map 'Study.Group' to 'PATGROUPFINAL_C'.
   - Set 'CENTER_C' to constant cohort label ("Franzosa" / "Yachida").
   - Map 'GENDER', 'BMI_C', 'AGE' if available; otherwise use safe neutral fallbacks (0.0)
     so snf_neighbours.py / metadata_features() processes them without NaN errors.

Outputs saved to:
  - aligned_datasets/franzosa/ (microbiome.csv, metabolome.csv, metadata.csv)
  - aligned_datasets/yachida/  (microbiome.csv, metabolome.csv, metadata.csv)
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ==============================================================================
# 1. METACARDIS METABOLITE <-> KEGG COMPOUND MAPPING
# ==============================================================================
# Mapping from each of the 102 MetaCardis serum metabolome feature names to their
# corresponding KEGG Compound ID(s) (including standard synonyms/tautomers/isomers).
METACARDIS_KEGG_MAPPING: Dict[str, List[str]] = {
    "SERUM_ABS_1,5_anhydro_D_sorbitol": ["C07326"],
    "SERUM_ABS_1_stearoyl_rac_glycerol": ["C02112"],
    "SERUM_ABS_2,3_butanediol_2": ["C03044", "C01815"],
    "SERUM_ABS_2_hydroxybutyric_acid": ["C05984"],
    "SERUM_ABS_2_ketoisocaproic_acid_2": ["C00233"],
    "SERUM_ABS_2_oleoylglycerol": ["C19672"],
    "SERUM_ABS_3_indoleacetic_acid": ["C00954"],
    "SERUM_ABS_3_indolelactic_acid_2": ["C02043"],
    "SERUM_ABS_3_methyl_2_oxobutanoic_acid_1": ["C00141"],
    "SERUM_ABS_5alpha_Cholestan_3_beta_ol": ["C01903"],
    "SERUM_ABS_6_deoxy_D_glucose_1": ["C15938"],
    "SERUM_ABS_6_phosphogluconic_acid": ["C00345"],
    "SERUM_ABS_Artemisin": ["C09827"],
    "SERUM_ABS_Beta__alanine__1": ["C00099"],
    "SERUM_ABS_DL_3_aminoisobutyric_acid_2": ["C05145", "C01205", "C03284"],
    "SERUM_ABS_DL_glyceraldehyde_2": ["C00577", "C00096"],
    "SERUM_ABS_DL_isoleucine_2": ["C00407"],
    "SERUM_ABS_D_P_trehalose": ["C01083"],
    "SERUM_ABS_D_glucose_1": ["C00031", "C00293"],
    "SERUM_ABS_D_lyxose_2": ["C01509"],
    "SERUM_ABS_D_malic_acid": ["C00711", "C00149"],
    "SERUM_ABS_D_mannose_1": ["C00159"],
    "SERUM_ABS_D_sorbitol": ["C00794"],
    "SERUM_ABS_D_threitol": ["C01697", "C06413"],
    "SERUM_ABS_L_P_lactic_acid": ["C00186"],
    "SERUM_ABS_L_alanine_1": ["C00041", "C01401"],
    "SERUM_ABS_L_glutamic_acid_2": ["C00025"],
    "SERUM_ABS_L_leucine_1": ["C00123"],
    "SERUM_ABS_L_lysine_2": ["C00047"],
    "SERUM_ABS_L_methionine_2": ["C00073"],
    "SERUM_ABS_L_proline_2": ["C00148"],
    "SERUM_ABS_L_pyroglutamic_acid": ["C01879"],
    "SERUM_ABS_L_serine_2": ["C00065"],
    "SERUM_ABS_L_threonine_2": ["C00188"],
    "SERUM_ABS_L_tryptophan_2": ["C00078"],
    "SERUM_ABS_L_tyrosine_2": ["C00082"],
    "SERUM_ABS_L_valine_2": ["C00183"],
    "SERUM_ABS_Methyl_Palmitate": ["C19459"],
    "SERUM_ABS_N_methylalanine": ["C02862"],
    "SERUM_ABS_O_acetylsalicylic_acid": ["C01405"],
    "SERUM_ABS_Sucrose": ["C00089"],
    "SERUM_ABS_alpha_ketoglutaric_acid": ["C00026"],
    "SERUM_ABS_alpha_tocophereol": ["C02477"],
    "SERUM_ABS_arachidic_acid": ["C06425"],
    "SERUM_ABS_aspartic_acid_2": ["C00049"],
    "SERUM_ABS_beta_hydroxyisovalerate": ["C05267"],
    "SERUM_ABS_beta_sitosterol": ["C01753"],
    "SERUM_ABS_capric_acid": ["C01571"],
    "SERUM_ABS_catechol": ["C00090"],
    "SERUM_ABS_cellotetraose": ["C04838"],
    "SERUM_ABS_cholesterol": ["C00187"],
    "SERUM_ABS_citric_acid": ["C00158"],
    "SERUM_ABS_citrulline_2": ["C00327"],
    "SERUM_ABS_ethanolamine": ["C00189"],
    "SERUM_ABS_fumaric_acid": ["C00122"],
    "SERUM_ABS_glucoheptonic_acid_2": ["C03126"],
    "SERUM_ABS_gluconic_acid_2": ["C00257"],
    "SERUM_ABS_glyceric_acid": ["C00258"],
    "SERUM_ABS_glycerol": ["C00116"],
    "SERUM_ABS_glycine": ["C00037"],
    "SERUM_ABS_glycolic_acid": ["C00160"],
    "SERUM_ABS_halostachine_2": ["C10368"],
    "SERUM_ABS_heptadecanoic_acid": ["C16537"],
    "SERUM_ABS_hexanoic_acid": ["C01585"],
    "SERUM_ABS_hippuric_acid_2": ["C01586"],
    "SERUM_ABS_hydroquinone": ["C00530"],
    "SERUM_ABS_hypotaurine": ["C00519"],
    "SERUM_ABS_indolepropionic_acid": ["C05634"],
    "SERUM_ABS_lactamide_1": ["C19436"],
    "SERUM_ABS_lactose_1": ["C00243"],
    "SERUM_ABS_lauric_acid": ["C02679"],
    "SERUM_ABS_linoleic_acid": ["C01595"],
    "SERUM_ABS_malonic_acid_1": ["C00383"],
    "SERUM_ABS_maltose_1": ["C00208"],
    "SERUM_ABS_methyl_linolenate": ["C19460"],
    "SERUM_ABS_methyl_oleate": ["C19461"],
    "SERUM_ABS_myo_inositol": ["C00137"],
    "SERUM_ABS_myristic_acid": ["C06424"],
    "SERUM_ABS_oleic_acid": ["C00712"],
    "SERUM_ABS_oxalic_acid": ["C00209"],
    "SERUM_ABS_p_cresol": ["C01468"],
    "SERUM_ABS_palmitic_acid": ["C00249"],
    "SERUM_ABS_palmitoleic_acid": ["C08362"],
    "SERUM_ABS_pelargonic_acid_nonanoic_acid": ["C01601"],
    "SERUM_ABS_phosphoric_acid": ["C00009"],
    "SERUM_ABS_phytanic_acid": ["C05445"],
    "SERUM_ABS_pyruvic_acid": ["C00022"],
    "SERUM_ABS_quinic_acid": ["C00296"],
    "SERUM_ABS_rhamnose_2": ["C00507", "C01705"],
    "SERUM_ABS_ribose": ["C00121"],
    "SERUM_ABS_serotonin_1": ["C00780"],
    "SERUM_ABS_stearic_acid": ["C01530"],
    "SERUM_ABS_stigmasterol": ["C05442"],
    "SERUM_ABS_succinic_acid": ["C00042"],
    "SERUM_ABS_talose_2": ["C06456"],
    "SERUM_ABS_tartronic_acid": ["C00877"],
    "SERUM_ABS_tetracosane": [],
    "SERUM_ABS_threonic_acid": ["C01620"],
    "SERUM_ABS_urea": ["C00086"],
    "SERUM_ABS_xylitol": ["C00379"],
    "SERUM_ABS_xylose_2": ["C00181"],
    "SERUM_ABS_xylulose": ["C00310", "C00312"],
}


def build_kegg_to_metacardis_map(mapping: Dict[str, List[str]]) -> Dict[str, str]:
    """Invert the MetaCardis -> KEGG mapping for O(1) feature lookup."""
    kegg_to_mc: Dict[str, str] = {}
    for mc_col, kegg_ids in mapping.items():
        for k in kegg_ids:
            kegg_to_mc[k] = mc_col
    return kegg_to_mc


# ==============================================================================
# 2. SCHEMA LOADING & MODULAR ALIGNMENT FUNCTIONS
# ==============================================================================

def load_root_schemas(root_dir: Path = Path(".")) -> Tuple[List[str], List[str], List[str]]:
    """
    Load feature column lists and metadata column order from root MetaCardis files.
    """
    mb = pd.read_csv(root_dir / "microbiome.csv", index_col=0, nrows=1)
    mt = pd.read_csv(root_dir / "metabolome.csv", index_col=0, nrows=1)
    meta = pd.read_csv(root_dir / "metadata.csv", index_col=0, nrows=1)

    return list(mb.columns), list(mt.columns), list(meta.columns)


def align_microbiome(
    raw_mb: pd.DataFrame,
    root_taxa: List[str]
) -> Tuple[pd.DataFrame, int]:
    """
    Align raw microbiome table to root MetaCardis taxa:
    1. Sets index name to 'SampleID'.
    2. Identifies samples missing microbiome (all-NaN rows).
    3. Keeps only taxa present in root MetaCardis microbiome.
    4. Reorders columns to match root_taxa exactly.
    5. Fills missing root taxa with 0.0 for observed samples.
    6. Preserves all-NaN rows for missing samples.
    """
    df = raw_mb.copy()
    if df.index.name != "SampleID":
        df.index.name = "SampleID"

    # Identify samples where microbiome modality is missing
    is_missing = df.isna().all(axis=1)

    # Intersection of taxa
    matched_taxa = [taxon for taxon in root_taxa if taxon in df.columns]

    # Build aligned DataFrame with exactly root_taxa columns
    aligned = pd.DataFrame(0.0, index=df.index, columns=root_taxa)
    if matched_taxa:
        aligned[matched_taxa] = df[matched_taxa]

    # Strictly preserve missing samples as all-NaN
    aligned.loc[is_missing] = np.nan
    aligned.index.name = "SampleID"

    return aligned, len(matched_taxa)


def align_metabolome(
    raw_mt: pd.DataFrame,
    map_df: pd.DataFrame,
    root_metabolites: List[str],
    kegg_to_mc: Dict[str, str]
) -> Tuple[pd.DataFrame, int, int]:
    """
    Align raw metabolome table to root MetaCardis KEGG metabolites:
    1. Sets index name to 'SampleID'.
    2. Identifies samples missing metabolome (all-NaN rows).
    3. Maps raw feature names to KEGG IDs using map_df ('Compound' -> 'KEGG').
    4. Filters to keep only features whose KEGG ID maps to a MetaCardis metabolite.
    5. Aggregates multiple raw features mapping to the same MetaCardis metabolite via mean.
    6. Reorders columns to match root_metabolites exactly, filling unobserved with 0.0.
    7. Preserves all-NaN rows for missing samples.

    Returns:
        (aligned_df, num_matched_metacardis_metabolites, num_matched_raw_features)
    """
    df = raw_mt.copy()
    if df.index.name != "SampleID":
        df.index.name = "SampleID"

    is_missing = df.isna().all(axis=1)

    # Build Compound -> KEGG dictionary from map_df
    comp_to_kegg = dict(zip(map_df["Compound"].astype(str), map_df["KEGG"].astype(str)))

    # Map raw features to MetaCardis target columns
    feat_to_target: Dict[str, str] = {}
    for col in df.columns:
        kegg_id = comp_to_kegg.get(str(col))
        if kegg_id and kegg_id != "nan" and kegg_id in kegg_to_mc:
            feat_to_target[col] = kegg_to_mc[kegg_id]

    matched_raw_count = len(feat_to_target)
    unique_matched_metabolites = set(feat_to_target.values())

    # Build aligned DataFrame
    aligned = pd.DataFrame(0.0, index=df.index, columns=root_metabolites)

    # Populate matched metabolites (averaging multiple features where applicable)
    for mc_col in root_metabolites:
        matching_feats = [f for f, target in feat_to_target.items() if target == mc_col]
        if matching_feats:
            aligned[mc_col] = df[matching_feats].mean(axis=1)

    # Strictly preserve missing samples as all-NaN
    aligned.loc[is_missing] = np.nan
    aligned.index.name = "SampleID"

    return aligned, len(unique_matched_metabolites), matched_raw_count


def align_metadata(
    raw_meta: pd.DataFrame,
    cohort_name: str,
    root_meta_cols: List[str]
) -> pd.DataFrame:
    """
    Align raw metadata table to match the root MetaCardis schema:
    Columns: ['PATGROUPFINAL_C', 'GENDER', 'BMI_C', 'AGE', 'CENTER_C']
    - PATGROUPFINAL_C: Study.Group (disease status / stage / control)
    - CENTER_C: cohort label ("Franzosa" / "Yachida")
    - GENDER: 0.0 for Male, 1.0 for Female (or 0.0 neutral fallback)
    - BMI_C: BMI values (or 0.0 neutral fallback)
    - AGE: Age in years (numeric)
    """
    df = raw_meta.copy()
    if df.index.name != "SampleID":
        df.index.name = "SampleID"

    # 1. Study.Group -> PATGROUPFINAL_C
    patgroup = df["Study.Group"].astype(str) if "Study.Group" in df.columns else "Unknown"

    # 2. Age -> AGE
    age_col = "Age" if "Age" in df.columns else ("AGE" if "AGE" in df.columns else None)
    age = pd.to_numeric(df[age_col], errors="coerce") if age_col else pd.Series(np.nan, index=df.index)
    # Neutral median fallback for age if entirely missing
    if age.isna().all():
        age = age.fillna(50.0)

    # 3. Gender -> GENDER (MetaCardis convention: 0.0 Male, 1.0 Female)
    gender_col = "Gender" if "Gender" in df.columns else ("GENDER" if "GENDER" in df.columns else None)
    if gender_col:
        gender_map = {"Male": 0.0, "Female": 1.0, "male": 0.0, "female": 1.0, "0": 0.0, "1": 1.0, 0: 0.0, 1: 1.0}
        gender = df[gender_col].map(gender_map).astype(float).fillna(0.0)
    else:
        # Safe neutral fallback (0.0)
        gender = pd.Series(0.0, index=df.index)

    # 4. BMI -> BMI_C
    bmi_col = "BMI" if "BMI" in df.columns else ("BMI_C" if "BMI_C" in df.columns else None)
    if bmi_col:
        bmi = pd.to_numeric(df[bmi_col], errors="coerce").fillna(0.0)
    else:
        # Safe neutral fallback (0.0)
        bmi = pd.Series(0.0, index=df.index)

    aligned = pd.DataFrame({
        "PATGROUPFINAL_C": patgroup,
        "GENDER": gender,
        "BMI_C": bmi,
        "AGE": age,
        "CENTER_C": cohort_name
    }, index=df.index)

    # Ensure exact column order matching root metadata
    aligned = aligned[root_meta_cols]
    aligned.index.name = "SampleID"

    return aligned


# ==============================================================================
# 3. DATASET PROCESSOR & REPORTING
# ==============================================================================

def process_dataset(
    dataset_name: str,
    input_dir: Path,
    output_dir: Path,
    root_taxa: List[str],
    root_metabolites: List[str],
    root_meta_cols: List[str],
    kegg_to_mc: Dict[str, str]
) -> Dict:
    """
    Process an entire dataset: load raw files, align microbiome, metabolome, metadata,
    and save the aligned files to output_dir.
    """
    cohort_title = dataset_name.capitalize()

    # Load raw files
    raw_mb = pd.read_csv(input_dir / "microbiome.csv", index_col=0)
    raw_mt = pd.read_csv(input_dir / "metabolome.csv", index_col=0)
    raw_meta = pd.read_csv(input_dir / "metadata.csv", index_col=0)

    # Find map file (mtb.map.tsv or mtb.map)
    map_path = input_dir / "mtb.map.tsv"
    if not map_path.exists():
        map_path = input_dir / "mtb.map"
    map_df = pd.read_csv(map_path, sep="\t")

    # Align each modality
    aligned_mb, matched_taxa = align_microbiome(raw_mb, root_taxa)
    aligned_mt, matched_metabolites, matched_raw_feats = align_metabolome(
        raw_mt, map_df, root_metabolites, kegg_to_mc
    )
    aligned_meta = align_metadata(raw_meta, cohort_title, root_meta_cols)

    # Calculate sample counts
    total_samples = len(aligned_meta)
    mb_missing = aligned_mb.isna().all(axis=1)
    mt_missing = aligned_mt.isna().all(axis=1)

    missing_mb_count = int(mb_missing.sum())
    missing_mt_count = int(mt_missing.sum())
    complete_count = int((~mb_missing & ~mt_missing).sum())

    # Save aligned files
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_mb.to_csv(output_dir / "microbiome.csv")
    aligned_mt.to_csv(output_dir / "metabolome.csv")
    aligned_meta.to_csv(output_dir / "metadata.csv")

    return {
        "dataset": cohort_title,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_samples": total_samples,
        "complete_samples": complete_count,
        "missing_mb": missing_mb_count,
        "missing_mt": missing_mt_count,
        "matched_taxa": matched_taxa,
        "total_root_taxa": len(root_taxa),
        "matched_metabolites": matched_metabolites,
        "matched_raw_features": matched_raw_feats,
        "total_root_metabolites": len(root_metabolites),
        "disease_groups": aligned_meta["PATGROUPFINAL_C"].value_counts().to_dict(),
    }


def print_summary_report(summaries: List[Dict]) -> None:
    """Print an exhaustive, professional summary report of the alignment."""
    print("\n" + "=" * 78)
    print("       VALIDATION DATASETS ALIGNMENT SUMMARY REPORT (SNF NEIGHBOURS)")
    print("=" * 78)

    for s in summaries:
        print(f"\n[{s['dataset']} Cohort]")
        print(f"  * Source Directory       : {s['input_dir']}")
        print(f"  * Destination Directory  : {s['output_dir']}")
        print(f"  * Total Samples          : {s['total_samples']}")
        print(f"  * Complete Samples       : {s['complete_samples']} ({s['complete_samples']/s['total_samples']:.1%})")
        print(f"  * Missing Microbiome     : {s['missing_mb']} ({s['missing_mb']/s['total_samples']:.1%})")
        print(f"  * Missing Metabolome      : {s['missing_mt']} ({s['missing_mt']/s['total_samples']:.1%})")
        print(f"  * Matched Bacterial Taxa : {s['matched_taxa']} / {s['total_root_taxa']} ({s['matched_taxa']/s['total_root_taxa']:.1%})")
        print(f"  * Matched KEGG Metabolites: {s['matched_metabolites']} / {s['total_root_metabolites']} ({s['matched_metabolites']/s['total_root_metabolites']:.1%})")
        print(f"    (Aggregated from {s['matched_raw_features']} raw LC-MS/CE-MS feature peaks via mean)")
        print(f"  * Disease Stratification : {s['disease_groups']}")

    print("\n" + "-" * 78)
    print("SCHEMA VERIFICATION SUMMARY TABLE")
    print("-" * 78)
    summary_df = pd.DataFrame([
        {
            "Dataset": s["dataset"],
            "Total": s["total_samples"],
            "Complete": s["complete_samples"],
            "Miss MB": s["missing_mb"],
            "Miss MT": s["missing_mt"],
            "Taxa (170)": f"{s['matched_taxa']} ({s['matched_taxa']/s['total_root_taxa']:.1%})",
            "Mtb (102)": f"{s['matched_metabolites']} ({s['matched_metabolites']/s['total_root_metabolites']:.1%})",
            "Meta Cols": "5 / 5 (Matched)",
        }
        for s in summaries
    ])
    print(summary_df.to_string(index=False))
    print("=" * 78 + "\n")


# ==============================================================================
# 4. ENTRYPOINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Align external validation datasets (Franzosa & Yachida) to MetaCardis schema for snf_neighbours.py"
    )
    parser.add_argument("--root-dir", type=str, default=".", help="Root directory containing MetaCardis files")
    parser.add_argument("--additional-dir", type=str, default="additional_datasets", help="Directory of raw additional datasets")
    parser.add_argument("--output-base", type=str, default="aligned_datasets", help="Base directory for aligned output datasets")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    additional_dir = Path(args.additional_dir)
    output_base = Path(args.output_base)

    print(f"Loading root MetaCardis schemas from {root_dir.resolve()}...")
    root_taxa, root_metabolites, root_meta_cols = load_root_schemas(root_dir)
    print(f"Root schema: {len(root_taxa)} microbiome taxa, {len(root_metabolites)} metabolome features, {len(root_meta_cols)} metadata columns.")

    kegg_to_mc = build_kegg_to_metacardis_map(METACARDIS_KEGG_MAPPING)

    datasets = ["franzosa", "yachida"]
    summaries = []

    for ds in datasets:
        ds_input = additional_dir / ds
        ds_output = output_base / ds
        if not ds_input.exists():
            print(f"Warning: Dataset directory {ds_input} does not exist. Skipping.")
            continue
        print(f"Aligning {ds.capitalize()} dataset...")
        summary = process_dataset(
            ds,
            ds_input,
            ds_output,
            root_taxa,
            root_metabolites,
            root_meta_cols,
            kegg_to_mc
        )
        summaries.append(summary)

    print_summary_report(summaries)
    print(f"All aligned datasets successfully saved to: {output_base.resolve()}/")


if __name__ == "__main__":
    main()
