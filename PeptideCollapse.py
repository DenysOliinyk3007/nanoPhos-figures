import re
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd


class PeptideCollapse:
    def __init__(self):
        self.required_columns = {
            "essential": [
                "R.FileName",
                "EG.PrecursorId",
                "EG.TotalQuantity (Settings)",
                "PEP.PeptidePosition",
                "EG.PTMAssayProbability",
                "PG.Genes",
                "PG.ProteinGroups",
            ],
            "optional": [
                "R.Condition",
                "EG.PTMLocalizationProbabilities",
                "EG.ProteinPTMLocations",
                "PEP.StrippedSequence",
                "PG.UniProtIds",
            ],
        }

        self.data = None
        self.processed_data = None
        self.fasta_dict = None
        self.peptide_data = None
        self.site_data = None

        self.processing_stats = {
            'initial_rows': 0,
            'phospho_rows': 0,
            'final_peptides': 0,
            'final_sites': 0,
            'processing_time': 0
        }
    
    def load_data(self, data: pd.DataFrame, validate: bool = True) -> None:

        self.data = data.copy()
        self.processing_stats['initial_rows'] = len(self.data)
        
        if validate:
            self._validate_input_data()
        

    
    def load_fasta(self, fasta_path: str) -> None:

        self.fasta_dict = self._load_fasta_to_dict(fasta_path)

    
    def preprocess_data(self) -> None:

        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")

        df = self.data.copy()
        underscore_count = df["PG.Genes"].str.contains("_", na=False).sum()
        if underscore_count > 0:
            df["PG.Genes"] = df["PG.Genes"].str.replace("_", "#", regex=False)
            warnings.warn(f"{underscore_count} gene names contained underscores and have been replaced with '#'.")

        modification_info = df["EG.PrecursorId"].apply(self._extract_sequence_modifications)

        df["clean_sequence"] = [info["clean_sequence"] for info in modification_info]
        df["phospho_positions"] = [info["phospho_positions"] for info in modification_info]
        df["phospho_count"] = [info["phospho_count"] for info in modification_info]
        df["all_modifications"] = [info["all_modifications"] for info in modification_info]
        df["phospho_sequence"] = [info["phospho_sequence"] for info in modification_info]

        phospho_rows = len(df)
        df = df[df["phospho_count"] > 0].copy()
        removed_non_phospho = phospho_rows - len(df)

        df["peptide_start_position"] = df["PEP.PeptidePosition"].astype(str).apply(
            self._extract_first_valid_position
        )

        initial_with_phospho = len(df)
        df = df.dropna(subset=["peptide_start_position"]).copy()
        df["peptide_start_position"] = df["peptide_start_position"].astype(int)
        
        removed_invalid_pos = initial_with_phospho - len(df)
        if removed_invalid_pos > 0:

            df["phospho_multiplicity"] = df["phospho_count"].apply(lambda x: min(x, 3))
        
        self.processed_data = df
        self.processing_stats['phospho_rows'] = len(df)
        
    
    def collapse_to_peptides(
        self,
        cutoff: float = 0.75,
        collapse_level: str = "PG",
        aggregation_method: str = "median",
        exclude_carbamidomethyl: bool = True,
        add_kinase_sequences: bool = False,
        kinase_window_size: int = 6
    ) -> pd.DataFrame:

        if self.processed_data is None:
            self.preprocess_data()

        self.peptide_data = self._create_peptide_level_collapse(
            self.processed_data, cutoff, collapse_level, aggregation_method, exclude_carbamidomethyl
        )
        
        if add_kinase_sequences:
            if self.fasta_dict is None:
                raise ValueError("FASTA data required for kinase sequences. Use load_fasta() first.")
            self.peptide_data = self._generate_kinase_sequences(
                self.peptide_data, kinase_window_size
            )
        
        self.processing_stats['final_peptides'] = len(self.peptide_data)

        return self.peptide_data
    
    def collapse_to_sites(
        self,
        cutoff: float = 0.75,
        collapse_level: str = "PG",
        aggregation_method: str = "median",
        add_kinase_sequences: bool = True,
        kinase_window_size: int = 6
    ) -> pd.DataFrame:

        if self.processed_data is None:
            self.preprocess_data()

        self.site_data = self._create_site_level_collapse(
            self.processed_data, cutoff, collapse_level, aggregation_method
        )
        
        if add_kinase_sequences:
            if self.fasta_dict is None:
                raise ValueError("FASTA data required for kinase sequences. Use load_fasta() first.")
            self.site_data = self._generate_kinase_sequences(
                self.site_data, kinase_window_size
            )
        
        self.processing_stats['final_sites'] = len(self.site_data)
        return self.site_data
    
    def process_complete_pipeline(
        self,
        data: pd.DataFrame,
        cutoff: float = 0.75,
        collapse_level: str = "PG",
        aggregation_method: str = "median",
        return_both: bool = False,
        exclude_carbamidomethyl: bool = True,
        fasta_path: Optional[str] = None,
        add_kinase_sequences: bool = True,
        kinase_window_size: int = 6
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:

        start_time = time.time()
        
        
        self.load_data(data)
        
        
        if add_kinase_sequences and fasta_path:
            self.load_fasta(fasta_path)
        
        
        self.preprocess_data()
        
        if return_both:
            # Create both levels
            peptide_data = self.collapse_to_peptides(
                cutoff, collapse_level, aggregation_method, exclude_carbamidomethyl,
                add_kinase_sequences, kinase_window_size
            )
            site_data = self.collapse_to_sites(
                cutoff, collapse_level, aggregation_method,
                add_kinase_sequences, kinase_window_size
            )
            
            self.processing_stats['processing_time'] = time.time() - start_time
            return peptide_data, site_data
        else:
            # Site-level only
            site_data = self.collapse_to_sites(
                cutoff, collapse_level, aggregation_method,
                add_kinase_sequences, kinase_window_size
            )
            
            self.processing_stats['processing_time'] = time.time() - start_time
            return site_data
    
    def reformat_for_analysis(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:

        if data is None:
            if self.site_data is None:
                raise ValueError("No site data available. Run collapse_to_sites() first.")
            data = self.site_data
        
        return self._clean_and_reformat_phospho_data(data)
    
    def get_processing_summary(self) -> dict:

        return self.processing_stats.copy()
    
    def get_quant_sample_data(self) -> list:
        
        return self.data['R.FileName'].unique().tolist()
    
    def get_precursor_condition_dataset(self) -> pd.DataFrame:
        return pd.DataFrame({'Sample': self.get_quant_sample_data(), 'Condition': np.nan})
    
    # Private methods (implementation of original functions)
    
    def _validate_input_data(self) -> None:
        
        missing_columns = []
        present_columns = []
        required_columns = self.required_columns['essential']
        
        for column in required_columns:
            if column in self.data.columns:
                present_columns.append(column)
            else:
                missing_columns.append(column)
                print(f"✗ {column} - MISSING")
        
        if missing_columns:
            print(f"\nERROR: Missing {len(missing_columns)} required column(s):")
            for col in missing_columns:
                print(f"  - {col}")
            print("\nAvailable columns in dataset:")
            for col in sorted(self.data.columns):
                print(f"  - {col}")
            sys.exit("Function stopped due to missing required columns")
    
    def _extract_sequence_modifications(self, sequence: str) -> Dict[str, Union[List[int], str, int]]:

        base_sequence = sequence
        
        if len(sequence) >= 4:
            if sequence.startswith("*.") or sequence.startswith("_."):
                base_sequence = base_sequence[2:]
            elif sequence.startswith("*") or sequence.startswith("_"):
                base_sequence = base_sequence[1:]
            
            if ".*" in base_sequence:
                base_sequence = base_sequence.split(".*")[0]
            elif "._" in base_sequence:
                base_sequence = base_sequence.split("._")[0]
            elif base_sequence.endswith("*"):
                base_sequence = base_sequence[:-1]
            elif base_sequence.endswith("_"):
                base_sequence = base_sequence[:-1]
            
            parts = base_sequence.split(".")
            if len(parts) > 1 and parts[-1].isdigit():
                base_sequence = ".".join(parts[:-1])
        
        pat_del_all = r"\[(?!Phospho \(STY\))[^\]]*\]"
        phospho_only_sequence = re.sub(pat_del_all, "", base_sequence)
        clean_sequence = re.sub(r"\[[^\]]*\]", "", base_sequence)
        clean_sequence = clean_sequence.replace("_", "")
        
        phospho_count = len(sequence.split("[Phospho (STY)]")) - 1
        all_modifications = re.findall(r"\[([^\]]+)\]", base_sequence)
        
        phospho_positions = []
        if phospho_count > 0:
            phospho_positions = self._calculate_phospho_positions(phospho_only_sequence)
        
        return {
            "clean_sequence": clean_sequence,
            "phospho_positions": phospho_positions,
            "phospho_count": phospho_count,
            "all_modifications": all_modifications,
            "phospho_sequence": phospho_only_sequence,
            "base_sequence": base_sequence,
        }
    
    def _calculate_phospho_positions(self, phospho_sequence: str) -> List[int]:
        
        if "[Phospho (STY)]" not in phospho_sequence:
            return []
        
        segments = phospho_sequence.split("[Phospho (STY)]")
        positions = []
        current_pos = 0
        
        for i in range(len(segments) - 1):
            current_pos += len(segments[i])
            positions.append(current_pos)
        return positions
    
    def _extract_first_valid_position(self, position_str: str) -> Optional[int]:

        if pd.isna(position_str) or position_str == "None" or position_str == "":
            return None
        
        try:
            parts = str(position_str).split(";")
            if parts:
                first_part = parts[0].split(",")[0].strip()
                if first_part and first_part != "None":
                    return int(first_part)
        except (ValueError, AttributeError):
            pass
        return None
    
    def _create_peptide_level_collapse(
        self, data: pd.DataFrame, cutoff: float, collapse_level: str,
        aggregation_method: str, exclude_carbamidomethyl: bool
    ) -> pd.DataFrame:
        df = data.copy()
        
        # Re-extract modifications
        modification_info = df["EG.PrecursorId"].apply(self._extract_sequence_modifications)
        df["clean_sequence"] = [info["clean_sequence"] for info in modification_info]
        df["phospho_positions"] = [info["phospho_positions"] for info in modification_info]
        df["phospho_count"] = [info["phospho_count"] for info in modification_info]
        df["all_modifications"] = [info["all_modifications"] for info in modification_info]
        
        df = df[df["phospho_count"] > 0].copy()
        
        if len(df) == 0:
            return pd.DataFrame()
        
        df["genes_processed"] = df["PG.Genes"].astype(str).str.replace("#", "_", regex=False)
        df["peptide_collapse_key"] = df.apply(
            lambda row: self._create_peptide_key(row, collapse_level, exclude_carbamidomethyl), axis=1
        )
        
        error_keys = df["peptide_collapse_key"].str.startswith("Error_")
        if error_keys.any():
            df = df[~error_keys].copy()
        
        if len(df) == 0:
            return pd.DataFrame()

        quant_pivot = df.pivot_table(
            index="peptide_collapse_key",
            columns="R.FileName",
            values="EG.TotalQuantity (Settings)",
            aggfunc="first",
        )
        quant_pivot = np.log2(quant_pivot.replace([0], np.nan))
        
        loc_pivot = df.pivot_table(
            index="peptide_collapse_key",
            columns="R.FileName",
            values="EG.PTMAssayProbability",
            aggfunc="first",
        )
        loc_pivot = loc_pivot.fillna(-1)
        loc_pivot = pd.DataFrame({"EG.PTMAssayProbability": loc_pivot.apply(max, axis=1)})
        
        metadata_cols = ["PG.Genes", "PG.ProteinGroups", "clean_sequence", "all_modifications", "phospho_count"]
        available_metadata_cols = [col for col in metadata_cols if col in df.columns]
        
        if available_metadata_cols:
            metadata_pivot = df.pivot_table(
                index="peptide_collapse_key", values=available_metadata_cols, aggfunc="first"
            )
        else:
            metadata_pivot = pd.DataFrame(index=quant_pivot.index)

        if aggregation_method == "median":
            quant_final = quant_pivot.groupby(level=0).median()
        elif aggregation_method == "mean":
            quant_final = quant_pivot.groupby(level=0).mean()
        else:
            quant_final = quant_pivot.groupby(level=0).max()
        
        quant_final = quant_final.replace(0, np.nan).replace("Filtered", np.nan)
        loc_final = loc_pivot.groupby(level=0).max()
        
        result = pd.concat([quant_final, metadata_pivot, loc_final], axis=1)
        result = result.reset_index()

        result["EG.PTMAssayProbability"] = result["EG.PTMAssayProbability"].replace(-1, np.nan)
        pre_filter_count = len(result)
        result = result[result["EG.PTMAssayProbability"] >= cutoff]


        new_columns = []
        for col in result.columns:
            if isinstance(col, tuple):
                new_columns.append(col[1] if len(col) > 1 else col[0])
            else:
                new_columns.append(col)
        result.columns = new_columns
        
        if "PG.Genes" in result.columns:
            result["PG.Genes"] = result["PG.Genes"].astype(str).str.replace("#", "_", regex=False)
        
        return result
    
    def _create_site_level_collapse(
        self, data: pd.DataFrame, cutoff: float, collapse_level: str, aggregation_method: str
    ) -> pd.DataFrame:
        """Create site-level collapsed data."""
        df = data.copy()

        modification_info = df["EG.PrecursorId"].apply(self._extract_sequence_modifications)
        df["PTM_base_seq"] = [info["clean_sequence"] for info in modification_info]
        df["PTM_0_pos_val"] = [info["phospho_positions"] for info in modification_info]
        df["PTM_0_num"] = [info["phospho_count"] for info in modification_info]
        df["PTM_group"] = df["EG.PrecursorId"]
        
        df = df[df["PTM_0_num"] > 0]
        df = df.explode("PTM_0_pos_val")
        
        df["UPD_seq"] = df.apply(
            lambda x: self._create_modified_sequence(x["PTM_base_seq"], x["PTM_0_pos_val"]), axis=1
        )
        
        df = df.reset_index()
        df["PTM_0_aa"] = df.apply(
            lambda x: self._get_phospho_amino_acid(x["PTM_base_seq"], x["PTM_0_pos_val"]), axis=1
        )
        df = df.set_index("index")
        df["PTM_localization"] = df["EG.PTMAssayProbability"].astype(np.float64)

        fine_names = list(data["R.FileName"].unique())
        
        df2 = pd.pivot_table(
            df, index=["PTM_group", "PTM_0_pos_val"], columns=["R.FileName"],
            values=["EG.TotalQuantity (Settings)"], aggfunc="first"
        )
        df2 = np.log2(df2)
        
        df3 = pd.pivot_table(
            df, index=["PTM_group", "PTM_0_pos_val"], columns=["R.FileName"],
            values=["PTM_localization"], aggfunc="first"
        )
        df3 = df3.fillna(-1)
        df3 = pd.DataFrame({"PTM_localization": df3.apply(max, axis=1)})
        
        keep = ["PEP.PeptidePosition", "PG.ProteinGroups", "PG.Genes", "PTM_0_num", "PTM_0_aa"]
        df4 = pd.pivot_table(df, index=["PTM_group", "PTM_0_pos_val"], values=keep, aggfunc="first")
        
        df5 = pd.pivot_table(
            df, index=["PTM_group", "PTM_0_pos_val"], values=["UPD_seq"], aggfunc="first"
        )
        
        data_combined = pd.concat([df2, df4, df3, df5], axis=1).reset_index()

        data_combined["PTM_Genprot"] = data_combined["PG.Genes"].astype(str)
        data_combined["PTM_pep_pos"] = (
            data_combined["PEP.PeptidePosition"].astype(str)
            .apply(lambda row: list(filter(None, row.split(";")))[0])
        )
        data_combined["PTM_pep_pos"] = (
            data_combined["PTM_pep_pos"].astype(str)
            .apply(lambda row: list(filter(None, row.split(",")))[0])
        )
        data_combined = data_combined[data_combined["PTM_pep_pos"] != "None"]
        data_combined["PTM_pep_pos"] = data_combined["PTM_pep_pos"].astype(np.int64)
        
        data_combined["PTM_mult123"] = data_combined["PTM_0_num"].astype(np.int64)
        data_combined["PTM_mult123"] = data_combined["PTM_mult123"].apply(lambda x: min(x, 3))
        
        data_combined["PTM_Collapse_key"] = data_combined.apply(
            lambda x: self._create_collapse_key(
                x["PG.ProteinGroups"], x["PTM_Genprot"], x["PTM_0_aa"],
                x["PTM_0_pos_val"], x["PTM_pep_pos"], x["PTM_mult123"]
            ), axis=1
        )

        cols = []
        for c in fine_names:
            cols.append([col for col in data_combined.columns if c in col][0])
        
        df1 = data_combined.set_index("PTM_Collapse_key")
        df2_agg = df1[df1.columns[df1.columns.isin(cols)]].reset_index()
        
        if aggregation_method == "median":
            df3_agg = df2_agg.groupby("PTM_Collapse_key").median()
        elif aggregation_method == "mean":
            df3_agg = df2_agg.groupby("PTM_Collapse_key").mean()
        else:
            df3_agg = df2_agg.groupby("PTM_Collapse_key").max()
        
        df3_agg = df3_agg.replace(0, np.nan).replace(1, np.nan).replace("Filtered", np.nan)
        
        df4_agg = df1[df1.columns[df1.columns == "PTM_localization"]].reset_index()
        df5_agg = df4_agg.groupby("PTM_Collapse_key").max()
        
        protgen = ["PG.Genes", "PG.ProteinGroups", "UPD_seq"]
        df6_agg = df1[df1.columns[df1.columns.isin(protgen)]].reset_index()
        df7_agg = df6_agg.groupby("PTM_Collapse_key").first()
        
        result = pd.concat([df3_agg, df7_agg, df5_agg], axis=1).reset_index()

        result["PTM_localization"] = result["PTM_localization"].replace(-1, np.nan)
        result = result[result["PTM_localization"] >= cutoff]

        cols = []
        for c in list(result.columns):
            if type(c) == tuple:
                cols.append(c[1])
            else:
                cols.append(c)
        result.columns = cols
        
        final_result = self._finalize_collapsed_output(result, collapse_level)
        return final_result
    
    def _create_peptide_key(self, row: pd.Series, collapse_level: str, exclude_carbamidomethyl: bool) -> str:

        try:
            protein_groups = row["PG.ProteinGroups"]
            if pd.isna(protein_groups):
                protein_part = "Unknown"
            else:
                protein_groups_str = str(protein_groups)
                if collapse_level == "PG":
                    protein_part = protein_groups_str.split(";")[0]
                else:
                    protein_part = protein_groups_str
            
            genes = row["PG.Genes"]
            if pd.isna(genes):
                gene_part = "Unknown"
            else:
                genes_str = str(genes).replace("#", "_")
                gene_part = genes_str.split(";")[0]
            
            clean_seq = row["clean_sequence"]
            sequence_part = str(clean_seq) if pd.notna(clean_seq) else "Unknown"
            
            peptide_pos = row["peptide_start_position"]
            position_part = str(int(peptide_pos)) if pd.notna(peptide_pos) else "0"
            
            all_mods = row["all_modifications"]
            try:
                if all_mods is None or (isinstance(all_mods, float) and np.isnan(all_mods)):
                    mod_part = "NoMods"
                elif isinstance(all_mods, (list, tuple, np.ndarray)):
                    if len(all_mods) > 0:
                        filtered_mods = all_mods
                        if exclude_carbamidomethyl:
                            filtered_mods = [
                                mod for mod in all_mods
                                if not (isinstance(mod, str) and "Carbamidomethyl" in mod)
                            ]
                        
                        if len(filtered_mods) > 0:
                            mod_part = "_".join(sorted([str(mod) for mod in filtered_mods]))
                        else:
                            mod_part = "NoMods"
                    else:
                        mod_part = "NoMods"
                else:
                    single_mod = str(all_mods) if all_mods else ""
                    if exclude_carbamidomethyl and "Carbamidomethyl" in single_mod:
                        mod_part = "NoMods"
                    else:
                        mod_part = single_mod if single_mod else "NoMods"
            except (TypeError, AttributeError, ValueError):
                mod_part = "NoMods"
            
            key = f"{protein_part}~{gene_part}_{sequence_part}_{position_part}_{mod_part}"
            return key
        
        except Exception as e:
            return "Error_peptide_key"
    
    def _create_collapse_key(self, entry0, entry1, entry2, entry3, entry4, entry5) -> str:

        try:
            absolute_position = int(entry3 + entry4 - 1)
            result = (
                str(entry0) + "~" + str(entry1) + "_" + str(entry2) + 
                str(absolute_position) + "_M" + str(int(entry5))
            )
            return result
        except Exception:
            return "Error_key"
    
    def _create_modified_sequence(self, clean_sequence: str, phospho_position: int) -> str:

        if phospho_position < 1 or phospho_position > len(clean_sequence):
            return clean_sequence
        
        pos_idx = phospho_position - 1
        modified_seq = (
            clean_sequence[:pos_idx] + clean_sequence[pos_idx].lower() + "*" +
            clean_sequence[pos_idx + 1:]
        )
        return modified_seq
    
    def _get_phospho_amino_acid(self, sequence: str, position: int) -> str:

        try:
            if position < 1 or position > len(sequence):
                return "X"
            result = sequence[position - 1 : position]
            return result if result else "X"
        except (IndexError, TypeError):
            return "X"
    
    def _finalize_collapsed_output(self, data: pd.DataFrame, collapse_level: str) -> pd.DataFrame:

        if collapse_level == "P":
            result = self._process_protein_level_output(data)
        elif collapse_level == "PG":
            result = self._process_protein_group_output(data)
        else:
            raise ValueError(f"collapse_level must be 'P' or 'PG', got '{collapse_level}'")
        return result
    
    def _process_protein_level_output(self, data: pd.DataFrame) -> pd.DataFrame:

        data["Protein_name"] = data["PTM_Collapse_key"].apply(lambda row: row.split("~")[0])
        data["Protein_group"] = data["Protein_name"].str.split(";")
        data["Protein_group"] = data["Protein_group"].apply(lambda row: row[0])
        data["Protein_name"] = data["Protein_name"].str.split(";")
        
        data["PTM"] = data["PTM_Collapse_key"].apply(lambda row: row.split("~")[1])
        data[["Gene_name", "Site", "Mult"]] = data["PTM"].str.split("_", expand=True)
        data["Gene_name"] = data["Gene_name"].str.split(";")
        data["Gene_group"] = data["Gene_name"].apply(lambda row: row[0])
        
        data = data.drop(["Gene_name", "PTM", "PTM_Collapse_key", "PG.Genes", "PG.ProteinGroups"], axis=1)
        data = data.explode("Protein_name", ignore_index=True)
        
        data["PTM_Collapse_key"] = (
            data["Protein_group"] + "~" + data["Gene_group"] + "_" + data["Site"] + "_" + data["Mult"]
        )
        data["Protein_Collapse_key"] = (
            data["Protein_name"] + "~" + data["Gene_group"] + "_" + data["Site"] + "_" + data["Mult"]
        )
        
        data = data.drop(["Site", "Mult"], axis=1)
        data["PTM_Collapse_key"] = data["PTM_Collapse_key"].str.replace("#", "_", regex=False)
        data["Gene_group"] = data["Gene_group"].str.replace("#", "_", regex=False)
        return data
    
    def _process_protein_group_output(self, data: pd.DataFrame) -> pd.DataFrame:

        data["Protein_name"] = data["PTM_Collapse_key"].apply(lambda row: row.split("~")[0])
        data["Protein_group"] = data["Protein_name"].str.split(";")
        data["Protein_group"] = data["Protein_group"].apply(lambda row: row[0])
        
        data["PTM"] = data["PTM_Collapse_key"].apply(lambda row: row.split("~")[1])
        data[["Gene_name", "Site", "Mult"]] = data["PTM"].str.split("_", expand=True)
        data["Gene_name"] = data["Gene_name"].str.split(";")
        data["Gene_group"] = data["Gene_name"].apply(lambda row: row[0])
        
        data = data.drop(
            ["Gene_name", "Protein_name", "PTM", "PTM_Collapse_key", "PG.Genes", "PG.ProteinGroups"],
            axis=1
        )
        
        data["PTM_Collapse_key"] = (
            data["Protein_group"] + "~" + data["Gene_group"] + "_" + data["Site"] + "_" + data["Mult"]
        )
        
        data = data.drop(["Site", "Mult"], axis=1)
        data["PTM_Collapse_key"] = data["PTM_Collapse_key"].str.replace("#", "_", regex=False)
        data["Gene_group"] = data["Gene_group"].str.replace("#", "_", regex=False)
        return data
    
    def _load_fasta_to_dict(self, fasta_path: str) -> dict:

        try:
            with open(fasta_path, "r") as file:
                fasta_dict = {}
                current_id = None
                current_sequence = []
                
                for line in file:
                    line = line.strip()
                    if line.startswith(">"):
                        if current_id is not None:
                            fasta_dict[current_id] = "".join(current_sequence)
                        
                        try:
                            header_parts = line[1:].split("|")
                            if len(header_parts) >= 2:
                                current_id = header_parts[1]
                            else:
                                current_id = line[1:].split()[0]
                            current_sequence = []
                        except Exception as e:
                            raise ValueError(f"Invalid FASTA header format: {line}") from e
                    
                    elif line and current_id is not None:
                        current_sequence.append(line.upper())
                
                if current_id is not None:
                    fasta_dict[current_id] = "".join(current_sequence)
                
                if not fasta_dict:
                    raise ValueError("No valid protein sequences found in FASTA file")
                
                return fasta_dict
        
        except FileNotFoundError:
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        except Exception as e:
            raise ValueError(f"Error reading FASTA file: {e}")
    
    def _generate_kinase_sequences(self, data: pd.DataFrame, window_size: int = 6) -> pd.DataFrame:
        result_data = data.copy()
        kinase_sequences = []
    
        success_count = 0
        error_count = 0
        mismatch_count = 0
        
        for index, row in data.iterrows():
            try:
                protein_id = row["Protein_group"]
                collapse_key = row["PTM_Collapse_key"]
                ptm_part = collapse_key.split("~")[1]
                site_info = ptm_part.split("_")[1]
                
                amino_acid = site_info[0]
                position = int(site_info[1:])
                
                kinase_seq = self._create_kinase_sequence(
                    protein_id, position, amino_acid, window_size
                )
                
                if kinase_seq.startswith(("FASTA_ERROR:", "POSITION_ERROR:")):
                    error_count += 1
                elif kinase_seq.startswith("SEQUENCE_MISMATCH:"):
                    mismatch_count += 1
                    error_count += 1
                else:
                    success_count += 1
                
                kinase_sequences.append(kinase_seq)
            
            except Exception as e:
                error_msg = f"PARSING_ERROR: Row {index} - {str(e)}"
                kinase_sequences.append(error_msg)
                error_count += 1
        
        result_data["kinase_sequence"] = kinase_sequences
        
        return result_data
    
    def _create_kinase_sequence(self, protein_id: str, position: int, amino_acid: str, window_size: int = 6) -> str:
        if protein_id not in self.fasta_dict:
            warning_msg = f"FASTA_ERROR: Protein '{protein_id}' not found in FASTA dictionary"
            warnings.warn(warning_msg)
            return warning_msg
        
        protein_sequence = self.fasta_dict[protein_id]
        sequence_length = len(protein_sequence)
        zero_indexed_position = position - 1
        
        if zero_indexed_position < 0 or zero_indexed_position >= sequence_length:
            warning_msg = f"POSITION_ERROR: Position {position} out of bounds for protein '{protein_id}' (length: {sequence_length})"
            warnings.warn(warning_msg)
            return warning_msg
        
        actual_amino_acid = protein_sequence[zero_indexed_position]
        if actual_amino_acid != amino_acid.upper():
            warning_msg = f"SEQUENCE_MISMATCH: Expected '{amino_acid}' at position {position} in '{protein_id}', found '{actual_amino_acid}'"
            warnings.warn(warning_msg)
            return warning_msg
        
        start_pos = zero_indexed_position - window_size
        end_pos = zero_indexed_position + window_size
        
        sequence_parts = []
        
        if start_pos < 0:
            sequence_parts.append("_" * abs(start_pos))
            actual_start = 0
        else:
            actual_start = start_pos
        
        actual_end = min(end_pos, sequence_length - 1)
        sequence_parts.append(protein_sequence[actual_start:zero_indexed_position])
        sequence_parts.append(f"*{amino_acid.upper()}*")
        sequence_parts.append(protein_sequence[zero_indexed_position + 1 : actual_end + 1])
        
        if end_pos >= sequence_length:
            missing_chars = end_pos - sequence_length + 1
            sequence_parts.append("_" * missing_chars)
        
        kinase_sequence = "_" + "".join(sequence_parts) + "_"
        return kinase_sequence
    
    def _clean_and_reformat_phospho_data(self, df: pd.DataFrame) -> pd.DataFrame:
        metadata_cols = [
            "UPD_seq", "PTM_localization", "Protein_group", "Gene_group", 
            "PTM_Collapse_key", "kinase_sequence"
        ]
        sample_cols = [col for col in df.columns if col not in metadata_cols]
        
        df_melted = df.melt(
            id_vars=["PTM_Collapse_key"],
            value_vars=sample_cols,
            var_name="sample_name",
            value_name="intensity",
        )
        
        df_reformatted = df_melted.pivot_table(
            index="sample_name", columns="PTM_Collapse_key", values="intensity", aggfunc="first"
        )
        
        df_reformatted.columns.name = None
        df_reformatted.index.name = None
        return df_reformatted