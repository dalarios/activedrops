from pymol import cmd, stored
import pandas as pd
import os
import glob
from multiprocessing import Pool, set_start_method, get_start_method
set_start_method('fork', force=True)

# Dictionaries for single-letter amino acid codes and full atom names
aa_dict = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

atom_dict = {
    'H': 'Hydrogen', 'HE': 'Helium', 'LI': 'Lithium', 'BE': 'Beryllium', 'B': 'Boron',
    'C': 'Carbon', 'N': 'Nitrogen', 'O': 'Oxygen', 'F': 'Fluorine', 'NE': 'Neon',
    'NA': 'Sodium', 'MG': 'Magnesium', 'AL': 'Aluminium', 'SI': 'Silicon', 'P': 'Phosphorus',
    'S': 'Sulfur', 'CL': 'Chlorine', 'K': 'Potassium', 'CA': 'Calcium', 'MN': 'Manganese',
    'FE': 'Iron', 'CO': 'Cobalt', 'NI': 'Nickel', 'CU': 'Copper', 'ZN': 'Zinc',
    # Specific atoms in amino acids
    'CA': 'Carbon alpha', 'CB': 'Carbon beta', 'CG': 'Carbon gamma', 'CD': 'Carbon delta', 
    'CE': 'Carbon epsilon', 'CZ': 'Carbon zeta', 'ND1': 'Nitrogen delta 1', 'NE2': 'Nitrogen epsilon 2', 
    'CD2': 'Carbon delta 2', 'CE1': 'Carbon epsilon 1', 'CG2': 'Carbon gamma 2',
    'OG': 'Oxygen gamma', 'OG1': 'Oxygen gamma 1', 'OD1': 'Oxygen delta 1', 'OD2': 'Oxygen delta 2', 
    'OE1': 'Oxygen epsilon 1', 'OE2': 'Oxygen epsilon 2', 'ND2': 'Nitrogen delta 2',
    'NE': 'Nitrogen epsilon', 'NZ': 'Nitrogen zeta', 'SG': 'Sulfur gamma', 'CE2': 'Carbon epsilon 2', 'CD1': 'Carbon delta 1',
    'NH1': 'Nitrogen eta 1', 'NH2': 'Nitrogen eta 2',
    # Specific atoms in ATP
    'PA': 'Phosphorus alpha', 'PB': 'Phosphorus beta', 'PG': 'Phosphorus gamma',
    'O1A': 'Oxygen 1 alpha', 'O2A': 'Oxygen 2 alpha', 'O3A': 'Oxygen 3 alpha',
    'O1B': 'Oxygen 1 beta', 'O2B': 'Oxygen 2 beta', 'O3B': 'Oxygen 3 beta',
    'O1G': 'Oxygen 1 gamma', 'O2G': 'Oxygen 2 gamma', 'O3G': 'Oxygen 3 gamma',
    'N1': 'Nitrogen 1', 'N3': 'Nitrogen 3', 'N6': 'Nitrogen 6', 'N7': 'Nitrogen 7', 'N9': 'Nitrogen 9',
    'C1': 'Carbon 1', 'C2': 'Carbon 2', 'C3': 'Carbon 3', 'C4': 'Carbon 4', 'C5': 'Carbon 5',
    'C6': 'Carbon 6', 'C7': 'Carbon 7', 'C8': 'Carbon 8', 'C9': 'Carbon 9',
    # Prime atoms (commonly in nucleotides and carbohydrates)
    "C1'": "Carbon 1 prime", "C2'": "Carbon 2 prime", "C3'": "Carbon 3 prime",
    "C4'": "Carbon 4 prime", "C5'": "Carbon 5 prime", "O2'": "Oxygen 2 prime",
    "O3'": "Oxygen 3 prime", "O4'": "Oxygen 4 prime", "O5'": "Oxygen 5 prime"
}

def three_to_one(three_letter_code):
    return aa_dict.get(three_letter_code.strip().upper(), '?')

def get_full_atom_name(atom_symbol):
    return atom_dict.get(atom_symbol.strip().upper(), 'Unknown')

def log_interactions(file_path, threshold=5.0):
    cmd.load(file_path)
    interactions = set()  # Use a set to avoid duplicates

    # Select interacting atoms, including magnesium interactions
    cmd.select("interacting_atoms", f"(chain A within {threshold} of chain C) or (chain A within {threshold} of chain D) or (chain B within {threshold} of chain C) or (chain B within {threshold} of chain D) or (chain A within {threshold} of resn ATP) or (chain B within {threshold} of resn ATP) or (chain A within {threshold} of resn ADP) or (chain B within {threshold} of resn ADP) or (chain A within {threshold} of chain B) or (resn ATP within {threshold} of resn MG) or (resn ADP within {threshold} of resn MG) or (elem MG within {threshold} of all)")
    
    # Iterate over selected atoms and log interactions
    stored.list = []
    cmd.iterate("interacting_atoms", "stored.list.append((chain, resi, resn, name, index))")

    for chain, resi, resn, name, index in stored.list:
        # Log Magnesium interactions
        cmd.select("mg_near", f"elem MG within {threshold} of index {index}")
        stored.mg_list = []
        cmd.iterate("mg_near", "stored.mg_list.append((chain, resi, resn, name, index))")
        for mg_chain, mg_resi, mg_resn, mg_name, mg_index in stored.mg_list:
            distance = cmd.get_distance(f"index {index}", f"index {mg_index}")
            if distance <= threshold:  # Only log interactions within threshold
                interactions.add((os.path.splitext(os.path.basename(file_path))[0], chain, resi, resn, name, mg_name, mg_resn, mg_chain, mg_resi, distance))

        # Interaction with chains C and D
        cmd.select("interaction_near", f"(chain C within {threshold} of index {index}) or (chain D within {threshold} of index {index})")
        stored.interaction_list = []
        cmd.iterate("interaction_near", "stored.interaction_list.append((chain, resi, resn, name, index))")
        for inter_chain, inter_resi, inter_resn, inter_name, inter_index in stored.interaction_list:
            distance = cmd.get_distance(f"index {index}", f"index {inter_index}")
            if distance <= threshold:  # Only log interactions within threshold
                interactions.add((os.path.splitext(os.path.basename(file_path))[0], chain, resi, resn, name, inter_name, inter_resn, inter_chain, inter_resi, distance))
        
        # Interaction with ATP
        cmd.select("atp_near", f"(resn ATP within {threshold} of index {index})")
        stored.atp_list = []
        cmd.iterate("atp_near", "stored.atp_list.append((chain, resi, resn, name, index))")
        for atp_chain, atp_resi, atp_resn, atp_name, atp_index in stored.atp_list:
            distance = cmd.get_distance(f"index {index}", f"index {atp_index}")
            if distance <= threshold:  # Only log interactions within threshold
                interactions.add((os.path.splitext(os.path.basename(file_path))[0], chain, resi, resn, name, atp_name, atp_resn, atp_chain, atp_resi, distance))

        # Interaction with ADP
        cmd.select("adp_near", f"(resn ADP within {threshold} of index {index})")
        stored.adp_list = []
        cmd.iterate("adp_near", "stored.adp_list.append((chain, resi, resn, name, index))")
        for adp_chain, adp_resi, adp_resn, adp_name, adp_index in stored.adp_list:
            distance = cmd.get_distance(f"index {index}", f"index {adp_index}")
            if distance <= threshold:  # Only log interactions within threshold
                interactions.add((os.path.splitext(os.path.basename(file_path))[0], chain, resi, resn, name, adp_name, adp_resn, adp_chain, adp_resi, distance))

        # Interaction between chain A and chain B
        cmd.select("ab_near", f"(chain B within {threshold} of index {index})" if chain == 'A' else f"(chain A within {threshold} of index {index})")
        stored.ab_list = []
        cmd.iterate("ab_near", "stored.ab_list.append((chain, resi, resn, name, index))")
        for ab_chain, ab_resi, ab_resn, ab_name, ab_index in stored.ab_list:
            distance = cmd.get_distance(f"index {index}", f"index {ab_index}")
            if distance <= threshold:  # Only log interactions within threshold
                interactions.add((os.path.splitext(os.path.basename(file_path))[0], chain, resi, resn, name, ab_name, ab_resn, ab_chain, ab_resi, distance))

    cmd.delete("all")  # Clear all selections to prepare for the next file
    return list(interactions)  # Convert set to list

def process_file(file_path, threshold):
    return log_interactions(file_path, threshold)

def process_folder(folder_path, threshold=5.0):
    all_interactions = []
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    model_0_files = []
    for subfolder in subfolders:
        model_0_files.extend(glob.glob(os.path.join(subfolder, "*model_0*.pdb")) + glob.glob(os.path.join(subfolder, "*model_0*.cif")))

    # Use multiprocessing to process files in parallel
    with Pool() as pool:
        results = pool.starmap(process_file, [(file_path, threshold) for file_path in model_0_files])
    
    for result in results:
        all_interactions.extend(result)

    # Create a DataFrame from the interactions
    df = pd.DataFrame(all_interactions, columns=["file", "chain", "resi", "resn", "atom_name", "interacting_atom", "interacting_resn", "interacting_chain", "interacting_resinumber", "distance (angstroms)"])

    # Add columns for amino acid single-letter code and complete atom names
    df['residue_one_letter'] = df['resn'].apply(three_to_one)
    df['full_atom_name'] = df['atom_name'].apply(get_full_atom_name)
    df['interacting_full_atom_name'] = df['interacting_atom'].apply(get_full_atom_name)
    df['interactingresi_oneletter'] = df['interacting_resn'].apply(three_to_one)

    # Save the DataFrame to a CSV file but first remove the first 5 characters from the file name and capitalize the first letter
    df['file'] = df['file'].str[5:].str.capitalize()
    df.to_csv(os.path.join(folder_path, "interactions_chimeras.csv"), index=False)
    print(f"DataFrame has been saved to interactions_chimeras.csv")



folder_path = "../data/3d_predictions/species_seeds/"

# Run the function for a folder with a threshold
process_folder(folder_path, threshold=7)  # Replace with the path to your folder containing PDB or CIF files
