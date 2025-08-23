from pymol import cmd
import os
import glob

def extract_chain_a_to_folder():
    input_dir = "../data/3d_predictions/prodigy"
    output_dir = os.path.join(input_dir, "chain_A")
    os.makedirs(output_dir, exist_ok=True)

    # Find all .pdb and .cif files in input_dir
    pdb_files = glob.glob(os.path.join(input_dir, "*.pdb"))
    cif_files = glob.glob(os.path.join(input_dir, "*.cif"))
    all_files = pdb_files + cif_files

    for filepath in all_files:
        filename = os.path.basename(filepath)
        base_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]

        obj_name = f"{base_name}_obj"
        selection = f"{base_name}_chainA_sel"
        output_file = os.path.join(output_dir, f"{base_name}_chainA.pdb")

        # Load into a unique object
        cmd.load(filepath, obj_name)
        cmd.select(selection, f"{obj_name} and chain A")

        # Save selection (always to .pdb for consistency)
        cmd.save(output_file, selection)

        # Cleanup
        cmd.delete(obj_name)
        cmd.delete(selection)

# Run it
extract_chain_a_to_folder()