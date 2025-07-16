import os
import glob
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select

class ChainASelector(Select):
    """Select only chain A residues"""
    def accept_chain(self, chain):
        return chain.id == "A"
    
    def accept_residue(self, residue):
        return residue.get_parent().id == "A"

def extract_chain_a_to_folder():
    input_dir = "../data/3d_predictions/prodigy"
    output_dir = os.path.join(input_dir, "chain_A")
    os.makedirs(output_dir, exist_ok=True)

    # Find all .pdb and .cif files in input_dir
    pdb_files = glob.glob(os.path.join(input_dir, "*.pdb"))
    cif_files = glob.glob(os.path.join(input_dir, "*.cif"))
    all_files = pdb_files + cif_files

    print(f"Found {len(all_files)} files to process")
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        base_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1].lower()
        
        output_file = os.path.join(output_dir, f"{base_name}_chainA.pdb")
        
        print(f"Processing: {filename}")
        
        try:
            # Choose appropriate parser based on file extension
            if ext == ".pdb":
                parser = PDBParser(QUIET=True)
            elif ext == ".cif":
                parser = MMCIFParser(QUIET=True)
            else:
                print(f"  SKIP: Unsupported file format {ext}")
                continue
            
            # Parse the structure
            structure = parser.get_structure("structure", filepath)
            
            # Check available chains
            chains = list(structure.get_chains())
            chain_ids = [chain.id for chain in chains]
            print(f"  Available chains: {chain_ids}")
            
            # Check if chain A exists
            if "A" not in chain_ids:
                print(f"  WARNING: No chain A found in {filename}")
                continue
            
            # Get chain A
            chain_a = structure[0]["A"]
            residue_count = len(list(chain_a.get_residues()))
            print(f"  Chain A has {residue_count} residues")
            
            if residue_count == 0:
                print(f"  WARNING: Chain A is empty in {filename}")
                continue
            
            # Save chain A to PDB file
            io = PDBIO()
            io.set_structure(structure)
            io.save(output_file, ChainASelector())
            
            print(f"  Saved to: {output_file}")
            
        except Exception as e:
            print(f"  ERROR processing {filename}: {str(e)}")
    
    print("Processing complete!")

if __name__ == "__main__":
    extract_chain_a_to_folder() 