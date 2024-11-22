from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import random
import math
from rdkit import RDLogger
import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from visualizer import visualize_tree,print_path,build_tree_graph
import matplotlib.cm as cm
# Disable RDKit warning messages
RDLogger.DisableLog('rdApp.warning')
import pdb
# Disable RDKit error messages (if needed)
RDLogger.DisableLog('rdApp.error')


# Node class definition with score attribute
class Node:
    node_counter = 0  # Class variable to assign unique IDs
    all_nodes = []    # Class variable to store all nodes

    def __init__(self, smiles, parent=None):
        self.id = Node.node_counter
        Node.node_counter += 1

        self.smiles = smiles
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0  # Accumulated reward for MCTS
        self.score = 0.0  # Score for the SMILES
        self.is_terminal = False
        self.fragment = None  # Fragment added to reach this node
        Node.all_nodes.append(self)

    def is_fully_expanded(self, max_atoms):
        
        return count_atoms(self.smiles) >= max_atoms or len(self.children) >= 6
    
    def UCT_select_child(self):
        c = math.sqrt(2)
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            if child.visits == 0:
                return child
            exploitation = child.value / child.visits
            exploration = c * math.sqrt(math.log(self.visits) / child.visits)
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def get_path(self):
        node = self
        path = []
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()
        return path



def generate_formula_dict(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    atom_counts = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol != 'H':
            atom_counts[symbol] = atom_counts.get(symbol, 0) + 1

    return atom_counts

def subtract_formulas(target_formula, current_smiles):
    current_formula = generate_formula_dict(current_smiles)

    remaining_formula = target_formula.copy()
    is_valid = True

    for element, count in current_formula.items():
        if element in remaining_formula:
            remaining_formula[element] -= count
            if remaining_formula[element] < 0:
                is_valid = False
                break
        else:
            # Current molecule has an element not in the target
            is_valid = False
            break

    # Remove elements with zero count
    remaining_formula = {k: v for k, v in remaining_formula.items() if v > 0}

    return remaining_formula, is_valid

def filter_fragments_by_formula(fragment_pool, remaining_formula):
    valid_fragments = []
    for frag in fragment_pool:
        frag_formula = generate_formula_dict(frag)
        is_valid = True
        for element, count in frag_formula.items():
            if element not in remaining_formula or count > remaining_formula[element]:
                is_valid = False
                break
        if is_valid:
            valid_fragments.append(frag)
    return valid_fragments

def select_fragments_using_llama(node, fragment_pool, k):
    """
    Uses LLAMA (language model) to select the next k fragments.
    """
    # Prepare the prompt for LLAMA
    prompt = f"Given the molecule {node.smiles}, select {k} diverse fragments to add from the fragment pool: {fragment_pool}. Return a list of SMILES strings of the selected fragments."

    # Call LLAMA to get the selected fragments
    # For demonstration purposes, we'll simulate the LLAMA response
    # Replace this with actual LLAMA API calls

    # Ensure we don't select more fragments than available
    k = min(k, len(fragment_pool))

    # Simulate LLAMA by randomly selecting k fragments
    selected_fragments = random.sample(fragment_pool, k)

    return selected_fragments

def combine_molecules(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return set()

    # Get atoms with implicit hydrogens in molecule 1
    atom_indices_1 = [atom.GetIdx() for atom in mol1.GetAtoms() if atom.GetNumImplicitHs() > 0]
    # Get atoms with implicit hydrogens in molecule 2
    atom_indices_2 = [atom.GetIdx() for atom in mol2.GetAtoms() if atom.GetNumImplicitHs() > 0]

    unique_molecules = set()

    for idx1 in atom_indices_1:
        for idx2 in atom_indices_2:
            # Combine molecules into one editable molecule
            combined_mol = Chem.EditableMol(Chem.CombineMols(mol1, mol2))
            # Adjust the atom index for the second molecule
            num_atoms1 = mol1.GetNumAtoms()
            # Add a bond between the specified atoms
            combined_mol.AddBond(idx1, num_atoms1 + idx2, order=Chem.rdchem.BondType.SINGLE)
            # Create the new combined molecule
            new_mol = combined_mol.GetMol()
            # Generate canonical SMILES to ensure uniqueness
            try:
                Chem.SanitizeMol(new_mol)
                canonical_smiles = Chem.MolToSmiles(new_mol, canonical=True)
                unique_molecules.add(canonical_smiles)
            except:
                continue  # Skip invalid molecules

    return unique_molecules

def count_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return mol.GetNumHeavyAtoms()

def calculate_reward(smiles, target_smiles):
    mol1 = Chem.MolFromSmiles(smiles)
    mol2 = Chem.MolFromSmiles(target_smiles)
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprint(mol1, 2)
    fp2 = AllChem.GetMorganFingerprint(mol2, 2)
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return similarity

def calculate_score(smiles, target_smiles):
    """
    Calculates a score for the given SMILES string.

    Parameters:
    - smiles: The SMILES string of the molecule to score.
    - target_smiles: The SMILES string of the target molecule.

    Returns:
    - score: A float representing the score.
    """
    # Example: Use the Tanimoto similarity as the score
    mol = Chem.MolFromSmiles(smiles)
    target_mol = Chem.MolFromSmiles(target_smiles)
    if mol is None or target_mol is None:
        return 0.0

    fp = AllChem.GetMorganFingerprint(mol, 2)
    target_fp = AllChem.GetMorganFingerprint(target_mol, 2)
    score = DataStructs.TanimotoSimilarity(fp, target_fp)
    return score

def expand(node, fragment_pool, target_formula, max_atoms, k):
    # Compute the remaining formula
    remaining_formula, is_valid = subtract_formulas(target_formula, node.smiles)
    if not is_valid:
        node.is_terminal = True
        return node  # Cannot proceed further

    # Filter fragment pool based on remaining formula
    # filtered_fragments = filter_fragments_by_formula(fragment_pool, remaining_formula)
    filtered_fragments = fragment_pool

    if not filtered_fragments:
        node.is_terminal = True
        return node  # No valid fragments to expand

    # Select k fragments using LLAMA from the filtered fragments
    fragments = select_fragments_using_llama(node, filtered_fragments, k)

    if not fragments:
        node.is_terminal = True
        return node  # No fragments selected

    new_nodes = []

    for fragment in fragments:
        # Combine molecules, getting multiple SMILES strings
        new_smiles_set = combine_molecules(node.smiles, fragment)
        if not new_smiles_set:
            continue  # No valid combinations

        for new_smiles in new_smiles_set:
            # Check if the molecule exceeds the atom count
            if count_atoms(new_smiles) > max_atoms:
                continue  # Skip this molecule

            # Create a new node for each unique SMILES
            new_node = Node(new_smiles, parent=node)
            new_node.fragment = fragment

            # Calculate the score for the new SMILES
            new_node.score = calculate_score(new_smiles, target_smiles)

            node.children.append(new_node)
            new_nodes.append(new_node)

            # Print the path of this new branch along with the score
            print("\nNew branch created:")
            print_path(new_node)
            print(f"Score for SMILES '{new_smiles}': {new_node.score:.4f}")

            # Check for termination condition
            if count_atoms(new_smiles) >= max_atoms:
                new_node.is_terminal = True

    # If no valid new nodes were created, mark node as terminal
    if not new_nodes:
        node.is_terminal = True
        return node

    # Select one of the new nodes for further simulation
    selected_node = random.choice(new_nodes)
    return selected_node

def tree_policy(node, fragment_pool, target_formula, max_atoms, k):
    while not node.is_terminal:
        if not node.is_fully_expanded(max_atoms):
            return expand(node, fragment_pool, target_formula, max_atoms, k)
        else:
            node = node.UCT_select_child()
    return node

def default_policy(node_smiles, fragment_pool, target_smiles, max_atoms):
    current_smiles = node_smiles
    while count_atoms(current_smiles) < max_atoms:
        fragment = random.choice(fragment_pool)
        new_smiles_set = combine_molecules(current_smiles, fragment)
        if not new_smiles_set:
            break
        # Randomly select one of the new SMILES for simulation
        current_smiles = random.choice(list(new_smiles_set))
    reward = calculate_reward(current_smiles, target_smiles)
    return reward

def backup(node, reward):
    while node is not None:
        node.visits += 1
        node.value += reward
        node = node.parent

def MCTS(root_nodes, fragment_pool, target_smiles, target_formula, max_atoms, iterations, k):
    all_nodes = []  # List to collect all nodes
    Node.all_nodes.extend(root_nodes)

    for _ in range(iterations):
        root = random.choice(root_nodes)
        node = tree_policy(root, fragment_pool, target_formula, max_atoms, k)
        reward = default_policy(node.smiles, fragment_pool, target_smiles, max_atoms)
        backup(node, reward)

        # Collect nodes and their scores
        all_nodes.append(node)

    best_roots = sorted(root_nodes, key=lambda n: n.visits, reverse=True)
    return best_roots, all_nodes


# Main execution
if __name__ == "__main__":
    fragment_pool = ['COC', 'O', 'C', 'CO', 'CC', 'CCC']
    target_smiles = 'CCCCOCCCC'
    max_atoms = count_atoms(target_smiles)  # Adjusted for testing purposes
    iterations = 25  # Adjusted for testing purposes
    k = len(fragment_pool)  # Number of fragments to select for diversity

    # Generate target formula
    target_formula = generate_formula_dict(target_smiles)
    root_nodes = [Node(frag) for frag in fragment_pool]
    root_node_ids = [node.id for node in root_nodes]
    print(f"Root node IDs: {root_node_ids}")
    for node in root_nodes:
        node.score = calculate_score(node.smiles, target_smiles)
        if count_atoms(node.smiles) >= max_atoms:
            node.is_terminal = True
    # pdb.set_trace()
    best_roots, all_nodes = MCTS(root_nodes, fragment_pool, target_smiles, target_formula, max_atoms, iterations, k)
    pdb.set_trace()
    # Output results
    for root in best_roots:
        avg_reward = root.value / root.visits if root.visits > 0 else 0
        print(f"\nFragment: {root.smiles}, Visits: {root.visits}, Average Reward: {avg_reward:.4f}, Score: {root.score:.4f}")

    # Save scores to a file
    with open("node_scores.txt", "w") as f:
        for node in all_nodes:
            f.write(f"Node ID: {node.id}, SMILES: {node.smiles}, Score: {node.score:.4f}\n")

    G = build_tree_graph(root_nodes,Node)
    visualize_tree(G)
