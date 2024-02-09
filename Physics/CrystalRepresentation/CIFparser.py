
import numpy as np
import sys
from gemmi import cif
import os
import copy

def RemoveParenthesis(string_a):  
    modified_string = list(string_a)
    try:
        modified_string.remove('(')
        modified_string.remove(')')      
        tmp = ''
        for element in modified_string:
            tmp += element
            modified_string = float(tmp)
    except Exception as e:
        modified_string = string_a
    return modified_string

# Open a file location
__location__ = os.path.join(os.getcwd(),os.path.dirname(__file__))
filepath = os.path.join(__location__,'2010109.cif')
print('')
print('')
print(filepath)

# Select the CIF file to work
doc = cif.read_file(filepath)  # copy all the data from mmCIF file

# Define only one block in CIF file
block = doc.sole_block()  # mmCIF has exactly one block
CIFnumber = block.name
print('CIF Number = ' + block.name)

# Chemical Formula
chemFormula = block.find_value('_chemical_formula_sum')
print('Chemical Formula = ' + chemFormula)

# Space Group
spaceGroup = block.find_value('_space_group_IT_number')
print('Space Group = ' + spaceGroup)
 
# Symmertry Hall
SymmetryHall = block.find_value('_symmetry_space_group_name_Hall')
print('Symmetry Hall = ' + SymmetryHall)

# Length of cell
cell_a = block.find_value('_cell_length_a')
cell_a = RemoveParenthesis(cell_a)

cell_b = block.find_value('_cell_length_b')
cell_b = RemoveParenthesis(cell_b)

cell_c = block.find_value('_cell_length_c')
cell_c = RemoveParenthesis(cell_c)

# Angles of a cell
cell_alpha = block.find_value('_cell_angle_alpha')
cell_alpha = RemoveParenthesis(cell_alpha)

cell_beta = block.find_value('_cell_angle_beta')
cell_beta = RemoveParenthesis(cell_beta)

cell_gamma = block.find_value('_cell_angle_gamma')
cell_gamma = RemoveParenthesis(cell_gamma)

matrix_cell_data = np.array([cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma])
print('')
print('Cell_a Cell_b Cell_c Cell_alpha Cell_beta Cell_gamma')
print(matrix_cell_data)

# Atomic Positions
loop_atoms_positions = block.find_loop('_atom_site_label').get_loop()
loop_atoms_positionsTags = loop_atoms_positions.tags
tableAtomPosition = block.find(['_atom_site_label', '_atom_site_symmetry_multiplicity', '_atom_site_Wyckoff_symbol', 
'_atom_site_fract_x', '_atom_site_fract_y','_atom_site_fract_z', '_atom_site_occupancy'])
tableAtomPosition_rows = len(tableAtomPosition)
tableAtomPosition_col = tableAtomPosition.width()
atoms_positions = []
temp = []
for row in block.find(['_atom_site_label', '_atom_site_symmetry_multiplicity', '_atom_site_Wyckoff_symbol', 
'_atom_site_fract_x', '_atom_site_fract_y','_atom_site_fract_z', '_atom_site_occupancy']):    
    temp = list(row)
    atoms_positions.append(temp)

print('')
print('Elem\t Mult\t Wyck\t x\t y\t z\t sof')
for i in range(len(atoms_positions)) :  
    for j in range(len(atoms_positions[i])) :  
        atoms_positions[i][j] = RemoveParenthesis(atoms_positions[i][j])
        print(atoms_positions[i][j], end="\t") 
    print()     


