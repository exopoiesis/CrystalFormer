element_list = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']

element_dict = {value: index for index, value in enumerate(element_list)}

# radioactive elements
radioactive_elements = [ 'Tc', 'Pm', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu',
                         'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh',
                         'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
radioactive_elements_dict = {e: element_dict[e] for e in radioactive_elements}

# noble gas elements
noble_gas = ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', 'Og']
noble_gas_dict = {e: element_dict[e] for e in noble_gas}


def parse_composition_bias(bias_string, atom_types=119):
    """Parse a composition bias string into a numeric array.

    Parameters:
        bias_string: Comma-separated ``Element:weight`` pairs, e.g.
            ``"Fe:2.0,S:1.5,O:-1.0"``.  Positive weight increases
            sampling probability; negative decreases.  ``None`` or
            empty string returns all zeros (no bias).
        atom_types: Length of the returned array (default 119).

    Returns:
        numpy array of shape ``(atom_types,)`` with bias values.

    Raises:
        ValueError: Unknown element symbol or malformed entry.
    """
    import numpy as np
    bias = np.zeros(atom_types)
    if not bias_string:
        return bias
    for item in bias_string.split(","):
        parts = item.strip().split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid bias format: {item!r}. Expected 'Element:weight'"
            )
        element, weight = parts[0].strip(), parts[1].strip()
        if element not in element_dict:
            raise ValueError(f"Unknown element: {element!r}")
        idx = element_dict[element]
        if idx >= atom_types:
            raise ValueError(
                f"Element {element!r} (index {idx}) is outside model "
                f"vocabulary (atom_types={atom_types})"
            )
        bias[idx] = float(weight)
    return bias


if __name__=="__main__":
    print (len(element_list))
    print (element_dict["H"])
        
    atom_types = 119
    wyck_types = 3
    aw_types = (atom_types -1)*(wyck_types -1) + 1
    print (aw_types)
    idx = [element_dict[e] for e in ['H', 'C', 'O']]
    aw_mask = [1] + [1 if ((i-1)%(atom_types-1)+1 in idx) else 0 for i in range(1, aw_types)] # 1 for possible elements
    print (idx )
    print (aw_mask)
    print(radioactive_elements_dict)
    print(noble_gas_dict)
    atom_mask = [1] + [1 if i not in radioactive_elements_dict.values() and i not in noble_gas_dict.values() else 0 for i in range(1, atom_types)]
    print('sampling structure formed by non-radioactive elements and non-noble gas')
    print(atom_mask)



