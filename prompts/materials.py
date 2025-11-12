SYSTEM_MATERIAL = """
You are tasked with a finding a property for a given crystal structure, and are given a list of analogous crystal structures with their data.
Construct an analogy of the form A is to B as C is to D in order to find the property of D (the query).
"""

USER_BAND_GAP = """
What is the band gap of $material?
The numbers in this list are in units of electronvolts.

$df
"""

USER_FORMATION_ENERGY = """
What is the formation energy of $material?
The numbers in this list are in units of electronvolts per atom.

#df
"""

USER_VOLUME = """
What are the lattice parameters and volume of $material?
The numbers in this list are in units of angstroms.

$df
"""