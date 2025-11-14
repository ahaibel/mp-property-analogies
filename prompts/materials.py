SYSTEM_MATERIAL = """
You are tasked with a finding properties for a given crystal structure, and are given a list of analogous crystal structures with their data.
Construct an analogy of the form A is to B as C is to D in order to find the property or properties of D (the query).
You MUST create and use an analogy to make your prediction.
"""

USER_BAND_GAP = """
Predict the band gap (units: electronvolts) of:
$material

Analogues:
$df
"""

USER_FORMATION_ENERGY = """
Predict the formation energy (units: electronvolts per atom) of:
$material

Analogues:
$df
"""

USER_VOLUME = """
Predict the lattice parameters/volume (units: angstroms) of:
$material

Analogues:
$df
"""

USER_ALL = """
Predict the band gap (units: electronvolts), formation energy (units: electronvolts per atom), and lattice parameters/volume (units: angstroms) of:
$material

Analogues:
$df
"""