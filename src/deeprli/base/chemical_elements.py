class ChemicalElements():
  chemical_symbols = [
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

  atomic_numbers = {}
  for Z, symbol in enumerate(chemical_symbols):
    atomic_numbers[symbol] = Z

  periodic_table = """
H,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,He
Li,Be,1,1,1,1,1,1,1,1,1,1,B,C,N,O,F,Ne
Na,Mg,1,1,1,1,1,1,1,1,1,1,Al,Si,P,S,Cl,Ar
K,Ca,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Ga,Ge,As,Se,Br,Kr
Rb,Sr,Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,In,Sn,Sb,Te,I,Xe
Cs,Ba,Lu,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi,Po,At,Rn
"""

  periodic_table_loc = dict()
  for i, period in enumerate(periodic_table.split()):
    for j, element in enumerate(period.split(",")):
      periodic_table_loc[element] = (i, j)

  alkali_metals = [
    'Li',
    'Na',
    'K' ,
    'Rb',
    'Cs',
    'Hf']

  alkaline_earth_metals = [
    'Be',
    'Mg',
    'Ca',
    'Sr',
    'Ba',
    'Ra']

  transition_metals = [
    'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y' , 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
          'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
          'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn']

  other_metals = [
    'Al',
    'Ga',
    'In', 'Sn',
    'Tl', 'Pb', 'Bi',
    'Nh', 'Fl', 'Mc', 'Lv']

  metalloids = [
    'B' ,
          'Si',
          'Ge', 'As',
                'Sb', 'Te',
                      'Po']

  non_metals = [
    'C' , 'N' , 'O' ,
          'P' , 'S' ,
                'Se']

  halogens = [
    'F' ,
    'Cl',
    'Br',
    'I' ,
    'At',
    'Ts']

  noble_gases = [
    'He',
    'Ne',
    'Ar',
    'Kr',
    'Xe',
    'Rn',
    'Og']

  lanthanides = [
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']

  actinides = [
    'Ac', 'Th', 'Pa', 'U' , 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']

  metals = alkali_metals + alkaline_earth_metals + transition_metals + other_metals + lanthanides + actinides
