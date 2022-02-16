"""Global constants."""

periodic_table = {
    1: 'H',
    6: 'C',
    7: 'N',
    8: 'O'
}
"""Maps atomic number to atomic symbol"""

periodic_table_inv = dict(zip(periodic_table.values(), periodic_table.keys()))
"""Maps atomic symbol to atomic number"""