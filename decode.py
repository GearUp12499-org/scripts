from datetime import datetime
import re
import sqlite3
from json import loads

LOG_SOURCE = "robot.log"
DESTINATION_DB = "Motion.db"


with open(LOG_SOURCE, 'r', encoding='utf-8') as f:
    data = f.read()

columns = {"$timestamp"}
arry_names = ['x', 'y', 'z', 'w']

packs = []
for match in re.finditer(r'##([0-9.]*?)##(.*?)##$', data, re.MULTILINE):
    timestamp = int(float(match.group(1)) * 1000)
    json_bundle = loads(match.group(2))
    json_bundle["$timestamp"] = timestamp

    output_bundle = {}
    for k, v in json_bundle.items():
        safecolname = re.sub(r'["\']', '_', k)
        if type(v) == list:
            for i in range(0, len(v)):
                index_name = f'{safecolname}_{arry_names[i] if i < len(arry_names) else str(i+1)}'
                columns.add(index_name)
                output_bundle[index_name] = v[i]
        elif type(v) == int or type(v) == float or type(v) == str or type(v) == bool:
            columns.add(safecolname)
            output_bundle[safecolname] = v
        else:
            raise ValueError(f"cannot generate columns for type {type(v)}")
    packs.append(output_bundle)

tablename = f'Logs_{datetime.now().strftime("%d-%m-%Y_%H:%M:%S")}'
# tablename=input('table name? ')
sorted_cols = sorted(columns)

coldecl = ', '.join(map(lambda x: f"\"{x}\"", sorted_cols))

conn = sqlite3.connect(DESTINATION_DB)
cursor = conn.cursor()
cursor.execute(f"CREATE TABLE \"{tablename}\"({coldecl})")

sql_cmd = f"INSERT INTO \"{tablename}\" VALUES({', '.join(['?'] * len(sorted_cols))})"
print(sql_cmd)
ordered_packages = []
for package in packs:
    splat = []
    for colname in sorted_cols:
        if colname in package:
            splat.append(package[colname])
        else:
            splat.append(None)
    cursor.execute(sql_cmd, splat)

conn.commit()
conn.close()
