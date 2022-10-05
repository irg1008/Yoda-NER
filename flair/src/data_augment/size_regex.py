import pandas as pd
from utils import get_data
import re
from tqdm import tqdm


def find_units(x: str):
    # Find units and sizes:
    # examples: 125mm, 5 inches, 5kg, 5 kg, 5x6x89cm, 5x6x89 cm, 4cm x 5cm x 6cm

    units = []

    # First find units like 2x2x2cm or 2x2x2 cm or 600x8x8mm or 600x8x8 mm
    # Then find units like 2cm x 2cm x 2cm or 2 cm x 2 cm x 2 cm not in previous or 2cm X 2cm X 2cm
    # Then find single units like 2cm or 2 cm or 6kg or 7 inches, etc. Not in tandem like previous (do not accept x or X after number)
    # Then find decimal values or values like 44 1/6 or 44 1/2 or 44 1/4 or 44 1/8
    # Accept decimal values with dot or comma
    units.extend(
        re.findall(
            r"\b([\d\.,]+[\s]?[xX][\s]?[\d\.,]+[\s]?[xX][\s]?[\d\.,]+[\s]?[a-zA-Z]{1,2})\b",
            x,
        )
    )
    units.extend(
        re.findall(
            r"\b([\d\.,]+[\s]?[a-zA-Z]+[\s]?[xX][\s]?[\s]?[a-zA-Z]+[\s]?[xX][\s]?[\s]?[a-zA-Z]{1,2})\b",
            x,
        )
    )
    units.extend(re.findall(r"\b([\d\.,]+[\s]?[a-wyzA-WYZ]{1,2})\b", x))
    units.extend(re.findall(r"\b([\d\.,]+[\s]?[\d\.,/]+)\b", x))

    # Add also sizes XS, S, M, ...
    # One or more X before S or L
    units.extend(re.findall(r"\b(M|X{0,2}[SL])\b", x))

    # Remove units included in others. i.e. 5x5x5cm and 5x5cm
    to_remove = []
    for unit in units:
        for u in units:
            if unit != u and unit in u:
                to_remove.append(unit)

    for r in set(to_remove):
        units.remove(r)

    return units


def size_pass(data: pd.DataFrame):
    # Iter every row and get title and size columns.
    # Find unit values like 125mm or 5kg with and without spaces between value and unit.
    # Correct and match the ones in the size column.
    # Add to the column if does not exist

    print("Making size pass")
    for row in tqdm(data.itertuples(), total=len(data)):
        title, size, color = str(row.title), str(row.size), str(row.color)
        title_units, size_units = find_units(title), find_units(size)

        # Add title units to size units and remove duplicates.
        size_units = list(set(size_units + title_units))

        # TODO?: Add units from size not appearing in original title, maybe at the end?

        # Remove all colors from size_units strings, we don't want double labeled words.
        # We need for color column to be corerct on this one.
        # This is why we run the regex pass on size column after color column.
        # Example: 5kg red, 5kg blue, 5kg green, 5kg black, 5kg white, 5kg yellow
        colors = color.split(";")
        for i, su in enumerate(size_units):
            for c in colors:
                su = str(su).lower().replace(c.lower(), "").strip()
            size_units[i] = su

        str_units = ";".join(set(size_units))
        data.at[row.Index, "size"] = str_units

    return data


if __name__ == "__main__":
    data, aug_dir = get_data()
    revised = size_pass(data)
