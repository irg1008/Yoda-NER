import pandas as pd
from os import path
from utils import get_data
import re
from tqdm import tqdm


def clean_column(column: pd.Series) -> pd.Series:
    # Create new row for multiple values marked with "/" o "-"
    column = column.str.split("/", expand=True).stack().reset_index(drop=True)
    column = column.str.split("-", expand=True).stack().reset_index(drop=True)

    # Remove rows with numbers in them
    column = column[~column.str.contains(r"\d")]

    # Remove with less than three characters
    column = column[column.str.len() > 3]

    # Remove rows with sentence length less than 4 words
    column = column[column.str.split().str.len() < 4]

    # Remove all special characters like ", . ; : ! ? \" \'"
    column = column.str.replace(r"[^\w\s]", "")

    # Strip whitespaces
    column = column.str.strip().str.lower()

    # Remove duplicates
    column = column.drop_duplicates().dropna()

    return column


def get_colors(data_colors, base, blacklisted):
    # Append base colors
    data_colors = data_colors.append(base)

    # Create dict for colors
    colors = clean_column(data_colors)

    # Remove blacklisted
    clean_blacklisted_colors = clean_column(blacklisted)
    colors = colors[~colors.isin(clean_blacklisted_colors)]

    return colors


def load_colors(aug_dir):
    colors = pd.read_csv(path.join(aug_dir, "dictionaries/colors/colors.txt"))
    # Return first column
    return colors.iloc[:, 0]


def color_pass(data: pd.DataFrame, colors: pd.Series):
    print("Making color pass")
    for row in tqdm(data.itertuples(), total=len(data)):
        title = row.title

        # Find all colors in title
        found_colors = []

        for c in colors:
            # Check color is in title with regex, find entire word and make it case insensitive
            if re.search(r"\b" + c + r"\b", title, re.IGNORECASE):
                found_colors.append(c)

        found_colors_str = ";".join(found_colors)

        # Change color column
        data.at[row.Index, "color"] = found_colors_str

    return data


def create_dictionary():
    data, aug_dir = get_data()

    # Load base colros (txt with one column with colors)
    base_colors = pd.read_csv(
        path.join(aug_dir, "dictionaries/colors/base.txt"), header=None
    )

    # Blacklisted colors
    blacklisted_colors = pd.read_csv(
        path.join(aug_dir, "dictionaries/colors/blacklist.txt"), header=None
    )

    colors = get_colors(data["color"], base_colors[0], blacklisted_colors[0])

    # Save to file
    colors.to_csv(path.join(aug_dir, "dictionaries/colors/colors.txt"), index=False)


if __name__ == "__main__":
    create_dictionary()
