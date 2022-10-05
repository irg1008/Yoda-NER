from color_dicts import color_pass, load_colors
from size_regex import size_pass
from utils import get_data
from os import path


def main():
    data, aug_dir = get_data()

    # Color pass
    colors = load_colors(aug_dir)
    color_data = color_pass(data, colors)

    # Size pass
    size_data = size_pass(color_data)

    # Save to file
    size_data.to_csv(path.join(aug_dir, "augmented_data.csv"), index=False)


if __name__ == "__main__":
    main()
