"""Plot disease counts from the Maine workbook."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "Maine TBD Counts by Month 2001-2025.xlsx"
)
MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}
DISEASES = {"Anaplasmosis": "Anaplasmosis", "Lyme": "Lyme disease"}
COUNTIES = {
    "Androscoggin",
    "Aroostook",
    "Cumberland",
    "Franklin",
    "Hancock",
    "Kennebec",
    "Knox",
    "Lincoln",
    "Oxford",
    "Penobscot",
    "Piscataquis",
    "Sagadahoc",
    "Somerset",
    "Waldo",
    "Washington",
    "York",
}


def _row_index(data: pd.DataFrame, label: str, start: int = 0) -> int:
    """Return the first row whose label in column A matches exactly."""
    for row in range(start, len(data)):
        value = data.iloc[row, 0]
        if not pd.isna(value) and str(value).strip() == label:
            return row
    raise ValueError(f"Could not find row label {label!r} in the workbook.")


def _monthly_columns(data: pd.DataFrame) -> list[tuple[int, pd.Timestamp]]:
    """Convert the workbook's two header rows into dated monthly columns."""
    years = data.iloc[0].ffill()
    months = data.iloc[1]
    columns = []

    for column in range(1, data.shape[1]):
        month = str(months.iloc[column]).strip()
        year_text = str(years.iloc[column]).strip()
        if month not in MONTHS or "total" in year_text.lower():
            continue
        try:
            year = int(float(year_text))
        except ValueError:
            continue
        columns.append((column, pd.Timestamp(year=year, month=MONTHS[month], day=1)))

    if not columns:
        raise ValueError("No monthly columns were found in the workbook.")
    return columns


def _disease_rows(data: pd.DataFrame) -> dict[str, int]:
    """Find disease aggregate rows by detecting rows followed by a county."""
    rows = {}
    labels = [
        "" if pd.isna(value) else str(value).strip()
        for value in data.iloc[:, 0].tolist()
    ]
    for row in range(2, len(labels) - 1):
        label = labels[row]
        if (
            label
            and label not in COUNTIES
            and label != "Grand Total"
            and labels[row + 1] in COUNTIES
        ):
            rows[label] = row
    if not rows:
        raise ValueError("Could not find disease blocks in the workbook.")
    return rows


def _load_counts(
    path: Path,
    disease_rows: dict[str, int],
    county: str | None = None,
) -> pd.DataFrame:
    """Load counts for selected disease rows, optionally using county detail rows."""
    data = pd.read_excel(path, sheet_name=0, header=None)
    monthly_columns = _monthly_columns(data)
    values: dict[str, list[float]] = {}

    for display_name, disease_row in disease_rows.items():
        row = disease_row
        if county is not None:
            row = _row_index(data, county, start=disease_row + 1)

        disease_values = []
        for column, _ in monthly_columns:
            value = pd.to_numeric(data.iloc[row, column], errors="coerce")
            disease_values.append(0 if pd.isna(value) else value)
        values[display_name] = disease_values

    dates = [date for _, date in monthly_columns]
    counts = pd.DataFrame(values, index=pd.DatetimeIndex(dates, name="Month"))
    counts = counts.groupby(level=0).sum()
    full_index = pd.date_range(counts.index.min(), counts.index.max(), freq="MS", name="Month")
    return counts.reindex(full_index, fill_value=0)


def load_monthly_counts(path: Path, county: str | None = None) -> pd.DataFrame:
    """Load monthly Anaplasmosis and Lyme counts."""
    data = pd.read_excel(path, sheet_name=0, header=None)
    disease_rows = _disease_rows(data)
    selected_rows = {
        display_name: disease_rows[workbook_name]
        for display_name, workbook_name in DISEASES.items()
    }
    return _load_counts(path, selected_rows, county=county)


def load_annual_counts(path: Path, county: str | None = None) -> pd.DataFrame:
    """Load annual counts for every disease in the workbook."""
    data = pd.read_excel(path, sheet_name=0, header=None)
    disease_rows = _disease_rows(data)
    monthly_counts = _load_counts(path, disease_rows, county=county)
    annual_counts = monthly_counts.groupby(monthly_counts.index.to_series().dt.year).sum()
    annual_counts.index.name = "Year"
    return annual_counts


def plot_counts(counts: pd.DataFrame, title: str, output_path: Path) -> None:
    """Create and save a line plot for disease counts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(13, 6.5))
    counts.plot(ax=axis, linewidth=1.8)
    axis.set_title(title)
    axis.set_xlabel("Year" if counts.index.name == "Year" else "Month")
    axis.set_ylabel("Number of cases")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(title="Disease")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to the Excel workbook")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for the generated PNG files",
    )
    parser.add_argument("--show", action="store_true", help="Display the plots after saving them")
    args = parser.parse_args()

    all_counties = load_monthly_counts(args.input)
    waldo = load_monthly_counts(args.input, county="Waldo")
    annual_all_counties = load_annual_counts(args.input)
    annual_waldo = load_annual_counts(args.input, county="Waldo")
    all_counties_path = args.output_dir / "disease_cases_all_counties.png"
    waldo_path = args.output_dir / "disease_cases_waldo.png"
    annual_all_counties_path = args.output_dir / "annual_disease_cases_all_counties.png"
    annual_waldo_path = args.output_dir / "annual_disease_cases_waldo.png"
    plot_counts(all_counties, "Anaplasmosis and Lyme Cases in All Maine Counties", all_counties_path)
    plot_counts(waldo, "Anaplasmosis and Lyme Cases in Waldo County", waldo_path)
    plot_counts(
        annual_all_counties,
        "Annual Cases of All Diseases in All Maine Counties",
        annual_all_counties_path,
    )
    plot_counts(
        annual_waldo,
        "Annual Cases of All Diseases in Waldo County",
        annual_waldo_path,
    )
    print(f"Saved {all_counties_path}")
    print(f"Saved {waldo_path}")
    print(f"Saved {annual_all_counties_path}")
    print(f"Saved {annual_waldo_path}")

    if args.show:
        all_counties.plot(figsize=(13, 6.5), title="Anaplasmosis and Lyme Cases in All Maine Counties")
        waldo.plot(figsize=(13, 6.5), title="Anaplasmosis and Lyme Cases in Waldo County")
        plt.show()


if __name__ == "__main__":
    main()