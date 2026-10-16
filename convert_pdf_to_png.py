#!/usr/bin/env python3
"""Convert PDF images to PNG format."""

import os
from pathlib import Path
from pdf2image import convert_from_path

REPO = os.path.dirname(os.path.abspath(__file__))


def convert_pdfs_to_png(directory):
    """
    Convert all PDF files in a directory to PNG format.

    Args:
        directory: Path to directory containing PDF files
    """
    directory = Path(directory)

    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return

    # Find all PDF files
    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {directory}")
        return

    print(f"Found {len(pdf_files)} PDF files to convert")

    for pdf_file in pdf_files:
        try:
            print(f"Converting {pdf_file.name}...", end=" ", flush=True)

            # Convert PDF to images (one per page)
            images = convert_from_path(pdf_file, dpi=300)

            # Save each page as PNG
            base_name = pdf_file.stem

            if len(images) == 1:
                # Single page PDF - save with simple name
                png_path = directory / f"{base_name}.png"
                images[0].save(png_path, "PNG")
            else:
                # Multi-page PDF - add page number
                for page_num, image in enumerate(images, 1):
                    png_path = directory / f"{base_name}_page{page_num}.png"
                    image.save(png_path, "PNG")

            print(f"✓ ({len(images)} page{'s' if len(images) != 1 else ''})")

        except Exception as e:
            print(f"✗ Error: {e}")

    print("\nConversion complete!")


if __name__ == "__main__":
    figures_dir = os.path.join(REPO, "semantic_entropy_probes", "probe_results", "figures")
    convert_pdfs_to_png(figures_dir)
